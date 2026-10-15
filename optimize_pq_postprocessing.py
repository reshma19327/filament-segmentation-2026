"""Automated post-processing sweep for existing MAGFiLO binary predictions.


Uses the explicitly selected original provisional evaluator's GT preparation,
deduplication, IoU matrix, and matching functions. Does not retrain the model
or modify original prediction files.
"""
from pathlib import Path
from collections import defaultdict
import csv
import importlib.util


import cv2
import numpy as np




ROOT = Path(__file__).resolve().parent


# IMPORTANT: use the intended evaluator explicitly; do not auto-select another
# file that happens to match the same filename pattern.
EVALUATOR_FILENAME = "provisional_pq_evaluator_clean_cldice_best_512_v2.py"
eval_path = ROOT / EVALUATOR_FILENAME
if not eval_path.is_file():
    raise FileNotFoundError(
        f"Expected evaluator not found: {eval_path}\n"
        "Set EVALUATOR_FILENAME to the exact filename of your original evaluator."
    )


spec = importlib.util.spec_from_file_location("original_pq_evaluator", eval_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load evaluator module: {eval_path}")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


# Use the explicitly selected evaluator's configured paths.
data = ev.load_json(ev.ANNOTATION_JSON)
split = ev.load_json(ev.SPLIT_JSON)
validation = split.get("validation", [])
pred_dir = ev.PREDICTIONS_ROOT / ev.PREDICTION_SUBDIR


if not validation:
    raise RuntimeError("No validation images found in split JSON.")
if not pred_dir.is_dir():
    raise FileNotFoundError(f"Predictions folder not found: {pred_dir}")


records_by_filename = defaultdict(list)
for image_record in data.get("images", []):
    records_by_filename[Path(image_record.get("file_name", "")).name.lower()].append(image_record)


anns_by_id = defaultdict(list)
for annotation in data.get("annotations", []):
    anns_by_id[annotation.get("image_id")].append(annotation)


# Cache GT instances and unprocessed binary predictions once.
cache = []
for n, item in enumerate(validation, 1):
    filename = Path(item["image"]).name
    stem = Path(filename).stem


    possible_paths = [pred_dir / f"{stem}_filament.png"]
    mask_name = item.get("mask")
    if mask_name:
        possible_paths.append(pred_dir / mask_name)
    possible_paths.append(pred_dir / f"{stem}.png")
    pred_path = next((path for path in possible_paths if path.is_file()), None)
    if pred_path is None:
        raise FileNotFoundError(
            f"Missing prediction for {filename}. Tried: "
            + ", ".join(str(path) for path in possible_paths)
        )


    image = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read prediction: {pred_path}")


    binary = (image > 127).astype(np.uint8)
    if binary.shape != (ev.IMAGE_SIZE, ev.IMAGE_SIZE):
        binary = cv2.resize(
            binary,
            (ev.IMAGE_SIZE, ev.IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )


    gt_candidates = []
    for image_record in records_by_filename.get(filename.lower(), []):
        width = int(image_record.get("width", ev.IMAGE_SIZE))
        height = int(image_record.get("height", ev.IMAGE_SIZE))
        for annotation in anns_by_id.get(image_record.get("id"), []):
            raw_mask = ev.polygon_to_mask(
                annotation.get("segmentation", []), width, height
            )
            if raw_mask.any():
                gt_candidates.append(ev.resize_binary(raw_mask))


    gt_instances, _ = ev.deduplicate_gt_instances(gt_candidates)
    cache.append((filename, binary, gt_instances))


    if n == 1 or n % 40 == 0 or n == len(validation):
        print(f"Loaded {n}/{len(validation)} images")


# Conservative post-processing candidates.
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))




def transform(mask, mode):
    if mode == "none":
        return mask.copy()
    if mode == "open3":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    if mode == "close3":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    if mode == "openclose3":
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k3)
    if mode == "close5":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    if mode == "erode3":
        return cv2.erode(mask, k3, iterations=1)
    if mode == "dilate3":
        return cv2.dilate(mask, k3, iterations=1)
    raise ValueError(f"Unknown morphology mode: {mode}")




def instances_from_binary(binary, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=ev.CONNECTIVITY
    )
    return [
        labels == label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]




# Full sweep. To verify first, change to modes=["none"] and areas=[34].
modes = ["none", "open3", "close3", "openclose3", "close5", "erode3", "dilate3"]
areas = [20, 25, 30, 32, 33, 34, 35, 36, 38, 40]


results = []
for mode in modes:
    for min_area in areas:
        total_iou = 0.0
        total_tp = total_fp = total_fn = total_pred = 0


        for filename, original, gt_instances in cache:
            pred_binary = transform(original, mode)
            pred_instances = instances_from_binary(pred_binary, min_area)


            # Match the original evaluator API exactly:
            # build_iou_matrix -> match_instances(iou_matrix) -> (matches, tp, fp, fn)
            iou_matrix = ev.build_iou_matrix(gt_instances, pred_instances)
            matches, tp, fp, fn = ev.match_instances(iou_matrix)


            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += len(pred_instances)
            total_iou += sum(match[2] for match in matches)


        denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
        pq = total_iou / denominator if denominator else 0.0
        rq = total_tp / denominator if denominator else 0.0
        sq = total_iou / total_tp if total_tp else 0.0


        results.append({
            "morphology": mode,
            "min_area": min_area,
            "PQ": pq,
            "SQ": sq,
            "RQ": rq,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "predicted_components": total_pred,
        })


    print(f"Finished morphology: {mode}")


results.sort(key=lambda row: row["PQ"], reverse=True)
out_csv = ROOT / "pq_postprocessing_sweep_results.csv"
with out_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)


print("\n===== TOP 10 CONFIGURATIONS (PROVISIONAL) =====")
for row in results[:10]:
    print(
        f"{row['morphology']:10s} min_area={row['min_area']:2d} "
        f"PQ={row['PQ']:.6f} SQ={row['SQ']:.6f} RQ={row['RQ']:.6f} "
        f"TP={row['TP']} FP={row['FP']} FN={row['FN']}"
    )
print(f"\nFull results saved to: {out_csv}")
print("Original prediction files were not changed.")




