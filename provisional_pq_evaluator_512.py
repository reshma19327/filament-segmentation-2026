"""
Provisional PQ/SQ/RQ evaluator for MAGFiLO.
NOT the official competition evaluator.


Assumptions:
- Strict IoU > 0.5 and one-to-one Hungarian matching.
- Duplicate GT polygons across source records are greedily deduplicated at IoU >= 0.5.
- Binary predictions are converted to instances using 8-connected components.
These assumptions must be disclosed; duplicate-record handling and instance generation
are provisional and may differ from the organizer's protocol.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
ANNOTATION_JSON = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SPLIT_JSON = ROOT / "benchmark_512" / "benchmark_split.json"
PREDICTIONS_ROOT = ROOT / "unified_evaluation_512_new_cldice" / "predictions"
PREDICTION_SUBDIR = "unetpp_spine_cldice_smoothfix_best_cldice"
OUTPUT_DIR = ROOT / "pq_provisional_results_512"
IMAGE_SIZE = 512
MATCH_IOU_THRESHOLD = 0.5
DEDUP_IOU_THRESHOLD = 0.5
MIN_INSTANCE_PIXELS = 1
CONNECTIVITY = 8


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def polygon_to_mask(segmentation, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(segmentation, list):
        return mask
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    return mask


def resize_binary(mask):
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        mask = cv2.resize(mask.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def mask_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def deduplicate_gt_instances(instances):
    instances = sorted(instances, key=lambda x: int(x.sum()), reverse=True)
    kept, suppressed = [], 0
    for candidate in instances:
        if any(mask_iou(candidate, existing) >= DEDUP_IOU_THRESHOLD for existing in kept):
            suppressed += 1
        else:
            kept.append(candidate)
    return kept, suppressed


def extract_prediction_instances(pred_path):
    image = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read prediction: {pred_path}")
    binary = image > 127
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=CONNECTIVITY
    )
    instances = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= MIN_INSTANCE_PIXELS:
            instances.append(labels == label)
    return instances


def match_instances(gt_instances, pred_instances):
    if not gt_instances or not pred_instances:
        return [], len(pred_instances), len(gt_instances)
    ious = np.zeros((len(gt_instances), len(pred_instances)), dtype=np.float64)
    for i, gt in enumerate(gt_instances):
        for j, pred in enumerate(pred_instances):
            ious[i, j] = mask_iou(gt, pred)
    gi, pi = linear_sum_assignment(-ious)
    matches = [(int(i), int(j), float(ious[i, j]))
               for i, j in zip(gi, pi) if ious[i, j] > MATCH_IOU_THRESHOLD]
    tp = len(matches)
    return matches, len(pred_instances) - tp, len(gt_instances) - tp


def metric_values(matches, fp, fn):
    tp = len(matches)
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom else 0.0
    sq = float(np.mean([m[2] for m in matches])) if matches else 0.0
    pq = float(sum(m[2] for m in matches) / denom) if denom else 0.0
    return pq, sq, rq


def main():
    data = load_json(ANNOTATION_JSON)
    split = load_json(SPLIT_JSON)
    validation = split.get("validation", [])
    if not validation:
        raise RuntimeError("No validation entries found in split JSON.")


    records_by_filename = defaultdict(list)
    for im in data.get("images", []):
        records_by_filename[Path(im.get("file_name", "")).name.lower()].append(im)
    anns_by_id = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_id[ann.get("image_id")].append(ann)


    pred_dir = PREDICTIONS_ROOT / PREDICTION_SUBDIR
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    image_rows, match_rows, missing = [], [], []
    total_gt_raw = total_gt_kept = total_suppressed = total_pred = 0


    for n, item in enumerate(validation, 1):
        filename = Path(item["image"]).name
        pred_path = pred_dir / item["mask"]
        if not pred_path.exists():
            pred_path = pred_dir / (Path(filename).stem + ".png")
        if not pred_path.exists():
            missing.append(filename)
            continue


        gt_candidates = []
        for im in records_by_filename.get(filename.lower(), []):
            w, h = int(im.get("width", IMAGE_SIZE)), int(im.get("height", IMAGE_SIZE))
            for ann in anns_by_id.get(im.get("id"), []):
                raw = polygon_to_mask(ann.get("segmentation", []), w, h)
                if raw.any():
                    gt_candidates.append(resize_binary(raw))


        gt_before = len(gt_candidates)
        gt_instances, suppressed = deduplicate_gt_instances(gt_candidates)
        pred_instances = extract_prediction_instances(pred_path)
        matches, fp, fn = match_instances(gt_instances, pred_instances)
        pq, sq, rq = metric_values(matches, fp, fn)
        tp = len(matches)


        total_gt_raw += gt_before
        total_gt_kept += len(gt_instances)
        total_suppressed += suppressed
        total_pred += len(pred_instances)


        image_rows.append({
            "image": filename,
            "source_annotation_records": len(records_by_filename.get(filename.lower(), [])),
            "gt_instances_before_dedup": gt_before,
            "gt_instances_after_dedup": len(gt_instances),
            "gt_instances_suppressed_provisionally": suppressed,
            "predicted_connected_components": len(pred_instances),
            "TP": tp, "FP": fp, "FN": fn, "PQ": pq, "SQ": sq, "RQ": rq,
            "matched_IoU_mean": float(np.mean([m[2] for m in matches])) if matches else 0.0,
            "status": "OK"
        })
        for gi, pi, iou in matches:
            match_rows.append({"image": filename, "gt_instance_index": gi,
                               "pred_instance_index": pi, "IoU": iou,
                               "match_rule": "strict IoU > 0.5"})
        if n == 1 or n % 20 == 0 or n == len(validation):
            print(f"Processed {n}/{len(validation)} validation images")


    if missing:
        print("WARNING missing prediction files:", len(missing), missing[:20])
    if not image_rows:
        raise RuntimeError("No images were evaluated.")


    total_tp = sum(r["TP"] for r in image_rows)
    total_fp = sum(r["FP"] for r in image_rows)
    total_fn = sum(r["FN"] for r in image_rows)
    global_matches = [(None, None, row["IoU"]) for row in match_rows]
    pq, sq, rq = metric_values(global_matches, total_fp, total_fn)


    per_image_csv = OUTPUT_DIR / "provisional_pq_per_image.csv"
    matches_csv = OUTPUT_DIR / "provisional_pq_matches.csv"
    summary_json = OUTPUT_DIR / "provisional_pq_summary.json"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    with matches_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["image", "gt_instance_index", "pred_instance_index", "IoU", "match_rule"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(match_rows)


    summary = {
        "status": "PROVISIONAL_NOT_OFFICIAL",
        "validation_images_evaluated": len(image_rows),
        "validation_images_expected": len(validation),
        "missing_prediction_images": missing,
        "image_size": IMAGE_SIZE,
        "match_rule": "strict IoU > 0.5",
        "matching": "Hungarian one-to-one assignment maximizing IoU",
        "prediction_instance_generation": "8-connected components from binary semantic masks",
        "ground_truth_duplicate_handling": "Greedy suppression at mask IoU >= 0.5; provisional heuristic",
        "minimum_prediction_component_pixels": MIN_INSTANCE_PIXELS,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "PQ": pq, "SQ": sq, "RQ": rq,
        "mean_per_image_PQ": float(np.mean([r["PQ"] for r in image_rows])),
        "mean_per_image_SQ": float(np.mean([r["SQ"] for r in image_rows])),
        "mean_per_image_RQ": float(np.mean([r["RQ"] for r in image_rows])),
        "ground_truth_instances_before_dedup_total": total_gt_raw,
        "ground_truth_instances_after_dedup_total": total_gt_kept,
        "ground_truth_instances_suppressed_provisionally": total_suppressed,
        "predicted_connected_components_total": total_pred,
        "matched_instances_total": len(match_rows),
        "limitations": [
            "Not the official competition evaluator.",
            "Duplicate annotation suppression is heuristic and may remove valid instances.",
            "Connected components can split or merge filaments differently from native instance predictions.",
            "Polygon rasterization and resizing may differ from the official pipeline."
        ]
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


    print("\n===== PROVISIONAL PQ/SQ/RQ =====")
    print("Status: PROVISIONAL — NOT OFFICIAL")
    print(f"Images: {len(image_rows)}/{len(validation)}")
    print(f"GT instances raw/kept/suppressed: {total_gt_raw}/{total_gt_kept}/{total_suppressed}")
    print(f"Predicted connected components: {total_pred}")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"PQ={pq:.6f}, SQ={sq:.6f}, RQ={rq:.6f}")
    print(f"Mean per-image PQ={summary['mean_per_image_PQ']:.6f}")
    print("Saved outputs:")
    print(per_image_csv)
    print(matches_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
