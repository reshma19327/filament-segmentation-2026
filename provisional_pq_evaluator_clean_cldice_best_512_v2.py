"""
Provisional PQ/SQ/RQ evaluator for MAGFiLO.
NOT the official competition evaluator.


Assumptions:
- Strict IoU > MATCH_IOU_THRESHOLD and one-to-one matching.
- Duplicate GT masks across source records are greedily deduplicated at
  IoU >= DEDUP_IOU_THRESHOLD (a provisional heuristic).
- Binary predictions are converted to instances using connected components.
- MIN_INSTANCE_PIXELS filters tiny predicted components.


Run with the same settings for fair comparisons. This script deliberately writes
results to an output directory named for MIN_INSTANCE_PIXELS so experiments
do not overwrite each other.
"""


import json
import csv
from pathlib import Path
from collections import defaultdict


import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment




# ------------------------- CONFIGURATION -------------------------
ROOT = Path(__file__).resolve().parent


ANNOTATION_JSON = (
    ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
SPLIT_JSON = ROOT / "benchmark_512" / "benchmark_split.json"
PREDICTIONS_ROOT = (
    ROOT / "benchmark_models_512" / "post_training_analysis_stronger"
    / "predictions"
)
PREDICTION_SUBDIR = "cldice_best"


IMAGE_SIZE = 512
MATCH_IOU_THRESHOLD = 0.5
DEDUP_IOU_THRESHOLD = 0.5


# Test 1, 5, and 10 in separate runs. Start with 1 to reproduce baseline.
MIN_INSTANCE_PIXELS = 34
CONNECTIVITY = 8


# Separate output folder per component-area setting; avoids overwriting results.
OUTPUT_DIR = ROOT / (
    f"pq_provisional_results_clean_cldice_best_512_minarea_{MIN_INSTANCE_PIXELS}"
)
# ----------------------------------------------------------------




def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))




def polygon_to_mask(segmentation, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(segmentation, list):
        return mask


    for polygon in segmentation:
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2
        ):
            continue


        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)


    return mask




def resize_binary(mask):
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(bool)




def mask_iou(a, b):
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return intersection / union if union else 0.0




def deduplicate_gt_instances(instances):
    """Greedy GT deduplication; retained only as an explicitly provisional rule."""
    instances = sorted(instances, key=lambda x: int(x.sum()), reverse=True)
    kept = []
    suppressed = 0


    for candidate in instances:
        if any(
            mask_iou(candidate, existing) >= DEDUP_IOU_THRESHOLD
            for existing in kept
        ):
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
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_INSTANCE_PIXELS:
            instances.append(labels == label)


    return instances




def match_instances(iou_matrix, threshold=MATCH_IOU_THRESHOLD):
    """
    One-to-one matching that prioritizes the number of pairs with IoU > threshold,
    then maximizes IoU among those matches.


    Returns:
        matches: [(gt_index, pred_index, IoU), ...]
        tp, fp, fn
    """
    n_gt, n_pred = iou_matrix.shape


    if n_gt == 0 or n_pred == 0:
        return [], 0, n_pred, n_gt


    valid = iou_matrix > threshold


    # A valid match gets a reward > any possible total IoU tie-break contribution.
    reward = np.where(valid, 2.0 + iou_matrix, 0.0)
    gt_indices, pred_indices = linear_sum_assignment(-reward)


    matches = [
        (int(g), int(p), float(iou_matrix[g, p]))
        for g, p in zip(gt_indices, pred_indices)
        if valid[g, p]
    ]


    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    return matches, tp, fp, fn




def build_iou_matrix(gt_instances, pred_instances):
    matrix = np.zeros((len(gt_instances), len(pred_instances)), dtype=np.float64)
    for gi, gt in enumerate(gt_instances):
        for pi, pred in enumerate(pred_instances):
            matrix[gi, pi] = mask_iou(gt, pred)
    return matrix



def metric_values(matches, fp, fn):
    tp = len(matches)
    denominator = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denominator if denominator else 0.0
    sq = float(np.mean([m[2] for m in matches])) if matches else 0.0
    pq = float(sum(m[2] for m in matches) / denominator) if denominator else 0.0
    return pq, sq, rq




def main():
    data = load_json(ANNOTATION_JSON)
    split = load_json(SPLIT_JSON)
    validation = split.get("validation", [])
    if not validation:
        raise RuntimeError("No validation entries found in split JSON.")


    records_by_filename = defaultdict(list)
    for image_record in data.get("images", []):
        name = Path(image_record.get("file_name", "")).name.lower()
        records_by_filename[name].append(image_record)


    anns_by_id = defaultdict(list)
    for annotation in data.get("annotations", []):
        anns_by_id[annotation.get("image_id")].append(annotation)


    pred_dir = PREDICTIONS_ROOT / PREDICTION_SUBDIR
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    image_rows = []
    match_rows = []
    missing = []
    total_gt_raw = 0
    total_gt_kept = 0
    total_suppressed = 0
    total_pred = 0


    for n, item in enumerate(validation, 1):
        filename = Path(item["image"]).name
        stem = Path(filename).stem


        # Evaluate the filament mask only, not the paired spine mask.
        possible_paths = [
            pred_dir / f"{stem}_filament.png",
            pred_dir / item.get("mask", ""),
            pred_dir / f"{stem}.png",
        ]
        pred_path = next((p for p in possible_paths if p.is_file()), None)
        if pred_path is None:
            missing.append(filename)
            continue


        gt_candidates = []
        for image_record in records_by_filename.get(filename.lower(), []):
            width = int(image_record.get("width", IMAGE_SIZE))
            height = int(image_record.get("height", IMAGE_SIZE))


            for annotation in anns_by_id.get(image_record.get("id"), []):
                raw_mask = polygon_to_mask(
                    annotation.get("segmentation", []), width, height
                )
                if raw_mask.any():
                    gt_candidates.append(resize_binary(raw_mask))


        gt_before = len(gt_candidates)
        gt_instances, suppressed = deduplicate_gt_instances(gt_candidates)
        pred_instances = extract_prediction_instances(pred_path)


        iou_matrix = build_iou_matrix(gt_instances, pred_instances)
        matches, tp, fp, fn = match_instances(iou_matrix)
        pq, sq, rq = metric_values(matches, fp, fn)


        total_gt_raw += gt_before
        total_gt_kept += len(gt_instances)
        total_suppressed += suppressed
        total_pred += len(pred_instances)


        image_rows.append({
            "image": filename,
            "source_annotation_records": len(
                records_by_filename.get(filename.lower(), [])
            ),
            "gt_instances_before_dedup": gt_before,
            "gt_instances_after_dedup": len(gt_instances),
            "gt_instances_suppressed_provisionally": suppressed,
            "predicted_connected_components": len(pred_instances),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "PQ": pq,
            "SQ": sq,
            "RQ": rq,
            "matched_IoU_mean": (
                float(np.mean([m[2] for m in matches]) if matches else 0.0)
            ),
            "status": "OK",
        })


        for gi, pi, iou in matches:
            match_rows.append({
                "image": filename,
                "gt_instance_index": gi,
                "pred_instance_index": pi,
                "IoU": iou,
                "match_rule": f"strict IoU > {MATCH_IOU_THRESHOLD}",
            })


        if n == 1 or n % 20 == 0 or n == len(validation):
            print(f"Processed {n}/{len(validation)} validation images")


    if missing:
        print("ERROR: missing prediction files:", len(missing), missing[:20])
        raise RuntimeError(
            f"Only {len(image_rows)}/{len(validation)} validation images "
            "had predictions; refusing to report incomplete PQ."
        )
    if not image_rows:
        raise RuntimeError("No images were evaluated.")


    total_tp = sum(row["TP"] for row in image_rows)
    total_fp = sum(row["FP"] for row in image_rows)
    total_fn = sum(row["FN"] for row in image_rows)


    # Aggregate PQ from all matched IoUs and pooled TP/FP/FN.
    all_matched_ious = [row["IoU"] for row in match_rows]
    global_matches = [(None, None, iou) for iou in all_matched_ious]
    pq, sq, rq = metric_values(global_matches, total_fp, total_fn)


    per_image_csv = OUTPUT_DIR / "provisional_pq_per_image.csv"
    matches_csv = OUTPUT_DIR / "provisional_pq_matches.csv"
    summary_json = OUTPUT_DIR / "provisional_pq_summary.json"


    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)


    with matches_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "image", "gt_instance_index", "pred_instance_index", "IoU",
            "match_rule",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(match_rows)


    summary = {
        "status": "PROVISIONAL_NOT_OFFICIAL",
        "validation_images_evaluated": len(image_rows),
        "validation_images_expected": len(validation),
        "missing_prediction_images": missing,
        "image_size": IMAGE_SIZE,
        "match_rule": f"strict IoU > {MATCH_IOU_THRESHOLD}",
        "matching": (
            "One-to-one assignment prioritizing valid matches above the IoU "
            "threshold, then maximizing IoU"
        ),
        "prediction_instance_generation": (
            f"{CONNECTIVITY}-connected components from binary semantic masks"
        ),
        "ground_truth_duplicate_handling": (
            f"Greedy suppression at mask IoU >= {DEDUP_IOU_THRESHOLD}; "
            "provisional heuristic"
        ),
        "minimum_prediction_component_pixels": MIN_INSTANCE_PIXELS,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "PQ": pq,
        "SQ": sq,
        "RQ": rq,
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
            "Polygon rasterization and resizing may differ from the official pipeline.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


    print("\n===== PROVISIONAL PQ/SQ/RQ =====")
    print("Status: PROVISIONAL — NOT OFFICIAL")
    print(f"Minimum component area: {MIN_INSTANCE_PIXELS} pixels")
    print(f"Images: {len(image_rows)}/{len(validation)}")
    print(
        "GT instances raw/kept/suppressed: "
        f"{total_gt_raw}/{total_gt_kept}/{total_suppressed}"
    )
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
