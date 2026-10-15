"""
Verified evaluation for the existing seven MAGFiLO segmentation prediction sets.


This script intentionally DOES NOT load checkpoints or generate predictions.
It evaluates existing prediction PNGs against both:
  1) data/full_masks (original masks)
  2) data/audited_masks_v1/filament_masks (audited union masks)


It writes to evaluation_verified_512/ and does not overwrite existing results.
It reports global/micro metrics AND mean per-image metrics separately.
"""


from __future__ import annotations


import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


import cv2
import numpy as np




ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 512
THRESHOLD_U8 = 127
SPLIT_FILE = ROOT / "benchmark_512" / "benchmark_split.json"
ORIGINAL_MASK_DIR = ROOT / "data" / "full_masks"
UNION_MASK_DIR = ROOT / "data" / "audited_masks_v1" / "filament_masks"
PREDICTIONS_ROOT = ROOT / "unified_evaluation_512" / "predictions"
OUTPUT_DIR = ROOT / "evaluation_verified_512"


MODELS = [
    ("UNet++ + Spine + clDice (Improved)", "unetpp_spine_cldice_improved"),
    ("UNet++ Baseline", "unetpp_baseline"),
    ("UNet++ + Spine", "unetpp_spine"),
    ("Attention U-Net", "attention_unet"),
    ("UNeXt", "unext"),
    ("Mask R-CNN", "maskrcnn"),
    ("DeepLabV3", "deeplabv3"),
]




def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)




def basename(value: Any) -> str:
    return Path(str(value)).name




def get_first(record: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value:
            return basename(value)
    return None




def load_records() -> list[dict]:
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"Validation split not found: {SPLIT_FILE}")
    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)


    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = next(
            (data[k] for k in ("validation", "val", "valid") if k in data),
            None,
        )
        if records is None:
            raise ValueError("No validation/val/valid key found in split JSON.")
    else:
        raise ValueError("Unsupported split JSON format.")


    if not isinstance(records, list) or not records:
        raise ValueError("Validation split is empty or invalid.")
    return records




def resolve_file(folder: Path, name: str) -> Path:
    direct = folder / name
    if direct.is_file():
        return direct
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    matches = [p for p in folder.iterdir() if p.is_file() and p.name.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    stem_matches = [p for p in folder.iterdir() if p.is_file() and p.stem.lower() == Path(name).stem.lower()]
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(matches) > 1 or len(stem_matches) > 1:
        raise RuntimeError(f"Ambiguous filename {name!r} in {folder}")
    raise FileNotFoundError(f"Could not resolve {name!r} in {folder}")




def resolve_mask_name(record: dict, image_name: str) -> str:
    mask_name = get_first(record, ["mask", "mask_path", "mask_file", "mask_filename"])
    return mask_name or image_name




def read_binary_mask(path: Path, is_prediction: bool = False) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Could not read image/mask: {path}")
    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    if is_prediction:
        # The saved prediction PNGs are expected to be 0/255 binary masks.
        # >127 matches the original evaluator's interpretation.
        return arr > THRESHOLD_U8
    return arr > 0




def metric_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    return {
        "TP": int(np.logical_and(pred, gt).sum()),
        "TN": int(np.logical_and(~pred, ~gt).sum()),
        "FP": int(np.logical_and(pred, ~gt).sum()),
        "FN": int(np.logical_and(~pred, gt).sum()),
    }




def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0




def metrics_from_counts(c: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (c[k] for k in ("TP", "TN", "FP", "FN"))
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    mcc_den = math.sqrt(max(0.0, float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    mcc = safe_div(tp * tn - fp * fn, mcc_den)
    total = tp + tn + fp + fn
    observed = safe_div(tp + tn, total)
    actual_pos, actual_neg = tp + fn, tn + fp
    pred_pos, pred_neg = tp + fp, tn + fn
    expected = safe_div(actual_pos * pred_pos + actual_neg * pred_neg, total * total)
    kappa = safe_div(observed - expected, 1.0 - expected)
    return {
        "Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall,
        "MCC": mcc, "Cohen_Kappa": kappa,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()




def load_cldice_backend():
    try:
        from skimage.morphology import skeletonize
        return lambda mask: skeletonize(mask.astype(bool)), "skimage.morphology.skeletonize"
    except Exception:
        pass
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        def skeleton(mask):
            src = (mask.astype(np.uint8) * 255)
            return cv2.ximgproc.thinning(src) > 0
        return skeleton, "cv2.ximgproc.thinning"
    return None, "unavailable (install scikit-image for clDice)"




def cldice(pred: np.ndarray, gt: np.ndarray, skeletonize_fn) -> float:
    sp = skeletonize_fn(pred)
    sg = skeletonize_fn(gt)
    tprec = safe_div(float(np.logical_and(sp, gt).sum()), float(sp.sum()))
    tsens = safe_div(float(np.logical_and(sg, pred).sum()), float(sg.sum()))
    return safe_div(2 * tprec * tsens, tprec + tsens)




def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)




def main() -> int:
    banner("VERIFIED EVALUATION — EXISTING PREDICTIONS ONLY")
    print(f"Project root      : {ROOT}")
    print(f"Validation split  : {SPLIT_FILE}")
    print(f"Original masks    : {ORIGINAL_MASK_DIR}")
    print(f"Union masks       : {UNION_MASK_DIR}")
    print(f"Predictions       : {PREDICTIONS_ROOT}")
    print(f"Output            : {OUTPUT_DIR}")
    print("No checkpoint loading or prediction generation will occur.")
    print("Existing result CSV/JSON/TXT files will not be modified.")


    for folder in (ORIGINAL_MASK_DIR, UNION_MASK_DIR, PREDICTIONS_ROOT):
        if not folder.is_dir():
            raise FileNotFoundError(f"Required directory not found: {folder}")


    records = load_records()
    print(f"\nValidation records: {len(records)}")


    resolved = []
    seen_names = set()
    for i, record in enumerate(records, 1):
        image_name = get_first(record, ["image", "image_path", "image_file", "filename", "file_name", "img"])
        if not image_name:
            raise ValueError(f"Could not find image filename in record {i}: {record}")
        if image_name.lower() in seen_names:
            raise ValueError(f"Duplicate validation image filename in split: {image_name}")
        seen_names.add(image_name.lower())
        mask_name = resolve_mask_name(record, image_name)
        original_path = resolve_file(ORIGINAL_MASK_DIR, mask_name)
        union_path = resolve_file(UNION_MASK_DIR, image_name)
        resolved.append({
            "image_name": image_name,
            "original_mask": original_path,
            "union_mask": union_path,
        })
    print(f"Resolved validation image/mask pairs: {len(resolved)}")


    skeletonize_fn, skeleton_backend = load_cldice_backend()
    print(f"clDice skeleton backend: {skeleton_backend}")


    all_rows = []
    per_image_rows = []
    file_checks = []
    for model_name, model_key in MODELS:
        pred_dir = PREDICTIONS_ROOT / model_key
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory missing for {model_name}: {pred_dir}")


        global_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        per_image_metric_rows = {"original": [], "union": []}
        cldice_values = []
        missing = []
        invalid = []


        for item in resolved:
            pred_path = pred_dir / f"{Path(item['image_name']).stem}.png"
            if not pred_path.is_file():
                # Case-insensitive fallback for Windows-safe filename matching.
                try:
                    pred_path = resolve_file(pred_dir, pred_path.name)
                except FileNotFoundError:
                    missing.append(pred_path.name)
                    continue


            raw_pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if raw_pred is None:
                invalid.append(str(pred_path))
                continue
            if raw_pred.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raw_pred = cv2.resize(raw_pred, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            unique_values = np.unique(raw_pred)
            if not np.isin(unique_values, [0, 255]).all():
                invalid.append(f"{pred_path} (non-binary values: {unique_values[:10].tolist()})")
            pred = raw_pred > THRESHOLD_U8


            gt_masks = {
                "original": read_binary_mask(item["original_mask"]),
                "union": read_binary_mask(item["union_mask"]),
            }
            for version, gt in gt_masks.items():
                c = metric_counts(pred, gt)
                for key in global_counts:
                    global_counts[key] += c[key]
                per_image_metric_rows[version].append(metrics_from_counts(c))


            if skeletonize_fn is not None:
                cldice_values.append(cldice(pred, gt_masks["union"], skeletonize_fn))


        if missing or invalid:
            raise RuntimeError(
                f"Prediction integrity failure for {model_name}: "
                f"missing={len(missing)}, invalid={len(invalid)}. "
                f"Missing examples={missing[:5]}, invalid examples={invalid[:3]}"
            )


        for version in ("original", "union"):
            global_version_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            # Re-accumulate this version independently (not shared between GT versions).
            for item in resolved:
                pred_path = pred_dir / f"{Path(item['image_name']).stem}.png"
                if not pred_path.is_file():
                    pred_path = resolve_file(pred_dir, pred_path.name)
                pred = read_binary_mask(pred_path, is_prediction=True)
                gt = read_binary_mask(item[f"{version}_mask"])
                c = metric_counts(pred, gt)
                for key in global_version_counts:
                    global_version_counts[key] += c[key]


            global_metrics = metrics_from_counts(global_version_counts)
            image_rows = per_image_metric_rows[version]
            mean_metrics = {
                f"MeanImage_{key}": float(np.mean([r[key] for r in image_rows]))
                for key in ("Dice", "IoU", "Precision", "Recall", "MCC", "Cohen_Kappa")
            }
            row = {
                "Model": model_name,
                "Model_Key": model_key,
                "Ground_Truth": version,
                "N_Images": len(image_rows),
                "Aggregation": "global_pixel_micro",
                **global_metrics,
                **mean_metrics,
                "Mean_Per_Image_clDice_Union": (
                    float(np.mean(cldice_values)) if version == "union" and cldice_values else ""
                ),
                "clDice_Backend": skeleton_backend if version == "union" else "",
                "Status": "SUCCESS",
            }
            all_rows.append(row)


        for idx, item in enumerate(resolved):
            p = pred_dir / f"{Path(item['image_name']).stem}.png"
            pred = read_binary_mask(p, is_prediction=True)
            orig = read_binary_mask(item["original_mask"])
            union = read_binary_mask(item["union_mask"])
            per_image_rows.append({
                "Model": model_name,
                "Model_Key": model_key,
                "Image": item["image_name"],
                "Original_Dice": metrics_from_counts(metric_counts(pred, orig))["Dice"],
                "Original_IoU": metrics_from_counts(metric_counts(pred, orig))["IoU"],
                "Union_Dice": metrics_from_counts(metric_counts(pred, union))["Dice"],
                "Union_IoU": metrics_from_counts(metric_counts(pred, union))["IoU"],
                "Union_clDice": cldice(pred, union, skeletonize_fn) if skeletonize_fn else "",
            })


        file_checks.append({
            "Model": model_name,
            "Model_Key": model_key,
            "Validation_Images": len(resolved),
            "Missing_Predictions": len(missing),
            "Invalid_Predictions": len(invalid),
            "Prediction_Directory": str(pred_dir),
        })


        union_row = next(r for r in all_rows if r["Model_Key"] == model_key and r["Ground_Truth"] == "union")
        orig_row = next(r for r in all_rows if r["Model_Key"] == model_key and r["Ground_Truth"] == "original")
        print(
            f"{model_name}: original global Dice={orig_row['Dice']:.6f}; "
            f"union global Dice={union_row['Dice']:.6f}; "
            f"union mean-image Dice={union_row['MeanImage_Dice']:.6f}; "
            f"union mean-image clDice={union_row['Mean_Per_Image_clDice_Union'] if union_row['Mean_Per_Image_clDice_Union'] != '' else 'N/A'}"
        )


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "verified_model_metrics.csv", all_rows)
    write_csv(OUTPUT_DIR / "verified_per_image_metrics.csv", per_image_rows)
    write_csv(OUTPUT_DIR / "prediction_integrity.csv", file_checks)


    report = {
        "project_root": str(ROOT),
        "validation_count": len(resolved),
        "models": len(MODELS),
        "prediction_integrity": file_checks,
        "metric_definition": {
            "global_metrics": "Confusion counts pooled across all validation pixels (micro/global aggregation).",
            "mean_image_metrics": "Calculate each image's metrics separately, then arithmetic mean across validation images.",
            "prediction_threshold": f"Grayscale prediction PNG > {THRESHOLD_U8} is foreground.",
            "ground_truth_threshold": "Grayscale ground-truth mask > 0 is foreground.",
            "mask_resize": "Nearest-neighbor to 512x512 when needed.",
            "cldice": "Mean per-image clDice against union masks; skeleton backend is recorded.",
        },
        "limitations": [
            "This script evaluates saved predictions only; it cannot prove which checkpoint or preprocessing produced them.",
            "Checkpoint/architecture loading correctness must be verified against training scripts and checkpoint metadata.",
            "SHA-256 exact duplicate checking is not a near-duplicate/perceptual leakage test.",
            "PQ is not computed; instance-ID annotations and a verified matching protocol are required.",
        ],
        "outputs": [
            str(OUTPUT_DIR / "verified_model_metrics.csv"),
            str(OUTPUT_DIR / "verified_per_image_metrics.csv"),
            str(OUTPUT_DIR / "prediction_integrity.csv"),
        ],
    }
    with (OUTPUT_DIR / "verified_evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


    banner("VERIFIED EVALUATION COMPLETE")
    print(f"Metrics CSV         : {OUTPUT_DIR / 'verified_model_metrics.csv'}")
    print(f"Per-image CSV       : {OUTPUT_DIR / 'verified_per_image_metrics.csv'}")
    print(f"Integrity CSV       : {OUTPUT_DIR / 'prediction_integrity.csv'}")
    print(f"Audit JSON          : {OUTPUT_DIR / 'verified_evaluation_report.json'}")
    print("Existing results and predictions were not modified.")
    print("Checkpoint provenance and PQ remain separate verification tasks.")
    return 0




if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise



