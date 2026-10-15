"""
Consolidated MAGFiLO evaluation audit.
Run from the project root with the project's active virtual environment:
    python consolidated_project_audit.py


This script reuses existing predictions, never retrains models, and writes new
reports/visualizations under consolidated_audit_512/. It does not alter the
existing baseline or union-mask CSVs.
"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


import cv2
import numpy as np


ROOT = Path.cwd()
EVALUATOR_FILE = ROOT / "unified_evaluation_7_models_512.py"
SPLIT_FILE = ROOT / "benchmark_512" / "benchmark_split.json"
ORIGINAL_MASK_DIR = ROOT / "data" / "full_masks"
UNION_MASK_DIR = ROOT / "data" / "audited_masks_v1" / "filament_masks"
OUTPUT_DIR = ROOT / "consolidated_audit_512"
PANEL_DIR = OUTPUT_DIR / "visual_panels"
REPORT_JSON = OUTPUT_DIR / "consolidated_audit_report.json"
REPORT_CSV = OUTPUT_DIR / "consolidated_model_metrics.csv"
THRESHOLD_PIXELS = 127


def stop(message):
    raise SystemExit(f"\nSTOP: {message}")


for required in (EVALUATOR_FILE, SPLIT_FILE, ORIGINAL_MASK_DIR, UNION_MASK_DIR):
    if not required.exists():
        stop(f"Required path not found: {required}")


spec = importlib.util.spec_from_file_location("project_evaluator", EVALUATOR_FILE)
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)


def read_binary(path, size=512, threshold=0):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return img > threshold


def metrics_from_counts(tp, tn, fp, fn):
    tp, tn, fp, fn = map(int, (tp, tn, fp, fn))
    total = tp + tn + fp + fn
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    factors = [float(tp + fp), float(tp + fn), float(tn + fp), float(tn + fn)]
    mcc_den = math.sqrt(math.prod(factors))
    observed = (tp + tn) / total if total else 0.0
    expected = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) /
                (total * total)) if total else 0.0
    return {
        "Dice": 2 * tp / dice_den if dice_den else 1.0,
        "IoU": tp / iou_den if iou_den else 1.0,
        "Precision": tp / precision_den if precision_den else 0.0,
        "Recall": tp / recall_den if recall_den else 0.0,
        "MCC": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
        "Cohen_Kappa": ((observed - expected) / (1 - expected)
                        if total and expected < 1 else 0.0),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def skeletonize(mask):
    """Return a one-pixel skeleton if OpenCV contrib or scikit-image is installed."""
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(mask)
    except Exception:
        return None


def cldice_score(pred, gt):
    """Binary clDice using skeleton precision and sensitivity."""
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    if sp is None or sg is None:
        return None
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    tprec = float(np.logical_and(sp, gt).sum()) / max(int(sp.sum()), 1)
    tsens = float(np.logical_and(sg, pred).sum()) / max(int(sg.sum()), 1)
    den = tprec + tsens
    return (2 * tprec * tsens / den) if den else 0.0


def overlay_panel(image_path, original_path, union_path, prediction_path, out_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    orig = read_binary(original_path)
    union = read_binary(union_path)
    pred_img = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    if image is None or orig is None or union is None or pred_img is None:
        return False
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    pred = cv2.resize(pred_img, (512, 512), interpolation=cv2.INTER_NEAREST) > THRESHOLD_PIXELS
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Color conventions: original GT green; union-only additions blue; prediction red.
    panels = []
    for title, gt in (("Original GT", orig), ("Union GT", union), ("Prediction vs union", union)):
        canvas = base.copy()
        if title == "Prediction vs union":
            canvas[gt & pred] = (0, 190, 0)
            canvas[gt & ~pred] = (255, 0, 0)
            canvas[~gt & pred] = (0, 0, 255)
        elif title == "Union GT":
            canvas[orig] = (0, 190, 0)
            canvas[union & ~orig] = (255, 120, 0)
        else:
            canvas[gt] = (0, 190, 0)
        cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        panels.append(canvas)
    strip = np.hstack(panels)
    if not cv2.imwrite(str(out_path), strip):
        return False
    return True


print("\n[1/6] Loading the fixed validation split and checking prediction coverage...")
records = e.load_validation_records()
resolved = e.resolve_validation_files(records)
coverage = []
for cfg in e.MODELS:
    expected = [e.prediction_path(cfg, item["image_name"]) for item in resolved]
    missing = [p.name for p in expected if not p.is_file()]
    coverage.append({
        "model": cfg["name"], "expected": len(expected),
        "present": len(expected) - len(missing), "missing": missing[:10],
        "complete": not missing,
    })
    print(f'  {cfg["name"]}: {len(expected)-len(missing)}/{len(expected)}')
if any(not c["complete"] for c in coverage):
    print("WARNING: one or more prediction sets are incomplete; metrics will mark these as incomplete.")


print("\n[2/6] Comparing original and audited masks on the validation split...")
mask_audit = {"compared": 0, "changed": 0, "mean_iou": None,
              "original_foreground_pixels": 0, "union_foreground_pixels": 0,
              "missing": []}
mask_ious = []
for item in resolved:
    name = Path(item["mask"]).name
    orig_path = ORIGINAL_MASK_DIR / name
    union_path = UNION_MASK_DIR / name
    a = read_binary(orig_path)
    b = read_binary(union_path)
    if a is None or b is None:
        mask_audit["missing"].append(name)
        continue
    inter = int(np.logical_and(a, b).sum())
    denom = int(np.logical_or(a, b).sum())
    mask_ious.append(inter / denom if denom else 1.0)
    mask_audit["compared"] += 1
    mask_audit["changed"] += int(not np.array_equal(a, b))
    mask_audit["original_foreground_pixels"] += int(a.sum())
    mask_audit["union_foreground_pixels"] += int(b.sum())
mask_audit["mean_iou"] = float(np.mean(mask_ious)) if mask_ious else None
mask_audit["additional_union_foreground_pixels"] = (
    mask_audit["union_foreground_pixels"] - mask_audit["original_foreground_pixels"])
print(f'  Compared={mask_audit["compared"]}, changed={mask_audit["changed"]}, '
      f'mean IoU={mask_audit["mean_iou"]}')


print("\n[3/6] Evaluating existing predictions against both mask versions...")
all_rows = []
per_image_cldice = []
for cfg, cov in zip(e.MODELS, coverage):
    counts = {
        "original": [0, 0, 0, 0],
        "union": [0, 0, 0, 0],
    }
    cldice_values = []
    missing = []
    for item in resolved:
        pred_path = e.prediction_path(cfg, item["image_name"])
        original_path = ORIGINAL_MASK_DIR / Path(item["mask"]).name
        union_path = UNION_MASK_DIR / Path(item["mask"]).name
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) if pred_path.is_file() else None
        pred = None if pred_img is None else cv2.resize(
            pred_img, (e.IMAGE_SIZE, e.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) > THRESHOLD_PIXELS
        orig = read_binary(original_path)
        union = read_binary(union_path)
        if pred is None or orig is None or union is None:
            missing.append(item["image_name"])
            continue
        for label, gt in (("original", orig), ("union", union)):
            counts[label][0] += int(np.logical_and(pred, gt).sum())
            counts[label][1] += int(np.logical_and(~pred, ~gt).sum())
            counts[label][2] += int(np.logical_and(pred, ~gt).sum())
            counts[label][3] += int(np.logical_and(~pred, gt).sum())
        score = cldice_score(pred, union)
        if score is not None:
            cldice_values.append(score)
            per_image_cldice.append({
                "Model": cfg["name"], "Image": item["image_name"], "clDice_union": score
            })
    for label in ("original", "union"):
        row = {
            "Model": cfg["name"], "Mask_Version": label,
            **metrics_from_counts(*counts[label]),
            "Images_Evaluated": len(resolved) - len(missing),
            "Status": "SUCCESS" if not missing and cov["complete"] else "INCOMPLETE",
        }
        all_rows.append(row)
    all_rows[-1]["Mean_Per_Image_clDice_Union"] = (
        float(np.mean(cldice_values)) if cldice_values else None)
    print(f'  {cfg["name"]}: original Dice={all_rows[-2]["Dice"]:.6f}; '
          f'union Dice={all_rows[-1]["Dice"]:.6f}; '
          f'mean per-image union clDice=' +
          (f'{np.mean(cldice_values):.6f}' if cldice_values else "UNAVAILABLE"))
    if missing:
        print("    Missing/unreadable examples:", missing[:5])


# Save a compact paired metrics CSV.
metric_fields = [
    "Model", "Mask_Version", "Dice", "IoU", "Precision", "Recall", "MCC",
    "Cohen_Kappa", "TP", "TN", "FP", "FN", "Images_Evaluated", "Status",
    "Mean_Per_Image_clDice_Union"
]
with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metric_fields)
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row.get(k, "") for k in metric_fields})


cldice_csv = OUTPUT_DIR / "per_image_cldice_union.csv"
with cldice_csv.open("w", encoding="utf-8", newline="") as f:
    fields = ["Model", "Image", "clDice_union"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(per_image_cldice)


print("\n[4/6] Checking train/validation image-content leakage with SHA-256...")
split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
train_records = split.get("train", [])
val_records = split.get("validation", split.get("val", []))
image_dir = Path(getattr(e, "IMAGE_DIR", ROOT / "data"))
def locate_image(record):
    name = record.get("image") or record.get("image_path") or record.get("filename") or record.get("file_name")
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return p
    candidates = [image_dir / p.name, ROOT / p]
    for c in candidates:
        if c.is_file():
            return c
    return None
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
train_hashes, missing_train, missing_val = {}, [], []
for r in train_records:
    p = locate_image(r)
    if p is None:
        missing_train.append(str(r))
    else:
        train_hashes.setdefault(sha(p), []).append(p.name)
leaks = []
for r in val_records:
    p = locate_image(r)
    if p is None:
        missing_val.append(str(r))
    else:
        digest = sha(p)
        if digest in train_hashes:
            leaks.append({"validation_image": p.name, "matching_train_names": train_hashes[digest]})
leak_report = {
    "train_records": len(train_records), "validation_records": len(val_records),
    "missing_train_images": len(missing_train), "missing_validation_images": len(missing_val),
    "exact_content_hash_matches": leaks,
    "status": "CHECKED_NO_EXACT_HASH_MATCHES" if not leaks and not missing_train and not missing_val else "REVIEW_REQUIRED",
}
print(f'  Train={len(train_records)}, validation={len(val_records)}, exact duplicate matches={len(leaks)}')
if missing_train or missing_val:
    print(f'  Images not resolved for hashing: train={len(missing_train)}, validation={len(missing_val)}')


print("\n[5/6] Creating visual comparison panels...")
panel_rows = []
# Prioritize masks changed by union; create a panel for each changed validation image
# using the improved model's prediction, then cap at 80 panels for manageable output.
improved_cfg = next((c for c in e.MODELS if "Improved" in c["name"]), e.MODELS[0])
for item in resolved:
    name = Path(item["mask"]).name
    orig_path = ORIGINAL_MASK_DIR / name
    union_path = UNION_MASK_DIR / name
    a, b = read_binary(orig_path), read_binary(union_path)
    if a is None or b is None or np.array_equal(a, b):
        continue
    pred_path = e.prediction_path(improved_cfg, item["image_name"])
    out_path = PANEL_DIR / f'{Path(item["image_name"]).stem}_comparison.png'
    ok = overlay_panel(item["image"], orig_path, union_path, pred_path, out_path)
    if ok:
        panel_rows.append({"image": item["image_name"], "panel": str(out_path.relative_to(ROOT))})
print(f"  Created {len(panel_rows)} panels for changed validation masks.")


print("\n[6/6] Preparing audit report...")
# PQ is intentionally not computed from binary semantic masks. Valid PQ requires
# consistent instance IDs and a documented matching/IoU protocol.
report = {
    "validation_images": len(resolved),
    "prediction_coverage": coverage,
    "mask_audit": mask_audit,
    "metrics_csv": str(REPORT_CSV),
    "per_image_cldice_csv": str(cldice_csv),
    "leakage_check": leak_report,
    "visual_panels_dir": str(PANEL_DIR),
    "visual_panels_created": len(panel_rows),
    "visual_panels": panel_rows,
    "panoptic_quality": {
        "status": "NOT_COMPUTED",
        "reason": "Current evaluation inputs are binary semantic masks. PQ needs a verified instance-ID representation and matching protocol; deriving instances from connected components without validation could produce misleading results."
    },
    "manual_review_required": [
        "Review visual panels for annotation union errors and model failure modes.",
        "Choose and document the primary ground-truth mask version.",
        "Confirm split JSON image paths and any missing hash checks.",
        "Validate clDice implementation and topology conventions before reporting it.",
        "Implement PQ only after instance-ID annotations/predictions and matching rules are established."
    ],
    "notes": [
        "Existing predictions were reused; no inference or training was performed.",
        "Existing baseline and union-mask result files were not modified.",
        "Metrics are aggregate pixel-level metrics; clDice is mean per-image clDice against union masks when a skeletonization backend is available."
    ],
}
REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nDONE.")
print("Metrics CSV:", REPORT_CSV)
print("Per-image clDice CSV:", cldice_csv)
print("Audit JSON:", REPORT_JSON)
print("Visual panels:", PANEL_DIR)
print("PQ: deliberately left pending until instance-ID protocol is verified.")
print("Existing result CSVs and predictions were not modified.")



