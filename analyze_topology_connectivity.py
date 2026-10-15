from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "benchmark_models_512" / "post_training_analysis_stronger"
PRED_DIR = BASE / "predictions" / "cldice_best"
REF_DIR = BASE / "reference"
OUT = BASE / "topology_diagnostic.csv"

def read_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8)

def component_count(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(n - 1), int(np.sum(areas < 5)), int(np.sum(areas))

def skeletonize(mask):
    # OpenCV morphological skeleton; no extra packages required.
    img = mask.copy() * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
    return (skel > 0).astype(np.uint8)

def skeleton_stats(mask):
    sk = skeletonize(mask)
    n, _, _, _ = cv2.connectedComponentsWithStats(sk, 8)
    # Count skeleton endpoints using 8-neighbour degree.
    neighbours = cv2.filter2D(
        sk, cv2.CV_16S, np.ones((3, 3), np.uint8)
    ) - sk
    endpoints = int(np.sum((sk > 0) & (neighbours == 1)))
    return int(n - 1), endpoints, int(sk.sum())

rows = []
for pred_path in sorted(PRED_DIR.glob("*_filament.png")):
    stem = pred_path.name.replace("_filament.png", "")
    ref_path = REF_DIR / f"{stem}_ground_truth.png"
    pred = read_mask(pred_path)
    gt = read_mask(ref_path)
    if pred is None or gt is None:
        print(f"SKIP missing/unreadable: {stem}")
        continue

    pred_cc, pred_tiny, pred_area = component_count(pred)
    gt_cc, gt_tiny, gt_area = component_count(gt)
    pred_skcc, pred_end, pred_skl = skeleton_stats(pred)
    gt_skcc, gt_end, gt_skl = skeleton_stats(gt)

    rows.append({
        "image": stem,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "pred_components": pred_cc,
        "gt_components": gt_cc,
        "pred_tiny_components": pred_tiny,
        "pred_skeleton_components": pred_skcc,
        "gt_skeleton_components": gt_skcc,
        "pred_skeleton_endpoints": pred_end,
        "gt_skeleton_endpoints": gt_end,
        "pred_skeleton_pixels": pred_skl,
        "gt_skeleton_pixels": gt_skl,
    })

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Images analyzed: {len(rows)}")
print(f"Saved: {OUT}")
print("\nLargest predicted component counts:")
for r in sorted(rows, key=lambda x: x["pred_components"], reverse=True)[:10]:
    print(
        f'{r["image"]}: pred_cc={r["pred_components"]}, '
        f'GT_cc={r["gt_components"]}, '
        f'pred_skel_cc={r["pred_skeleton_components"]}, '
        f'GT_skel_cc={r["gt_skeleton_components"]}'
    )
