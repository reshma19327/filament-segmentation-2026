from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

ROOT = Path.cwd()
SPLIT = ROOT / "benchmark_512" / "benchmark_split.json"
PRED_DIR = ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "predictions_all142"
OUT = PRED_DIR.parent / "cldice_evaluation_all142.csv"

with SPLIT.open("r", encoding="utf-8-sig") as f:
    data = json.load(f)

records = data["validation"]
mask_dir = Path(data["mask_dir"])
rows = []

def read_binary(path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Cannot read: {path}")
    return arr > 127

for rec in records:
    name = rec["image"]
    stem = Path(name).stem
    pred_path = PRED_DIR / f"{stem}_filament.png"
    gt_path = mask_dir / rec["mask"]

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction missing: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth mask missing: {gt_path}")

    pred = read_binary(pred_path)
    gt = read_binary(gt_path)

    if pred.shape != gt.shape:
        gt = cv2.resize(
            gt.astype(np.uint8),
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) > 0

    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)
    eps = 1e-8

    top_precision = np.logical_and(pred_skel, gt).sum() / (pred_skel.sum() + eps)
    top_recall = np.logical_and(gt_skel, pred).sum() / (gt_skel.sum() + eps)
    cldice = 2 * top_precision * top_recall / (top_precision + top_recall + eps)
    dice = 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps)

    rows.append({"image": name, "dice": float(dice), "cldice": float(cldice)})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\n========== FULL VALIDATION RESULTS ==========")
print(f"Images evaluated: {len(df)} / {len(records)}")
print(f"Mean Dice:        {df['dice'].mean():.6f}")
print(f"Mean clDice:      {df['cldice'].mean():.6f}")
print(f"Median clDice:    {df['cldice'].median():.6f}")
print(f"Images clDice >= 0.85: {(df['cldice'] >= 0.85).sum()} / {len(df)}")
print(f"Results saved to: {OUT}")
