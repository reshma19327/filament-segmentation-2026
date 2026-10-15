import csv
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

root = Path("benchmark_models_512/post_training_analysis_stronger")
pred_dir = root / "predictions/cldice_best"
ref_dir = root / "reference"
out = root / "morphology_measurements_verified.csv"

rows = []
for pred_path in sorted(pred_dir.glob("*_filament.png")):
    stem = pred_path.name.removesuffix("_filament.png")
    gt_path = ref_dir / f"{stem}_ground_truth.png"
    if not gt_path.exists():
        print("Missing ground truth:", gt_path)
        continue

    gt = np.asarray(Image.open(gt_path).convert("L")) > 0
    pred = np.asarray(Image.open(pred_path).convert("L")) > 0
    row = {"image": f"{stem}.jpeg"}

    for prefix, mask in [("ground_truth", gt), ("prediction", pred)]:
        row[f"{prefix}_area_pixels"] = int(mask.sum())
        ys, xs = np.where(mask)
        row[f"{prefix}_bbox_width_pixels"] = int(xs.max()-xs.min()+1) if len(xs) else 0
        row[f"{prefix}_bbox_height_pixels"] = int(ys.max()-ys.min()+1) if len(ys) else 0
        length = int(skeletonize(mask).sum())
        row[f"{prefix}_skeleton_length_pixels_approx"] = length
        row[f"{prefix}_mean_width_pixels_approx"] = float(mask.sum()/length) if length else 0.0

    rows.append(row)

if not rows:
    raise RuntimeError("No morphology rows created. Check the reference directory.")

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("Verified morphology CSV:", out)
print("Images processed:", len(rows))
check = next((r for r in rows if r["image"] == "20110221173814Mh.jpeg"), None)
print("Diagnostic image:", check)
