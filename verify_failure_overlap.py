from pathlib import Path
import numpy as np
from PIL import Image

base = Path("benchmark_models_512/post_training_analysis_stronger")
gt_dir = base / "reference"
pred_dir = base / "predictions" / "cldice_best"

for stem in [
    "20110221173814Mh",
    "20190805054730Lh",
    "20200105073250Uh",
]:
    gt = np.asarray(Image.open(
        gt_dir / f"{stem}_ground_truth.png"
    ).convert("L")) > 0
    pred = np.asarray(Image.open(
        pred_dir / f"{stem}_filament.png"
    ).convert("L")) > 0

    intersection = int(np.logical_and(gt, pred).sum())
    denom = int(gt.sum() + pred.sum())
    dice = 2 * intersection / denom if denom else 1.0

    print(
        f"{stem}: GT={int(gt.sum())}, Pred={int(pred.sum())}, "
        f"Intersection={intersection}, Dice={dice:.6f}"
    )
