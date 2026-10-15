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
    gt_path = gt_dir / f"{stem}_ground_truth.png"
    pred_path = pred_dir / f"{stem}_filament.png"

    print(f"\n{stem}")
    for name, path in [("GT", gt_path), ("Prediction", pred_path)]:
        if not path.exists():
            print(f"  {name}: FILE MISSING — {path}")
            continue
        mask = np.asarray(Image.open(path).convert("L")) > 0
        print(f"  {name}: {int(mask.sum())} foreground pixels")
