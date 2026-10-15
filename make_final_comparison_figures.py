from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path.cwd()
BASE = ROOT / "benchmark_models_512" / "post_training_analysis_stronger"
PRED_DIR = BASE / "predictions" / "cldice_best"
REF_DIR = BASE / "reference"
CSV_PATH = BASE / "morphology_measurements_verified.csv"
OUT_DIR = BASE / "final_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURES = ["20110221173814Mh", "20190805054730Lh"]
SIZE = 256

def read_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 0

def get_image(stem):
    candidates = [
        REF_DIR / f"{stem}_input.png",
        ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images" / f"{stem}.jpeg",
        ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images" / f"{stem}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return Image.open(p).convert("L").resize((SIZE, SIZE))
    return Image.new("L", (SIZE, SIZE), 0)

rows = []
with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

scored = []
for row in rows:
    stem = Path(row["image"]).stem
    gt_path = REF_DIR / f"{stem}_ground_truth.png"
    pred_path = PRED_DIR / f"{stem}_filament.png"
    if not gt_path.exists() or not pred_path.exists():
        continue
    gt, pred = read_mask(gt_path), read_mask(pred_path)
    if gt.shape != pred.shape:
        print("Skipping size mismatch:", stem, gt.shape, pred.shape)
        continue
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    dice = (2 * intersection / denom) if denom else 1.0
    scored.append((dice, stem, gt_path, pred_path))

if not scored:
    raise SystemExit(
        "No matched GT/prediction pairs found. Check REF_DIR and prediction filenames."
    )

scored.sort(key=lambda x: x[0], reverse=True)
selected = []
used = set()

def add(item, label):
    if item[1] not in used:
        selected.append((label, *item))
        used.add(item[1])

# Three highest-Dice examples
for i, item in enumerate(scored[:3], 1):
    add(item, f"Strong prediction {i}")

# One median example
add(scored[len(scored) // 2], "Middle example")

# Explicit failure cases, if present
for stem in FAILURES:
    item = next((x for x in scored if x[1] == stem), None)
    if item:
        add(item, "Failure case: " + stem)

# Lowest-Dice example if not already included
add(scored[-1], "Lowest-Dice example")

panel_w, panel_h = SIZE, SIZE + 30
labels = ["Input", "Ground truth", "Prediction", "Overlay"]
sheet = Image.new("RGB", (panel_w * 4, panel_h * len(selected)), "white")
draw = ImageDraw.Draw(sheet)

for r, (label, dice, stem, gt_path, pred_path) in enumerate(selected):
    inp = get_image(stem).convert("RGB")
    gt = read_mask(gt_path)
    pred = read_mask(pred_path)

    gt_img = Image.fromarray((gt * 255).astype(np.uint8)).resize((SIZE, SIZE))
    pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((SIZE, SIZE))

    # Green = ground truth; red = prediction; yellow = overlap
    overlay = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    overlay[..., 0] = pred_img
    overlay[..., 1] = gt_img
    panels = [inp, gt_img.convert("RGB"), pred_img.convert("RGB"),
              Image.fromarray(overlay)]

    y = r * panel_h
    draw.text((5, y + 5), f"{stem} | {label} | Dice={dice:.3f}", fill="black")
    for c, (title, panel) in enumerate(zip(labels, panels)):
        x = c * panel_w
        sheet.paste(panel, (x, y + 30))
        draw.text((x + 5, y + 10), title, fill="black")

out = OUT_DIR / "representative_comparison_contact_sheet.png"
sheet.save(out)
print("Images processed for selection:", len(scored))
print("Selected examples:")
for label, dice, stem, *_ in selected:
    print(f"  {stem}: Dice={dice:.4f} ({label})")
print("Saved:", out)
