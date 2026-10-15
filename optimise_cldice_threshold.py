"""Threshold sweep for the existing MAGFiLO clDice checkpoint.


This does NOT train the model or modify any checkpoint/prediction files.
It evaluates the fixed 142-image validation split at multiple probability
thresholds using the same soft_skel-based hard clDice metric as the training
script. Threshold selection on validation can be optimistic; confirm the
chosen threshold on a separate holdout if available.
"""
from pathlib import Path
import importlib.util
import csv


import torch
from torch.utils.data import DataLoader




ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_unetpp_spine_512_cldice_smoothfix.py"


if not TRAIN_SCRIPT.is_file():
    raise FileNotFoundError(
        f"Training script not found: {TRAIN_SCRIPT}\n"
        "Place this script beside train_unetpp_spine_512_cldice_smoothfix.py."
    )


spec = importlib.util.spec_from_file_location("cldice_training_module", TRAIN_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import training module: {TRAIN_SCRIPT}")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


# Prefer the checkpoint selected by validation clDice.
CHECKPOINT = train.BEST_CHECKPOINT
if not CHECKPOINT.is_file():
    # Fall back to the latest checkpoint only if the best-clDice checkpoint
    # does not exist.
    CHECKPOINT = train.LAST_CHECKPOINT
if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        "No clDice checkpoint found. Expected one of:\n"
        f"  {train.BEST_CHECKPOINT}\n"
        f"  {train.LAST_CHECKPOINT}"
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Checkpoint: {CHECKPOINT}")


# Use exactly the training script's dataset matching and fixed benchmark split.
records = train.discover_dataset()
_, val_records = train.split_records(records)
val_dataset = train.FilamentSpineDataset(
    val_records, image_size=train.IMAGE_SIZE, augment=False
)
val_loader = DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda")
)


model = train.UNetPPSpine(in_channels=1, base_channels=16).to(DEVICE)
train.load_checkpoint(model, CHECKPOINT, DEVICE)
model.eval()


# Dense threshold sweep: enough resolution to identify a useful operating point.
thresholds = [round(x, 3) for x in
              [0.10, 0.15, 0.20, 0.25, 0.30, 0.325, 0.35, 0.375,
               0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575,
               0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775,
               0.80, 0.825, 0.85, 0.875, 0.90]]


# Cache probabilities and targets once so the model is only run one time per image.
cached = []
with torch.inference_mode():
    for idx, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        outputs = model(images)
        filament_logits, _ = train.extract_outputs(outputs)
        probabilities = torch.sigmoid(filament_logits).float().cpu()
        cached.append((probabilities, masks.float().cpu(), list(batch["name"])))
        if idx == 1 or idx % 20 == 0 or idx == len(val_loader):
            print(f"Inference batches: {idx}/{len(val_loader)}")


def metric_for_threshold(probabilities, target, threshold, target_skeleton):
    pred = (probabilities >= threshold).float()
    # Use the training script's own skeletonization implementation.
    pred_skeleton = train.soft_skel(pred, iterations=train.CLDICE_ITERATIONS)

    smooth = 1e-6
    skel_precision = ((pred_skeleton * target).sum(dim=(1, 2, 3)) + smooth) / (
        pred_skeleton.sum(dim=(1, 2, 3)) + smooth
    )
    skel_sensitivity = ((target_skeleton * pred).sum(dim=(1, 2, 3)) + smooth) / (
        target_skeleton.sum(dim=(1, 2, 3)) + smooth
    )
    cldice = (2.0 * skel_precision * skel_sensitivity) / (
        skel_precision + skel_sensitivity + smooth
    )


    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred) * target).sum(dim=(1, 2, 3))
    dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
    return cldice, dice


rows = []
for threshold in thresholds:
    per_image_cldice = []
    per_image_dice = []
    for probabilities, targets, names in cached:
        cldice, dice = metric_for_threshold(probabilities, targets, threshold, train.soft_skel(targets, iterations=train.CLDICE_ITERATIONS))
        per_image_cldice.extend(cldice.tolist())
        per_image_dice.extend(dice.tolist())
    rows.append({
        "threshold": threshold,
        "mean_clDice": sum(per_image_cldice) / len(per_image_cldice),
        "mean_Dice": sum(per_image_dice) / len(per_image_dice),
        "images": len(per_image_cldice),
    })


rows.sort(key=lambda row: row["mean_clDice"], reverse=True)
out_path = ROOT / "cldice_threshold_sweep_results.csv"
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["threshold", "mean_clDice", "mean_Dice", "images"])
    writer.writeheader()
    writer.writerows(rows)


print("\n" + "=" * 76)
print("TOP THRESHOLDS BY VALIDATION clDice (PROVISIONAL)")
print("=" * 76)
for row in rows[:10]:
    print(
        f"threshold={row['threshold']:.3f} | "
        f"clDice={row['mean_clDice']:.6f} | "
        f"Dice={row['mean_Dice']:.6f} | images={row['images']}"
    )
default_row = next((row for row in rows if row["threshold"] == 0.5), None)
if default_row:
    print(
        f"\nThreshold 0.500 baseline: clDice={default_row['mean_clDice']:.6f}, "
        f"Dice={default_row['mean_Dice']:.6f}"
    )
print(f"\nSaved full sweep: {out_path}")
print("No model weights or prediction files were changed.")









