from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
OUT = ROOT / "benchmark_models_512" / "threshold_sweep_all"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "original": ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "best_unetpp_spine_topology_512.pth",
    "full10": ROOT / "benchmark_models_512" / "unetpp_spine_topology_full10" / "best_by_cldice.pth",
    "exp075": ROOT / "benchmark_models_512" / "unetpp_spine_topology_exp075" / "best_by_cldice.pth",
}
THRESHOLDS = [round(x, 3) for x in np.arange(0.03, 0.151, 0.01)]

missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

_, val_records = base.load_split()
loader = DataLoader(
    base.FilamentSpineDataset(val_records),
    batch_size=base.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(base.DEVICE.type == "cuda"),
)

def scores(prob, gt, threshold):
    pred = prob >= threshold
    ps, gs = skeletonize(pred), skeletonize(gt)
    eps = 1e-8
    precision = np.logical_and(ps, gt).sum() / (ps.sum() + eps)
    recall = np.logical_and(gs, pred).sum() / (gs.sum() + eps)
    cldice = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps)
    return float(cldice), float(dice)

summary = []
per_image = []

for label, path in CHECKPOINTS.items():
    print(f"\nEvaluating {label} across {len(THRESHOLDS)} thresholds...")
    model = UNetPPSpine(in_channels=1, base_channels=16).to(base.DEVICE)
    checkpoint = torch.load(path, map_location=base.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    cached = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(base.DEVICE)
            masks = batch["mask"].cpu().numpy()[:, 0] > 0.5
            logits, _ = base.get_outputs(model(images))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            for name, prob, gt in zip(batch["name"], probs, masks):
                cached.append((Path(name).name, prob, gt))

    for threshold in THRESHOLDS:
        cl_values, dice_values = [], []
        for name, prob, gt in cached:
            cldice, dice = scores(prob, gt, threshold)
            cl_values.append(cldice)
            dice_values.append(dice)
            per_image.append({
                "checkpoint": label,
                "image": name,
                "threshold": threshold,
                "cldice": cldice,
                "dice": dice,
            })
        summary.append({
            "checkpoint": label,
            "threshold": threshold,
            "mean_cldice": float(np.mean(cl_values)),
            "mean_dice": float(np.mean(dice_values)),
            "images": len(cl_values),
        })
    del model
    if base.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

summary.sort(key=lambda r: r["mean_cldice"], reverse=True)
summary_file = OUT / "threshold_sweep_summary.csv"
image_file = OUT / "threshold_sweep_per_image.csv"

with summary_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

with image_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
    writer.writeheader()
    writer.writerows(per_image)

print("\n" + "=" * 72)
print("TOP 10 CHECKPOINT / THRESHOLD COMBINATIONS BY MEAN VALIDATION CLDICE")
print("=" * 72)
for i, row in enumerate(summary[:10], 1):
    print(f"{i:2}. {row['checkpoint']:8s} threshold={row['threshold']:.3f} "
          f"clDice={row['mean_cldice']:.6f} Dice={row['mean_dice']:.6f}")

best = summary[0]
print("\nBEST VALIDATION CONFIGURATION")
print(f"Checkpoint: {best['checkpoint']}")
print(f"Threshold:  {best['threshold']:.3f}")
print(f"clDice:     {best['mean_cldice']:.6f}")
print(f"Dice:       {best['mean_dice']:.6f}")
print("\nReports saved:")
print(summary_file)
print(image_file)
print("\nNo training was performed. No model checkpoints were modified.")
print("Threshold is selected on this validation set; confirm on independent test data if available.")
