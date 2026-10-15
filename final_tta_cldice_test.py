from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
OUT = ROOT / "benchmark_models_512" / "final_tta_test"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "original": ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "best_unetpp_spine_topology_512.pth",
    "full10": ROOT / "benchmark_models_512" / "unetpp_spine_topology_full10" / "best_by_cldice.pth",
    "exp075": ROOT / "benchmark_models_512" / "unetpp_spine_topology_exp075" / "best_by_cldice.pth",
}
THRESHOLDS = np.round(np.arange(0.03, 0.151, 0.0025), 4)
DEVICE = base.DEVICE

missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

_, val_records = base.load_split()
loader = DataLoader(
    base.FilamentSpineDataset(val_records),
    batch_size=base.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

def score(prob, gt, threshold):
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
    print(f"\nEvaluating {label} with 4-view flip TTA...")
    model = UNetPPSpine(in_channels=1, base_channels=16).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    cached = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            image = batch["image"].to(DEVICE)
            gt_batch = batch["mask"].cpu().numpy()[:, 0] > 0.5

            # Original, horizontal flip, vertical flip, both flips.
            views = [
                (image, ()),
                (torch.flip(image, dims=[3]), (3,)),
                (torch.flip(image, dims=[2]), (2,)),
                (torch.flip(image, dims=[2, 3]), (2, 3)),
            ]
            restored = []
            for view, dims in views:
                logits, _ = base.get_outputs(model(view))
                prob = torch.sigmoid(logits)
                if dims:
                    prob = torch.flip(prob, dims=list(dims))
                restored.append(prob)

            avg_prob = torch.stack(restored, dim=0).mean(dim=0).cpu().numpy()[:, 0]

            for name, prob, gt in zip(batch["name"], avg_prob, gt_batch):
                cached.append((Path(name).name, prob, gt))

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(f"  Batches {batch_idx}/{len(loader)}")

    for threshold in THRESHOLDS:
        cls, dices = [], []
        for name, prob, gt in cached:
            cl, di = score(prob, gt, float(threshold))
            cls.append(cl)
            dices.append(di)
            per_image.append({
                "checkpoint": label,
                "image": name,
                "threshold": float(threshold),
                "cldice": cl,
                "dice": di,
            })
        summary.append({
            "checkpoint": label,
            "threshold": float(threshold),
            "mean_cldice": float(np.mean(cls)),
            "mean_dice": float(np.mean(dices)),
            "images": len(cls),
            "method": "4-view flip TTA",
        })

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

summary.sort(key=lambda r: r["mean_cldice"], reverse=True)

summary_path = OUT / "final_tta_threshold_summary.csv"
per_image_path = OUT / "final_tta_per_image.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

with per_image_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
    writer.writeheader()
    writer.writerows(per_image)

print("\n" + "=" * 68)
print("FINAL TTA RESULTS — VALIDATION SET")
print("=" * 68)
for i, row in enumerate(summary[:10], 1):
    print(
        f"{i:2}. {row['checkpoint']:8s} "
        f"threshold={row['threshold']:.4f} "
        f"clDice={row['mean_cldice']:.6f} "
        f"Dice={row['mean_dice']:.6f}"
    )

best = summary[0]
print("\nBEST RESULT")
print(f"Checkpoint: {best['checkpoint']}")
print(f"Threshold:  {best['threshold']:.4f}")
print(f"Mean clDice: {best['mean_cldice']:.6f}")
print(f"Mean Dice:   {best['mean_dice']:.6f}")
print(f"Target >= 0.800000: {'REACHED' if best['mean_cldice'] >= 0.80 else 'NOT REACHED'}")
print(f"\nReports:\n{summary_path}\n{per_image_path}")
print("No training was performed; existing checkpoints were not modified.")
print("This threshold is tuned on the validation set, not an independent test set.")
