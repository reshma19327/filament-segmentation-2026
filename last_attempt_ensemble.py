from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
OUT = ROOT / "benchmark_models_512" / "last_attempt_ensemble"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "full10": ROOT / "benchmark_models_512" / "unetpp_spine_topology_full10" / "best_by_cldice.pth",
    "exp075": ROOT / "benchmark_models_512" / "unetpp_spine_topology_exp075" / "best_by_cldice.pth",
}
THRESHOLDS = np.round(np.arange(0.03, 0.15001, 0.0025), 4)
DEVICE = base.DEVICE

for path in CHECKPOINTS.values():
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

_, val_records = base.load_split()
loader = DataLoader(
    base.FilamentSpineDataset(val_records),
    batch_size=base.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

models = {}
for label, path in CHECKPOINTS.items():
    model = UNetPPSpine(in_channels=1, base_channels=16).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    models[label] = model

cached = []
print("Generating predictions without TTA...")
with torch.no_grad():
    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].cpu().numpy()[:, 0] > 0.5
        probs_by_model = {}

        for label, model in models.items():
            logits, _ = base.get_outputs(model(images))
            probs_by_model[label] = torch.sigmoid(logits).cpu().numpy()[:, 0]

        ensemble = (probs_by_model["full10"] + probs_by_model["exp075"]) / 2.0

        for j, name in enumerate(batch["name"]):
            cached.append({
                "image": Path(name).name,
                "gt": masks[j],
                "full10": probs_by_model["full10"][j],
                "exp075": probs_by_model["exp075"][j],
                "ensemble": ensemble[j],
            })

        if i % 20 == 0 or i == len(loader):
            print(f"Batches {i}/{len(loader)}")

def score(prob, gt, threshold):
    pred = prob >= threshold
    ps, gs = skeletonize(pred), skeletonize(gt)
    eps = 1e-8
    precision = np.logical_and(ps, gt).sum() / (ps.sum() + eps)
    recall = np.logical_and(gs, pred).sum() / (gs.sum() + eps)
    cldice = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps)
    return float(cldice), float(dice)

results = []
per_image = []

for method in ["full10", "exp075", "ensemble"]:
    for threshold in THRESHOLDS:
        cls, dices = [], []
        for item in cached:
            cl, di = score(item[method], item["gt"], float(threshold))
            cls.append(cl)
            dices.append(di)
            per_image.append({
                "method": method,
                "image": item["image"],
                "threshold": float(threshold),
                "cldice": cl,
                "dice": di,
            })
        results.append({
            "method": method,
            "threshold": float(threshold),
            "mean_cldice": float(np.mean(cls)),
            "mean_dice": float(np.mean(dices)),
            "images": len(cls),
        })

results.sort(key=lambda x: x["mean_cldice"], reverse=True)

summary_path = OUT / "final_ensemble_summary.csv"
image_path = OUT / "final_ensemble_per_image.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

with image_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
    writer.writeheader()
    writer.writerows(per_image)

print("\n" + "=" * 65)
print("FINAL RESULTS: TOP 10")
print("=" * 65)
for rank, r in enumerate(results[:10], 1):
    print(
        f"{rank:2}. {r['method']:8s} threshold={r['threshold']:.4f} "
        f"clDice={r['mean_cldice']:.6f} Dice={r['mean_dice']:.6f}"
    )

best = results[0]
print("\nBEST OBSERVED VALIDATION CONFIGURATION")
print(f"Method: {best['method']}")
print(f"Threshold: {best['threshold']:.4f}")
print(f"Mean clDice: {best['mean_cldice']:.6f}")
print(f"Mean Dice: {best['mean_dice']:.6f}")
print(f"Target 0.80: {'REACHED' if best['mean_cldice'] >= 0.80 else 'NOT REACHED'}")
print(f"\nReports: {summary_path}\n         {image_path}")
print("No checkpoints were modified.")
print("Results are validation-tuned, not an independent test score.")
