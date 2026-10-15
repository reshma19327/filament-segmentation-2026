from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
OUT = ROOT / "benchmark_models_512" / "checkpoint_comparison"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.07

CHECKPOINTS = {
    "original": ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "best_unetpp_spine_topology_512.pth",
    "full10": ROOT / "benchmark_models_512" / "unetpp_spine_topology_full10" / "best_by_cldice.pth",
    "exp075": ROOT / "benchmark_models_512" / "unetpp_spine_topology_exp075" / "best_by_cldice.pth",
}

def score(prob, gt):
    pred = prob >= THRESHOLD
    ps, gs = skeletonize(pred), skeletonize(gt)
    eps = 1e-8
    precision = np.logical_and(ps, gt).sum() / (ps.sum() + eps)
    recall = np.logical_and(gs, pred).sum() / (gs.sum() + eps)
    cl = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps)
    return float(cl), float(dice)

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

results = {}
for label, path in CHECKPOINTS.items():
    model = UNetPPSpine(in_channels=1, base_channels=16).to(base.DEVICE)
    ckpt = torch.load(path, map_location=base.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    rows = []
    print(f"\nEvaluating {label}...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(base.DEVICE)
            masks = batch["mask"].cpu().numpy()[:, 0] > 0.5
            logits, _ = base.get_outputs(model(images))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            for name, prob, gt in zip(batch["name"], probs, masks):
                cl, dice = score(prob, gt)
                rows.append({
                    "checkpoint": label,
                    "image": Path(name).name,
                    "threshold": THRESHOLD,
                    "cldice": cl,
                    "dice": dice,
                })
    results[label] = rows
    print(f"Images: {len(rows)} | clDice: {np.mean([r['cldice'] for r in rows]):.6f} | Dice: {np.mean([r['dice'] for r in rows]):.6f}")

csv_path = OUT / "three_checkpoint_comparison_007.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["checkpoint", "image", "threshold", "cldice", "dice"])
    writer.writeheader()
    for label in CHECKPOINTS:
        writer.writerows(results[label])

by_image = {
    label: {r["image"]: r for r in rows}
    for label, rows in results.items()
}
comparison_path = OUT / "per_image_change_full10_vs_exp075.csv"
with comparison_path.open("w", newline="", encoding="utf-8") as f:
    fields = ["image", "full10_cldice", "exp075_cldice", "cldice_change",
              "full10_dice", "exp075_dice", "dice_change"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for image in by_image["full10"]:
        a, b = by_image["full10"][image], by_image["exp075"][image]
        writer.writerow({
            "image": image,
            "full10_cldice": a["cldice"],
            "exp075_cldice": b["cldice"],
            "cldice_change": b["cldice"] - a["cldice"],
            "full10_dice": a["dice"],
            "exp075_dice": b["dice"],
            "dice_change": b["dice"] - a["dice"],
        })

print("\n" + "=" * 60)
print("SUMMARY (same 142-image validation set, threshold 0.07)")
print("=" * 60)
for label, rows in results.items():
    print(f"{label:10s} clDice={np.mean([r['cldice'] for r in rows]):.6f}  Dice={np.mean([r['dice'] for r in rows]):.6f}")

changes = [
    by_image["exp075"][name]["cldice"] - by_image["full10"][name]["cldice"]
    for name in by_image["full10"]
]
print(f"\nexp075 vs full10: improved={sum(x > 1e-8 for x in changes)}, worsened={sum(x < -1e-8 for x in changes)}, unchanged={sum(abs(x) <= 1e-8 for x in changes)}")
print("\nFive largest clDice improvements (exp075 vs full10):")
for name, delta in sorted(zip(by_image["full10"], changes), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {delta:+.6f}")
print("\nFive largest clDice declines (exp075 vs full10):")
for name, delta in sorted(zip(by_image["full10"], changes), key=lambda x: x[1])[:5]:
    print(f"  {name}: {delta:+.6f}")
print(f"\nSaved reports:\n{csv_path}\n{comparison_path}")
print("No checkpoints were modified.")
