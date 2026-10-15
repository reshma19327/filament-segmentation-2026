from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
OUT = ROOT / "benchmark_models_512" / "hard_case_visual_inspection"
OUT.mkdir(parents=True, exist_ok=True)

CASES = ["20110205082634Lh", "20181102104630Ch"]
THRESHOLD = 0.08

CHECKPOINTS = {
    "full10": ROOT / "benchmark_models_512" / "unetpp_spine_topology_full10" / "best_by_cldice.pth",
    "exp075": ROOT / "benchmark_models_512" / "unetpp_spine_topology_exp075" / "best_by_cldice.pth",
}

for path in CHECKPOINTS.values():
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

_, val_records = base.load_split()
records = [
    r for r in val_records
    if Path(str(r.get("name", r.get("image", "")))).stem in CASES
]
if len(records) != len(CASES):
    print("Available record keys:", list(val_records[0].keys()) if val_records else [])
    raise RuntimeError(
        f"Found {len(records)} of {len(CASES)} requested images. "
        "Check dataset record naming."
    )

loader = DataLoader(
    base.FilamentSpineDataset(records),
    batch_size=1, shuffle=False, num_workers=0
)

def score(prob, gt):
    pred = prob >= THRESHOLD
    ps, gs = skeletonize(pred), skeletonize(gt)
    eps = 1e-8
    precision = np.logical_and(ps, gt).sum() / (ps.sum() + eps)
    recall = np.logical_and(gs, pred).sum() / (gs.sum() + eps)
    cldice = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps)
    return float(cldice), float(dice), pred

models = {}
for label, path in CHECKPOINTS.items():
    model = UNetPPSpine(in_channels=1, base_channels=16).to(base.DEVICE)
    ckpt = torch.load(path, map_location=base.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    models[label] = model

rows = []
with torch.no_grad():
    for batch in loader:
        name = Path(str(batch["name"][0])).stem
        image = batch["image"].to(base.DEVICE)
        gt = batch["mask"].cpu().numpy()[0, 0] > 0.5
        outputs = {}

        for label, model in models.items():
            logits, _ = base.get_outputs(model(image))
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            cldice, dice, pred = score(prob, gt)
            outputs[label] = (prob, pred, cldice, dice)
            rows.append({
                "image": name, "checkpoint": label,
                "threshold": THRESHOLD, "cldice": cldice, "dice": dice,
                "predicted_pixels": int(pred.sum()),
                "ground_truth_pixels": int(gt.sum()),
            })

        image_np = batch["image"].cpu().numpy()[0, 0]
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        panels = [
            (image_np, "Input image", "gray"),
            (gt, "Ground-truth mask", "gray"),
            (outputs["full10"][1], f"full10 prediction\nclDice={outputs['full10'][2]:.4f}", "gray"),
            (outputs["exp075"][1], f"exp075 prediction\nclDice={outputs['exp075'][2]:.4f}", "gray"),
            (skeletonize(gt), "GT skeleton", "gray"),
            (skeletonize(outputs["full10"][1]), "full10 skeleton", "gray"),
            (skeletonize(outputs["exp075"][1]), "exp075 skeleton", "gray"),
            (np.abs(outputs["exp075"][1].astype(float) - outputs["full10"][1].astype(float)),
             "Prediction disagreement", "magma"),
        ]
        for ax, (arr, title, cmap) in zip(axes.flat, panels):
            ax.imshow(arr, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{name} | threshold={THRESHOLD}")
        fig.tight_layout()
        path = OUT / f"{name}_comparison.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved overlay: {path}")

csv_path = OUT / "hard_case_scores.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== SCORES ===")
for r in rows:
    print(f"{r['image']} | {r['checkpoint']:6s} | "
          f"clDice={r['cldice']:.6f} | Dice={r['dice']:.6f} | "
          f"predicted pixels={r['predicted_pixels']} | GT pixels={r['ground_truth_pixels']}")

print(f"\nReport: {csv_path}")
print(f"All inspection files: {OUT}")
print("Checkpoints were not modified.")
