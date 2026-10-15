
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_models_512" / "checkpoint_comparison"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "original": ROOT / "benchmark_models_512"
    / "unetpp_spine_topology_finetune_cldice"
    / "best_unetpp_spine_topology_512.pth",

    "continuation_epoch9": ROOT / "benchmark_models_512"
    / "unetpp_spine_topology_full10"
    / "best_by_cldice.pth",
}

THRESHOLD = 0.07


def score_masks(prob, gt, threshold):
    pred = prob >= threshold
    eps = 1e-8

    ps = skeletonize(pred)
    gs = skeletonize(gt)

    precision = np.logical_and(ps, gt).sum() / (ps.sum() + eps)
    recall = np.logical_and(gs, pred).sum() / (gs.sum() + eps)
    cldice = 2 * precision * recall / (precision + recall + eps)

    dice = 2 * np.logical_and(pred, gt).sum() / (
        pred.sum() + gt.sum() + eps
    )

    return float(cldice), float(dice)


def main():
    device = base.DEVICE
    _, val_records = base.load_split()

    loader = DataLoader(
        base.FilamentSpineDataset(val_records),
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    rows = []

    for label, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        model = UNetPPSpine(in_channels=1, base_channels=16).to(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        print(f"\nEvaluating {label} at threshold {THRESHOLD:.2f}")
        count = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].cpu().numpy()[:, 0] > 0.5
                names = batch["name"]

                logits, _ = base.get_outputs(model(images))
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

                for name, prob, gt in zip(names, probs, masks):
                    cldice, dice = score_masks(prob, gt, THRESHOLD)
                    rows.append({
                        "checkpoint": label,
                        "image": Path(name).name,
                        "threshold": THRESHOLD,
                        "cldice": cldice,
                        "dice": dice,
                    })
                    count += 1

        print(f"Images evaluated: {count}")

    out_csv = OUT / "per_image_threshold_007.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "image", "threshold", "cldice", "dice"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-image results: {out_csv}")

    for label in CHECKPOINTS:
        subset = [r for r in rows if r["checkpoint"] == label]
        print(f"\n{label}:")
        print(f"Mean clDice: {np.mean([r['cldice'] for r in subset]):.6f}")
        print(f"Mean Dice:   {np.mean([r['dice'] for r in subset]):.6f}")

        print("Five lowest clDice images:")
        for r in sorted(subset, key=lambda x: x["cldice"])[:5]:
            print(
                f"  {r['image']}: clDice={r['cldice']:.6f}, "
                f"Dice={r['dice']:.6f}"
            )

    print("\nExisting checkpoints and comparison CSV were not modified.")


if __name__ == "__main__":
    main()


