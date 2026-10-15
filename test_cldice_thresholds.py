
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    ROOT / "benchmark_models_512"
    / "unetpp_spine_topology_full10"
    / "best_by_cldice.pth"
)

OUT_DIR = ROOT / "benchmark_models_512" / "threshold_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = [0.06, 0.07, 0.08, 0.09, 0.10]






def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = base.DEVICE
    _, val_records = base.load_split()

    loader = DataLoader(
        base.FilamentSpineDataset(val_records),
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetPPSpine(in_channels=1, base_channels=16).to(device)
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Keep probability maps in memory so inference runs only once.
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()[:, 0] > 0.5

            logits, _ = base.get_outputs(model(images))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            all_probs.extend(list(probs))
            all_masks.extend(list(masks))

    if len(all_probs) != len(val_records):
        raise RuntimeError(
            f"Expected {len(val_records)} predictions, got {len(all_probs)}"
        )

    rows = []

    for threshold in THRESHOLDS:
        image_cldice = []
        image_dice = []

        for prob, gt in zip(all_probs, all_masks):
            pred = prob >= threshold

            pred_skeleton = skeletonize(pred)
            gt_skeleton = skeletonize(gt)

            eps = 1e-8
            precision = np.logical_and(pred_skeleton, gt).sum() / (
                pred_skeleton.sum() + eps
            )
            recall = np.logical_and(gt_skeleton, pred).sum() / (
                gt_skeleton.sum() + eps
            )
            cldice = 2 * precision * recall / (
                precision + recall + eps
            )

            dice = 2 * np.logical_and(pred, gt).sum() / (
                pred.sum() + gt.sum() + eps
            )

            image_cldice.append(float(cldice))
            image_dice.append(float(dice))

        row = {
            "threshold": threshold,
            "mean_cldice": float(np.mean(image_cldice)),
            "mean_dice": float(np.mean(image_dice)),
        }
        rows.append(row)

        print(
            f"Threshold={threshold:.2f} | "
            f"Mean clDice={row['mean_cldice']:.6f} | "
            f"Mean Dice={row['mean_dice']:.6f}"
        )

    csv_path = OUT_DIR / "threshold_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["mean_cldice"])

    print("\n===== THRESHOLD TEST COMPLETE =====")
    print(f"Best tested threshold: {best['threshold']:.2f}")
    print(f"Best mean clDice: {best['mean_cldice']:.6f}")
    print(f"Results saved to: {csv_path}")
    print("Original checkpoint and prediction files were not modified.")


if __name__ == "__main__":
    main()


