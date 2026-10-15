
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

import train_unetpp_spine_topology_exp as base
from unetpp_spine import UNetPPSpine


ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT
    / "benchmark_models_512"
    / "unetpp_spine_topology_full10"
    / "best_by_cldice.pth"
)

OUT = (
    ROOT
    / "benchmark_models_512"
    / "unetpp_spine_topology_exp075"
)

EPOCHS = 10
LR = 1e-6
PATIENCE = 4
DEVICE = base.DEVICE

# Use the threshold that gave the current best validation score.
THRESHOLD = 0.07


def evaluate_cldice(model, loader):
    model.eval()
    scores, dices = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].cpu().numpy()[:, 0] > 0.5

            logits, _ = base.get_outputs(model(images))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            preds = probs >= THRESHOLD

            for pred, gt in zip(preds, masks):
                pred_skel = skeletonize(pred)
                gt_skel = skeletonize(gt)
                eps = 1e-8

                precision = (
                    np.logical_and(pred_skel, gt).sum()
                    / (pred_skel.sum() + eps)
                )
                recall = (
                    np.logical_and(gt_skel, pred).sum()
                    / (gt_skel.sum() + eps)
                )

                cldice = (
                    2 * precision * recall
                    / (precision + recall + eps)
                )
                dice = (
                    2 * np.logical_and(pred, gt).sum()
                    / (pred.sum() + gt.sum() + eps)
                )

                scores.append(float(cldice))
                dices.append(float(dice))

    return float(np.mean(scores)), float(np.mean(dices))


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {SOURCE}"
        )

    # Prevent accidentally reusing an existing experiment directory.
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUT}\n"
            "Choose a new experiment folder to avoid overwriting results."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    base.seed_everything(base.SEED)
    train_records, val_records = base.load_split()

    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Starting checkpoint: {SOURCE}")
    print(f"Output folder: {OUT}")
    print(f"Topology weight: {base.LAMBDA_TOPOLOGY}")
    print(f"Evaluation threshold: {THRESHOLD}")

    train_loader = DataLoader(
        base.FilamentSpineDataset(train_records),
        batch_size=base.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        base.FilamentSpineDataset(val_records),
        batch_size=base.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = UNetPPSpine(
        in_channels=1,
        base_channels=16,
    ).to(DEVICE)

    checkpoint = torch.load(
        SOURCE,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    baseline_cldice, baseline_dice = evaluate_cldice(
        model, val_loader
    )

    best_cldice = baseline_cldice
    best_epoch = 0
    stale_epochs = 0

    print(f"Baseline clDice: {baseline_cldice:.6f}")
    print(f"Baseline Dice: {baseline_dice:.6f}")

    # Preserve the exact starting checkpoint for this experiment.
    torch.save(
        checkpoint,
        OUT / "starting_checkpoint.pth",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    history = []
    start = time.time()
    best_path = OUT / "best_by_cldice.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Experiment epoch {epoch}/{EPOCHS} =====")

        train_metrics = base.train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
        )

        val_metrics = base.validate(model, val_loader)
        val_cldice, val_dice = evaluate_cldice(
            model, val_loader
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "validation_loss": float(val_metrics["loss"]),
            "validation_dice": val_dice,
            "validation_cldice": val_cldice,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
        }
        history.append(record)

        print(f"Validation Dice: {val_dice:.6f}")
        print(f"Validation clDice: {val_cldice:.6f}")

        if val_cldice > best_cldice:
            best_cldice = val_cldice
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_cldice": val_cldice,
                    "validation_dice": val_dice,
                    "source_checkpoint": str(SOURCE),
                    "learning_rate": LR,
                    "lambda_spine": base.LAMBDA_SPINE,
                    "lambda_topology": base.LAMBDA_TOPOLOGY,
                    "threshold_used_for_selection": THRESHOLD,
                },
                best_path,
            )
            print("New best experimental checkpoint saved.")
        else:
            stale_epochs += 1
            print(f"No improvement: {stale_epochs}/{PATIENCE}")

        with (OUT / "history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "source_checkpoint": str(SOURCE),
                    "baseline_cldice": baseline_cldice,
                    "baseline_dice": baseline_dice,
                    "lambda_spine": base.LAMBDA_SPINE,
                    "lambda_topology": base.LAMBDA_TOPOLOGY,
                    "threshold": THRESHOLD,
                    "best_cldice": best_cldice,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                f,
                indent=2,
            )

        if stale_epochs >= PATIENCE:
            print("Early stopping: validation clDice did not improve.")
            break

    print("\n===== EXPERIMENT COMPLETE =====")
    print(f"Baseline clDice: {baseline_cldice:.6f}")
    print(f"Best clDice: {best_cldice:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Elapsed hours: {(time.time() - start) / 3600:.2f}")
    print(f"Output folder: {OUT}")

    if best_epoch == 0:
        print(
            "No epoch beat the starting checkpoint. "
            "Keep the existing best checkpoint."
        )


if __name__ == "__main__":
    main()


