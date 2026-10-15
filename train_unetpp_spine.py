"""
Train U-Net++ with filament and spine outputs at 512x512.

Project structure:
    train_unetpp_spine.py
    model.py
    dataset.py
    MAGFiLO_1.0_Kaggle_2026/
        train/
            train_images/
    data/
        full_masks/
        full_spine_masks/

Run:
    python train_unetpp_spine.py
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Change these imports if your classes are defined in other files.
from unetpp_spine import UNetPPSpine
from dataset_unetpp_spine import UNetPPSpineDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 50

LEARNING_RATE = 1e-3
SPINE_LOSS_WEIGHT = 0.5

NUM_WORKERS = 0
USE_AMP = True

PROJECT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_DIR
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_DIR / "data" / "full_masks"

SPINE_DIR = PROJECT_DIR / "data" / "full_spine_masks"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_512"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_unetpp_spine_512.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last_unetpp_spine_512.pth"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_image_names(image_dir, mask_dir, spine_dir):
    """
    Find images with matching filament and spine masks.

    Assumes masks are named:
        <image_stem>.png
    """

    for directory in (image_dir, mask_dir, spine_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

    image_names = []

    missing_masks = []
    missing_spines = []

    image_paths = sorted(
        image_dir.iterdir(),
        key=lambda path: path.name.lower()
    )

    for image_path in image_paths:

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = mask_dir / f"{image_path.stem}.png"

        spine_path = spine_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            missing_masks.append(image_path.name)
            continue

        if not spine_path.exists():
            missing_spines.append(image_path.name)
            continue

        image_names.append(image_path.name)

    print(f"Images with both masks: {len(image_names)}")

    if missing_masks:
        print(
            f"Warning: {len(missing_masks)} images "
            "are missing filament masks."
        )
        print("Examples:", missing_masks[:5])

    if missing_spines:
        print(
            f"Warning: {len(missing_spines)} images "
            "are missing spine masks."
        )
        print("Examples:", missing_spines[:5])

    if not image_names:
        raise RuntimeError(
            "No matching image/mask/spine triplets found. "
            "Check your paths and mask filenames."
        )

    return image_names


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Soft Dice loss.

    Computes Dice independently for each image, then averages
    across the batch.
    """

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.contiguous().view(
        probabilities.shape[0], -1
    )

    targets = targets.float().contiguous().view(
        targets.shape[0], -1
    )

    intersection = (probabilities * targets).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


def combined_loss(
    mask_logits,
    masks,
    spine_logits,
    spines,
    bce_loss,
):
    """
    Filament loss:
        BCE + Dice

    Spine loss:
        BCE + Dice

    Total:
        filament_loss + weight * spine_loss
    """

    filament_loss = (
        bce_loss(mask_logits, masks)
        + dice_loss_from_logits(mask_logits, masks)
    )

    spine_loss = (
        bce_loss(spine_logits, spines)
        + dice_loss_from_logits(spine_logits, spines)
    )

    total_loss = (
        filament_loss
        + SPINE_LOSS_WEIGHT * spine_loss
    )

    return total_loss


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    bce_loss,
    amp_enabled,
):
    model.eval()

    total_loss = 0.0
    batch_count = 0

    # Global pixel counts for filament metrics.
    intersection = 0
    pred_positive = 0
    target_positive = 0

    # Global pixel counts for spine metrics.
    spine_intersection = 0
    spine_pred_positive = 0
    spine_target_positive = 0

    for images, masks, spines, _names in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        ).float()

        spines = spines.to(
            device,
            non_blocking=True,
        ).float()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            mask_logits, spine_logits = model(images)

            loss = combined_loss(
                mask_logits,
                masks,
                spine_logits,
                spines,
                bce_loss,
            )

        total_loss += loss.item()
        batch_count += 1

        # Filament predictions.
        predictions = torch.sigmoid(mask_logits) >= 0.5
        targets = masks >= 0.5

        intersection += (
            predictions & targets
        ).sum().item()

        pred_positive += predictions.sum().item()
        target_positive += targets.sum().item()

        # Spine predictions.
        spine_predictions = (
            torch.sigmoid(spine_logits) >= 0.5
        )

        spine_targets = spines >= 0.5

        spine_intersection += (
            spine_predictions & spine_targets
        ).sum().item()

        spine_pred_positive += (
            spine_predictions.sum().item()
        )

        spine_target_positive += (
            spine_targets.sum().item()
        )

    eps = 1e-8

    filament_dice = (
        2 * intersection + eps
    ) / (
        pred_positive + target_positive + eps
    )

    filament_iou = (
        intersection + eps
    ) / (
        pred_positive
        + target_positive
        - intersection
        + eps
    )

    precision = (
        intersection + eps
    ) / (
        pred_positive + eps
    )

    recall = (
        intersection + eps
    ) / (
        target_positive + eps
    )

    spine_dice = (
        2 * spine_intersection + eps
    ) / (
        spine_pred_positive
        + spine_target_positive
        + eps
    )

    spine_iou = (
        spine_intersection + eps
    ) / (
        spine_pred_positive
        + spine_target_positive
        - spine_intersection
        + eps
    )

    return {
        "loss": total_loss / max(batch_count, 1),
        "dice": filament_dice,
        "iou": filament_iou,
        "precision": precision,
        "recall": recall,
        "spine_dice": spine_dice,
        "spine_iou": spine_iou,
    }


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_dice,
    config,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_dice,
            "config": config,
        },
        path,
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    else:

        device = torch.device("cpu")

        print(
            "WARNING: CUDA is unavailable. "
            "Training will be significantly slower."
        )

    amp_enabled = USE_AMP and device.type == "cuda"

    print("Mixed precision:", amp_enabled)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_names = discover_image_names(
        IMAGE_DIR,
        MASK_DIR,
        SPINE_DIR,
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # For a strict comparison against your original baseline,
    # use the exact same train_names and val_names from the
    # original training script.
    #
    # This split is deterministic for the same ordered list.
    # --------------------------------------------------------

    train_names, val_names = train_test_split(
        image_names,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
    )

    print(f"Training images: {len(train_names)}")
    print(f"Validation images: {len(val_names)}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = UNetPPSpineDataset(
        image_dir=str(IMAGE_DIR),
        mask_dir=str(MASK_DIR),
        spine_dir=str(SPINE_DIR),
        image_names=list(train_names),
        image_size=IMAGE_SIZE,
    )

    val_dataset = UNetPPSpineDataset(
        image_dir=str(IMAGE_DIR),
        mask_dir=str(MASK_DIR),
        spine_dir=str(SPINE_DIR),
        image_names=list(val_names),
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Model, optimizer and losses
    # --------------------------------------------------------

    model = UNetPPSpine().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    bce_loss = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    config = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "spine_loss_weight": SPINE_LOSS_WEIGHT,
        "seed": SEED,
        "train_names": list(train_names),
        "val_names": list(val_names),
    }

    best_dice = -1.0
    best_epoch = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\nStarting 512 x 512 training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        batch_count = 0

        epoch_start = time.time()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for images, masks, spines, _names in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            ).float()

            spines = spines.to(
                device,
                non_blocking=True,
            ).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):

                mask_logits, spine_logits = model(images)

                loss = combined_loss(
                    mask_logits,
                    masks,
                    spine_logits,
                    spines,
                    bce_loss,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss / max(batch_count, 1)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        metrics = evaluate(
            model,
            val_loader,
            device,
            bce_loss,
            amp_enabled,
        )

        elapsed = time.time() - epoch_start

        print(
            f"\nEpoch {epoch:03d}/{EPOCHS}\n"
            f"  Train loss:       {train_loss:.5f}\n"
            f"  Validation loss:  {metrics['loss']:.5f}\n"
            f"  Filament Dice:    {metrics['dice']:.6f}\n"
            f"  Filament IoU:     {metrics['iou']:.6f}\n"
            f"  Precision:        {metrics['precision']:.6f}\n"
            f"  Recall:           {metrics['recall']:.6f}\n"
            f"  Spine Dice:       {metrics['spine_dice']:.6f}\n"
            f"  Spine IoU:        {metrics['spine_iou']:.6f}\n"
            f"  Epoch time:       {elapsed:.1f} seconds\n"
        )

        # Save latest checkpoint every epoch.
        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            epoch,
            best_dice,
            config,
        )

        # Save best checkpoint by filament Dice.
        if metrics["dice"] > best_dice:

            best_dice = metrics["dice"]
            best_epoch = epoch

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                epoch,
                best_dice,
                config,
            )

            print(
                f"New best checkpoint saved: "
                f"{BEST_CHECKPOINT}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nTraining complete.")
    print(f"Best validation Dice: {best_dice:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")
    print(f"Last checkpoint: {LAST_CHECKPOINT}")


if __name__ == "__main__":
    main()