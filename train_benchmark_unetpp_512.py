# ================================================================
# train_benchmark_unetpp_512.py
#
# 512x512 FULL DATASET BENCHMARK
# MODEL 1: UNET++ BASELINE
#
# FIXED:
# - Uses benchmark_512/benchmark_split.json
# - Correctly resolves image paths
# - Correctly resolves mask paths
# - Handles original 2048x2048 images
# - Resizes to 512x512
# - Uses CUDA automatically
# - Validates ALL files before training
# - Saves best checkpoint
# - Saves final metrics
# ================================================================

import os
import json
import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unetpp_spine import UNetPPSpine


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"

SPLIT_FILE = PROJECT_ROOT / "benchmark_512" / "benchmark_split.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_models_512"
    / "unetpp_baseline"
)

CHECKPOINT_FILE = OUTPUT_DIR / "best_unetpp_baseline_512.pth"
FINAL_CHECKPOINT_FILE = OUTPUT_DIR / "final_unetpp_baseline_512.pth"
METRICS_FILE = OUTPUT_DIR / "training_metrics_512.json"

IMAGE_SIZE = 512

RANDOM_SEED = 42

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 25

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

AMP_ENABLED = True

BASE_CHANNELS = 16


# ================================================================
# RANDOM SEED
# ================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# DEVICE
# ================================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("=" * 70)
        print("CUDA GPU ENABLED")
        print("=" * 70)

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)

        print(
            f"VRAM: "
            f"{props.total_memory / (1024 ** 3):.1f} GB"
        )

        print("=" * 70)

        return device

    print("=" * 70)
    print("CUDA NOT AVAILABLE")
    print("Using CPU")
    print("=" * 70)

    return torch.device("cpu")


# ================================================================
# PRINT HEADER
# ================================================================

def print_header():

    print("=" * 70)
    print("512x512 FULL DATASET BENCHMARK")
    print("MODEL 1: UNET++ BASELINE")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Image dir    : {IMAGE_DIR}")
    print(f"Mask dir     : {MASK_DIR}")
    print(f"Split        : {SPLIT_FILE}")
    print(f"Output       : {OUTPUT_DIR}")

    print("=" * 70)


# ================================================================
# FILE HELPERS
# ================================================================

def normalize_name(name):

    return Path(str(name)).name


def find_file(directory, filename):

    """
    Robustly resolve a filename inside the intended directory.

    Handles:
      filename
      absolute path
      accidental relative path
      case differences on Windows
    """

    filename = normalize_name(filename)

    direct = directory / filename

    if direct.exists():
        return direct

    # Case-insensitive fallback
    target = filename.lower()

    try:

        for p in directory.iterdir():

            if p.is_file() and p.name.lower() == target:
                return p

    except Exception:
        pass

    return None


def get_record_filename(record, possible_keys):

    for key in possible_keys:

        if key in record and record[key]:

            return normalize_name(record[key])

    return None


# ================================================================
# LOAD SPLIT
# ================================================================

def load_split():

    print()
    print("=" * 70)
    print("LOADING FIXED BENCHMARK SPLIT")
    print("=" * 70)

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark split not found:\n{SPLIT_FILE}\n\n"
            f"Run benchmark_setup_512.py first."
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    print()

    # ------------------------------------------------------------
    # Supported split formats
    # ------------------------------------------------------------

    if "train" in data and "validation" in data:

        train_records = data["train"]
        val_records = data["validation"]

    elif "training" in data and "validation" in data:

        train_records = data["training"]
        val_records = data["validation"]

    else:

        raise RuntimeError(
            "Could not find train/validation records in benchmark_split.json."
        )

    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")

    return train_records, val_records


# ================================================================
# RESOLVE RECORD
# ================================================================

def resolve_record(record):

    """
    Converts benchmark JSON record into actual filesystem paths.

    IMPORTANT:
    The split stores filenames/stems.
    Paths MUST be constructed from IMAGE_DIR and MASK_DIR.
    """

    image_name = get_record_filename(
        record,
        [
            "image",
            "image_file",
            "image_filename",
            "image_name",
            "filename",
        ],
    )

    mask_name = get_record_filename(
        record,
        [
            "mask",
            "mask_file",
            "mask_filename",
            "mask_name",
        ],
    )

    stem = record.get("stem", None)

    if stem is not None:
        stem = Path(str(stem)).stem

    # ------------------------------------------------------------
    # Image
    # ------------------------------------------------------------

    if image_name:

        image_path = find_file(
            IMAGE_DIR,
            image_name
        )

    elif stem:

        image_path = None

        for ext in [
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".bmp",
        ]:

            candidate = find_file(
                IMAGE_DIR,
                stem + ext
            )

            if candidate is not None:

                image_path = candidate
                break

    else:

        image_path = None

    # ------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------

    if mask_name:

        mask_path = find_file(
            MASK_DIR,
            mask_name
        )

    elif stem:

        mask_path = None

        for ext in [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
        ]:

            candidate = find_file(
                MASK_DIR,
                stem + ext
            )

            if candidate is not None:

                mask_path = candidate
                break

    else:

        mask_path = None

    return image_path, mask_path


# ================================================================
# VERIFY SPLIT FILES
# ================================================================

def verify_split_files(train_records, val_records):

    print()
    print("=" * 70)
    print("VERIFYING BENCHMARK FILE PATHS")
    print("=" * 70)

    all_records = list(train_records) + list(val_records)

    missing_images = []
    missing_masks = []

    resolved = []

    total = len(all_records)

    for i, record in enumerate(all_records, 1):

        image_path, mask_path = resolve_record(record)

        if image_path is None:

            missing_images.append(record)

        if mask_path is None:

            missing_masks.append(record)

        if image_path is not None and mask_path is not None:

            resolved.append(
                (record, image_path, mask_path)
            )

        if (
            i == 1
            or i % 100 == 0
            or i == total
        ):

            print(
                f"Checked {i}/{total}"
            )

    print()

    print(f"Resolved image/mask pairs: {len(resolved)}")
    print(f"Missing images:             {len(missing_images)}")
    print(f"Missing masks:               {len(missing_masks)}")

    if missing_images:

        print()
        print("FIRST MISSING IMAGES:")

        for record in missing_images[:10]:

            print(record)

    if missing_masks:

        print()
        print("FIRST MISSING MASKS:")

        for record in missing_masks[:10]:

            print(record)

    if missing_images or missing_masks:

        raise RuntimeError(
            "Benchmark file verification failed. "
            "No training was started."
        )

    print()
    print("FILE PATH VERIFICATION PASSED")

    return resolved


# ================================================================
# DATASET
# ================================================================

class BenchmarkDataset(Dataset):

    def __init__(
        self,
        records,
    ):

        self.items = []

        for record in records:

            image_path, mask_path = resolve_record(record)

            if image_path is None:

                raise FileNotFoundError(
                    f"Image could not be resolved:\n"
                    f"{record}"
                )

            if mask_path is None:

                raise FileNotFoundError(
                    f"Mask could not be resolved:\n"
                    f"{record}"
                )

            self.items.append(
                (
                    image_path,
                    mask_path,
                )
            )

    def __len__(self):

        return len(self.items)

    @staticmethod
    def load_image(path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image:\n{path}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        return image

    @staticmethod
    def load_mask(path):

        mask = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise FileNotFoundError(
                f"Could not read mask:\n{path}"
            )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 127
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return mask

    def __getitem__(self, index):

        image_path, mask_path = self.items[index]

        image = self.load_image(image_path)

        mask = self.load_mask(mask_path)

        return image, mask


# ================================================================
# DICE
# ================================================================

def dice_score(
    prediction,
    target,
    threshold=0.5,
    eps=1e-7,
):

    prediction = (
        prediction >= threshold
    ).float()

    target = (
        target >= 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return float(dice.item())


# ================================================================
# IOU
# ================================================================

def iou_score(
    prediction,
    target,
    threshold=0.5,
    eps=1e-7,
):

    prediction = (
        prediction >= threshold
    ).float()

    target = (
        target >= 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return float(iou.item())


# ================================================================
# BCE + DICE LOSS
# ================================================================

class DiceLoss(nn.Module):

    def __init__(self, eps=1e-7):

        super().__init__()

        self.eps = eps

    def forward(
        self,
        logits,
        target,
    ):

        probability = torch.sigmoid(
            logits
        )

        intersection = (
            probability * target
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probability.sum(
                dim=(1, 2, 3)
            )
            + target.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            2.0 * intersection
            + self.eps
        ) / (
            denominator
            + self.eps
        )

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        target,
    ):

        return (
            0.5 * self.bce(
                logits,
                target
            )
            +
            0.5 * self.dice(
                logits,
                target
            )
        )


# ================================================================
# MODEL OUTPUT HANDLING
# ================================================================

def extract_filament_output(output):

    if torch.is_tensor(output):

        return output

    if isinstance(output, dict):

        for key in [
            "filament",
            "mask",
            "out",
            "segmentation",
        ]:

            if key in output:

                return output[key]

        raise RuntimeError(
            "Could not identify filament output "
            "in model dictionary."
        )

    if isinstance(output, (tuple, list)):

        if len(output) == 0:

            raise RuntimeError(
                "Model returned empty output."
            )

        # UNet++ + spine returns filament first
        return output[0]

    raise RuntimeError(
        f"Unsupported model output type: "
        f"{type(output)}"
    )


# ================================================================
# MODEL SANITY CHECK
# ================================================================

@torch.no_grad()
def model_sanity_check(
    model,
    device,
    loader,
):

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    images, masks = next(
        iter(loader)
    )

    images = images.to(
        device,
        non_blocking=True
    )

    masks = masks.to(
        device,
        non_blocking=True
    )

    print(
        f"Input: {tuple(images.shape)}"
    )

    output = model(images)

    logits = extract_filament_output(
        output
    )

    print(
        f"Output: {tuple(logits.shape)}"
    )

    print(
        f"Target: {tuple(masks.shape)}"
    )

    if logits.shape != masks.shape:

        raise RuntimeError(
            "Model output and target shapes "
            "do not match."
        )

    print()
    print("MODEL SANITY CHECK PASSED")


# ================================================================
# TRAIN
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
):

    model.train()

    running_loss = 0.0

    total_batches = len(loader)

    for batch_idx, (
        images,
        masks,
    ) in enumerate(loader, 1):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if (
            AMP_ENABLED
            and device.type == "cuda"
        ):

            with torch.amp.autocast(
                device_type="cuda"
            ):

                output = model(images)

                logits = extract_filament_output(
                    output
                )

                loss = criterion(
                    logits,
                    masks
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            output = model(images)

            logits = extract_filament_output(
                output
            )

            loss = criterion(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item()
        )

        if (
            batch_idx == 1
            or batch_idx % 50 == 0
            or batch_idx == total_batches
        ):

            print(
                f"Batch {batch_idx:4d}/{total_batches:4d} "
                f"| Loss {loss.item():.4f}"
            )

    return (
        running_loss
        / max(total_batches, 1)
    )


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    dice_values = []
    iou_values = []

    running_loss = 0.0

    criterion = CombinedLoss()

    total = len(loader)

    for batch_idx, (
        images,
        masks,
    ) in enumerate(loader, 1):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        output = model(images)

        logits = extract_filament_output(
            output
        )

        loss = criterion(
            logits,
            masks
        )

        running_loss += loss.item()

        probabilities = torch.sigmoid(
            logits
        )

        dice_values.append(
            dice_score(
                probabilities,
                masks,
                threshold=0.5
            )
        )

        iou_values.append(
            iou_score(
                probabilities,
                masks,
                threshold=0.5
            )
        )

        if (
            batch_idx % 20 == 0
            or batch_idx == total
        ):

            print(
                f"Validation "
                f"{batch_idx}/{total}"
            )

    return {
        "loss": float(
            running_loss
            / max(total, 1)
        ),
        "dice": float(
            np.mean(dice_values)
        ),
        "iou": float(
            np.mean(iou_values)
        ),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print_header()

    device = get_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    train_records, val_records = (
        load_split()
    )

    # ------------------------------------------------------------
    # Verify every path BEFORE GPU training
    # ------------------------------------------------------------

    verify_split_files(
        train_records,
        val_records
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = BenchmarkDataset(
        train_records
    )

    val_dataset = BenchmarkDataset(
        val_records
    )

    print(
        f"Train dataset: {len(train_dataset)}"
    )

    print(
        f"Validation dataset: {len(val_dataset)}"
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = UNetPPSpine(
        in_channels=1,
        base_channels=BASE_CHANNELS,
    )

    model = model.to(
        device
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Loaded model from:"
    )

    print(
        "  unetpp_spine.py"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ------------------------------------------------------------
    # Model sanity check
    # ------------------------------------------------------------

    model_sanity_check(
        model,
        device,
        train_loader
    )

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.05,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            AMP_ENABLED
            and device.type == "cuda"
        )
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"AMP: {AMP_ENABLED}"
    )

    history = []

    best_dice = -1.0

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d}"
        )

        print("-" * 70)

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
        )

        validation = validate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        epoch_time = (
            time.time()
            - epoch_start
        )

        lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{validation['loss']:.6f}"
        )

        print(
            f"Validation Dice: "
            f"{validation['dice']:.6f}"
        )

        print(
            f"Validation IoU: "
            f"{validation['iou']:.6f}"
        )

        print(
            f"LR: {lr:.8f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "dice": validation["dice"],
            "iou": validation["iou"],
            "learning_rate": lr,
            "epoch_time_seconds": epoch_time,
        }

        history.append(
            record
        )

        # --------------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------------

        if validation["dice"] > best_dice:

            best_dice = validation["dice"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "best_iou": validation["iou"],
                "config": {
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "random_seed": RANDOM_SEED,
                    "base_channels": BASE_CHANNELS,
                },
            }

            torch.save(
                checkpoint,
                CHECKPOINT_FILE
            )

            print()
            print("*** NEW BEST MODEL ***")

            print(
                f"Dice: "
                f"{best_dice:.6f}"
            )

            print(
                f"IoU: "
                f"{validation['iou']:.6f}"
            )

            print(
                f"Saved: "
                f"{CHECKPOINT_FILE}"
            )

    # ------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
        },
        FINAL_CHECKPOINT_FILE
    )

    total_time = (
        time.time()
        - training_start
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    results = {
        "model": "UNet++ baseline",
        "image_size": IMAGE_SIZE,
        "dataset_size": 707,
        "training_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_dice": best_dice,
        "history": history,
        "best_checkpoint": str(
            CHECKPOINT_FILE
        ),
        "final_checkpoint": str(
            FINAL_CHECKPOINT_FILE
        ),
        "training_time_seconds": total_time,
        "training_time_hours": (
            total_time / 3600.0
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    print(
        f"Training time: "
        f"{total_time / 3600.0:.2f} hours"
    )

    print()
    print(
        f"Best checkpoint:"
    )

    print(
        CHECKPOINT_FILE
    )

    print()
    print(
        f"Final checkpoint:"
    )

    print(
        FINAL_CHECKPOINT_FILE
    )

    print()
    print(
        f"Metrics:"
    )

    print(
        METRICS_FILE
    )

    print("=" * 70)


if __name__ == "__main__":

    main()