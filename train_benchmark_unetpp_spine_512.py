# ================================================================
# train_benchmark_unetpp_spine_512.py
#
# MODEL 2:
#   UNet++ + Spine
#
# FULL DATASET BENCHMARK
#   707 total images
#   565 training
#   142 validation
#   fixed split: benchmark_512/benchmark_split.json
#
# Input:
#   2048x2048 source images
#   resized to 512x512
#
# GPU:
#   CUDA
#
# Epochs:
#   50
#
# Outputs:
#   benchmark_models_512/unetpp_spine/
#       best_unetpp_spine_512.pth
#       final_unetpp_spine_512.pth
#       training_metrics_512.json
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"

SPLIT_FILE = PROJECT_ROOT / "benchmark_512" / "benchmark_split.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_models_512"
    / "unetpp_spine"
)

BEST_CHECKPOINT = OUTPUT_DIR / "best_unetpp_spine_512.pth"
FINAL_CHECKPOINT = OUTPUT_DIR / "final_unetpp_spine_512.pth"
METRICS_FILE = OUTPUT_DIR / "training_metrics_512.json"

IMAGE_SIZE = 512

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 25

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

SEED = 42

# Loss weights
FILAMENT_BCE_WEIGHT = 1.0
FILAMENT_DICE_WEIGHT = 1.0
SPINE_BCE_WEIGHT = 1.0
SPINE_DICE_WEIGHT = 1.0

THRESHOLD = 0.5


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# DATASET
# ================================================================

class FilamentSpineDataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):

        return len(self.records)

    def _get_value(self, record, keys):

        for key in keys:
            if key in record:
                return record[key]

        return None

    def _resolve_path(self, value, directory):

        if value is None:
            raise RuntimeError(
                f"Could not find filename in record:\n{value}"
            )

        path = Path(value)

        # If JSON contains an absolute path
        if path.is_absolute():

            if path.exists():
                return path

        # Otherwise use the filename only
        candidate = directory / path.name

        if candidate.exists():
            return candidate

        # Case-insensitive fallback
        target = path.name.lower()

        for file in directory.iterdir():

            if file.name.lower() == target:
                return file

        raise FileNotFoundError(
            f"Could not find file:\n"
            f"{path.name}\n"
            f"in:\n{directory}"
        )

    def _load_gray(self, path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image:\n{path}"
            )

        return image

    def __getitem__(self, index):

        record = self.records[index]

        image_name = self._get_value(
            record,
            [
                "image",
                "image_file",
                "image_filename",
                "image_path",
                "filename",
            ],
        )

        mask_name = self._get_value(
            record,
            [
                "mask",
                "mask_file",
                "mask_filename",
                "mask_path",
            ],
        )

        spine_name = self._get_value(
            record,
            [
                "spine",
                "spine_file",
                "spine_filename",
                "spine_path",
            ],
        )

        image_path = self._resolve_path(
            image_name,
            IMAGE_DIR
        )

        mask_path = self._resolve_path(
            mask_name,
            MASK_DIR
        )

        spine_path = self._resolve_path(
            spine_name,
            SPINE_DIR
        )

        image = self._load_gray(image_path)
        mask = self._load_gray(mask_path)
        spine = self._load_gray(spine_path)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        spine = cv2.resize(
            spine,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0

        mask = (
            mask.astype(np.float32) / 255.0
        )

        spine = (
            spine.astype(np.float32) / 255.0
        )

        mask = (mask >= 0.5).astype(np.float32)
        spine = (spine >= 0.5).astype(np.float32)

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        spine = torch.from_numpy(
            spine
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "spine": spine,
            "name": image_path.stem,
        }


# ================================================================
# LOAD FIXED SPLIT
# ================================================================

def load_split():

    print()
    print("=" * 70)
    print("LOADING FIXED BENCHMARK SPLIT")
    print("=" * 70)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    if "train" in data:
        train_records = data["train"]
    elif "training" in data:
        train_records = data["training"]
    else:
        raise KeyError(
            "Could not find training records in benchmark_split.json"
        )

    if "validation" in data:
        val_records = data["validation"]
    elif "val" in data:
        val_records = data["val"]
    else:
        raise KeyError(
            "Could not find validation records in benchmark_split.json"
        )

    print(
        f"Training records:   {len(train_records)}"
    )

    print(
        f"Validation records: {len(val_records)}"
    )

    return train_records, val_records


# ================================================================
# MODEL COMPONENTS
# ================================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)


class UNetPPSpine(nn.Module):
    """
    UNet++ style encoder/decoder with two output heads:

        filament output
        spine output
    """

    def __init__(self):

        super().__init__()

        filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv0_0 = ConvBlock(
            1,
            filters[0]
        )

        self.conv1_0 = ConvBlock(
            filters[0],
            filters[1]
        )

        self.conv2_0 = ConvBlock(
            filters[1],
            filters[2]
        )

        self.conv3_0 = ConvBlock(
            filters[2],
            filters[3]
        )

        self.conv4_0 = ConvBlock(
            filters[3],
            filters[4]
        )

        self.conv0_1 = ConvBlock(
            filters[0] + filters[1],
            filters[0]
        )

        self.conv1_1 = ConvBlock(
            filters[1] + filters[2],
            filters[1]
        )

        self.conv2_1 = ConvBlock(
            filters[2] + filters[3],
            filters[2]
        )

        self.conv3_1 = ConvBlock(
            filters[3] + filters[4],
            filters[3]
        )

        self.conv0_2 = ConvBlock(
            filters[0] * 2 + filters[1],
            filters[0]
        )

        self.conv1_2 = ConvBlock(
            filters[1] * 2 + filters[2],
            filters[1]
        )

        self.conv2_2 = ConvBlock(
            filters[2] * 2 + filters[3],
            filters[2]
        )

        self.conv0_3 = ConvBlock(
            filters[0] * 3 + filters[1],
            filters[0]
        )

        self.conv1_3 = ConvBlock(
            filters[1] * 3 + filters[2],
            filters[1]
        )

        self.conv0_4 = ConvBlock(
            filters[0] * 4 + filters[1],
            filters[0]
        )

        self.filament_head = nn.Conv2d(
            filters[0],
            1,
            kernel_size=1
        )

        self.spine_head = nn.Conv2d(
            filters[0],
            1,
            kernel_size=1
        )

    def up(self, x, target):

        x = F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return x

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(
            self.pool(x0_0)
        )

        x2_0 = self.conv2_0(
            self.pool(x1_0)
        )

        x3_0 = self.conv3_0(
            self.pool(x2_0)
        )

        x4_0 = self.conv4_0(
            self.pool(x3_0)
        )

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    self.up(x1_0, x0_0),
                ],
                dim=1,
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    self.up(x2_0, x1_0),
                ],
                dim=1,
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    self.up(x3_0, x2_0),
                ],
                dim=1,
            )
        )

        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    self.up(x4_0, x3_0),
                ],
                dim=1,
            )
        )

        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    self.up(x1_1, x0_0),
                ],
                dim=1,
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    self.up(x2_1, x1_0),
                ],
                dim=1,
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    self.up(x3_1, x2_0),
                ],
                dim=1,
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    self.up(x1_2, x0_0),
                ],
                dim=1,
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    self.up(x2_2, x1_0),
                ],
                dim=1,
            )
        )

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    self.up(x1_3, x0_0),
                ],
                dim=1,
            )
        )

        filament = self.filament_head(
            x0_4
        )

        spine = self.spine_head(
            x0_4
        )

        return filament, spine


# ================================================================
# LOSSES
# ================================================================

def dice_loss(logits, target):

    probs = torch.sigmoid(logits)

    smooth = 1.0

    intersection = (
        probs * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        probs.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


def segmentation_loss(logits, target):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target
    )

    dice = dice_loss(
        logits,
        target
    )

    return bce + dice


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(logits, target):

    probs = torch.sigmoid(logits)

    pred = (
        probs >= THRESHOLD
    ).float()

    target = target.float()

    intersection = (
        pred * target
    ).sum().item()

    pred_sum = pred.sum().item()
    target_sum = target.sum().item()

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        pred_sum + target_sum + 1e-7
    )

    union = (
        pred_sum
        + target_sum
        - intersection
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return dice, iou


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(model, loader):

    model.eval()

    total_loss = 0.0

    dice_sum = 0.0
    iou_sum = 0.0

    count = 0

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        masks = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        spines = batch["spine"].to(
            DEVICE,
            non_blocking=True
        )

        filament_logits, spine_logits = model(
            images
        )

        filament_loss = segmentation_loss(
            filament_logits,
            masks
        )

        spine_loss = segmentation_loss(
            spine_logits,
            spines
        )

        loss = (
            FILAMENT_BCE_WEIGHT
            * filament_loss
            +
            SPINE_BCE_WEIGHT
            * spine_loss
        )

        total_loss += loss.item()

        dice, iou = calculate_metrics(
            filament_logits,
            masks
        )

        dice_sum += dice
        iou_sum += iou

        count += 1

        if (
            batch_index % 20 == 0
            or batch_index == len(loader)
        ):

            print(
                f"Validation "
                f"{batch_index}/{len(loader)}"
            )

    return (
        total_loss / max(count, 1),
        dice_sum / max(count, 1),
        iou_sum / max(count, 1),
    )


# ================================================================
# TRAINING
# ================================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0

    count = 0

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        masks = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        spines = batch["spine"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        filament_logits, spine_logits = model(
            images
        )

        filament_loss = segmentation_loss(
            filament_logits,
            masks
        )

        spine_loss = segmentation_loss(
            spine_logits,
            spines
        )

        loss = (
            FILAMENT_BCE_WEIGHT
            * filament_loss
            +
            SPINE_BCE_WEIGHT
            * spine_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        count += 1

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or batch_index == len(loader)
        ):

            print(
                f"Batch "
                f"{batch_index:4d}/"
                f"{len(loader):4d} "
                f"| Loss {loss.item():.4f}"
            )

    return total_loss / max(count, 1)


# ================================================================
# MAIN
# ================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("512x512 FULL DATASET BENCHMARK")
    print("MODEL 2: UNET++ + SPINE")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Image dir    : {IMAGE_DIR}"
    )

    print(
        f"Mask dir     : {MASK_DIR}"
    )

    print(
        f"Spine dir    : {SPINE_DIR}"
    )

    print(
        f"Split        : {SPLIT_FILE}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print("=" * 70)

    print("CUDA GPU ENABLED"
          if torch.cuda.is_available()
          else "WARNING: CPU MODE")

    print("=" * 70)

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(0)

        print(
            f"VRAM: "
            f"{props.total_memory / 1024**3:.1f} GB"
        )

    print("=" * 70)

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    train_records, val_records = load_split()

    # ------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = FilamentSpineDataset(
        train_records
    )

    val_dataset = FilamentSpineDataset(
        val_records
    )

    print(
        f"Train dataset:       {len(train_dataset)}"
    )

    print(
        f"Validation dataset:  {len(val_dataset)}"
    )

    # ------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = UNetPPSpine().to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ------------------------------------------------------------
    # SANITY CHECK
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    test_batch = next(
        iter(train_loader)
    )

    test_images = test_batch["image"].to(
        DEVICE
    )

    test_masks = test_batch["mask"].to(
        DEVICE
    )

    test_spines = test_batch["spine"].to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        test_filament, test_spine = model(
            test_images
        )

    print(
        f"Input:          "
        f"{tuple(test_images.shape)}"
    )

    print(
        f"Target mask:    "
        f"{tuple(test_masks.shape)}"
    )

    print(
        f"Target spine:   "
        f"{tuple(test_spines.shape)}"
    )

    print(
        f"Filament output:"
        f" {tuple(test_filament.shape)}"
    )

    print(
        f"Spine output:   "
        f"{tuple(test_spine.shape)}"
    )

    assert (
        test_filament.shape
        == test_masks.shape
    )

    assert (
        test_spine.shape
        == test_spines.shape
    )

    print()
    print("MODEL SANITY CHECK PASSED")

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    history = []

    best_dice = -1.0

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        print()
        print("-" * 70)
        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d}"
        )
        print("-" * 70)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Train loss:       "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss:  "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation Dice:  "
            f"{val_dice:.6f}"
        )

        print(
            f"Validation IoU:   "
            f"{val_iou:.6f}"
        )

        print(
            f"LR:               "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch time:       "
            f"{epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_dice": val_dice,
                "validation_iou": val_iou,
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time,
            }
        )

        # --------------------------------------------------------
        # BEST CHECKPOINT
        # --------------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "epoch": epoch,

                    "best_validation_dice":
                        best_dice,

                    "validation_iou":
                        val_iou,

                    "model_name":
                        "UNet++ + Spine",

                    "image_size":
                        IMAGE_SIZE,

                    "seed":
                        SEED,

                    "train_images":
                        len(train_records),

                    "validation_images":
                        len(val_records),
                },
                BEST_CHECKPOINT,
            )

            print()
            print(
                f"NEW BEST CHECKPOINT"
            )

            print(
                f"Dice: {best_dice:.6f}"
            )

            print(
                f"Saved to:"
            )

            print(
                BEST_CHECKPOINT
            )

        scheduler.step()

    # ------------------------------------------------------------
    # FINAL CHECKPOINT
    # ------------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epoch":
                EPOCHS,

            "model_name":
                "UNet++ + Spine",

            "image_size":
                IMAGE_SIZE,

            "seed":
                SEED,

            "train_images":
                len(train_records),

            "validation_images":
                len(val_records),

            "best_validation_dice":
                best_dice,
        },
        FINAL_CHECKPOINT,
    )

    total_time = (
        time.time()
        - total_start
    )

    # ------------------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------------------

    results = {

        "model":
            "UNet++ + Spine",

        "model_id":
            "unetpp_spine",

        "image_size":
            IMAGE_SIZE,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "seed":
            SEED,

        "train_images":
            len(train_records),

        "validation_images":
            len(val_records),

        "trainable_parameters":
            parameter_count,

        "best_validation_dice":
            best_dice,

        "total_training_time_seconds":
            total_time,

        "total_training_time_hours":
            total_time / 3600.0,

        "best_checkpoint":
            str(BEST_CHECKPOINT),

        "final_checkpoint":
            str(FINAL_CHECKPOINT),

        "history":
            history,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model: UNet++ + Spine"
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    print(
        f"Training time: "
        f"{total_time / 3600:.2f} hours"
    )

    print()
    print("Best checkpoint:")

    print(
        BEST_CHECKPOINT
    )

    print()
    print("Final checkpoint:")

    print(
        FINAL_CHECKPOINT
    )

    print()
    print("Metrics:")

    print(
        METRICS_FILE
    )

    print("=" * 70)


if __name__ == "__main__":

    main()