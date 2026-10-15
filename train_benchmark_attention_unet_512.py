"""
======================================================================
512x512 FULL DATASET BENCHMARK
MODEL 5: ATTENTION U-NET
======================================================================

Dataset:
    707 images
    565 training
    142 validation
    Fixed seed = 42
    Resolution = 512x512

Model:
    Attention U-Net

Training:
    50 epochs
    Batch size = 2
    AdamW
    CUDA / RTX 3050 6GB

Outputs:
    benchmark_models_512/
        attention_unet/
            best_attention_unet_512.pth
            final_attention_unet_512.pth
            training_metrics_512.json
            predictions/
======================================================================
"""

import os
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"

SPLIT_JSON = (
    PROJECT_ROOT
    / "benchmark_512"
    / "benchmark_split.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_models_512"
    / "attention_unet"
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 50
SEED = 42

LEARNING_RATE = 1e-4
MIN_LR = 1e-6

NUM_WORKERS = 0

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =====================================================================
# DATASET
# =====================================================================

class FilamentDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def _resolve(self, value, base_dir):

        p = Path(value)

        if p.is_absolute():
            return p

        return base_dir / p

    def load_image(self, path):

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

        image = image.astype(
            np.float32
        ) / 255.0

        return image

    def load_mask(self, path):

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

        return mask

    def __getitem__(self, index):

        record = self.records[index]

        image_name = (
            record.get("image")
            or record.get("image_path")
            or record.get("image_file")
        )

        mask_name = (
            record.get("mask")
            or record.get("mask_path")
            or record.get("mask_file")
        )

        if image_name is None:
            raise KeyError(
                f"No image field in record:\n{record}"
            )

        if mask_name is None:
            raise KeyError(
                f"No mask field in record:\n{record}"
            )

        image_path = self._resolve(
            image_name,
            IMAGE_DIR
        )

        mask_path = self._resolve(
            mask_name,
            MASK_DIR
        )

        image = self.load_image(
            image_path
        )

        mask = self.load_mask(
            mask_path
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0).float()

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "name": Path(image_name).stem
        }


# =====================================================================
# LOAD FIXED BENCHMARK SPLIT
# =====================================================================

def load_split():

    print("=" * 70)
    print("LOADING FIXED BENCHMARK SPLIT")
    print("=" * 70)

    with open(
        SPLIT_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "train" in data:
        train_records = data["train"]

    elif "training" in data:
        train_records = data["training"]

    else:
        raise KeyError(
            "Could not find training records."
        )

    if "validation" in data:
        val_records = data["validation"]

    elif "val" in data:
        val_records = data["val"]

    else:
        raise KeyError(
            "Could not find validation records."
        )

    print(
        f"Training records   : {len(train_records)}"
    )

    print(
        f"Validation records : {len(val_records)}"
    )

    if len(train_records) != 565:
        raise RuntimeError(
            f"Expected 565 training records, "
            f"found {len(train_records)}"
        )

    if len(val_records) != 142:
        raise RuntimeError(
            f"Expected 142 validation records, "
            f"found {len(val_records)}"
        )

    return train_records, val_records


# =====================================================================
# ATTENTION GATE
# =====================================================================

class AttentionGate(nn.Module):

    def __init__(
        self,
        gate_channels,
        skip_channels,
        inter_channels
    ):

        super().__init__()

        self.W_g = nn.Sequential(

            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.W_x = nn.Sequential(

            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.psi = nn.Sequential(

            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid()
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        gate,
        skip
    ):

        g = self.W_g(gate)
        x = self.W_x(skip)

        if g.shape[-2:] != x.shape[-2:]:

            g = F.interpolate(
                g,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        psi = self.relu(
            g + x
        )

        psi = self.psi(
            psi
        )

        return skip * psi


# =====================================================================
# DOUBLE CONVOLUTION BLOCK
# =====================================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# =====================================================================
# ATTENTION U-NET
# =====================================================================

class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1
    ):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionGate(
            gate_channels=512,
            skip_channels=512,
            inter_channels=256
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionGate(
            gate_channels=256,
            skip_channels=256,
            inter_channels=128
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionGate(
            gate_channels=128,
            skip_channels=128,
            inter_channels=64
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionGate(
            gate_channels=64,
            skip_channels=64,
            inter_channels=32
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder 4
        d4 = self.up4(b)

        if d4.shape[-2:] != e4.shape[-2:]:

            d4 = F.interpolate(
                d4,
                size=e4.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        a4 = self.att4(
            d4,
            e4
        )

        d4 = torch.cat(
            [d4, a4],
            dim=1
        )

        d4 = self.dec4(d4)

        # Decoder 3
        d3 = self.up3(d4)

        if d3.shape[-2:] != e3.shape[-2:]:

            d3 = F.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        a3 = self.att3(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, a3],
            dim=1
        )

        d3 = self.dec3(d3)

        # Decoder 2
        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:

            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        a2 = self.att2(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, a2],
            dim=1
        )

        d2 = self.dec2(d2)

        # Decoder 1
        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:

            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        a1 = self.att1(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, a1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# =====================================================================
# DICE LOSS
# =====================================================================

def dice_loss_from_logits(
    logits,
    targets,
    eps=1e-7
):

    probs = torch.sigmoid(
        logits
    )

    probs = probs.reshape(
        probs.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    intersection = (
        probs * targets
    ).sum(dim=1)

    denominator = (
        probs.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


# =====================================================================
# TOTAL LOSS
# =====================================================================

def segmentation_loss(
    logits,
    targets
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss_from_logits(
        logits,
        targets
    )

    return bce + dice


# =====================================================================
# METRICS
# =====================================================================

def calculate_metrics(
    logits,
    target,
    threshold=0.5
):

    probs = torch.sigmoid(
        logits
    )

    pred = (
        probs >= threshold
    ).float()

    target = target.float()

    tp = (
        pred * target
    ).sum().item()

    tn = (
        (1.0 - pred)
        * (1.0 - target)
    ).sum().item()

    fp = (
        pred
        * (1.0 - target)
    ).sum().item()

    fn = (
        (1.0 - pred)
        * target
    ).sum().item()

    dice = (
        2.0 * tp
        / max(
            2.0 * tp + fp + fn,
            1.0
        )
    )

    iou = (
        tp
        / max(
            tp + fp + fn,
            1.0
        )
    )

    precision = (
        tp
        / max(
            tp + fp,
            1.0
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1.0
        )
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# =====================================================================
# TRAIN
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    for batch_idx, batch in enumerate(
        loader,
        1
    ):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        mask = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
        ):

            logits = model(
                image
            )

            loss = segmentation_loss(
                logits,
                mask
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += loss.item()

        if (
            batch_idx == 1
            or batch_idx % 50 == 0
            or batch_idx == len(loader)
        ):

            print(
                f"Batch {batch_idx:4d}/{len(loader):4d} "
                f"| Loss {loss.item():.4f}"
            )

    return (
        running_loss
        / len(loader)
    )


# =====================================================================
# VALIDATION
# =====================================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    all_tp = 0.0
    all_tn = 0.0
    all_fp = 0.0
    all_fn = 0.0

    for batch_idx, batch in enumerate(
        loader,
        1
    ):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        mask = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            image
        )

        loss = segmentation_loss(
            logits,
            mask
        )

        total_loss += loss.item()

        metrics = calculate_metrics(
            logits,
            mask,
            THRESHOLD
        )

        all_tp += metrics["tp"]
        all_tn += metrics["tn"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]

        if (
            batch_idx % 20 == 0
            or batch_idx == len(loader)
        ):

            print(
                f"Validation "
                f"{batch_idx}/{len(loader)}"
            )

    dice = (
        2.0 * all_tp
        / max(
            2.0 * all_tp
            + all_fp
            + all_fn,
            1.0
        )
    )

    iou = (
        all_tp
        / max(
            all_tp
            + all_fp
            + all_fn,
            1.0
        )
    )

    precision = (
        all_tp
        / max(
            all_tp + all_fp,
            1.0
        )
    )

    recall = (
        all_tp
        / max(
            all_tp + all_fn,
            1.0
        )
    )

    return {
        "loss":
            total_loss / len(loader),

        "dice":
            dice,

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall
    }


# =====================================================================
# SAVE VALIDATION SEGMENTATIONS
# =====================================================================

@torch.no_grad()
def save_predictions(
    model,
    loader
):

    model.eval()

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = 0

    for batch in loader:

        image = batch["image"].to(
            DEVICE
        )

        logits = model(
            image
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

        prediction = (
            probability >= THRESHOLD
        ).astype(
            np.uint8
        ) * 255

        input_image = (
            batch["image"][0, 0]
            .cpu()
            .numpy()
            * 255.0
        ).astype(
            np.uint8
        )

        name = batch["name"][0]

        # Input
        Image.fromarray(
            input_image
        ).save(
            PREDICTION_DIR
            / f"{name}_input.png"
        )

        # Binary segmentation
        Image.fromarray(
            prediction
        ).save(
            PREDICTION_DIR
            / f"{name}_segmentation.png"
        )

        # Overlay
        overlay = cv2.cvtColor(
            input_image,
            cv2.COLOR_GRAY2BGR
        )

        overlay[
            prediction > 0
        ] = (0, 255, 0)

        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        Image.fromarray(
            overlay_rgb
        ).save(
            PREDICTION_DIR
            / f"{name}_overlay.png"
        )

        saved += 1

    print()
    print(
        f"Saved {saved} validation prediction sets."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    seed_everything(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("512x512 FULL DATASET BENCHMARK")
    print("MODEL 5: ATTENTION U-NET")
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
        f"Split        : {SPLIT_JSON}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # GPU
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        print("CUDA GPU ENABLED")

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    else:

        print(
            "WARNING: CUDA NOT AVAILABLE"
        )

    print("=" * 70)

    # ---------------------------------------------------------------
    # SPLIT
    # ---------------------------------------------------------------

    train_records, val_records = (
        load_split()
    )

    # ---------------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = FilamentDataset(
        train_records
    )

    val_dataset = FilamentDataset(
        val_records
    )

    print(
        f"Train dataset: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation dataset: "
        f"{len(val_dataset)}"
    )

    # ---------------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = AttentionUNet(
        in_channels=1,
        out_channels=1
    )

    model = model.to(
        DEVICE
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ---------------------------------------------------------------
    # MODEL SANITY CHECK
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    test_batch = next(
        iter(train_loader)
    )

    test_image = test_batch[
        "image"
    ].to(DEVICE)

    print(
        f"Input: "
        f"{tuple(test_image.shape)}"
    )

    with torch.no_grad():

        test_output = model(
            test_image
        )

    print(
        f"Output: "
        f"{tuple(test_output.shape)}"
    )

    expected_shape = (
        test_image.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert test_output.shape == (
        expected_shape
    )

    print()
    print(
        "MODEL SANITY CHECK PASSED"
    )

    # ---------------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=MIN_LR
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        )
    )

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------

    best_dice = -1.0

    history = []

    training_start = time.time()

    best_path = (
        OUTPUT_DIR
        / "best_attention_unet_512.pth"
    )

    final_path = (
        OUTPUT_DIR
        / "final_attention_unet_512.pth"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print("-" * 70)
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d}"
        )
        print("-" * 70)

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        val_metrics = validate(
            model,
            val_loader
        )

        scheduler.step()

        lr = optimizer.param_groups[
            0
        ]["lr"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                val_metrics["loss"],

            "validation_dice":
                val_metrics["dice"],

            "validation_iou":
                val_metrics["iou"],

            "validation_precision":
                val_metrics["precision"],

            "validation_recall":
                val_metrics["recall"],

            "learning_rate":
                lr,

            "epoch_time_seconds":
                epoch_time
        }

        history.append(
            record
        )

        print()
        print(
            f"Train loss:       "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss:  "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Validation Dice:  "
            f"{val_metrics['dice']:.6f}"
        )

        print(
            f"Validation IoU:   "
            f"{val_metrics['iou']:.6f}"
        )

        print(
            f"Precision:        "
            f"{val_metrics['precision']:.6f}"
        )

        print(
            f"Recall:           "
            f"{val_metrics['recall']:.6f}"
        )

        print(
            f"LR:               "
            f"{lr:.8f}"
        )

        print(
            f"Epoch time:       "
            f"{epoch_time:.1f}s"
        )

        # -----------------------------------------------------------
        # BEST CHECKPOINT
        # -----------------------------------------------------------

        if (
            val_metrics["dice"]
            > best_dice
        ):

            best_dice = (
                val_metrics["dice"]
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_dice":
                        best_dice,

                    "config": {
                        "model":
                            "Attention U-Net",

                        "image_size":
                            IMAGE_SIZE,

                        "batch_size":
                            BATCH_SIZE,

                        "epochs":
                            NUM_EPOCHS,

                        "seed":
                            SEED,

                        "threshold":
                            THRESHOLD
                    }
                },
                best_path
            )

            print()
            print(
                "NEW BEST CHECKPOINT"
            )

            print(
                f"Best Dice: "
                f"{best_dice:.6f}"
            )

    # ---------------------------------------------------------------
    # FINAL CHECKPOINT
    # ---------------------------------------------------------------

    torch.save(
        {
            "epoch":
                NUM_EPOCHS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_dice":
                best_dice
        },
        final_path
    )

    training_time = (
        time.time()
        - training_start
    )

    # ---------------------------------------------------------------
    # SAVE METRICS
    # ---------------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "training_metrics_512.json"
    )

    output_metrics = {

        "model":
            "Attention U-Net",

        "dataset":
            "MAGFiLO",

        "total_images":
            707,

        "training_images":
            565,

        "validation_images":
            142,

        "image_size":
            "512x512",

        "batch_size":
            BATCH_SIZE,

        "epochs":
            NUM_EPOCHS,

        "seed":
            SEED,

        "threshold":
            THRESHOLD,

        "best_validation_dice":
            best_dice,

        "training_time_seconds":
            training_time,

        "training_time_hours":
            training_time / 3600.0,

        "history":
            history
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_metrics,
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # LOAD BEST MODEL
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Dice: "
        f"{checkpoint['best_dice']:.6f}"
    )

    # ---------------------------------------------------------------
    # SAVE SEGMENTATIONS
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING VALIDATION SEGMENTATIONS")
    print("=" * 70)

    save_predictions(
        model,
        val_loader
    )

    # ---------------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Model: Attention U-Net"
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    print(
        f"Training time: "
        f"{training_time / 3600:.2f} hours"
    )

    print()
    print("Best checkpoint:")
    print(best_path)

    print()
    print("Final checkpoint:")
    print(final_path)

    print()
    print("Metrics:")
    print(metrics_path)

    print()
    print("Segmented validation images:")
    print(PREDICTION_DIR)

    print("=" * 70)


if __name__ == "__main__":
    main()