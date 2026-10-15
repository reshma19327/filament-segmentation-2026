"""
======================================================================
512x512 FULL DATASET BENCHMARK
MODEL 3: UNET++ + SPINE + TOPOLOGY
======================================================================

Dataset:
    707 images
    565 training
    142 validation
    Fixed seed = 42
    Resolution = 512x512

Model:
    UNet++ + Spine
    + topology-aware clDice loss

Epochs:
    50

GPU:
    CUDA / RTX 3050 6GB

Outputs:
    benchmark_models_512/
        unetpp_spine_topology/
            best_unetpp_spine_topology_512.pth
            final_unetpp_spine_topology_512.pth
            training_metrics_512.json
            predictions/
======================================================================
"""

import os
import json
import random
import time
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# IMPORTANT:
# This is your existing UNet++ + Spine architecture.
from unetpp_spine import UNetPPSpine



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

SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"

SPLIT_JSON = PROJECT_ROOT / "benchmark_512" / "benchmark_split.json"

OUTPUT_DIR = PROJECT_ROOT / "benchmark_models_512" / "unetpp_spine_cldice_finetune"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 20
SEED = 42

LEARNING_RATE = 1e-5
MIN_LR = 1e-6

NUM_WORKERS = 0

# Loss weights
LAMBDA_SPINE = 1.0
LAMBDA_TOPOLOGY = 0.5

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

class FilamentSpineDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def _resolve(self, value, base_dir):
        p = Path(value)

        if p.is_absolute():
            return p

        return base_dir / p

    def load_gray(self, path):
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

        mask = (mask > 127).astype(np.float32)

        return mask

    def __getitem__(self, index):

        record = self.records[index]

        # The benchmark JSON may contain either full paths
        # or relative filenames.
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

        spine_name = (
            record.get("spine")
            or record.get("spine_path")
            or record.get("spine_file")
        )

        image_path = self._resolve(
            image_name,
            IMAGE_DIR
        )

        mask_path = self._resolve(
            mask_name,
            MASK_DIR
        )

        spine_path = self._resolve(
            spine_name,
            SPINE_DIR
        )

        image = self.load_gray(image_path)
        mask = self.load_mask(mask_path)
        spine = self.load_mask(spine_path)

        image = torch.from_numpy(
            image
        ).unsqueeze(0).float()

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0).float()

        spine = torch.from_numpy(
            spine
        ).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "spine": spine,
            "name": Path(image_name).stem
        }


# =====================================================================
# LOAD FIXED SPLIT
# =====================================================================

def load_split():

    print("=" * 70)
    print("LOADING FIXED BENCHMARK SPLIT")
    print("=" * 70)

    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
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

    print(f"Training records   : {len(train_records)}")
    print(f"Validation records : {len(val_records)}")

    assert len(train_records) == 565
    assert len(val_records) == 142

    return train_records, val_records


# =====================================================================
# DICE LOSS
# =====================================================================

def dice_loss_from_logits(logits, targets, eps=1e-7):

    probs = torch.sigmoid(logits)

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
# SOFT SKELETONIZATION
# =====================================================================

def soft_erode(img):

    p1 = -F.max_pool2d(
        -img,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0)
    )

    p2 = -F.max_pool2d(
        -img,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1)
    )

    return torch.min(p1, p2)


def soft_dilate(img):

    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1
    )


def soft_open(img):

    return soft_dilate(
        soft_erode(img)
    )


def soft_skeletonize(
    img,
    iterations=10
):

    img = img.clamp(0.0, 1.0)

    skeleton = F.relu(
        img - soft_open(img)
    )

    for _ in range(iterations):

        img = soft_erode(img)

        delta = F.relu(
            img - soft_open(img)
        )

        skeleton = skeleton + (
            F.relu(
                delta - skeleton * delta
            )
        )

    return skeleton.clamp(0.0, 1.0)


# =====================================================================
# CLDICE
# =====================================================================

def cldice_score(
    pred,
    target,
    iterations=10,
    eps=1e-7
):

    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    pred_skel = soft_skeletonize(
        pred,
        iterations
    )

    target_skel = soft_skeletonize(
        target,
        iterations
    )

    # Topology precision:
    # predicted skeleton inside ground-truth mask
    tprec_num = (
        pred_skel * target
    ).sum(dim=(1, 2, 3))

    tprec_den = pred_skel.sum(
        dim=(1, 2, 3)
    )

    tprec = (
        tprec_num + eps
    ) / (
        tprec_den + eps
    )

    # Topology sensitivity:
    # ground-truth skeleton inside prediction
    tsens_num = (
        target_skel * pred
    ).sum(dim=(1, 2, 3))

    tsens_den = target_skel.sum(
        dim=(1, 2, 3)
    )

    tsens = (
        tsens_num + eps
    ) / (
        tsens_den + eps
    )

    cldice = (
        2.0 * tprec * tsens
    ) / (
        tprec + tsens + eps
    )

    return cldice.mean()


def cldice_loss(
    logits,
    targets,
    iterations=10
):

    pred = torch.sigmoid(logits)

    score = cldice_score(
        pred,
        targets,
        iterations=iterations
    )

    return 1.0 - score


# =====================================================================
# TOTAL LOSS
# =====================================================================

def total_loss(
    filament_logits,
    spine_logits,
    mask,
    spine
):

    # Filament segmentation loss
    filament_bce = F.binary_cross_entropy_with_logits(
        filament_logits,
        mask
    )

    filament_dice = dice_loss_from_logits(
        filament_logits,
        mask
    )

    filament_loss = (
        filament_bce
        + filament_dice
    )

    # Spine supervision
    spine_bce = F.binary_cross_entropy_with_logits(
        spine_logits,
        spine
    )

    spine_dice = dice_loss_from_logits(
        spine_logits,
        spine
    )

    spine_loss = (
        spine_bce
        + spine_dice
    )

    # Topology-aware loss
    topology_loss = cldice_loss(
        filament_logits,
        mask,
        iterations=10
    )

    total = (
        filament_loss
        + LAMBDA_SPINE * spine_loss
        + LAMBDA_TOPOLOGY * topology_loss
    )

    return (
        total,
        filament_loss.detach(),
        spine_loss.detach(),
        topology_loss.detach()
    )


# =====================================================================
# MODEL OUTPUT HANDLING
# =====================================================================

def get_outputs(output):

    if isinstance(output, (tuple, list)):

        if len(output) < 2:
            raise RuntimeError(
                "Model returned only one output. "
                "Expected filament + spine outputs."
            )

        return output[0], output[1]

    if isinstance(output, dict):

        filament = (
            output.get("filament")
            or output.get("mask")
            or output.get("out")
        )

        spine = output.get("spine")

        if filament is None or spine is None:
            raise RuntimeError(
                "Could not find filament/spine outputs."
            )

        return filament, spine

    raise RuntimeError(
        f"Unsupported model output type: {type(output)}"
    )


# =====================================================================
# METRICS
# =====================================================================

def calculate_metrics(
    logits,
    target,
    threshold=0.5
):

    probs = torch.sigmoid(logits)

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
# TRAIN ONE EPOCH
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    epoch
):

    model.train()

    running_total = 0.0
    running_filament = 0.0
    running_spine = 0.0
    running_topology = 0.0

    for batch_idx, batch in enumerate(loader, 1):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        mask = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        spine = batch["spine"].to(
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

            output = model(image)

            filament_logits, spine_logits = get_outputs(
                output
            )

            loss, filament_loss, spine_loss, topology_loss = total_loss(
                filament_logits,
                spine_logits,
                mask,
                spine
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_total += loss.item()
        running_filament += filament_loss.item()
        running_spine += spine_loss.item()
        running_topology += topology_loss.item()

        if (
            batch_idx == 1
            or batch_idx % 50 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"Batch {batch_idx:4d}/{len(loader):4d} "
                f"| Loss {loss.item():.4f}"
            )

    n = len(loader)

    return {
        "loss": running_total / n,
        "filament_loss": running_filament / n,
        "spine_loss": running_spine / n,
        "topology_loss": running_topology / n
    }


# =====================================================================
# VALIDATION
# =====================================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    running_val_loss = 0.0

    all_tp = 0.0
    all_tn = 0.0
    all_fp = 0.0
    all_fn = 0.0

    for batch_idx, batch in enumerate(loader, 1):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        mask = batch["mask"].to(
            DEVICE,
            non_blocking=True
        )

        spine = batch["spine"].to(
            DEVICE,
            non_blocking=True
        )

        output = model(image)

        filament_logits, spine_logits = get_outputs(
            output
        )

        loss, _, _, _ = total_loss(
            filament_logits,
            spine_logits,
            mask,
            spine
        )

        running_val_loss += loss.item()

        metrics = calculate_metrics(
            filament_logits,
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
        "loss": running_val_loss / len(loader),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


# =====================================================================
# SAVE PREDICTIONS
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

        output = model(image)

        filament_logits, spine_logits = get_outputs(
            output
        )

        filament_prob = torch.sigmoid(
            filament_logits
        )[0, 0].cpu().numpy()

        spine_prob = torch.sigmoid(
            spine_logits
        )[0, 0].cpu().numpy()

        filament_mask = (
            filament_prob >= THRESHOLD
        ).astype(np.uint8) * 255

        spine_mask = (
            spine_prob >= THRESHOLD
        ).astype(np.uint8) * 255

        name = batch["name"][0]

        # Input
        img = (
            batch["image"][0, 0]
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)

        Image.fromarray(img).save(
            PREDICTION_DIR
            / f"{name}_input.png"
        )

        Image.fromarray(
            filament_mask
        ).save(
            PREDICTION_DIR
            / f"{name}_filament.png"
        )

        Image.fromarray(
            spine_mask
        ).save(
            PREDICTION_DIR
            / f"{name}_spine.png"
        )

        # Overlay
        overlay = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

        overlay[
            filament_mask > 0
        ] = (0, 255, 0)

        overlay[
            spine_mask > 0
        ] = (0, 0, 255)

        Image.fromarray(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )
        ).save(
            PREDICTION_DIR
            / f"{name}_overlay.png"
        )

        saved += 1

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
    print("MODEL 3: UNET++ + SPINE + TOPOLOGY")
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
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    else:

        print("WARNING: CUDA NOT AVAILABLE")

    print("=" * 70)

    # ---------------------------------------------------------------
    # SPLIT
    # ---------------------------------------------------------------

    train_records, val_records = load_split()

    train_dataset = FilamentSpineDataset(
        train_records
    )

    val_dataset = FilamentSpineDataset(
        val_records
    )

    print()
    print("DATASETS")
    print(
        f"Train dataset: {len(train_dataset)}"
    )

    print(
        f"Validation dataset: {len(val_dataset)}"
    )

    # ---------------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = UNetPPSpine(
    in_channels=1,
    base_channels=16
)

    model = model.to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameters:,}"
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

    test_image = test_batch["image"].to(
        DEVICE
    )

    print(
        f"Input: {tuple(test_image.shape)}"
    )

    with torch.no_grad():

        test_output = model(
            test_image
        )

        test_filament, test_spine = get_outputs(
            test_output
        )

    print(
        f"Filament output: "
        f"{tuple(test_filament.shape)}"
    )

    print(
        f"Spine output: "
        f"{tuple(test_spine.shape)}"
    )

    assert test_filament.shape == (
        test_image.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert test_spine.shape == (
        test_image.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print()
    print("MODEL SANITY CHECK PASSED")

    # ---------------------------------------------------------------
    baseline_path = PROJECT_ROOT / 'benchmark_models_512' / 'unetpp_spine_only' / 'best_unetpp_spine_only_512.pth'
    if not baseline_path.is_file():
        raise FileNotFoundError(f'Baseline checkpoint missing: {baseline_path}')
    baseline_checkpoint = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(baseline_checkpoint['model_state_dict'], strict=True)
    print(f'Loaded clean baseline checkpoint from {baseline_path}')
    print(f'Baseline epoch: {baseline_checkpoint.get("epoch", "unknown")}')

    # OPTIMIZER
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LR
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------

    best_dice = -1.0

    history = []

    training_start = time.time()

    best_path = (
        OUTPUT_DIR
        / "best_unetpp_spine_topology_512.pth"
    )

    final_path = (
        OUTPUT_DIR
        / "final_unetpp_spine_topology_512.pth"
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

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch
        )

        val_metrics = validate(
            model,
            val_loader
        )

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_filament_loss":
                train_metrics["filament_loss"],

            "train_spine_loss":
                train_metrics["spine_loss"],

            "train_topology_loss":
                train_metrics["topology_loss"],

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

        history.append(record)

        print()
        print(
            f"Train loss:       "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train filament:   "
            f"{train_metrics['filament_loss']:.6f}"
        )

        print(
            f"Train spine:      "
            f"{train_metrics['spine_loss']:.6f}"
        )

        print(
            f"Train topology:   "
            f"{train_metrics['topology_loss']:.6f}"
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

        if val_metrics["dice"] > best_dice:

            best_dice = (
                val_metrics["dice"]
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_dice":
                        best_dice,
                    "config": {
                        "image_size":
                            IMAGE_SIZE,
                        "batch_size":
                            BATCH_SIZE,
                        "epochs":
                            NUM_EPOCHS,
                        "seed":
                            SEED,
                        "lambda_spine":
                            LAMBDA_SPINE,
                        "lambda_topology":
                            LAMBDA_TOPOLOGY,
                        "threshold":
                            THRESHOLD
                    }
                },
                best_path
            )

            print()
            print(
                f"NEW BEST CHECKPOINT"
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
            "epoch": NUM_EPOCHS,
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
            "UNet++ + Spine + Topology",

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

        "lambda_spine":
            LAMBDA_SPINE,

        "lambda_topology":
            LAMBDA_TOPOLOGY,

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
    print("LOADING BEST MODEL FOR FINAL PREDICTIONS")
    print("=" * 70)

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
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
    # SAVE SEGMENTATION OUTPUTS
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
        f"Model: UNet++ + Spine + Topology"
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
