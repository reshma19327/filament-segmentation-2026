"""
======================================================================
512x512 FULL DATASET BENCHMARK
MODEL: MASK R-CNN
======================================================================

Dataset:
    707 images
    565 training
    142 validation
    Fixed benchmark split
    Seed = 42
    Resolution = 512x512

Model:
    Mask R-CNN
    ResNet-50 FPN backbone

Training:
    50 epochs
    Batch size = 1
    RTX 3050 6GB

Outputs:
    benchmark_models_512/
        maskrcnn/
            best_maskrcnn_512.pth
            final_maskrcnn_512.pth
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

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "full_masks"
)

SPLIT_JSON = (
    PROJECT_ROOT
    / "benchmark_512"
    / "benchmark_split.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_models_512"
    / "maskrcnn"
)

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predictions"
)

IMAGE_SIZE = 512

# IMPORTANT:
# Mask R-CNN is memory intensive.
BATCH_SIZE = 1

NUM_EPOCHS = 25

SEED = 42

LEARNING_RATE = 1e-4

MIN_LR = 1e-6

NUM_WORKERS = 0

NUM_CLASSES = 2
# class 0 = background
# class 1 = filament

SCORE_THRESHOLD = 0.50


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# PATH RESOLUTION
# =====================================================================

def resolve_path(value, base_dir):

    if value is None:

        return None

    p = Path(value)

    if p.is_absolute():

        return p

    return base_dir / p


# =====================================================================
# DATASET
# =====================================================================

class MaskRCNNDataset(Dataset):

    def __init__(
        self,
        records
    ):

        self.records = records

    def __len__(self):

        return len(self.records)

    def load_image(
        self,
        path
    ):

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
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # Mask R-CNN expects 3-channel images.
        image = np.stack(
            [
                image,
                image,
                image
            ],
            axis=0
        )

        return torch.from_numpy(
            image
        ).float()

    def load_mask(
        self,
        path
    ):

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
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 127
        ).astype(
            np.uint8
        )

        return mask

    def mask_to_boxes(
        self,
        mask
    ):

        # Find connected components.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        boxes = []

        component_masks = []

        for label in range(
            1,
            num_labels
        ):

            x = stats[
                label,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                label,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                label,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area <= 0:
                continue

            # Ignore extremely tiny components.
            if area < 4:
                continue

            x1 = float(x)

            y1 = float(y)

            x2 = float(
                x + w
            )

            y2 = float(
                y + h
            )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

            component_masks.append(
                (
                    labels == label
                ).astype(
                    np.uint8
                )
            )

        # If there are no components, create one box
        # around the entire image only when the mask is nonempty.
        if len(boxes) == 0:

            if mask.sum() > 0:

                ys, xs = np.where(
                    mask > 0
                )

                x1 = float(xs.min())

                y1 = float(ys.min())

                x2 = float(xs.max() + 1)

                y2 = float(ys.max() + 1)

                boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                )

                component_masks.append(
                    mask.astype(
                        np.uint8
                    )
                )

        return (
            boxes,
            component_masks
        )

    def __getitem__(
        self,
        index
    ):

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
                "No image field found in benchmark record."
            )

        if mask_name is None:

            raise KeyError(
                "No mask field found in benchmark record."
            )

        image_path = resolve_path(
            image_name,
            IMAGE_DIR
        )

        mask_path = resolve_path(
            mask_name,
            MASK_DIR
        )

        image = self.load_image(
            image_path
        )

        mask = self.load_mask(
            mask_path
        )

        (
            boxes,
            component_masks
        ) = self.mask_to_boxes(
            mask
        )

        if len(boxes) > 0:

            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            masks_tensor = torch.from_numpy(
                np.stack(
                    component_masks,
                    axis=0
                )
            ).to(
                dtype=torch.uint8
            )

            labels = torch.ones(
                (
                    len(boxes),
                ),
                dtype=torch.int64
            )

            areas = (
                (
                    boxes_tensor[:, 2]
                    - boxes_tensor[:, 0]
                )
                *
                (
                    boxes_tensor[:, 3]
                    - boxes_tensor[:, 1]
                )
            )

            iscrowd = torch.zeros(
                (
                    len(boxes),
                ),
                dtype=torch.int64
            )

        else:

            boxes_tensor = torch.zeros(
                (
                    0,
                    4
                ),
                dtype=torch.float32
            )

            masks_tensor = torch.zeros(
                (
                    0,
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                dtype=torch.uint8
            )

            labels = torch.zeros(
                (
                    0,
                ),
                dtype=torch.int64
            )

            areas = torch.zeros(
                (
                    0,
                ),
                dtype=torch.float32
            )

            iscrowd = torch.zeros(
                (
                    0,
                ),
                dtype=torch.int64
            )

        target = {

            "boxes":
                boxes_tensor,

            "labels":
                labels,

            "masks":
                masks_tensor,

            "image_id":
                torch.tensor(
                    [index],
                    dtype=torch.int64
                ),

            "area":
                areas,

            "iscrowd":
                iscrowd
        }

        return (
            image,
            target,
            Path(
                image_name
            ).stem
        )


# =====================================================================
# COLLATE FUNCTION
# =====================================================================

def collate_fn(
    batch
):

    images = []

    targets = []

    names = []

    for image, target, name in batch:

        images.append(image)

        targets.append(target)

        names.append(name)

    return (
        images,
        targets,
        names
    )


# =====================================================================
# LOAD FIXED SPLIT
# =====================================================================

def load_split():

    print("=" * 70)

    print(
        "LOADING FIXED BENCHMARK SPLIT"
    )

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
            "Training records not found."
        )

    if "validation" in data:

        val_records = data["validation"]

    elif "val" in data:

        val_records = data["val"]

    else:

        raise KeyError(
            "Validation records not found."
        )

    print(
        f"Training records   : "
        f"{len(train_records)}"
    )

    print(
        f"Validation records : "
        f"{len(val_records)}"
    )

    assert len(train_records) == 565

    assert len(val_records) == 142

    return (
        train_records,
        val_records
    )


# =====================================================================
# CREATE MODEL
# =====================================================================

def create_model():

    print("=" * 70)

    print(
        "CREATING MASK R-CNN"
    )

    print("=" * 70)

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE
    )

    # ---------------------------------------------------------------
    # Replace box classifier
    # ---------------------------------------------------------------

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    # ---------------------------------------------------------------
    # Replace mask classifier
    # ---------------------------------------------------------------

    in_features_mask = (
        model.roi_heads
        .mask_predictor
        .conv5_mask
        .in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            NUM_CLASSES
        )
    )

    model = model.to(
        DEVICE
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    return model


# =====================================================================
# MOVE TARGETS TO DEVICE
# =====================================================================

def move_targets_to_device(
    targets
):

    device_targets = []

    for target in targets:

        device_target = {}

        for key, value in target.items():

            if torch.is_tensor(value):

                device_target[key] = value.to(
                    DEVICE,
                    non_blocking=True
                )

            else:

                device_target[key] = value

        device_targets.append(
            device_target
        )

    return device_targets


# =====================================================================
# MODEL SANITY CHECK
# =====================================================================

def model_sanity_check(
    model,
    loader
):

    print()

    print("=" * 70)

    print(
        "MODEL SANITY CHECK"
    )

    print("=" * 70)

    images, targets, names = next(
        iter(loader)
    )

    # ---------------------------------------------------------------
    # MOVE EVERYTHING TO CUDA
    # ---------------------------------------------------------------

    images = [
        image.to(
            DEVICE,
            non_blocking=True
        )
        for image in images
    ]

    targets = move_targets_to_device(
        targets
    )

    print(
        f"Number of images: "
        f"{len(images)}"
    )

    print(
        f"Image shape: "
        f"{tuple(images[0].shape)}"
    )

    print(
        f"Device: "
        f"{images[0].device}"
    )

    print(
        f"Boxes device: "
        f"{targets[0]['boxes'].device}"
    )

    print(
        f"Masks device: "
        f"{targets[0]['masks'].device}"
    )

    model.train()

    with torch.enable_grad():

        loss_dict = model(
            images,
            targets
        )

    total_loss = sum(
        loss
        for loss in loss_dict.values()
    )

    print()

    print(
        "Loss components:"
    )

    for key, value in loss_dict.items():

        print(
            f"  {key}: "
            f"{value.item():.6f}"
        )

    print()

    print(
        f"Total loss: "
        f"{total_loss.item():.6f}"
    )

    assert torch.isfinite(
        total_loss
    ), "Sanity-check loss is not finite."

    print()

    print(
        "MODEL SANITY CHECK PASSED"
    )


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

    running_loss = 0.0

    num_batches = len(loader)

    for batch_idx, batch in enumerate(
        loader,
        start=1
    ):

        images, targets, names = batch

        # -----------------------------------------------------------
        # MOVE IMAGES TO DEVICE
        # -----------------------------------------------------------

        images = [
            image.to(
                DEVICE,
                non_blocking=True
            )
            for image in images
        ]

        # -----------------------------------------------------------
        # MOVE ALL TARGET TENSORS TO DEVICE
        # -----------------------------------------------------------

        targets = move_targets_to_device(
            targets
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # -----------------------------------------------------------
        # MASK R-CNN TRAINING
        # -----------------------------------------------------------

        # Detection models can have issues with AMP on some
        # torchvision/PyTorch combinations, so keep the core
        # Mask R-CNN forward pass in FP32 for stability.
        with torch.amp.autocast(
            device_type="cuda",
            enabled=False
        ):

            loss_dict = model(
                images,
                targets
            )

            losses = sum(
                loss
                for loss in loss_dict.values()
            )

        if not torch.isfinite(
            losses
        ):

            print(
                "WARNING: non-finite loss encountered."
            )

            print(
                loss_dict
            )

            raise RuntimeError(
                "Non-finite Mask R-CNN loss."
            )

        losses.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        loss_value = losses.item()

        running_loss += loss_value

        if (
            batch_idx == 1
            or batch_idx % 50 == 0
            or batch_idx == num_batches
        ):

            print(
                f"Batch "
                f"{batch_idx:4d}/"
                f"{num_batches:4d} "
                f"| Loss "
                f"{loss_value:.4f}"
            )

    return (
        running_loss
        / num_batches
    )


# =====================================================================
# MASK IOU
# =====================================================================

def mask_iou(
    pred,
    target
):

    pred = pred.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    union = np.logical_or(
        pred,
        target
    ).sum()

    if union == 0:

        return 1.0

    return (
        intersection
        / union
    )


# =====================================================================
# MASK DICE
# =====================================================================

def mask_dice(
    pred,
    target
):

    pred = pred.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    pred_sum = pred.sum()

    target_sum = target.sum()

    denominator = (
        pred_sum
        + target_sum
    )

    if denominator == 0:

        return 1.0

    return (
        2.0
        * intersection
        / denominator
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

    dice_scores = []

    iou_scores = []

    precision_scores = []

    recall_scores = []

    validation_loss_total = 0.0

    # ---------------------------------------------------------------
    # Validation loss requires train mode for torchvision detection
    # models. We calculate metrics separately in eval mode.
    # ---------------------------------------------------------------

    for batch_idx, batch in enumerate(
        loader,
        start=1
    ):

        images, targets, names = batch

        images = [
            image.to(
                DEVICE,
                non_blocking=True
            )
            for image in images
        ]

        predictions = model(
            images
        )

        for prediction, target in zip(
            predictions,
            targets
        ):

            # -------------------------------------------------------
            # Ground-truth union mask
            # -------------------------------------------------------

            gt_masks = target[
                "masks"
            ].numpy()

            if len(gt_masks) > 0:

                gt_mask = (
                    gt_masks.max(
                        axis=0
                    )
                    > 0
                )

            else:

                gt_mask = np.zeros(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    dtype=bool
                )

            # -------------------------------------------------------
            # Prediction masks
            # -------------------------------------------------------

            if (
                "masks" in prediction
                and len(prediction["masks"]) > 0
            ):

                scores = (
                    prediction[
                        "scores"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                masks = (
                    prediction[
                        "masks"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                keep = (
                    scores
                    >= SCORE_THRESHOLD
                )

                masks = masks[
                    keep
                ]

                if len(masks) > 0:

                    pred_mask = (
                        masks[:, 0]
                        > 0.5
                    ).any(
                        axis=0
                    )

                else:

                    pred_mask = np.zeros(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        ),
                        dtype=bool
                    )

            else:

                pred_mask = np.zeros(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    dtype=bool
                )

            # -------------------------------------------------------
            # Metrics
            # -------------------------------------------------------

            pred_bool = (
                pred_mask
                .astype(bool)
            )

            gt_bool = (
                gt_mask
                .astype(bool)
            )

            intersection = np.logical_and(
                pred_bool,
                gt_bool
            ).sum()

            pred_sum = pred_bool.sum()

            gt_sum = gt_bool.sum()

            tp = intersection

            fp = max(
                pred_sum - tp,
                0
            )

            fn = max(
                gt_sum - tp,
                0
            )

            dice = (
                2.0 * tp
                / max(
                    2.0 * tp
                    + fp
                    + fn,
                    1
                )
            )

            iou = (
                tp
                / max(
                    tp
                    + fp
                    + fn,
                    1
                )
            )

            precision = (
                tp
                / max(
                    tp + fp,
                    1
                )
            )

            recall = (
                tp
                / max(
                    tp + fn,
                    1
                )
            )

            dice_scores.append(
                dice
            )

            iou_scores.append(
                iou
            )

            precision_scores.append(
                precision
            )

            recall_scores.append(
                recall
            )

        if (
            batch_idx % 20 == 0
            or batch_idx == len(loader)
        ):

            print(
                f"Validation "
                f"{batch_idx}/"
                f"{len(loader)}"
            )

    return {

        "loss":
            validation_loss_total
            / max(
                len(loader),
                1
            ),

        "dice":
            float(
                np.mean(
                    dice_scores
                )
            ),

        "iou":
            float(
                np.mean(
                    iou_scores
                )
            ),

        "precision":
            float(
                np.mean(
                    precision_scores
                )
            ),

        "recall":
            float(
                np.mean(
                    recall_scores
                )
            )
    }


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

@torch.no_grad()
def save_predictions(
    model,
    loader
):

    print()

    print("=" * 70)

    print(
        "SAVING VALIDATION SEGMENTATIONS"
    )

    print("=" * 70)

    model.eval()

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = 0

    for batch in loader:

        images, targets, names = batch

        images = [
            image.to(
                DEVICE,
                non_blocking=True
            )
            for image in images
        ]

        predictions = model(
            images
        )

        for image, prediction, name in zip(
            images,
            predictions,
            names
        ):

            # -------------------------------------------------------
            # Input image
            # -------------------------------------------------------

            img = (
                image[0]
                .detach()
                .cpu()
                .numpy()
                * 255.0
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            Image.fromarray(
                img
            ).save(
                PREDICTION_DIR
                / f"{name}_input.png"
            )

            # -------------------------------------------------------
            # Prediction mask
            # -------------------------------------------------------

            if (
                "masks" in prediction
                and len(prediction["masks"]) > 0
            ):

                scores = (
                    prediction[
                        "scores"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                masks = (
                    prediction[
                        "masks"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                keep = (
                    scores
                    >= SCORE_THRESHOLD
                )

                masks = masks[
                    keep
                ]

                scores_kept = scores[
                    keep
                ]

                if len(masks) > 0:

                    pred_mask = (
                        masks[:, 0]
                        > 0.5
                    ).any(
                        axis=0
                    )

                    pred_mask_uint8 = (
                        pred_mask
                        .astype(
                            np.uint8
                        )
                        * 255
                    )

                else:

                    pred_mask_uint8 = np.zeros(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        ),
                        dtype=np.uint8
                    )

            else:

                pred_mask_uint8 = np.zeros(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    dtype=np.uint8
                )

            Image.fromarray(
                pred_mask_uint8
            ).save(
                PREDICTION_DIR
                / f"{name}_filament.png"
            )

            # -------------------------------------------------------
            # Overlay
            # -------------------------------------------------------

            overlay = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

            overlay[
                pred_mask_uint8 > 0
            ] = (
                0,
                255,
                0
            )

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
        f"Saved {saved} validation "
        f"prediction sets."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    seed_everything(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)

    print(
        "512x512 FULL DATASET BENCHMARK"
    )

    print(
        "MODEL: MASK R-CNN"
    )

    print("=" * 70)

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Image dir    : "
        f"{IMAGE_DIR}"
    )

    print(
        f"Mask dir     : "
        f"{MASK_DIR}"
    )

    print(
        f"Split        : "
        f"{SPLIT_JSON}"
    )

    print(
        f"Output       : "
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # GPU
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        print(
            "CUDA GPU ENABLED"
        )

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

    print(
        "CREATING DATASETS"
    )

    print("=" * 70)

    train_dataset = MaskRCNNDataset(
        train_records
    )

    val_dataset = MaskRCNNDataset(
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
        ),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        collate_fn=collate_fn
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = create_model()

    # ---------------------------------------------------------------
    # SANITY CHECK
    # ---------------------------------------------------------------

    model_sanity_check(
        model,
        train_loader
    )

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # CHECKPOINT PATHS
    # ---------------------------------------------------------------

    best_path = (
        OUTPUT_DIR
        / "best_maskrcnn_512.pth"
    )

    final_path = (
        OUTPUT_DIR
        / "final_maskrcnn_512.pth"
    )

    metrics_path = (
        OUTPUT_DIR
        / "training_metrics_512.json"
    )

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------

    best_dice = -1.0

    history = []

    training_start = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()

        print("-" * 70)

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{NUM_EPOCHS:03d}"
        )

        print("-" * 70)

        epoch_start = time.time()

        # -----------------------------------------------------------
        # TRAIN
        # -----------------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            None,
            epoch
        )

        # -----------------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------------

        val_metrics = validate(
            model,
            val_loader
        )

        scheduler.step()

        lr = (
            optimizer
            .param_groups[0]["lr"]
        )

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
                val_metrics[
                    "loss"
                ],

            "validation_dice":
                val_metrics[
                    "dice"
                ],

            "validation_iou":
                val_metrics[
                    "iou"
                ],

            "validation_precision":
                val_metrics[
                    "precision"
                ],

            "validation_recall":
                val_metrics[
                    "recall"
                ],

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
                val_metrics[
                    "dice"
                ]
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

                    "best_iou":
                        val_metrics[
                            "iou"
                        ],

                    "config": {

                        "image_size":
                            IMAGE_SIZE,

                        "batch_size":
                            BATCH_SIZE,

                        "epochs":
                            NUM_EPOCHS,

                        "seed":
                            SEED,

                        "score_threshold":
                            SCORE_THRESHOLD,

                        "num_classes":
                            NUM_CLASSES
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
    # METRICS
    # ---------------------------------------------------------------

    output_metrics = {

        "model":
            "Mask R-CNN",

        "backbone":
            "ResNet-50 FPN",

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

        "score_threshold":
            SCORE_THRESHOLD,

        "best_validation_dice":
            best_dice,

        "training_time_seconds":
            training_time,

        "training_time_hours":
            training_time
            / 3600.0,

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

    print(
        "LOADING BEST MODEL"
    )

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
    # SAVE PREDICTIONS
    # ---------------------------------------------------------------

    save_predictions(
        model,
        val_loader
    )

    # ---------------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        "Model: Mask R-CNN"
    )

    print(
        "Backbone: ResNet-50 FPN"
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

    print(
        "Best checkpoint:"
    )

    print(
        best_path
    )

    print()

    print(
        "Final checkpoint:"
    )

    print(
        final_path
    )

    print()

    print(
        "Metrics:"
    )

    print(
        metrics_path
    )

    print()

    print(
        "Segmented validation images:"
    )

    print(
        PREDICTION_DIR
    )

    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()