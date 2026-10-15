from pathlib import Path
import random
import time
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from dataset_unetpp_spine import UNetPPSpineDataset
from unetpp_spine import UNetPPSpine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"
SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

OLD_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints_512"
    / "best_unetpp_spine_512.pth"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_512_improved"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_unetpp_spine_512_improved.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "last_unetpp_spine_512_improved.pth"
)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

IMAGE_SIZE = 512

EXPECTED_IMAGES = 707

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 4

NUM_EPOCHS = 60

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP = 1.0


# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

# Filament segmentation is the primary objective.
MASK_WEIGHT = 1.0

# Spine is an auxiliary structural task.
SPINE_WEIGHT = 0.25

# BCE helps pixel classification.
BCE_WEIGHT = 0.5

# Dice is the main segmentation overlap objective.
DICE_WEIGHT = 1.0


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

NUM_WORKERS = 0

PIN_MEMORY = True


# ------------------------------------------------------------
# AMP
# ------------------------------------------------------------

USE_AMP = True


# ------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------

USE_AUGMENTATION = True

HORIZONTAL_FLIP_PROB = 0.5

VERTICAL_FLIP_PROB = 0.5

ROTATE_90_PROB = 0.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("=" * 70)
        print("CUDA GPU ENABLED")
        print("=" * 70)

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

        print("=" * 70)

        return device

    print("=" * 70)
    print("WARNING: CUDA NOT AVAILABLE")
    print("Training will use CPU.")
    print("=" * 70)

    return torch.device("cpu")


# ============================================================
# IMAGE DISCOVERY
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def discover_image_names():

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_DIR}"
        )

    if not MASK_DIR.exists():

        raise FileNotFoundError(
            f"Mask directory does not exist:\n{MASK_DIR}"
        )

    if not SPINE_DIR.exists():

        raise FileNotFoundError(
            f"Spine directory does not exist:\n{SPINE_DIR}"
        )

    image_files = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        )
    )

    if len(image_files) == 0:

        raise RuntimeError(
            f"No images found in:\n{IMAGE_DIR}"
        )

    image_names = []

    missing_masks = []
    missing_spines = []

    for image_path in image_files:

        stem = image_path.stem

        mask_path = MASK_DIR / f"{stem}.png"
        spine_path = SPINE_DIR / f"{stem}.png"

        if not mask_path.exists():

            missing_masks.append(stem)

        if not spine_path.exists():

            missing_spines.append(stem)

        if (
            mask_path.exists()
            and spine_path.exists()
        ):

            image_names.append(
                image_path.name
            )

    print(
        f"Images found:           {len(image_files)}"
    )

    print(
        f"Images with both masks: {len(image_names)}"
    )

    if missing_masks:

        print(
            f"Missing masks: {len(missing_masks)}"
        )

        print(
            missing_masks[:10]
        )

    if missing_spines:

        print(
            f"Missing spine masks: {len(missing_spines)}"
        )

        print(
            missing_spines[:10]
        )

    if len(image_names) != EXPECTED_IMAGES:

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} usable images, "
            f"but found {len(image_names)}."
        )

    return image_names


# ============================================================
# PAIRED AUGMENTATION
# ============================================================

def augment_batch(
    images,
    masks,
    spines,
):

    if not USE_AUGMENTATION:

        return images, masks, spines

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    if random.random() < HORIZONTAL_FLIP_PROB:

        images = torch.flip(
            images,
            dims=[3],
        )

        masks = torch.flip(
            masks,
            dims=[3],
        )

        spines = torch.flip(
            spines,
            dims=[3],
        )

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    if random.random() < VERTICAL_FLIP_PROB:

        images = torch.flip(
            images,
            dims=[2],
        )

        masks = torch.flip(
            masks,
            dims=[2],
        )

        spines = torch.flip(
            spines,
            dims=[2],
        )

    # --------------------------------------------------------
    # 90 degree rotation
    # --------------------------------------------------------

    if random.random() < ROTATE_90_PROB:

        k = random.randint(1, 3)

        images = torch.rot90(
            images,
            k=k,
            dims=[2, 3],
        )

        masks = torch.rot90(
            masks,
            k=k,
            dims=[2, 3],
        )

        spines = torch.rot90(
            spines,
            k=k,
            dims=[2, 3],
        )

    return images, masks, spines


# ============================================================
# SOFT DICE LOSS
# ============================================================

class SoftDiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0,
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets,
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        targets = targets.reshape(
            targets.shape[0],
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class ImprovedUNetPPSpineLoss(nn.Module):

    def __init__(
        self,
        mask_weight=1.0,
        spine_weight=0.25,
        bce_weight=0.5,
        dice_weight=1.0,
    ):

        super().__init__()

        self.mask_weight = mask_weight

        self.spine_weight = spine_weight

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = SoftDiceLoss()

    def forward(
        self,
        mask_logits,
        spine_logits,
        mask_target,
        spine_target,
    ):

        # ----------------------------------------------------
        # Filament
        # ----------------------------------------------------

        mask_bce = self.bce(
            mask_logits,
            mask_target,
        )

        mask_dice = self.dice(
            mask_logits,
            mask_target,
        )

        mask_loss = (
            self.bce_weight * mask_bce
            +
            self.dice_weight * mask_dice
        )

        # ----------------------------------------------------
        # Spine
        # ----------------------------------------------------

        spine_bce = self.bce(
            spine_logits,
            spine_target,
        )

        spine_dice = self.dice(
            spine_logits,
            spine_target,
        )

        spine_loss = (
            self.bce_weight * spine_bce
            +
            self.dice_weight * spine_dice
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_loss = (
            self.mask_weight * mask_loss
            +
            self.spine_weight * spine_loss
        )

        return (
            total_loss,
            mask_loss,
            spine_loss,
            mask_bce,
            mask_dice,
            spine_bce,
            spine_dice,
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion(
    probability,
    target,
    threshold,
    confusion,
):

    prediction = (
        probability >= threshold
    )

    target = (
        target >= 0.5
    )

    tp = (
        prediction & target
    ).sum().item()

    tn = (
        (~prediction) & (~target)
    ).sum().item()

    fp = (
        prediction & (~target)
    ).sum().item()

    fn = (
        (~prediction) & target
    ).sum().item()

    confusion[0] += tp
    confusion[1] += tn
    confusion[2] += fp
    confusion[3] += fn


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    confusion,
):

    tp, tn, fp, fn = [
        float(x)
        for x in confusion
    ]

    eps = 1e-12

    dice = (
        2.0 * tp
    ) / (
        2.0 * tp
        + fp
        + fn
        + eps
    )

    iou = (
        tp
    ) / (
        tp
        + fp
        + fn
        + eps
    )

    precision = (
        tp
    ) / (
        tp
        + fp
        + eps
    )

    recall = (
        tp
    ) / (
        tp
        + fn
        + eps
    )

    accuracy = (
        tp + tn
    ) / (
        tp
        + tn
        + fp
        + fn
        + eps
    )

    numerator = (
        tp * tn
        - fp * fn
    )

    denominator = math.sqrt(
        max(
            (
                tp + fp
            )
            * (
                tp + fn
            )
            * (
                tn + fp
            )
            * (
                tn + fn
            ),
            eps,
        )
    )

    mcc = numerator / denominator

    total = (
        tp + tn + fp + fn
    )

    if total > 0:

        observed = (
            tp + tn
        ) / total

        actual_positive = (
            tp + fn
        )

        actual_negative = (
            tn + fp
        )

        predicted_positive = (
            tp + fp
        )

        predicted_negative = (
            tn + fn
        )

        expected = (
            actual_positive
            * predicted_positive
            +
            actual_negative
            * predicted_negative
        ) / (
            total * total
        )

        if abs(
            1.0 - expected
        ) > eps:

            kappa = (
                observed - expected
            ) / (
                1.0 - expected
            )

        else:

            kappa = 0.0

    else:

        kappa = 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "mcc": mcc,
        "kappa": kappa,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# DATASET SANITY CHECK
# ============================================================

def sanity_check(
    dataset,
    device,
):

    print("\n" + "=" * 70)
    print("DATASET SANITY CHECK")
    print("=" * 70)

    image, mask, spine, name = dataset[0]

    print(
        f"Image: {name}"
    )

    print(
        f"Image shape: {tuple(image.shape)}"
    )

    print(
        f"Mask shape:  {tuple(mask.shape)}"
    )

    print(
        f"Spine shape: {tuple(spine.shape)}"
    )

    print(
        f"Image dtype: {image.dtype}"
    )

    print(
        f"Mask dtype:  {mask.dtype}"
    )

    print(
        f"Spine dtype: {spine.dtype}"
    )

    print(
        f"Image range: "
        f"{image.min().item():.4f} -> "
        f"{image.max().item():.4f}"
    )

    print(
        f"Mask values: "
        f"{torch.unique(mask).tolist()}"
    )

    print(
        f"Spine values: "
        f"{torch.unique(spine).tolist()}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    images, masks, spines, names = next(
        iter(loader)
    )

    print("\nBatch:")

    print(
        f"Images: {tuple(images.shape)}"
    )

    print(
        f"Masks:  {tuple(masks.shape)}"
    )

    print(
        f"Spines: {tuple(spines.shape)}"
    )

    images = images.to(
        device,
        non_blocking=True,
    )

    masks = masks.to(
        device,
        non_blocking=True,
    )

    spines = spines.to(
        device,
        non_blocking=True,
    )

    model = UNetPPSpine().to(
        device
    )

    model.eval()

    with torch.no_grad():

        mask_logits, spine_logits = model(
            images
        )

    print("\nModel outputs:")

    print(
        f"Mask:  {tuple(mask_logits.shape)}"
    )

    print(
        f"Spine: {tuple(spine_logits.shape)}"
    )

    if mask_logits.shape != masks.shape:

        raise RuntimeError(
            "Mask output shape mismatch."
        )

    if spine_logits.shape != spines.shape:

        raise RuntimeError(
            "Spine output shape mismatch."
        )

    print(
        "\nSANITY CHECK PASSED"
    )


# ============================================================
# LOAD INITIAL CHECKPOINT
# ============================================================

def load_initial_checkpoint(
    model,
    device,
):

    if not OLD_CHECKPOINT.exists():

        print(
            "\nNo previous checkpoint found."
        )

        print(
            "Training will start from scratch."
        )

        return 0

    print("\n" + "=" * 70)
    print("LOADING PREVIOUS 512x512 CHECKPOINT")
    print("=" * 70)

    print(
        OLD_CHECKPOINT
    )

    checkpoint = torch.load(
        OLD_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        epoch = checkpoint.get(
            "epoch",
            0,
        )

        saved_dice = checkpoint.get(
            "best_val_dice",
            None,
        )

    else:

        model.load_state_dict(
            checkpoint
        )

        epoch = 0

        saved_dice = None

    print(
        f"Checkpoint epoch: {epoch}"
    )

    if saved_dice is not None:

        print(
            f"Saved Dice: {saved_dice:.6f}"
        )

    print(
        "Checkpoint loaded successfully."
    )

    return epoch


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):

    model.train()

    total_sum = 0.0
    mask_sum = 0.0
    spine_sum = 0.0

    sample_count = 0

    for (
        images,
        masks,
        spines,
        image_names,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        spines = spines.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Paired augmentation
        # ----------------------------------------------------

        images, masks, spines = augment_batch(
            images,
            masks,
            spines,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            mask_logits, spine_logits = model(
                images
            )

            (
                total_loss,
                mask_loss,
                spine_loss,
                _,
                _,
                _,
                _,
            ) = criterion(
                mask_logits,
                spine_logits,
                masks,
                spines,
            )

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "Non-finite training loss."
            )

        if (
            USE_AMP
            and device.type == "cuda"
        ):

            scaler.scale(
                total_loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

        batch_size = images.size(0)

        total_sum += (
            total_loss.item()
            * batch_size
        )

        mask_sum += (
            mask_loss.item()
            * batch_size
        )

        spine_sum += (
            spine_loss.item()
            * batch_size
        )

        sample_count += batch_size

    return (
        total_sum / sample_count,
        mask_sum / sample_count,
        spine_sum / sample_count,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):

    model.eval()

    total_sum = 0.0
    mask_sum = 0.0
    spine_sum = 0.0

    sample_count = 0

    mask_confusion = [
        0,
        0,
        0,
        0,
    ]

    spine_confusion = [
        0,
        0,
        0,
        0,
    ]

    inference_time = 0.0

    image_count = 0

    for (
        images,
        masks,
        spines,
        image_names,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        spines = spines.to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):

            mask_logits, spine_logits = model(
                images
            )

            (
                total_loss,
                mask_loss,
                spine_loss,
                _,
                _,
                _,
                _,
            ) = criterion(
                mask_logits,
                spine_logits,
                masks,
                spines,
            )

        if device.type == "cuda":

            torch.cuda.synchronize()

        inference_time += (
            time.perf_counter() - start
        )

        batch_size = images.size(0)

        total_sum += (
            total_loss.item()
            * batch_size
        )

        mask_sum += (
            mask_loss.item()
            * batch_size
        )

        spine_sum += (
            spine_loss.item()
            * batch_size
        )

        sample_count += batch_size

        image_count += batch_size

        mask_probability = torch.sigmoid(
            mask_logits
        )

        spine_probability = torch.sigmoid(
            spine_logits
        )

        update_confusion(
            mask_probability,
            masks,
            threshold,
            mask_confusion,
        )

        update_confusion(
            spine_probability,
            spines,
            threshold,
            spine_confusion,
        )

    mask_metrics = calculate_metrics(
        mask_confusion
    )

    spine_metrics = calculate_metrics(
        spine_confusion
    )

    return {
        "loss": total_sum / sample_count,
        "mask_loss": mask_sum / sample_count,
        "spine_loss": spine_sum / sample_count,
        "mask": mask_metrics,
        "spine": spine_metrics,
        "inference_seconds_per_image": (
            inference_time / image_count
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 70)
    print("512x512 UNET++ + SPINE IMPROVED TRAINING")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Image directory: {IMAGE_DIR}"
    )

    print(
        f"Mask directory: {MASK_DIR}"
    )

    print(
        f"Spine directory: {SPINE_DIR}"
    )

    print("=" * 70)

    device = get_device()

    # --------------------------------------------------------
    # Discover dataset
    # --------------------------------------------------------

    image_names = discover_image_names()

    print("\nDataset:")

    print(
        f"Total images:     {len(image_names)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    full_dataset = UNetPPSpineDataset(
        image_dir=str(
            IMAGE_DIR
        ),
        mask_dir=str(
            MASK_DIR
        ),
        spine_dir=str(
            SPINE_DIR
        ),
        image_names=image_names,
        image_size=IMAGE_SIZE,
    )

    if len(full_dataset) != EXPECTED_IMAGES:

        raise RuntimeError(
            "Unexpected dataset size."
        )

    # --------------------------------------------------------
    # Exact same deterministic split
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_size = int(
        TRAIN_RATIO
        * len(full_dataset)
    )

    val_size = (
        len(full_dataset)
        - train_size
    )

    train_subset, val_subset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
        ],
        generator=generator,
    )

    print(
        f"Training images:  {len(train_subset)}"
    )

    print(
        f"Validation images:{len(val_subset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    sanity_check(
        full_dataset,
        device,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNetPPSpine().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Load existing 512 checkpoint
    # --------------------------------------------------------

    load_initial_checkpoint(
        model,
        device,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = ImprovedUNetPPSpineLoss(
        mask_weight=MASK_WEIGHT,
        spine_weight=SPINE_WEIGHT,
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Cosine scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            USE_AMP
            and device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        f"Image size:        {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size:        {BATCH_SIZE}"
    )

    print(
        f"Epochs:            {NUM_EPOCHS}"
    )

    print(
        f"Learning rate:     {LEARNING_RATE}"
    )

    print(
        f"Weight decay:      {WEIGHT_DECAY}"
    )

    print(
        f"Mask weight:       {MASK_WEIGHT}"
    )

    print(
        f"Spine weight:      {SPINE_WEIGHT}"
    )

    print(
        f"BCE weight:        {BCE_WEIGHT}"
    )

    print(
        f"Dice weight:       {DICE_WEIGHT}"
    )

    print(
        f"Gradient clip:     {GRADIENT_CLIP}"
    )

    print(
        f"Augmentation:      {USE_AUGMENTATION}"
    )

    print(
        f"AMP:               {USE_AMP}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Best tracking
    # --------------------------------------------------------

    best_dice = -float("inf")

    best_iou = -float("inf")

    best_epoch = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.perf_counter()

        train_total, train_mask, train_spine = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
        )

        validation = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )

        scheduler.step()

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        mask = validation["mask"]

        spine = validation["spine"]

        row = {
            "epoch": epoch,

            "train_loss": train_total,

            "train_mask_loss": train_mask,

            "train_spine_loss": train_spine,

            "val_loss": validation["loss"],

            "val_mask_loss": validation["mask_loss"],

            "val_spine_loss": validation["spine_loss"],

            "dice": mask["dice"],

            "iou": mask["iou"],

            "precision": mask["precision"],

            "recall": mask["recall"],

            "accuracy": mask["accuracy"],

            "mcc": mask["mcc"],

            "kappa": mask["kappa"],

            "spine_dice": spine["dice"],

            "spine_iou": spine["iou"],

            "lr": current_lr,

            "epoch_time": epoch_time,
        }

        history.append(row)

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}"
        )

        print("-" * 70)

        print(
            f"Train loss:       {train_total:.6f}"
        )

        print(
            f"Train mask loss:  {train_mask:.6f}"
        )

        print(
            f"Train spine loss: {train_spine:.6f}"
        )

        print(
            f"Val loss:         {validation['loss']:.6f}"
        )

        print(
            f"Filament Dice:    {mask['dice']:.6f}"
        )

        print(
            f"Filament IoU:     {mask['iou']:.6f}"
        )

        print(
            f"Precision:        {mask['precision']:.6f}"
        )

        print(
            f"Recall:           {mask['recall']:.6f}"
        )

        print(
            f"Spine Dice:       {spine['dice']:.6f}"
        )

        print(
            f"Spine IoU:        {spine['iou']:.6f}"
        )

        print(
            f"LR:               {current_lr:.8f}"
        )

        print(
            f"Epoch time:       {epoch_time:.1f} seconds"
        )

        print(
            f"Inference:        "
            f"{validation['inference_seconds_per_image']:.5f} "
            f"s/image"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if mask["dice"] > best_dice:

            best_dice = mask["dice"]

            best_iou = mask["iou"]

            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_dice":
                    best_dice,

                "best_val_iou":
                    best_iou,

                "history":
                    history,

                "config": {
                    "image_size":
                        IMAGE_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "num_epochs":
                        NUM_EPOCHS,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "mask_weight":
                        MASK_WEIGHT,

                    "spine_weight":
                        SPINE_WEIGHT,

                    "bce_weight":
                        BCE_WEIGHT,

                    "dice_weight":
                        DICE_WEIGHT,

                    "random_seed":
                        RANDOM_SEED,

                    "threshold":
                        0.5,

                    "augmentation":
                        USE_AUGMENTATION,
                },
            }

            torch.save(
                checkpoint,
                BEST_CHECKPOINT,
            )

            print(
                "\n*** NEW BEST MODEL ***"
            )

            print(
                f"Dice: {best_dice:.6f}"
            )

            print(
                f"IoU:  {best_iou:.6f}"
            )

            print(
                f"Saved: {BEST_CHECKPOINT}"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        last_checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_dice":
                best_dice,

            "best_val_iou":
                best_iou,

            "history":
                history,
        }

        torch.save(
            last_checkpoint,
            LAST_CHECKPOINT,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("IMPROVED TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    print(
        f"Best validation IoU:  "
        f"{best_iou:.6f}"
    )

    print(
        f"Best epoch:            "
        f"{best_epoch}"
    )

    print("\nBest checkpoint:")

    print(
        BEST_CHECKPOINT
    )

    print("\nLast checkpoint:")

    print(
        LAST_CHECKPOINT
    )

    print("\nNext step:")

    print(
        "Run threshold tuning on the BEST checkpoint."
    )

    print(
        "Do NOT use the last checkpoint unless needed."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()