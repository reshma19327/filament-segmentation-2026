from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_unetpp_spine import UNetPPSpineDataset
from unetpp_spine import UNetPPSpine
from loss_unetpp_spine_cldice import UNetPPSpineCLDiceLoss


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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_cldice"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_unetpp_spine_cldice.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last_unetpp_spine_cldice.pth"
)

IMAGE_SIZE = 256

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

NUM_WORKERS = 0
PIN_MEMORY = True

# ------------------------------------------------------------
# LOSS CONFIGURATION
# ------------------------------------------------------------

MASK_WEIGHT = 1.0
SPINE_WEIGHT = 0.5

BCE_WEIGHT = 1.0
CLDICE_WEIGHT = 1.0
SKELETON_ITERATIONS = 10


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
            f"CUDA version reported by PyTorch: "
            f"{torch.version.cuda}"
        )

        print("=" * 70)

        return device

    print("=" * 70)
    print("WARNING: CUDA NOT AVAILABLE")
    print("Training will run on CPU.")
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
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_files:
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
            image_names.append(image_path.name)

    print(
        f"Images found:           {len(image_files)}"
    )

    print(
        f"Images with both masks: {len(image_names)}"
    )

    if missing_masks:

        print(
            f"Missing filament masks: "
            f"{len(missing_masks)}"
        )

        print(
            "First missing masks:",
            missing_masks[:10]
        )

    if missing_spines:

        print(
            f"Missing spine masks:    "
            f"{len(missing_spines)}"
        )

        print(
            "First missing spines:",
            missing_spines[:10]
        )

    if len(image_names) != 707:

        raise RuntimeError(
            f"Expected 707 usable physical images, "
            f"but found {len(image_names)}."
        )

    return image_names


# ============================================================
# LOSS FACTORY
# ============================================================

def create_criterion():

    return UNetPPSpineCLDiceLoss(
        mask_weight=MASK_WEIGHT,
        spine_weight=SPINE_WEIGHT,
        bce_weight=BCE_WEIGHT,
        cldice_weight=CLDICE_WEIGHT,
        skeleton_iterations=SKELETON_ITERATIONS,
    )


# ============================================================
# DATASET / MODEL SANITY CHECK
# ============================================================

def sanity_check_dataset(dataset, device):

    print("\n" + "=" * 70)
    print("DATASET / MODEL SANITY CHECK")
    print("=" * 70)

    image, mask, spine, image_name = dataset[0]

    print(f"First image:  {image_name}")
    print(f"Image shape:  {tuple(image.shape)}")
    print(f"Mask shape:   {tuple(mask.shape)}")
    print(f"Spine shape:  {tuple(spine.shape)}")

    print(f"Image dtype:  {image.dtype}")
    print(f"Mask dtype:   {mask.dtype}")
    print(f"Spine dtype:  {spine.dtype}")

    print(
        f"Image range:  "
        f"{image.min().item():.6f} -> "
        f"{image.max().item():.6f}"
    )

    print(
        f"Mask values:  "
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

    images, masks, spines, image_names = next(
        iter(loader)
    )

    print("\nFirst batch:")
    print(f"Images: {tuple(images.shape)}")
    print(f"Masks:  {tuple(masks.shape)}")
    print(f"Spines: {tuple(spines.shape)}")

    images = images.to(
        device,
        non_blocking=True
    )

    masks = masks.to(
        device,
        non_blocking=True
    )

    spines = spines.to(
        device,
        non_blocking=True
    )

    model = UNetPPSpine().to(device)

    # IMPORTANT:
    # This is the clDice loss, not the old Dice loss.
    criterion = create_criterion()

    model.eval()

    with torch.no_grad():

        mask_prediction, spine_prediction = model(
            images
        )

        (
            total_loss,
            mask_loss,
            spine_loss,
            mask_bce,
            mask_cldice,
            spine_bce,
            spine_cldice,
        ) = criterion(
            mask_prediction,
            spine_prediction,
            masks,
            spines,
        )

    print("\nModel outputs:")
    print(
        f"Mask prediction:  "
        f"{tuple(mask_prediction.shape)}"
    )

    print(
        f"Spine prediction: "
        f"{tuple(spine_prediction.shape)}"
    )

    print("\nLoss:")
    print(
        f"Total:       {total_loss.item():.6f}"
    )

    print(
        f"Mask total:  {mask_loss.item():.6f}"
    )

    print(
        f"Mask BCE:    {mask_bce.item():.6f}"
    )

    print(
        f"Mask clDice: {mask_cldice.item():.6f}"
    )

    print(
        f"Spine total: {spine_loss.item():.6f}"
    )

    print(
        f"Spine BCE:   {spine_bce.item():.6f}"
    )

    print(
        f"Spine clDice:{spine_cldice.item():.6f}"
    )

    if not torch.isfinite(total_loss):

        raise RuntimeError(
            "Sanity check failed: "
            "loss is not finite."
        )

    if mask_prediction.shape != masks.shape:

        raise RuntimeError(
            "Mask output shape mismatch: "
            f"{mask_prediction.shape} "
            f"vs {masks.shape}"
        )

    if spine_prediction.shape != spines.shape:

        raise RuntimeError(
            "Spine output shape mismatch: "
            f"{spine_prediction.shape} "
            f"vs {spines.shape}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("\nSANITY CHECK PASSED")
    print("=" * 70)


# ============================================================
# METRICS
# ============================================================

def update_confusion_matrix(
    prediction,
    target,
    confusion,
):

    prediction = (
        prediction >= 0.5
    ).to(torch.bool)

    target = (
        target >= 0.5
    ).to(torch.bool)

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


def calculate_metrics(confusion):

    tp, tn, fp, fn = [
        float(x) for x in confusion
    ]

    eps = 1e-12

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + eps)
    )

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    precision = (
        tp
        / (tp + fp + eps)
    )

    recall = (
        tp
        / (tp + fn + eps)
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + eps)
    )

    mcc_numerator = (
        tp * tn
        - fp * fn
    )

    mcc_denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if mcc_denominator > eps:

        mcc = (
            mcc_numerator
            / (mcc_denominator ** 0.5)
        )

    else:
        mcc = 0.0

    total = (
        tp + tn + fp + fn
    )

    if total > 0:

        observed_agreement = (
            tp + tn
        ) / total

        actual_positive = tp + fn
        actual_negative = tn + fp

        predicted_positive = tp + fp
        predicted_negative = tn + fn

        expected_agreement = (
            actual_positive
            * predicted_positive
            +
            actual_negative
            * predicted_negative
        ) / (total * total)

        if (
            abs(
                1.0 - expected_agreement
            ) > eps
        ):

            kappa = (
                observed_agreement
                - expected_agreement
            ) / (
                1.0 - expected_agreement
            )

        else:
            kappa = 0.0

    else:
        kappa = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_total = 0.0
    running_mask = 0.0
    running_spine = 0.0

    num_samples = 0

    for (
        images,
        masks,
        spines,
        image_names,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        spines = spines.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        mask_prediction, spine_prediction = model(
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
            mask_prediction,
            spine_prediction,
            masks,
            spines,
        )

        if not torch.isfinite(total_loss):

            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        total_loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_total += (
            total_loss.item()
            * batch_size
        )

        running_mask += (
            mask_loss.item()
            * batch_size
        )

        running_spine += (
            spine_loss.item()
            * batch_size
        )

        num_samples += batch_size

    return (
        running_total / num_samples,
        running_mask / num_samples,
        running_spine / num_samples,
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
):

    model.eval()

    running_total = 0.0
    running_mask = 0.0
    running_spine = 0.0

    num_samples = 0

    mask_confusion = [0, 0, 0, 0]
    spine_confusion = [0, 0, 0, 0]

    inference_time = 0.0
    inference_images = 0

    for (
        images,
        masks,
        spines,
        image_names,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        spines = spines.to(
            device,
            non_blocking=True
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        mask_prediction, spine_prediction = model(
            images
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time += (
            time.perf_counter()
            - start_time
        )

        inference_images += images.size(0)

        (
            total_loss,
            mask_loss,
            spine_loss,
            _,
            _,
            _,
            _,
        ) = criterion(
            mask_prediction,
            spine_prediction,
            masks,
            spines,
        )

        batch_size = images.size(0)

        running_total += (
            total_loss.item()
            * batch_size
        )

        running_mask += (
            mask_loss.item()
            * batch_size
        )

        running_spine += (
            spine_loss.item()
            * batch_size
        )

        num_samples += batch_size

        mask_probability = torch.sigmoid(
            mask_prediction
        )

        spine_probability = torch.sigmoid(
            spine_prediction
        )

        update_confusion_matrix(
            mask_probability,
            masks,
            mask_confusion,
        )

        update_confusion_matrix(
            spine_probability,
            spines,
            spine_confusion,
        )

    mask_metrics = calculate_metrics(
        mask_confusion
    )

    spine_metrics = calculate_metrics(
        spine_confusion
    )

    results = {
        "loss": (
            running_total
            / num_samples
        ),
        "mask_loss": (
            running_mask
            / num_samples
        ),
        "spine_loss": (
            running_spine
            / num_samples
        ),
        "mask_metrics": mask_metrics,
        "spine_metrics": spine_metrics,
        "inference_seconds_per_image": (
            inference_time
            / inference_images
        ),
    }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("UNET++ + SPINE + clDice TRAINING")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Image dir:    {IMAGE_DIR}"
    )

    print(
        f"Mask dir:     {MASK_DIR}"
    )

    print(
        f"Spine dir:    {SPINE_DIR}"
    )

    print("=" * 70)

    device = get_device()

    # --------------------------------------------------------
    # Discover images
    # --------------------------------------------------------

    image_names = discover_image_names()

    print("\nDataset confirmed:")
    print(
        f"Total usable images: {len(image_names)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    full_dataset = UNetPPSpineDataset(
        image_dir=str(IMAGE_DIR),
        mask_dir=str(MASK_DIR),
        spine_dir=str(SPINE_DIR),
        image_names=image_names,
        image_size=IMAGE_SIZE,
    )

    if len(full_dataset) != 707:

        raise RuntimeError(
            f"Dataset length is "
            f"{len(full_dataset)}, expected 707."
        )

    # --------------------------------------------------------
    # Deterministic split
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

    train_subset, val_subset = (
        torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )
    )

    print(
        f"Training images:   "
        f"{len(train_subset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_subset)}"
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

    sanity_check_dataset(
        full_dataset,
        device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNetPPSpine().to(device)

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
    # clDice criterion
    # --------------------------------------------------------

    criterion = create_criterion()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_dice = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    history = []

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(f"Epochs:              {NUM_EPOCHS}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Mask weight:         {MASK_WEIGHT}")
    print(f"Spine weight:        {SPINE_WEIGHT}")
    print(f"BCE weight:          {BCE_WEIGHT}")
    print(f"clDice weight:       {CLDICE_WEIGHT}")
    print(
        f"Skeleton iterations: "
        f"{SKELETON_ITERATIONS}"
    )
    print(f"Device:              {device}")

    print("=" * 70)

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.perf_counter()

        train_total, train_mask, train_spine = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_results = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            val_results["loss"]
        )

        val_mask = (
            val_results["mask_metrics"]
        )

        val_spine = (
            val_results["spine_metrics"]
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_total,
            "train_mask_loss": train_mask,
            "train_spine_loss": train_spine,
            "val_loss": val_results["loss"],
            "val_mask_loss": val_results["mask_loss"],
            "val_spine_loss": val_results["spine_loss"],
            "val_dice": val_mask["dice"],
            "val_iou": val_mask["iou"],
            "val_precision": val_mask["precision"],
            "val_recall": val_mask["recall"],
            "val_accuracy": val_mask["accuracy"],
            "val_mcc": val_mask["mcc"],
            "val_kappa": val_mask["kappa"],
            "val_spine_dice": val_spine["dice"],
            "val_spine_iou": val_spine["iou"],
            "lr": current_lr,
            "epoch_seconds": epoch_time,
        }

        history.append(row)

        print("\n" + "-" * 70)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        )
        print("-" * 70)

        print(
            f"Train | total={train_total:.6f} "
            f"mask={train_mask:.6f} "
            f"spine={train_spine:.6f}"
        )

        print(
            f"Val   | total={val_results['loss']:.6f} "
            f"mask={val_results['mask_loss']:.6f} "
            f"spine={val_results['spine_loss']:.6f}"
        )

        print(
            f"Mask  | Dice={val_mask['dice']:.4f} "
            f"IoU={val_mask['iou']:.4f} "
            f"Precision={val_mask['precision']:.4f} "
            f"Recall={val_mask['recall']:.4f}"
        )

        print(
            f"Mask  | Accuracy={val_mask['accuracy']:.4f} "
            f"MCC={val_mask['mcc']:.4f} "
            f"Kappa={val_mask['kappa']:.4f}"
        )

        print(
            f"Spine | Dice={val_spine['dice']:.4f} "
            f"IoU={val_spine['iou']:.4f}"
        )

        print(
            f"Pixels| TP={val_mask['tp']} "
            f"TN={val_mask['tn']} "
            f"FP={val_mask['fp']} "
            f"FN={val_mask['fn']}"
        )

        print(
            f"LR={current_lr:.6g} "
            f"Time={epoch_time:.1f}s "
            f"Inference="
            f"{val_results['inference_seconds_per_image']:.4f}"
            f"s/image"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_mask["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_mask["dice"]
            )

            best_val_loss = (
                val_results["loss"]
            )

            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "best_val_dice": (
                    best_val_dice
                ),
                "best_val_loss": (
                    best_val_loss
                ),
                "history": history,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "mask_weight": MASK_WEIGHT,
                    "spine_weight": SPINE_WEIGHT,
                    "bce_weight": BCE_WEIGHT,
                    "cldice_weight": CLDICE_WEIGHT,
                    "skeleton_iterations": (
                        SKELETON_ITERATIONS
                    ),
                    "random_seed": RANDOM_SEED,
                },
            }

            torch.save(
                checkpoint,
                BEST_CHECKPOINT
            )

            print(
                f"*** NEW BEST MODEL SAVED: "
                f"Dice={best_val_dice:.4f} ***"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "best_val_dice": (
                best_val_dice
            ),
            "best_val_loss": (
                best_val_loss
            ),
            "history": history,
            "config": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "mask_weight": MASK_WEIGHT,
                "spine_weight": SPINE_WEIGHT,
                "bce_weight": BCE_WEIGHT,
                "cldice_weight": CLDICE_WEIGHT,
                "skeleton_iterations": (
                    SKELETON_ITERATIONS
                ),
                "random_seed": RANDOM_SEED,
            },
        }

        torch.save(
            last_checkpoint,
            LAST_CHECKPOINT
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.6f}"
    )

    print(
        f"Best validation loss:  "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best epoch:            "
        f"{best_epoch}"
    )

    print("\nBest checkpoint:")
    print(BEST_CHECKPOINT)

    print("\nLast checkpoint:")
    print(LAST_CHECKPOINT)

    print("\nFinal validation metrics:")

    if history:

        final = history[-1]

        print(
            f"Dice:      "
            f"{final['val_dice']:.6f}"
        )

        print(
            f"IoU:       "
            f"{final['val_iou']:.6f}"
        )

        print(
            f"Precision: "
            f"{final['val_precision']:.6f}"
        )

        print(
            f"Recall:    "
            f"{final['val_recall']:.6f}"
        )

        print(
            f"Accuracy:  "
            f"{final['val_accuracy']:.6f}"
        )

        print(
            f"MCC:       "
            f"{final['val_mcc']:.6f}"
        )

        print(
            f"Kappa:     "
            f"{final['val_kappa']:.6f}"
        )

        print(
            f"Spine Dice:"
            f"{final['val_spine_dice']:.6f}"
        )

        print(
            f"Spine IoU: "
            f"{final['val_spine_iou']:.6f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()