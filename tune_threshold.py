from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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
# IMPORTANT:
# This is the improved 512x512 checkpoint.
# ------------------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints_512_improved"
    / "best_unetpp_spine_512_improved.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "threshold_tuning_512_improved"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "threshold_metrics_512_improved.csv"
BEST_THRESHOLD_PATH = OUTPUT_DIR / "best_threshold_512_improved.txt"

IMAGE_SIZE = 512

BATCH_SIZE = 4

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

NUM_WORKERS = 0

PIN_MEMORY = True

# Threshold search.
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01


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

        print("=" * 65)
        print("CUDA GPU ENABLED")
        print("=" * 65)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print("=" * 65)

        return device

    print("=" * 65)
    print("WARNING: CUDA NOT AVAILABLE")
    print("Using CPU.")
    print("=" * 65)

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

        if mask_path.exists() and spine_path.exists():
            image_names.append(image_path.name)

    print(f"Images found:           {len(image_files)}")
    print(f"Images with both masks: {len(image_names)}")

    if missing_masks:
        print(f"Missing masks:          {len(missing_masks)}")
        print("First missing masks:", missing_masks[:10])

    if missing_spines:
        print(f"Missing spine masks:    {len(missing_spines)}")
        print("First missing spines:", missing_spines[:10])

    if len(image_names) != 707:
        raise RuntimeError(
            f"Expected 707 usable images, "
            f"but found {len(image_names)}."
        )

    return image_names


# ============================================================
# CREATE EXACT SAME VALIDATION SPLIT
# ============================================================

def create_validation_loader(device):
    image_names = discover_image_names()

    print(f"Total images:       {len(image_names)}")

    dataset = UNetPPSpineDataset(
        image_dir=str(IMAGE_DIR),
        mask_dir=str(MASK_DIR),
        spine_dir=str(SPINE_DIR),
        image_names=image_names,
        image_size=IMAGE_SIZE,
    )

    if len(dataset) != 707:
        raise RuntimeError(
            f"Dataset length is {len(dataset)}, expected 707."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # This reproduces the split used by your training scripts.
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_size = int(
        TRAIN_RATIO * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(f"Training images:    {len(train_subset)}")
    print(f"Validation images:  {len(val_subset)}")
    print(f"Image size:         {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size:         {BATCH_SIZE}")

    loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    return loader, len(val_subset)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Make sure this file exists."
        )

    print("\n" + "=" * 65)
    print("LOADING CHECKPOINT")
    print("=" * 65)

    print(CHECKPOINT_PATH)

    model = UNetPPSpine().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # --------------------------------------------------------
    # Support normal checkpoint dictionary.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            # Sometimes the checkpoint itself is
            # a state_dict.
            state_dict = checkpoint

        epoch = checkpoint.get(
            "epoch",
            "unknown",
        )

        saved_dice = checkpoint.get(
            "best_val_dice",
            checkpoint.get(
                "best_dice",
                "unknown",
            ),
        )

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove DataParallel prefix if present.
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            new_key = key[
                len("module.") :
            ]
        else:
            new_key = key

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    print(f"Checkpoint epoch: {epoch}")
    print(f"Saved best Dice:  {saved_dice}")

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion(
    prediction,
    target,
):
    prediction = prediction.bool()
    target = target.bool()

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

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn),
    )


# ============================================================
# METRICS
# ============================================================

def metrics_from_confusion(
    tp,
    tn,
    fp,
    fn,
):
    eps = 1e-12

    dice = (
        2.0 * tp
        /
        (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        /
        (
            tp
            + fp
            + fn
            + eps
        )
    )

    precision = (
        tp
        /
        (
            tp
            + fp
            + eps
        )
    )

    recall = (
        tp
        /
        (
            tp
            + fn
            + eps
        )
    )

    total = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = (
        tp + tn
    ) / (
        total + eps
    )

    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator > eps:
        mcc = (
            tp * tn - fp * fn
        ) / (
            denominator ** 0.5
        )
    else:
        mcc = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):
    print("\n" + "=" * 65)
    print("RUNNING VALIDATION INFERENCE")
    print("=" * 65)

    prediction_batches = []
    target_batches = []

    total_images = 0

    start_time = time.perf_counter()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images = batch[0]
        masks = batch[1]

        images = images.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward pass.
        # ----------------------------------------------------

        mask_logits, spine_logits = model(
            images
        )

        # ----------------------------------------------------
        # We tune the FILAMENT mask only.
        # ----------------------------------------------------

        mask_probability = torch.sigmoid(
            mask_logits
        )

        prediction_batches.append(
            mask_probability.cpu()
        )

        target_batches.append(
            masks.cpu()
        )

        total_images += images.size(0)

        if (
            batch_index == 1
            or total_images % 20 == 0
            or total_images == len(loader.dataset)
        ):
            print(
                f"Processed "
                f"{total_images}/"
                f"{len(loader.dataset)}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    predictions = torch.cat(
        prediction_batches,
        dim=0,
    )

    targets = torch.cat(
        target_batches,
        dim=0,
    )

    print(
        f"\nInference completed in "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Predictions shape: "
        f"{tuple(predictions.shape)}"
    )

    print(
        f"Targets shape:     "
        f"{tuple(targets.shape)}"
    )

    return predictions, targets


# ============================================================
# THRESHOLD VALUES
# ============================================================

def create_thresholds():
    thresholds = []

    value = THRESHOLD_MIN

    while value <= (
        THRESHOLD_MAX + 1e-9
    ):
        thresholds.append(
            round(value, 3)
        )

        value += THRESHOLD_STEP

    return thresholds


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    predictions,
    targets,
    threshold,
):
    binary_predictions = (
        predictions >= threshold
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # --------------------------------------------------------
    # Process image batches to avoid unnecessary GPU memory.
    # Everything is already on CPU.
    # --------------------------------------------------------

    for index in range(
        predictions.shape[0]
    ):

        current_prediction = (
            binary_predictions[index]
        )

        current_target = (
            targets[index] >= 0.5
        )

        current_tp, current_tn, current_fp, current_fn = (
            calculate_confusion(
                current_prediction,
                current_target,
            )
        )

        tp += current_tp
        tn += current_tn
        fp += current_fp
        fn += current_fn

    return metrics_from_confusion(
        tp,
        tn,
        fp,
        fn,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):
    import csv

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "threshold",
                "dice",
                "iou",
                "precision",
                "recall",
                "accuracy",
                "mcc",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )

        for row in results:

            writer.writerow(
                [
                    row["threshold"],
                    row["dice"],
                    row["iou"],
                    row["precision"],
                    row["recall"],
                    row["accuracy"],
                    row["mcc"],
                    row["tp"],
                    row["tn"],
                    row["fp"],
                    row["fn"],
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print(
        "===========================================================================\n"
        "512x512 UNET++ + SPINE IMPROVED\n"
        "FILAMENT SEGMENTATION: THRESHOLD TUNING\n"
        "===========================================================================\n"
    )

    device = get_device()

    print("\nConfiguration:")
    print(
        f"Image size:       "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(
        f"Train ratio:      "
        f"{TRAIN_RATIO}"
    )
    print(
        f"Random seed:      "
        f"{RANDOM_SEED}"
    )
    print(
        f"Threshold range:  "
        f"{THRESHOLD_MIN} -> "
        f"{THRESHOLD_MAX}"
    )
    print(
        f"Threshold step:   "
        f"{THRESHOLD_STEP}"
    )

    # --------------------------------------------------------
    # Validation loader.
    # --------------------------------------------------------

    loader, validation_count = (
        create_validation_loader(
            device
        )
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Inference.
    # --------------------------------------------------------

    predictions, targets = (
        collect_predictions(
            model,
            loader,
            device,
        )
    )

    # --------------------------------------------------------
    # Threshold tuning.
    # --------------------------------------------------------

    thresholds = create_thresholds()

    print("\n")
    print("=" * 75)
    print("THRESHOLD RESULTS")
    print("=" * 75)

    print(
        f"{'Threshold':>10} "
        f"{'Dice':>12} "
        f"{'IoU':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>12}"
    )

    print("-" * 75)

    results = []

    best_result = None

    for threshold in thresholds:

        metrics = evaluate_threshold(
            predictions,
            targets,
            threshold,
        )

        row = {
            "threshold": threshold,
            **metrics,
        }

        results.append(row)

        print(
            f"{threshold:10.3f} "
            f"{metrics['dice']:12.6f} "
            f"{metrics['iou']:12.6f} "
            f"{metrics['precision']:12.6f} "
            f"{metrics['recall']:12.6f}"
        )

        # ----------------------------------------------------
        # Select highest Dice.
        # ----------------------------------------------------

        if (
            best_result is None
            or metrics["dice"]
            > best_result["dice"]
        ):
            best_result = row

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    save_csv(results)

    # --------------------------------------------------------
    # Save best threshold.
    # --------------------------------------------------------

    with open(
        BEST_THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{best_result['threshold']:.3f}\n"
        )

    # --------------------------------------------------------
    # Find threshold 0.50.
    # --------------------------------------------------------

    threshold_050 = min(
        results,
        key=lambda x: abs(
            x["threshold"] - 0.50
        ),
    )

    # --------------------------------------------------------
    # Display best result.
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("BEST VALIDATION RESULT")
    print("=" * 75)

    print(
        f"Best threshold: "
        f"{best_result['threshold']:.3f}"
    )

    print(
        f"Best Dice:      "
        f"{best_result['dice']:.6f}"
    )

    print(
        f"IoU:            "
        f"{best_result['iou']:.6f}"
    )

    print(
        f"Precision:      "
        f"{best_result['precision']:.6f}"
    )

    print(
        f"Recall:         "
        f"{best_result['recall']:.6f}"
    )

    print(
        f"Accuracy:       "
        f"{best_result['accuracy']:.6f}"
    )

    print(
        f"MCC:            "
        f"{best_result['mcc']:.6f}"
    )

    print(
        f"TP:             "
        f"{best_result['tp']}"
    )

    print(
        f"TN:             "
        f"{best_result['tn']}"
    )

    print(
        f"FP:             "
        f"{best_result['fp']}"
    )

    print(
        f"FN:             "
        f"{best_result['fn']}"
    )

    # --------------------------------------------------------
    # Compare against 0.50.
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("THRESHOLD 0.50 COMPARISON")
    print("=" * 75)

    print(
        f"Dice at 0.500: "
        f"{threshold_050['dice']:.6f}"
    )

    print(
        f"IoU at 0.500:  "
        f"{threshold_050['iou']:.6f}"
    )

    dice_difference = (
        best_result["dice"]
        - threshold_050["dice"]
    )

    iou_difference = (
        best_result["iou"]
        - threshold_050["iou"]
    )

    print(
        f"Dice improvement: "
        f"{dice_difference:+.6f}"
    )

    print(
        f"IoU improvement:  "
        f"{iou_difference:+.6f}"
    )

    # --------------------------------------------------------
    # Save locations.
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)

    print(
        "CSV saved to:"
    )
    print(CSV_PATH)

    print("\nBest threshold saved to:")
    print(BEST_THRESHOLD_PATH)

    # --------------------------------------------------------
    # Interpretation.
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("INTERPRETATION")
    print("=" * 75)

    print(
        "This experiment changes only the final "
        "binary threshold."
    )

    print(
        "It does NOT retrain the neural network."
    )

    print(
        "The checkpoint being evaluated is:"
    )

    print(CHECKPOINT_PATH)

    print(
        "\nThe current target is Dice >= 0.80."
    )

    if best_result["dice"] >= 0.80:

        print(
            "\nTARGET REACHED: "
            "validation Dice is >= 0.80."
        )

    else:

        print(
            f"\nTarget not reached yet."
        )

        print(
            f"Best tuned Dice: "
            f"{best_result['dice']:.6f}"
        )

        print(
            "Further improvement must come from "
            "training/model/data changes rather "
            "than threshold tuning alone."
        )

    print("\n")
    print("=" * 75)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()