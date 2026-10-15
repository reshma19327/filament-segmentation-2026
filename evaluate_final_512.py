# evaluate_final_512.py
# ================================================================
# 512x512 UNET++ + SPINE
# FINAL EVALUATION
#
# Corrected:
#   - robust filename matching
#   - correct UNetPPSpine constructor
#   - correct checkpoint loading
#   - robust confusion-matrix integer handling
#   - fixed MCC calculation
#   - filament + spine metrics
#   - threshold comparison
#   - CSV + JSON result saving
# ================================================================

from pathlib import Path
import json
import csv
import time
import random
import warnings

import cv2
import numpy as np
import torch
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
SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints_512_improved"
    / "best_unetpp_spine_512_improved.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "final_evaluation_512"

RESULT_JSON = OUTPUT_DIR / "final_evaluation_512.json"
RESULT_CSV = OUTPUT_DIR / "final_metrics_512.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 2
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

THRESHOLD = 0.80

NUM_WORKERS = 0

EPS = 1e-8


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        try:
            print(
                f"CUDA: "
                f"{torch.version.cuda}"
            )
        except Exception:
            pass

        print("=" * 70)

        return device

    print("=" * 70)
    print("CUDA NOT AVAILABLE - USING CPU")
    print("=" * 70)

    return torch.device("cpu")


# ================================================================
# FILE DISCOVERY
# ================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

MASK_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def build_stem_map(directory, extensions):

    result = {}

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    for path in sorted(directory.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        result[path.stem] = path

    return result


def discover_files():

    print()
    print("=" * 70)
    print("DATASET DISCOVERY")
    print("=" * 70)

    print(f"Images: {IMAGE_DIR}")
    print(f"Masks : {MASK_DIR}")
    print(f"Spines: {SPINE_DIR}")
    print()

    image_map = build_stem_map(
        IMAGE_DIR,
        IMAGE_EXTENSIONS
    )

    mask_map = build_stem_map(
        MASK_DIR,
        MASK_EXTENSIONS
    )

    spine_map = build_stem_map(
        SPINE_DIR,
        MASK_EXTENSIONS
    )

    print(f"Image files: {len(image_map)}")
    print(f"Mask files : {len(mask_map)}")
    print(f"Spine files: {len(spine_map)}")

    common = sorted(
        set(image_map)
        & set(mask_map)
        & set(spine_map)
    )

    print()
    print(f"Images with both masks: {len(common)}")

    if len(common) == 0:
        raise RuntimeError(
            "No images with both filament and spine masks."
        )

    print()
    print("First 5 matched files:")

    for stem in common[:5]:

        print(
            f"Image: {image_map[stem].name}"
        )

        print(
            f"Mask : {mask_map[stem].name}"
        )

        print(
            f"Spine: {spine_map[stem].name}"
        )

        print()

    return [
        (
            image_map[stem],
            mask_map[stem],
            spine_map[stem],
            stem,
        )
        for stem in common
    ]


# ================================================================
# DATASET
# ================================================================

class FinalDataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):

        return len(self.records)

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

        image_path, mask_path, spine_path, stem = (
            self.records[index]
        )

        image = self.load_image(image_path)

        mask = self.load_mask(mask_path)

        spine = self.load_mask(spine_path)

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
            "name": stem,
        }


# ================================================================
# DATASET SPLIT
# ================================================================

def create_split(records):

    records = list(records)

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(records)

    split_index = int(
        len(records) * TRAIN_RATIO
    )

    train_records = records[:split_index]

    val_records = records[split_index:]

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total images:     {len(records)}"
    )

    print(
        f"Training images:  {len(train_records)}"
    )

    print(
        f"Validation images:{len(val_records)}"
    )

    print(
        f"Image size:       "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size:       {BATCH_SIZE}"
    )

    print(
        f"Random seed:      {RANDOM_SEED}"
    )

    return train_records, val_records


# ================================================================
# SANITY CHECK
# ================================================================

def sanity_check(dataset):

    print()
    print("=" * 70)
    print("DATASET SANITY CHECK")
    print("=" * 70)

    sample = dataset[0]

    image = sample["image"]
    mask = sample["mask"]
    spine = sample["spine"]

    print(
        f"Image: {sample['name']}"
    )

    print(
        f"Image shape: {tuple(image.shape)}"
    )

    print(
        f"Mask shape : {tuple(mask.shape)}"
    )

    print(
        f"Spine shape: {tuple(spine.shape)}"
    )

    print(
        f"Image dtype: {image.dtype}"
    )

    print(
        f"Mask dtype : {mask.dtype}"
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

    assert image.shape == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert mask.shape == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert spine.shape == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print()
    print("SANITY CHECK PASSED")


# ================================================================
# MODEL
# ================================================================

def create_model(device):

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    # IMPORTANT:
    # Actual installed signature:
    #
    # UNetPPSpine(in_channels=1, base_channels=16)

    model = UNetPPSpine(
        in_channels=1,
        base_channels=16,
    )

    model = model.to(device)

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


# ================================================================
# CHECKPOINT LOADING
# ================================================================

def load_checkpoint(model, device):

    print()
    print("=" * 70)
    print("LOADING FINAL CHECKPOINT")
    print("=" * 70)

    print(CHECKPOINT)

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT}\n\n"
            f"Expected checkpoint:\n"
            f"checkpoints_512_improved/"
            f"best_unetpp_spine_512_improved.pth"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    checkpoint_epoch = None
    checkpoint_dice = None

    if isinstance(checkpoint, dict):

        checkpoint_epoch = (
            checkpoint.get("epoch")
        )

        checkpoint_dice = (
            checkpoint.get("best_dice")
        )

        if checkpoint_dice is None:
            checkpoint_dice = (
                checkpoint.get("dice")
            )

        if checkpoint_dice is None:
            checkpoint_dice = (
                checkpoint.get("val_dice")
            )

        state_dict = (
            checkpoint.get("model_state_dict")
        )

        if state_dict is None:
            state_dict = (
                checkpoint.get("state_dict")
            )

        if state_dict is None:

            # Sometimes the checkpoint itself
            # is a state dict.
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if checkpoint_epoch is not None:

        print(
            f"\nCheckpoint epoch: "
            f"{checkpoint_epoch}"
        )

    else:

        print(
            "\nCheckpoint epoch: unknown"
        )

    if checkpoint_dice is not None:

        try:

            print(
                f"Checkpoint Dice: "
                f"{float(checkpoint_dice):.6f}"
            )

        except Exception:

            print(
                f"Checkpoint Dice: "
                f"{checkpoint_dice}"
            )

    else:

        print(
            "Checkpoint Dice: unknown"
        )

    if missing:

        print(
            "\nWARNING - missing keys:"
        )

        for key in missing[:20]:
            print(
                f"  {key}"
            )

    if unexpected:

        print(
            "\nWARNING - unexpected keys:"
        )

        for key in unexpected[:20]:
            print(
                f"  {key}"
            )

    print(
        "\nCheckpoint loaded successfully."
    )


# ================================================================
# MODEL OUTPUT HANDLING
# ================================================================

def unpack_outputs(outputs):

    if isinstance(outputs, dict):

        filament = None
        spine = None

        for key in (
            "mask",
            "filament",
            "out",
            "segmentation",
            "main",
        ):

            if key in outputs:

                filament = outputs[key]

                break

        for key in (
            "spine",
            "spine_out",
            "skeleton",
        ):

            if key in outputs:

                spine = outputs[key]

                break

        if filament is None:

            raise RuntimeError(
                "Could not find filament output "
                "in model dictionary output."
            )

        if spine is None:

            raise RuntimeError(
                "Could not find spine output "
                "in model dictionary output."
            )

        return filament, spine

    if isinstance(outputs, (tuple, list)):

        if len(outputs) < 2:

            raise RuntimeError(
                "Model returned fewer than "
                "two outputs."
            )

        return outputs[0], outputs[1]

    raise RuntimeError(
        f"Unsupported model output type: "
        f"{type(outputs)}"
    )


# ================================================================
# MODEL SANITY CHECK
# ================================================================

def model_sanity_check(
    model,
    loader,
    device,
):

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    batch = next(
        iter(loader)
    )

    image = batch["image"].to(
        device,
        non_blocking=True,
    )

    mask = batch["mask"].to(
        device,
        non_blocking=True,
    )

    spine = batch["spine"].to(
        device,
        non_blocking=True,
    )

    with torch.no_grad():

        outputs = model(image)

        filament_output, spine_output = (
            unpack_outputs(outputs)
        )

    print(
        f"Input: {tuple(image.shape)}"
    )

    print(
        f"Target mask: {tuple(mask.shape)}"
    )

    print(
        f"Target spine: {tuple(spine.shape)}"
    )

    print(
        f"Filament output: "
        f"{tuple(filament_output.shape)}"
    )

    print(
        f"Spine output: "
        f"{tuple(spine_output.shape)}"
    )

    assert filament_output.shape == mask.shape

    assert spine_output.shape == spine.shape

    print()
    print("MODEL SANITY CHECK PASSED")


# ================================================================
# SAFE CONFUSION COUNTS
# ================================================================

def confusion_counts(
    probabilities,
    targets,
    threshold,
):

    predictions = (
        probabilities >= threshold
    )

    targets = (
        targets >= 0.5
    )

    tp = torch.logical_and(
        predictions,
        targets,
    ).sum().item()

    tn = torch.logical_and(
        ~predictions,
        ~targets,
    ).sum().item()

    fp = torch.logical_and(
        predictions,
        ~targets,
    ).sum().item()

    fn = torch.logical_and(
        ~predictions,
        targets,
    ).sum().item()

    # Explicit Python integer conversion.
    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn),
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    tp,
    tn,
    fp,
    fn,
):

    # Force ordinary Python integers.
    tp = int(tp)
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)

    dice_denominator = (
        2 * tp
        + fp
        + fn
    )

    iou_denominator = (
        tp
        + fp
        + fn
    )

    precision_denominator = (
        tp + fp
    )

    recall_denominator = (
        tp + fn
    )

    specificity_denominator = (
        tn + fp
    )

    accuracy_denominator = (
        tp + tn + fp + fn
    )

    dice = (
        (2.0 * tp)
        / max(
            float(dice_denominator),
            EPS,
        )
    )

    iou = (
        float(tp)
        / max(
            float(iou_denominator),
            EPS,
        )
    )

    precision = (
        float(tp)
        / max(
            float(precision_denominator),
            EPS,
        )
    )

    recall = (
        float(tp)
        / max(
            float(recall_denominator),
            EPS,
        )
    )

    specificity = (
        float(tn)
        / max(
            float(specificity_denominator),
            EPS,
        )
    )

    accuracy = (
        float(tp + tn)
        / max(
            float(accuracy_denominator),
            EPS,
        )
    )

    # ------------------------------------------------------------
    # MCC
    #
    # IMPORTANT:
    # Convert values to float BEFORE sqrt.
    # This fixes:
    #
    # TypeError: loop of ufunc does not support argument 0
    # of type int which has no callable sqrt method
    # ------------------------------------------------------------

    mcc_numerator = (
        float(tp * tn)
        - float(fp * fn)
    )

    mcc_a = float(tp + fp)

    mcc_b = float(tp + fn)

    mcc_c = float(tn + fp)

    mcc_d = float(tn + fn)

    mcc_product = (
        mcc_a
        * mcc_b
        * mcc_c
        * mcc_d
    )

    if mcc_product > 0.0:

        mcc_denominator = float(
            np.sqrt(
                mcc_product
            )
        )

        mcc = (
            mcc_numerator
            / mcc_denominator
        )

    else:

        mcc = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold,
):

    model.eval()

    filament_tp = 0
    filament_tn = 0
    filament_fp = 0
    filament_fn = 0

    spine_tp = 0
    spine_tn = 0
    spine_fp = 0
    spine_fn = 0

    total = 0

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        spines = batch["spine"].to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        filament_logits, spine_logits = (
            unpack_outputs(outputs)
        )

        filament_prob = torch.sigmoid(
            filament_logits
        )

        spine_prob = torch.sigmoid(
            spine_logits
        )

        (
            tp,
            tn,
            fp,
            fn,
        ) = confusion_counts(
            filament_prob,
            masks,
            threshold,
        )

        filament_tp += tp
        filament_tn += tn
        filament_fp += fp
        filament_fn += fn

        (
            tp,
            tn,
            fp,
            fn,
        ) = confusion_counts(
            spine_prob,
            spines,
            threshold,
        )

        spine_tp += tp
        spine_tn += tn
        spine_fp += fp
        spine_fn += fn

        total += images.shape[0]

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or total == len(loader.dataset)
        ):

            print(
                f"Processed "
                f"{total}/{len(loader.dataset)}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    filament = calculate_metrics(
        filament_tp,
        filament_tn,
        filament_fp,
        filament_fn,
    )

    spine = calculate_metrics(
        spine_tp,
        spine_tn,
        spine_fp,
        spine_fn,
    )

    return (
        filament,
        spine,
        elapsed,
    )


# ================================================================
# PRINT RESULTS
# ================================================================

def print_results(
    filament,
    spine,
    threshold,
    elapsed,
    number_images,
):

    print()
    print("=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Threshold: {threshold:.3f}"
    )

    print(
        f"Validation images: "
        f"{number_images}"
    )

    print(
        f"Inference time: "
        f"{elapsed:.2f} seconds"
    )

    if number_images > 0:

        print(
            f"Time/image: "
            f"{elapsed / number_images:.5f} seconds"
        )

    print()
    print("-" * 70)
    print("FILAMENT SEGMENTATION")
    print("-" * 70)

    print(
        f"Dice:        "
        f"{filament['dice']:.6f}"
    )

    print(
        f"IoU:         "
        f"{filament['iou']:.6f}"
    )

    print(
        f"Precision:   "
        f"{filament['precision']:.6f}"
    )

    print(
        f"Recall:      "
        f"{filament['recall']:.6f}"
    )

    print(
        f"Specificity: "
        f"{filament['specificity']:.6f}"
    )

    print(
        f"Accuracy:    "
        f"{filament['accuracy']:.6f}"
    )

    print(
        f"MCC:         "
        f"{filament['mcc']:.6f}"
    )

    print()
    print(
        f"TP: {filament['tp']:,}"
    )

    print(
        f"TN: {filament['tn']:,}"
    )

    print(
        f"FP: {filament['fp']:,}"
    )

    print(
        f"FN: {filament['fn']:,}"
    )

    print()
    print("-" * 70)
    print("SPINE SEGMENTATION")
    print("-" * 70)

    print(
        f"Dice:        "
        f"{spine['dice']:.6f}"
    )

    print(
        f"IoU:         "
        f"{spine['iou']:.6f}"
    )

    print(
        f"Precision:   "
        f"{spine['precision']:.6f}"
    )

    print(
        f"Recall:      "
        f"{spine['recall']:.6f}"
    )

    print(
        f"Specificity: "
        f"{spine['specificity']:.6f}"
    )

    print(
        f"Accuracy:    "
        f"{spine['accuracy']:.6f}"
    )

    print(
        f"MCC:         "
        f"{spine['mcc']:.6f}"
    )

    print()
    print(
        f"TP: {spine['tp']:,}"
    )

    print(
        f"TN: {spine['tn']:,}"
    )

    print(
        f"FP: {spine['fp']:,}"
    )

    print(
        f"FN: {spine['fn']:,}"
    )


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    filament,
    spine,
    threshold,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [

        {
            "target": "filament",
            "threshold": threshold,
            **filament,
        },

        {
            "target": "spine",
            "threshold": threshold,
            **spine,
        },

    ]

    fieldnames = [
        "target",
        "threshold",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "mcc",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print()
    print(
        f"CSV saved to:\n{RESULT_CSV}"
    )


# ================================================================
# SAVE JSON
# ================================================================

def save_json(
    filament,
    spine,
    threshold,
    elapsed,
    number_images,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {

        "project": (
            "512x512 UNET++ + SPINE"
        ),

        "evaluation": (
            "final validation"
        ),

        "checkpoint": str(
            CHECKPOINT
        ),

        "image_directory": str(
            IMAGE_DIR
        ),

        "mask_directory": str(
            MASK_DIR
        ),

        "spine_directory": str(
            SPINE_DIR
        ),

        "image_size": IMAGE_SIZE,

        "random_seed": RANDOM_SEED,

        "train_ratio": TRAIN_RATIO,

        "validation_images": (
            int(number_images)
        ),

        "threshold": float(
            threshold
        ),

        "inference_seconds": float(
            elapsed
        ),

        "seconds_per_image": float(
            elapsed / max(
                number_images,
                1,
            )
        ),

        "filament": filament,

        "spine": spine,

    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"JSON saved to:\n{RESULT_JSON}"
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

def print_summary(
    filament,
    spine,
):

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Filament Dice : "
        f"{filament['dice']:.6f}"
    )

    print(
        f"Filament IoU  : "
        f"{filament['iou']:.6f}"
    )

    print(
        f"Filament MCC   : "
        f"{filament['mcc']:.6f}"
    )

    print()

    print(
        f"Spine Dice    : "
        f"{spine['dice']:.6f}"
    )

    print(
        f"Spine IoU     : "
        f"{spine['iou']:.6f}"
    )

    print(
        f"Spine MCC      : "
        f"{spine['mcc']:.6f}"
    )

    print()
    print(
        "Checkpoint evaluated:"
    )

    print(
        CHECKPOINT
    )

    print()
    print(
        "FINAL EVALUATION COMPLETE"
    )

    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    seed_everything(
        RANDOM_SEED
    )

    print("=" * 70)
    print("512x512 UNET++ + SPINE")
    print("FINAL EVALUATION")
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

    # ------------------------------------------------------------
    # DISCOVER DATA
    # ------------------------------------------------------------

    records = discover_files()

    train_records, val_records = (
        create_split(records)
    )

    train_dataset = FinalDataset(
        train_records
    )

    val_dataset = FinalDataset(
        val_records
    )

    # ------------------------------------------------------------
    # SANITY CHECK
    # ------------------------------------------------------------

    sanity_check(
        val_dataset
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
    # MODEL
    # ------------------------------------------------------------

    model = create_model(
        device
    )

    model_sanity_check(
        model,
        val_loader,
        device,
    )

    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------

    load_checkpoint(
        model,
        device,
    )

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING FINAL VALIDATION")
    print("=" * 70)

    (
        filament,
        spine,
        elapsed,
    ) = evaluate(
        model,
        val_loader,
        device,
        THRESHOLD,
    )

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    print_results(
        filament,
        spine,
        THRESHOLD,
        elapsed,
        len(val_dataset),
    )

    save_csv(
        filament,
        spine,
        THRESHOLD,
    )

    save_json(
        filament,
        spine,
        THRESHOLD,
        elapsed,
        len(val_dataset),
    )

    print_summary(
        filament,
        spine,
    )


if __name__ == "__main__":

    main()