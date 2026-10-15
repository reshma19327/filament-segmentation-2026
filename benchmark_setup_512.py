# benchmark_setup_512.py
#
# 512x512 FULL DATASET BENCHMARK SETUP
#
# IMPORTANT:
#   Source images may be 2048x2048.
#   They are NOT required to already be 512x512.
#
#   The training/evaluation dataloader is responsible for resizing
#   images and masks to 512x512.
#
# This script ONLY:
#   1. discovers images/masks/spines
#   2. validates matching files
#   3. creates a fixed train/validation split
#   4. saves the split for all benchmark models
#
# It does NOT train anything.
# It does NOT modify the original dataset.

from pathlib import Path
import random
import json
import csv
import sys

import cv2


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

OUTPUT_DIR = PROJECT_ROOT / "benchmark_512"

IMAGE_SIZE = 512

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

EXPECTED_TOTAL = 707

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


# ============================================================
# PRINT HELPERS
# ============================================================

def print_line(char="=", n=70):
    print(char * n)


def print_header(title):
    print()
    print_line("=")
    print(title)
    print_line("=")


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files(directory, extensions):
    """
    Return files indexed by lowercase filename stem.

    Example:
        20110109104734Ch.jpeg
    becomes:
        20110109104734ch
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )

    result = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        stem = path.stem.lower()

        # Detect duplicate stems.
        if stem in result:
            raise RuntimeError(
                "Duplicate filename stem found:\n"
                f"Stem: {stem}\n"
                f"File 1: {result[stem]}\n"
                f"File 2: {path}\n\n"
                "Each image/mask/spine directory must contain "
                "only one file per stem."
            )

        result[stem] = path

    return result


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_dataset():
    print_header("DATASET DISCOVERY")

    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Images:")
    print(f"  {IMAGE_DIR}")

    print()
    print("Masks:")
    print(f"  {MASK_DIR}")

    print()
    print("Spines:")
    print(f"  {SPINE_DIR}")

    image_files = discover_files(
        IMAGE_DIR,
        IMAGE_EXTENSIONS,
    )

    mask_files = discover_files(
        MASK_DIR,
        MASK_EXTENSIONS,
    )

    spine_files = discover_files(
        SPINE_DIR,
        MASK_EXTENSIONS,
    )

    print()
    print(f"Image files: {len(image_files)}")
    print(f"Mask files : {len(mask_files)}")
    print(f"Spine files: {len(spine_files)}")

    common_stems = sorted(
        set(image_files)
        & set(mask_files)
        & set(spine_files)
    )

    missing_masks = sorted(
        set(image_files) - set(mask_files)
    )

    missing_spines = sorted(
        set(image_files) - set(spine_files)
    )

    extra_masks = sorted(
        set(mask_files) - set(image_files)
    )

    extra_spines = sorted(
        set(spine_files) - set(image_files)
    )

    print()
    print(f"Common filename stems: {len(common_stems)}")

    if missing_masks:
        print()
        print("Images missing filament masks:")
        for stem in missing_masks[:20]:
            print(f"  {stem}")

        if len(missing_masks) > 20:
            print(
                f"  ... and {len(missing_masks) - 20} more"
            )

    if missing_spines:
        print()
        print("Images missing spine masks:")
        for stem in missing_spines[:20]:
            print(f"  {stem}")

        if len(missing_spines) > 20:
            print(
                f"  ... and {len(missing_spines) - 20} more"
            )

    if extra_masks:
        print()
        print("Extra masks without images:")
        for stem in extra_masks[:20]:
            print(f"  {stem}")

    if extra_spines:
        print()
        print("Extra spine masks without images:")
        for stem in extra_spines[:20]:
            print(f"  {stem}")

    if len(common_stems) == 0:
        raise RuntimeError(
            "No common image/mask/spine filename stems found."
        )

    records = []

    for stem in common_stems:
        records.append(
            {
                "stem": stem,
                "image": image_files[stem],
                "mask": mask_files[stem],
                "spine": spine_files[stem],
            }
        )

    print()
    print(
        f"Confirmed: {len(records)} images "
        "with both masks."
    )

    print()
    print("First 5 matched files:")

    for record in records[:5]:
        print()
        print(f"Stem : {record['stem']}")
        print(
            f"Image: {record['image'].name}"
        )
        print(
            f"Mask : {record['mask'].name}"
        )
        print(
            f"Spine: {record['spine'].name}"
        )

    return records


# ============================================================
# IMAGE VALIDATION
# ============================================================

def read_image_info(path, grayscale=False):
    """
    Read image safely and return shape information.
    """

    if grayscale:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )
    else:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

    if image is None:
        return None

    return image


def validate_mask_values(path):
    """
    Validate that a mask is readable.

    We do NOT require the mask to already be 512x512.
    """

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return False, "cannot be read"

    if mask.size == 0:
        return False, "empty image"

    unique = set(
        int(x)
        for x in cv2.unique(mask)[1].flatten()
    )

    return True, unique


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(records):
    print_header("DATASET VALIDATION")

    print(
        f"Checking {len(records)} matched "
        "image/mask/spine sets..."
    )

    problems = []

    source_image_sizes = {}
    mask_sizes = {}
    spine_sizes = {}

    for index, record in enumerate(records, start=1):

        stem = record["stem"]

        image_path = record["image"]
        mask_path = record["mask"]
        spine_path = record["spine"]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = read_image_info(
            image_path,
            grayscale=False,
        )

        if image is None:
            problems.append(
                (
                    stem,
                    f"image cannot be read: "
                    f"{image_path.name}",
                )
            )
            continue

        if image.ndim == 2:
            image_h, image_w = image.shape
        else:
            image_h, image_w = image.shape[:2]

        source_image_sizes[
            (image_w, image_h)
        ] = source_image_sizes.get(
            (image_w, image_h),
            0,
        ) + 1

        # ----------------------------------------------------
        # FILAMENT MASK
        # ----------------------------------------------------

        mask = read_image_info(
            mask_path,
            grayscale=True,
        )

        if mask is None:
            problems.append(
                (
                    stem,
                    f"mask cannot be read: "
                    f"{mask_path.name}",
                )
            )
            continue

        mask_h, mask_w = mask.shape[:2]

        mask_sizes[
            (mask_w, mask_h)
        ] = mask_sizes.get(
            (mask_w, mask_h),
            0,
        ) + 1

        # ----------------------------------------------------
        # SPINE MASK
        # ----------------------------------------------------

        spine = read_image_info(
            spine_path,
            grayscale=True,
        )

        if spine is None:
            problems.append(
                (
                    stem,
                    f"spine cannot be read: "
                    f"{spine_path.name}",
                )
            )
            continue

        spine_h, spine_w = spine.shape[:2]

        spine_sizes[
            (spine_w, spine_h)
        ] = spine_sizes.get(
            (spine_w, spine_h),
            0,
        ) + 1

        # ----------------------------------------------------
        # SIZE VALIDATION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # We intentionally DO NOT require:
        #
        #     image == 512x512
        #
        # because your source images are 2048x2048.
        #
        # We also do not require masks to be 512x512.
        #
        # The training Dataset will resize everything to
        # 512x512.
        #
        # We only require all three to have valid dimensions.
        # ----------------------------------------------------

        if image_w <= 0 or image_h <= 0:
            problems.append(
                (
                    stem,
                    "invalid image dimensions",
                )
            )

        if mask_w <= 0 or mask_h <= 0:
            problems.append(
                (
                    stem,
                    "invalid filament mask dimensions",
                )
            )

        if spine_w <= 0 or spine_h <= 0:
            problems.append(
                (
                    stem,
                    "invalid spine mask dimensions",
                )
            )

        # ----------------------------------------------------
        # EXTREME SIZE CHECK
        # ----------------------------------------------------

        if image_w < 32 or image_h < 32:
            problems.append(
                (
                    stem,
                    (
                        f"image suspiciously small: "
                        f"{image_w}x{image_h}"
                    ),
                )
            )

        if mask_w < 32 or mask_h < 32:
            problems.append(
                (
                    stem,
                    (
                        f"mask suspiciously small: "
                        f"{mask_w}x{mask_h}"
                    ),
                )
            )

        if spine_w < 32 or spine_h < 32:
            problems.append(
                (
                    stem,
                    (
                        f"spine mask suspiciously small: "
                        f"{spine_w}x{spine_h}"
                    ),
                )
            )

        # Progress
        if (
            index % 100 == 0
            or index == len(records)
        ):
            print(
                f"Validated {index}/{len(records)}"
            )

    print()

    if problems:

        print(
            f"VALIDATION FAILED: "
            f"{len(problems)} problematic records."
        )

        print()

        for stem, reason in problems[:100]:
            print(
                f"{stem}: {reason}"
            )

        if len(problems) > 100:
            print()
            print(
                f"... and "
                f"{len(problems) - 100} more problems."
            )

        raise RuntimeError(
            "Dataset validation failed. "
            "Fix the files above before training."
        )

    print("VALIDATION PASSED")

    # --------------------------------------------------------
    # SOURCE SIZE SUMMARY
    # --------------------------------------------------------

    print()
    print("Source image dimensions:")

    for size, count in sorted(
        source_image_sizes.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {size[0]}x{size[1]} : {count}"
        )

    print()
    print("Filament mask dimensions:")

    for size, count in sorted(
        mask_sizes.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {size[0]}x{size[1]} : {count}"
        )

    print()
    print("Spine mask dimensions:")

    for size, count in sorted(
        spine_sizes.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {size[0]}x{size[1]} : {count}"
        )

    print()
    print(
        "IMPORTANT: Source dimensions are allowed to "
        "differ from 512x512."
    )

    print(
        f"Training/evaluation resolution will be "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}."
    )


# ============================================================
# CREATE FIXED SPLIT
# ============================================================

def create_split(records):
    print_header("CREATING FIXED BENCHMARK SPLIT")

    items = list(records)

    rng = random.Random(RANDOM_SEED)

    rng.shuffle(items)

    total = len(items)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = total - train_count

    train_records = items[:train_count]
    val_records = items[train_count:]

    print(f"Total images:      {total}")
    print(f"Training images:   {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Train ratio:       {TRAIN_RATIO}")
    print(f"Random seed:       {RANDOM_SEED}")

    return train_records, val_records


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    records,
    path,
    split_name,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "split",
                "stem",
                "image",
                "mask",
                "spine",
            ]
        )

        for record in records:

            writer.writerow(
                [
                    split_name,
                    record["stem"],
                    record["image"].name,
                    record["mask"].name,
                    record["spine"].name,
                ]
            )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    train_records,
    val_records,
    path,
):
    data = {
        "project_root": str(PROJECT_ROOT),
        "image_dir": str(IMAGE_DIR),
        "mask_dir": str(MASK_DIR),
        "spine_dir": str(SPINE_DIR),
        "image_size": IMAGE_SIZE,
        "source_images_are_resized": True,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "total_images": (
            len(train_records)
            + len(val_records)
        ),
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "train": [],
        "validation": [],
    }

    for record in train_records:

        data["train"].append(
            {
                "stem": record["stem"],
                "image": record["image"].name,
                "mask": record["mask"].name,
                "spine": record["spine"].name,
            }
        )

    for record in val_records:

        data["validation"].append(
            {
                "stem": record["stem"],
                "image": record["image"].name,
                "mask": record["mask"].name,
                "spine": record["spine"].name,
            }
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# SAVE PLAIN TEXT LISTS
# ============================================================

def save_list(records, path):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                record["stem"]
                + "\n"
            )


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    train_records,
    val_records,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_csv = (
        OUTPUT_DIR
        / "train_split.csv"
    )

    val_csv = (
        OUTPUT_DIR
        / "validation_split.csv"
    )

    all_csv = (
        OUTPUT_DIR
        / "benchmark_split.csv"
    )

    split_json = (
        OUTPUT_DIR
        / "benchmark_split.json"
    )

    train_txt = (
        OUTPUT_DIR
        / "train_stems.txt"
    )

    val_txt = (
        OUTPUT_DIR
        / "validation_stems.txt"
    )

    save_csv(
        train_records,
        train_csv,
        "train",
    )

    save_csv(
        val_records,
        val_csv,
        "validation",
    )

    save_csv(
        train_records,
        all_csv,
        "train",
    )

    # Append validation records to combined CSV.
    with open(
        all_csv,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        for record in val_records:

            writer.writerow(
                [
                    "validation",
                    record["stem"],
                    record["image"].name,
                    record["mask"].name,
                    record["spine"].name,
                ]
            )

    save_json(
        train_records,
        val_records,
        split_json,
    )

    save_list(
        train_records,
        train_txt,
    )

    save_list(
        val_records,
        val_txt,
    )

    print_header("BENCHMARK FILES SAVED")

    print(
        f"Output directory:\n"
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        f"Train CSV:\n"
        f"  {train_csv}"
    )

    print()
    print(
        f"Validation CSV:\n"
        f"  {val_csv}"
    )

    print()
    print(
        f"Combined CSV:\n"
        f"  {all_csv}"
    )

    print()
    print(
        f"JSON:\n"
        f"  {split_json}"
    )

    print()
    print(
        f"Train stems:\n"
        f"  {train_txt}"
    )

    print()
    print(
        f"Validation stems:\n"
        f"  {val_txt}"
    )


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(
    train_records,
    val_records,
):
    print_header("VERIFYING FIXED SPLIT")

    train_stems = {
        r["stem"]
        for r in train_records
    }

    val_stems = {
        r["stem"]
        for r in val_records
    }

    overlap = train_stems & val_stems

    if overlap:
        raise RuntimeError(
            "TRAIN/VALIDATION LEAKAGE DETECTED.\n"
            f"Overlapping stems: {len(overlap)}"
        )

    total_unique = (
        len(train_stems | val_stems)
    )

    expected_total = (
        len(train_records)
        + len(val_records)
    )

    if total_unique != expected_total:
        raise RuntimeError(
            "Split verification failed."
        )

    print(
        f"Train records:      {len(train_records)}"
    )

    print(
        f"Validation records: {len(val_records)}"
    )

    print(
        f"Unique records:     {total_unique}"
    )

    print()
    print("Train/validation overlap: 0")

    print()
    print("SPLIT VERIFICATION PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_line("=")
    print(
        "512x512 FULL DATASET BENCHMARK SETUP"
    )
    print_line("=")

    print()
    print(
        "This script DOES NOT train a model."
    )

    print(
        "It creates the fixed split for "
        "all benchmark models."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Source images can be 2048x2048."
    )

    print(
        "They will be resized to "
        f"{IMAGE_SIZE}x{IMAGE_SIZE} by the training pipeline."
    )

    print()
    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Random seed  : {RANDOM_SEED}"
    )

    print(
        f"Train ratio  : {TRAIN_RATIO}"
    )

    # --------------------------------------------------------
    # DISCOVER
    # --------------------------------------------------------

    records = discover_dataset()

    # --------------------------------------------------------
    # EXPECTED DATASET SIZE
    # --------------------------------------------------------

    if EXPECTED_TOTAL is not None:

        if len(records) != EXPECTED_TOTAL:

            raise RuntimeError(
                "\n"
                "Unexpected dataset size.\n"
                f"Expected: {EXPECTED_TOTAL}\n"
                f"Found:    {len(records)}\n\n"
                "Check the image/mask/spine directories."
            )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_dataset(records)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_records, val_records = create_split(
        records
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    verify_split(
        train_records,
        val_records,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_manifest(
        train_records,
        val_records,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_header("BENCHMARK SETUP COMPLETE")

    print(
        f"Total dataset:       {len(records)}"
    )

    print(
        f"Training images:     {len(train_records)}"
    )

    print(
        f"Validation images:   {len(val_records)}"
    )

    print(
        f"Training resolution: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Random seed:         {RANDOM_SEED}"
    )

    print()
    print(
        "Original 2048x2048 images were NOT modified."
    )

    print()
    print(
        "The fixed split is now ready for all models."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Use benchmark_512/benchmark_split.json "
        "in every training script."
    )

    print()
    print_line("=")


if __name__ == "__main__":
    main()