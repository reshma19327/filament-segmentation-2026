
import csv
from pathlib import Path
from collections import defaultdict
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
OLD_MASK_DIR = ROOT / "data" / "full_masks"
OLD_SPINE_DIR = ROOT / "data" / "full_spine_masks"

NEW_MASK_DIR = ROOT / "data" / "audited_masks_v1" / "filament_masks"
NEW_SPINE_DIR = ROOT / "data" / "audited_masks_v1" / "spine_masks"
MANIFEST = ROOT / "data" / "audited_masks_v1" / "mask_manifest.csv"

def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask

def metrics(old, new):
    old_fg = old > 0
    new_fg = new > 0

    intersection = int(np.count_nonzero(old_fg & new_fg))
    union = int(np.count_nonzero(old_fg | new_fg))
    old_count = int(np.count_nonzero(old_fg))
    new_count = int(np.count_nonzero(new_fg))
    added = int(np.count_nonzero(new_fg & ~old_fg))
    removed = int(np.count_nonzero(old_fg & ~new_fg))

    return {
        "old_pixels": old_count,
        "new_pixels": new_count,
        "added_pixels": added,
        "removed_pixels": removed,
        "iou": intersection / union if union else 1.0,
        "exact_match": bool(np.array_equal(old_fg, new_fg)),
    }

def summarize(name, rows):
    if not rows:
        print(f"{name}: no valid pairs")
        return

    print(f"\n{name}")
    print("Pairs:", len(rows))
    print("Exact matches:", sum(r["exact_match"] for r in rows))
    print("Changed masks:", sum(not r["exact_match"] for r in rows))
    print("Mean IoU:", round(float(np.mean([r["iou"] for r in rows])), 6))
    print("Median IoU:", round(float(np.median([r["iou"] for r in rows])), 6))
    print("Mean added foreground pixels:",
          round(float(np.mean([r["added_pixels"] for r in rows])), 2))
    print("Mean removed foreground pixels:",
          round(float(np.mean([r["removed_pixels"] for r in rows])), 2))

def main():
    for path in [MANIFEST, OLD_MASK_DIR, OLD_SPINE_DIR, NEW_MASK_DIR, NEW_SPINE_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    repeated_names = {
        row["file_name"]
        for row in manifest_rows
        if int(row["record_count"]) > 1
    }

    filament_rows = []
    spine_rows = []
    errors = []
    dimension_mismatches = 0
    non_binary = 0
    foreground_counts = defaultdict(list)

    for row in manifest_rows:
        filename = row["file_name"]
        stem = Path(filename).stem
        name = stem + ".png"

        image = read_mask(IMAGE_DIR / filename)
        old_mask = read_mask(OLD_MASK_DIR / name)
        new_mask = read_mask(NEW_MASK_DIR / name)
        old_spine = read_mask(OLD_SPINE_DIR / name)
        new_spine = read_mask(NEW_SPINE_DIR / name)

        if any(x is None for x in [image, old_mask, new_mask, old_spine, new_spine]):
            errors.append(filename)
            continue

        expected_shape = image.shape
        if any(x.shape != expected_shape for x in [
            old_mask, new_mask, old_spine, new_spine
        ]):
            dimension_mismatches += 1
            continue

        if any(
            not set(np.unique(x).tolist()).issubset({0, 255})
            for x in [old_mask, new_mask, old_spine, new_spine]
        ):
            non_binary += 1

        fm = metrics(old_mask, new_mask)
        sm = metrics(old_spine, new_spine)

        fm["file_name"] = filename
        sm["file_name"] = filename

        filament_rows.append(fm)
        spine_rows.append(sm)

        foreground_counts["new_filament"].append(fm["new_pixels"])
        foreground_counts["new_spine"].append(sm["new_pixels"])

    print("=== AUDITED MASK VALIDATION ===")
    print("Manifest entries:", len(manifest_rows))
    print("Repeated-record images:", len(repeated_names))
    print("Valid pairs checked:", len(filament_rows))
    print("Unreadable/missing pairs:", len(errors))
    print("Dimension mismatches:", dimension_mismatches)
    print("Masks with values other than 0/255:", non_binary)

    summarize("FILAMENT MASK COMPARISON", filament_rows)
    summarize("SPINE MASK COMPARISON", spine_rows)

    repeated_filament = [
        r for r in filament_rows if r["file_name"] in repeated_names
    ]
    single_filament = [
        r for r in filament_rows if r["file_name"] not in repeated_names
    ]

    summarize("FILAMENT — REPEATED-RECORD IMAGES", repeated_filament)
    summarize("FILAMENT — SINGLE-RECORD IMAGES", single_filament)

    if errors:
        print("\nExamples of unreadable/missing files:", errors[:10])

    print("\nRead-only validation complete. No files changed.")

if __name__ == "__main__":
    main()


