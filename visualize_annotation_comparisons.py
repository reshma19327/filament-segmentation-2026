import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

JSON_PATH = (
    ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
IMAGE_DIR = (
    ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
)
MASK_DIR = ROOT / "data" / "full_masks"
SPINE_DIR = ROOT / "data" / "full_spine_masks"

# New output location: existing project artifacts are not overwritten.
OUTPUT_DIR = ROOT / "audit_reports" / "annotation_comparison_panels"


def rasterize_record(record, annotations_by_image, spine=False):
    height = int(record["height"])
    width = int(record["width"])
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations_by_image[record["id"]]:
        if spine:
            coords = ann.get("spine")
            if not isinstance(coords, list) or len(coords) < 4:
                continue
            if len(coords) % 2:
                continue

            points = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
            if not np.isfinite(points).all():
                continue

            points = np.rint(points).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

            cv2.polylines(
                mask, [points], False, 255, thickness=5,
                lineType=cv2.LINE_AA
            )
            for point in points:
                cv2.circle(mask, tuple(point), 3, 255, -1)
        else:
            for polygon in ann.get("segmentation") or []:
                points = np.asarray(polygon, dtype=np.float32)
                if points.size < 6 or points.size % 2:
                    continue
                if not np.isfinite(points).all():
                    continue

                points = np.rint(points.reshape(-1, 2)).astype(np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)

                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 255)

    return mask


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {JSON_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records_by_filename = defaultdict(list)
    annotations_by_image = defaultdict(list)

    for image_record in data["images"]:
        records_by_filename[image_record["file_name"]].append(image_record)

    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    repeated = [
        (name, records)
        for name, records in records_by_filename.items()
        if len(records) > 1
    ]

    # Stable selection of five repeated filenames for visual review.
    selected = repeated[:5]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Repeated filenames available:", len(repeated))
    print("Comparison panels to create:", len(selected))
    print("Output directory:", OUTPUT_DIR)

    for index, (filename, records) in enumerate(selected, start=1):
        image_path = IMAGE_DIR / filename
        stem = Path(filename).stem
        existing_mask = load_gray(MASK_DIR / f"{stem}.png")
        existing_spine = load_gray(SPINE_DIR / f"{stem}.png")
        original = load_gray(image_path)

        if original is None or existing_mask is None or existing_spine is None:
            print(f"SKIP {filename}: missing/unreadable source or existing mask")
            continue

        reconstructed_masks = [
            rasterize_record(r, annotations_by_image, spine=False)
            for r in records
        ]
        reconstructed_spines = [
            rasterize_record(r, annotations_by_image, spine=True)
            for r in records
        ]

        if any(m.shape != existing_mask.shape for m in reconstructed_masks):
            print(f"SKIP {filename}: mask shape mismatch")
            continue

        union_mask = np.logical_or.reduce(
            [m > 0 for m in reconstructed_masks]
        ).astype(np.uint8) * 255

        union_spine = np.logical_or.reduce(
            [m > 0 for m in reconstructed_spines]
        ).astype(np.uint8) * 255

        difference = cv2.bitwise_xor(
            (existing_mask > 0).astype(np.uint8) * 255,
            union_mask
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = [
            (original, "Original image", "gray"),
            (existing_mask, "Existing filament mask", "gray"),
            (union_mask, "Union of all record masks", "gray"),
            (difference, "Existing vs union difference", "gray"),
            (existing_spine, "Existing spine mask", "gray"),
            (union_spine, "Union of record spine masks", "gray"),
        ]

        for ax, (panel, title, cmap) in zip(axes.flat, panels):
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"{filename} | {len(records)} annotation records",
            fontsize=12
        )
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{index:02d}_{stem}_comparison.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path.name}")

    print("Visualization complete. Existing masks were not modified.")


if __name__ == "__main__":
    main()
