
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

JSON_PATH = (
    ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
IMAGE_DIR = (
    ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
)

# New output directory. Existing masks are never modified.
OUT_DIR = ROOT / "data" / "audited_masks_v1"
MASK_DIR = OUT_DIR / "filament_masks"
SPINE_DIR = OUT_DIR / "spine_masks"


def draw_annotation(mask, annotation, width, height):
    """Draw one annotation's polygons into a binary semantic mask."""
    segmentation = annotation.get("segmentation") or []

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            raise ValueError(
                f"Invalid polygon in annotation {annotation.get('id')}"
            )
        if len(polygon) % 2:
            raise ValueError(
                f"Odd polygon coordinate count in {annotation.get('id')}"
            )

        coords = np.asarray(polygon, dtype=np.float64)
        if not np.isfinite(coords).all():
            raise ValueError(
                f"Non-finite polygon coordinates in {annotation.get('id')}"
            )

        points = np.rint(coords.reshape(-1, 2)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)


def draw_spine(mask, annotation, width, height):
    """Draw the supplied spine coordinates using the existing convention."""
    spine = annotation.get("spine")

    if not isinstance(spine, list) or not spine:
        raise ValueError(
            f"Missing spine in annotation {annotation.get('id')}"
        )
    if len(spine) < 4 or len(spine) % 2:
        raise ValueError(
            f"Invalid spine coordinate count in {annotation.get('id')}"
        )

    coords = np.asarray(spine, dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError(
            f"Non-finite spine coordinates in {annotation.get('id')}"
        )

    points = np.rint(coords.reshape(-1, 2)).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    cv2.polylines(
        mask, [points], isClosed=False,
        color=255, thickness=5, lineType=cv2.LINE_AA
    )
    for point in points:
        cv2.circle(mask, tuple(point), 3, 255, -1)


def main():
    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {JSON_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    # Never overwrite a previous candidate dataset.
    if OUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUT_DIR}\n"
            "Rename/archive it first, or choose a new output directory."
        )

    print("Loading annotation JSON...")
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records_by_filename = defaultdict(list)
    annotations_by_image = defaultdict(list)

    for record in data["images"]:
        filename = record.get("file_name")
        if not filename:
            raise ValueError(f"Image record missing file_name: {record.get('id')}")
        records_by_filename[filename].append(record)

    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    physical_names = {
        p.name for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    missing = sorted(set(records_by_filename) - physical_names)
    extra = sorted(physical_names - set(records_by_filename))

    if missing:
        raise RuntimeError(
            f"{len(missing)} JSON filenames are missing on disk. "
            f"Example: {missing[:5]}"
        )

    if extra:
        print(f"Note: {len(extra)} physical images have no JSON filename.")

    MASK_DIR.mkdir(parents=True)
    SPINE_DIR.mkdir(parents=True)

    manifest_path = OUT_DIR / "mask_manifest.csv"
    report_path = OUT_DIR / "build_report.json"

    manifest_rows = []
    category_totals = Counter()
    empty_filament = 0
    empty_spine = 0
    processed = 0

    filenames = sorted(records_by_filename)

    print("JSON image records:", len(data["images"]))
    print("Unique physical filenames:", len(filenames))
    print("Annotations:", len(data["annotations"]))
    print("Repeated filenames:", sum(
        len(records) > 1 for records in records_by_filename.values()
    ))
    print("Building candidate masks on CPU...")

    for index, filename in enumerate(filenames, start=1):
        records = records_by_filename[filename]
        image_path = IMAGE_DIR / filename

        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        height, width = original.shape
        filament_mask = np.zeros((height, width), dtype=np.uint8)
        spine_mask = np.zeros((height, width), dtype=np.uint8)

        annotation_ids = []
        record_ids = []
        per_category = Counter()
        annotation_count = 0

        for record in records:
            if (
                int(record["width"]) != width
                or int(record["height"]) != height
            ):
                raise ValueError(
                    f"JSON/image dimension mismatch for {filename}, "
                    f"record {record['id']}"
                )

            record_ids.append(str(record["id"]))

            for ann in annotations_by_image[record["id"]]:
                draw_annotation(filament_mask, ann, width, height)
                draw_spine(spine_mask, ann, width, height)

                annotation_ids.append(str(ann["id"]))
                per_category[str(ann.get("category_id"))] += 1
                category_totals[str(ann.get("category_id"))] += 1
                annotation_count += 1

        output_name = Path(filename).stem + ".png"
        mask_path = MASK_DIR / output_name
        spine_path = SPINE_DIR / output_name

        if not cv2.imwrite(str(mask_path), filament_mask):
            raise RuntimeError(f"Failed to write {mask_path}")
        if not cv2.imwrite(str(spine_path), spine_mask):
            raise RuntimeError(f"Failed to write {spine_path}")

        if not np.any(filament_mask):
            empty_filament += 1
        if not np.any(spine_mask):
            empty_spine += 1

        manifest_rows.append({
            "file_name": filename,
            "record_count": len(records),
            "record_ids": "|".join(record_ids),
            "annotation_count": annotation_count,
            "annotation_ids": "|".join(annotation_ids),
            "category_counts_json": json.dumps(
                dict(per_category), sort_keys=True
            ),
            "width": width,
            "height": height,
            "filament_foreground_pixels": int(
                np.count_nonzero(filament_mask)
            ),
            "spine_foreground_pixels": int(
                np.count_nonzero(spine_mask)
            ),
            "filament_mask": str(Path("filament_masks") / output_name),
            "spine_mask": str(Path("spine_masks") / output_name),
        })

        processed += 1

        if processed % 50 == 0 or processed == len(filenames):
            print(f"Processed {processed}/{len(filenames)}")

    # Write the manifest only after mask generation succeeds.
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    actual_masks = list(MASK_DIR.glob("*.png"))
    actual_spines = list(SPINE_DIR.glob("*.png"))

    report = {
        "source_json": str(JSON_PATH),
        "physical_images_expected": len(filenames),
        "physical_images_processed": processed,
        "json_image_records": len(data["images"]),
        "annotations_processed": sum(
            row["annotation_count"] for row in manifest_rows
        ),
        "repeated_filenames": sum(
            len(records) > 1 for records in records_by_filename.values()
        ),
        "category_counts": dict(category_totals),
        "filament_mask_files": len(actual_masks),
        "spine_mask_files": len(actual_spines),
        "empty_filament_masks": empty_filament,
        "empty_spine_masks": empty_spine,
        "missing_json_filenames_on_disk": missing,
        "extra_physical_filenames": extra,
        "note": (
            "Binary union candidate masks. Individual annotation IDs and "
            "categories are recorded in the CSV manifest. These masks "
            "are not yet approved ground truth for instance-level PQ."
        ),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== BUILD REPORT ===")
    print(json.dumps(report, indent=2))
    print("\nCandidate masks saved to:", OUT_DIR)
    print("Original masks and checkpoints were not modified.")


if __name__ == "__main__":
    main()


