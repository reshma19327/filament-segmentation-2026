import os
import json
import cv2
import numpy as np
from PIL import Image

# ============================================================
# MAGFiLO FULL DATASET MASK GENERATION
# Filament mask + Spine mask
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "MAGFiLO_1.0_Kaggle_2026"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

IMAGE_DIR = os.path.join(
    TRAIN_DIR,
    "train_images"
)

JSON_PATH = os.path.join(
    TRAIN_DIR,
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

# Output directories
MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "full_masks"
)

SPINE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "full_spine_masks"
)

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SPINE_DIR, exist_ok=True)

# ============================================================
# LOAD JSON
# ============================================================

print("=" * 60)
print("FULL DATASET MASK GENERATION")
print("=" * 60)

print("Loading JSON...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

print("Images in JSON:", len(images))
print("Annotations:", len(annotations))

# ============================================================
# GROUP ANNOTATIONS BY IMAGE
# ============================================================

annotations_by_image = {}

for ann in annotations:

    image_id = ann["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(ann)

print(
    "Images containing annotations:",
    len(annotations_by_image)
)

print("-" * 60)

# ============================================================
# GENERATE MASKS
# ============================================================

processed = 0
missing_images = 0
missing_segmentation = 0
missing_spine = 0

for index, image_info in enumerate(images):

    image_id = image_info["id"]

    width = image_info["width"]
    height = image_info["height"]

    # --------------------------------------------------------
    # Find actual image
    # --------------------------------------------------------

    filename = image_info.get("file_name")

    if filename is None:

        # Extract filename from URL
        url = image_info.get("url", "")
        filename = os.path.basename(url)

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(image_path):

        print(
            "WARNING: Image not found:",
            filename
        )

        missing_images += 1
        continue

    # --------------------------------------------------------
    # Create empty masks
    # --------------------------------------------------------

    filament_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    spine_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    image_annotations = annotations_by_image.get(
        image_id,
        []
    )

    # --------------------------------------------------------
    # Process annotations
    # --------------------------------------------------------

    for ann in image_annotations:

        # ====================================================
        # FILAMENT SEGMENTATION
        # ====================================================

        segmentation = ann.get(
            "segmentation"
        )

        if segmentation:

            # COCO polygon format
            for polygon in segmentation:

                if not polygon:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                points = np.round(
                    points
                ).astype(np.int32)

                # Keep points inside image
                points[:, 0] = np.clip(
                    points[:, 0],
                    0,
                    width - 1
                )

                points[:, 1] = np.clip(
                    points[:, 1],
                    0,
                    height - 1
                )

                if len(points) >= 3:

                    cv2.fillPoly(
                        filament_mask,
                        [points],
                        255
                    )

        else:

            missing_segmentation += 1

        # ====================================================
        # SPINE CENTERLINE
        # ====================================================

        spine = ann.get(
            "spine"
        )

        if spine:

            # Spine format:
            # [x1, y1, x2, y2, x3, y3, ...]

            spine_points = np.array(
                spine,
                dtype=np.float32
            ).reshape(-1, 2)

            spine_points = np.round(
                spine_points
            ).astype(np.int32)

            # Keep points inside image
            spine_points[:, 0] = np.clip(
                spine_points[:, 0],
                0,
                width - 1
            )

            spine_points[:, 1] = np.clip(
                spine_points[:, 1],
                0,
                height - 1
            )

            if len(spine_points) >= 2:

                cv2.polylines(
                    spine_mask,
                    [spine_points],
                    isClosed=False,
                    color=255,
                    thickness=5,
                    lineType=cv2.LINE_AA
                )

                # Make spine continuous
                for point in spine_points:

                    cv2.circle(
                        spine_mask,
                        tuple(point),
                        3,
                        255,
                        -1
                    )

        else:

            missing_spine += 1

    # --------------------------------------------------------
    # Save masks
    # --------------------------------------------------------

    mask_filename = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_filename
    )

    spine_path = os.path.join(
        SPINE_DIR,
        mask_filename
    )

    cv2.imwrite(
        mask_path,
        filament_mask
    )

    cv2.imwrite(
        spine_path,
        spine_mask
    )

    processed += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if processed % 25 == 0:

        print(
            f"Processed: {processed}/{len(images)}"
        )

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("MASK GENERATION COMPLETED")
print("=" * 60)

print(
    "Images in JSON:",
    len(images)
)

print(
    "Images processed:",
    processed
)

print(
    "Images missing:",
    missing_images
)

print(
    "Annotations without segmentation:",
    missing_segmentation
)

print(
    "Annotations without spine:",
    missing_spine
)

print(
    "Filament masks:",
    MASK_DIR
)

print(
    "Spine masks:",
    SPINE_DIR
)

print("=" * 60)

# ============================================================
# VERIFY OUTPUT COUNTS
# ============================================================

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

spine_files = [
    f for f in os.listdir(SPINE_DIR)
    if f.lower().endswith(".png")
]

print()
print("Verification")
print("-" * 60)

print(
    "Filament mask files:",
    len(mask_files)
)

print(
    "Spine mask files:",
    len(spine_files)
)

if (
    len(mask_files) == len(images)
    and
    len(spine_files) == len(images)
):

    print()
    print("SUCCESS: Masks generated for the full dataset.")

else:

    print()
    print("WARNING: Output count does not match image count.")

print("=" * 60)