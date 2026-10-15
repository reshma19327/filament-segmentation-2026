import json
import os
from collections import Counter

# ============================================================
# INSPECT MAGFiLO JSON ANNOTATIONS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(
    BASE_DIR,
    "MAGFiLO_1.0_Kaggle_2026",
    "train",
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

print("=" * 60)
print("INSPECTING MAGFiLO JSON ANNOTATIONS")
print("=" * 60)

print("JSON path:")
print(JSON_PATH)
print("-" * 60)

# ------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully")
print()

# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("Top-level keys:")
for key in data.keys():
    print("  ", key)

print("-" * 60)

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = data.get("categories", [])

print("Number of images:", len(images))
print("Number of annotations:", len(annotations))
print("Number of categories:", len(categories))

print("-" * 60)

# ------------------------------------------------------------
# Categories
# ------------------------------------------------------------

print("CATEGORIES")
print("-" * 60)

for category in categories:
    print(category)

print("-" * 60)

# ------------------------------------------------------------
# Annotation type distribution
# ------------------------------------------------------------

print("ANNOTATION TYPES")
print("-" * 60)

annotation_types = Counter()

for ann in annotations:
    if "category_id" in ann:
        annotation_types[ann["category_id"]] += 1

print("Annotations per category_id:")

for category_id, count in annotation_types.items():
    print(
        "  category_id =",
        category_id,
        "| count =",
        count
    )

print("-" * 60)

# ------------------------------------------------------------
# Example image
# ------------------------------------------------------------

if len(images) > 0:

    print("EXAMPLE IMAGE")
    print("-" * 60)

    example_image = images[0]

    for key, value in example_image.items():
        print(key, ":", value)

    example_image_id = example_image.get("id")

    print("-" * 60)

    # --------------------------------------------------------
    # Find annotations belonging to this image
    # --------------------------------------------------------

    image_annotations = [
        ann
        for ann in annotations
        if ann.get("image_id") == example_image_id
    ]

    print(
        "Annotations for example image:",
        len(image_annotations)
    )

    print("-" * 60)

    for i, ann in enumerate(image_annotations[:10]):

        print("Annotation", i + 1)

        for key, value in ann.items():

            if key == "segmentation":
                print(
                    "segmentation:",
                    type(value),
                    "length =",
                    len(value) if hasattr(value, "__len__") else "N/A"
                )
            else:
                print(key, ":", value)

        print("-" * 40)

# ------------------------------------------------------------
# Annotation fields
# ------------------------------------------------------------

print()
print("ANNOTATION FIELD CHECK")
print("-" * 60)

if len(annotations) > 0:

    fields = set()

    for ann in annotations:
        fields.update(ann.keys())

    print("Fields found in annotations:")

    for field in sorted(fields):
        print("  ", field)

# ------------------------------------------------------------
# Check segmentation
# ------------------------------------------------------------

print("-" * 60)
print("SEGMENTATION CHECK")
print("-" * 60)

segmentation_count = 0
bbox_count = 0
area_count = 0

for ann in annotations:

    if "segmentation" in ann:
        segmentation_count += 1

    if "bbox" in ann:
        bbox_count += 1

    if "area" in ann:
        area_count += 1

print("Annotations with segmentation:", segmentation_count)
print("Annotations with bbox:", bbox_count)
print("Annotations with area:", area_count)

print("-" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)