import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(
    BASE_DIR,
    "MAGFiLO_1.0_Kaggle_2026",
    "train",
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "MAGFiLO_1.0_Kaggle_2026",
    "train",
    "train_images"
)

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

print("=" * 60)
print("FULL DATASET VERIFICATION")
print("=" * 60)

# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

print("JSON image records:", len(images))
print("JSON annotations:", len(annotations))

# ---------------------------------------------------------
# PHYSICAL IMAGES
# ---------------------------------------------------------

physical_images = set()

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        physical_images.add(filename)

print("Physical image files:", len(physical_images))

# ---------------------------------------------------------
# MASK FILES
# ---------------------------------------------------------

mask_files = set()

if os.path.exists(MASK_DIR):
    for filename in os.listdir(MASK_DIR):
        if filename.lower().endswith(".png"):
            mask_files.add(filename)

spine_files = set()

if os.path.exists(SPINE_DIR):
    for filename in os.listdir(SPINE_DIR):
        if filename.lower().endswith(".png"):
            spine_files.add(filename)

print("Filament masks:", len(mask_files))
print("Spine masks:", len(spine_files))

# ---------------------------------------------------------
# JSON FILENAMES
# ---------------------------------------------------------

json_filenames = set()

for img in images:
    json_filenames.add(img["file_name"])

# ---------------------------------------------------------
# COMPARE
# ---------------------------------------------------------

json_not_on_disk = json_filenames - physical_images
images_not_in_json = physical_images - json_filenames

print()
print("-" * 60)
print("DATASET COMPARISON")
print("-" * 60)

print("JSON images:", len(json_filenames))
print("Images on disk:", len(physical_images))
print("JSON images missing from disk:", len(json_not_on_disk))
print("Disk images missing from JSON:", len(images_not_in_json))

# ---------------------------------------------------------
# SHOW MISSING EXAMPLES
# ---------------------------------------------------------

if json_not_on_disk:

    print()
    print("Examples of JSON images NOT found on disk:")

    for filename in sorted(json_not_on_disk)[:20]:
        print("  ", filename)

# ---------------------------------------------------------
# CHECK PAIRS
# ---------------------------------------------------------

paired = 0
missing_mask = 0
missing_spine = 0

for filename in sorted(physical_images):

    base = os.path.splitext(filename)[0]

    possible_masks = [
        base + ".png",
        base + "_mask.png"
    ]

    possible_spines = [
        base + ".png",
        base + "_spine.png"
    ]

    has_mask = any(x in mask_files for x in possible_masks)
    has_spine = any(x in spine_files for x in possible_spines)

    if has_mask and has_spine:
        paired += 1

    if not has_mask:
        missing_mask += 1

    if not has_spine:
        missing_spine += 1

print()
print("-" * 60)
print("TRAINING PAIR CHECK")
print("-" * 60)

print("Images:", len(physical_images))
print("Images with mask + spine:", paired)
print("Images missing mask:", missing_mask)
print("Images missing spine:", missing_spine)

# ---------------------------------------------------------
# ANNOTATION COUNTS
# ---------------------------------------------------------

image_ids = {img["id"] for img in images}

annotation_image_ids = {
    ann["image_id"]
    for ann in annotations
}

print()
print("-" * 60)
print("ANNOTATION CHECK")
print("-" * 60)

print("Images with annotations:", len(annotation_image_ids))

# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print("=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)

if len(physical_images) == len(mask_files) == len(spine_files):
    print("✓ Image / mask / spine counts match")

if paired == len(physical_images):
    print("✓ Every physical image has mask + spine")

if len(json_not_on_disk) > 0:
    print()
    print("IMPORTANT:")
    print(
        "The JSON contains image records that are not physically "
        "available in train_images."
    )

print("=" * 60)