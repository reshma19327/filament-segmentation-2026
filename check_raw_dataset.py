import os
import json

# ============================================================
# RAW DATASET CONFIGURATION
# ============================================================

BASE_DIR = "MAGFiLO_1.0_Kaggle_2026"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "train"
)

IMAGE_DIR = os.path.join(
    TRAIN_DIR,
    "train_images"
)


# ============================================================
# FIND JSON FILES
# ============================================================

print("=" * 60)
print("CHECKING RAW MAGFiLO DATASET")
print("=" * 60)

print("Base directory:")
print(os.path.abspath(BASE_DIR))

print("----------------------------------------")


if not os.path.exists(BASE_DIR):
    print("ERROR: Dataset folder not found.")
    print("Expected:")
    print(os.path.abspath(BASE_DIR))
    raise SystemExit


if not os.path.exists(TRAIN_DIR):
    print("ERROR: train folder not found.")
    raise SystemExit


if not os.path.exists(IMAGE_DIR):
    print("ERROR: train_images folder not found.")
    raise SystemExit


# ============================================================
# COUNT IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
]

print("Training images:", len(image_files))

print("----------------------------------------")


# ============================================================
# FIND JSON
# ============================================================

json_files = [
    f for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(".json")
]

print("JSON files found:", len(json_files))

for f in json_files:
    print("  ", f)

print("----------------------------------------")


# ============================================================
# INSPECT JSON
# ============================================================

for json_name in json_files:

    json_path = os.path.join(
        TRAIN_DIR,
        json_name
    )

    print("Checking JSON:")
    print(json_path)

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print("JSON loaded successfully")

        print("JSON type:", type(data).__name__)

        if isinstance(data, dict):

            print(
                "Top-level keys:"
            )

            for key in data.keys():
                print("  ", key)

        elif isinstance(data, list):

            print(
                "Number of records:",
                len(data)
            )

        print("----------------------------------------")

    except Exception as e:

        print("ERROR reading JSON:")
        print(e)

        print("----------------------------------------")


# ============================================================
# SHOW SAMPLE IMAGES
# ============================================================

print("Example training images:")

for filename in image_files[:10]:
    print("  ", filename)

print("----------------------------------------")

print("RAW DATASET CHECK COMPLETED")
print("=" * 60)