
import json
from pathlib import Path
from collections import Counter, defaultdict

# Read-only audit: this script does not create or modify project files.
ROOT = Path(__file__).resolve().parent
JSON_PATH = (
    ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)



if not JSON_PATH.exists():
    print("Annotation JSON not found at:")
    print(JSON_PATH)
    print("\nUpdate JSON_PATH to the actual location of your annotation file.")
    raise SystemExit(1)

print("Loading annotation JSON...")
with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = {
    item["id"]: item.get("name", "Unknown")
    for item in data.get("categories", [])
}

records_by_filename = defaultdict(list)
for image in images:
    records_by_filename[image.get("file_name", "")].append(image)

annotation_counts = Counter(
    annotation.get("image_id")
    for annotation in annotations
)

print("\n=== READ-ONLY MASK GENERATION AUDIT ===")
print(f"Image records: {len(images)}")
print(f"Annotation records: {len(annotations)}")
print(f"Unique filenames: {len(records_by_filename)}")
print(
    "Filenames with multiple records:",
    sum(len(records) > 1 for records in records_by_filename.values())
)
print("\nCategory counts:")
for category_id, count in Counter(
    str(a.get("category_id")) for a in annotations
).items():
    print(f"  {category_id} ({categories.get(int(category_id), 'Unknown')}): {count}")

print("\nExamples of repeated filenames:")
shown = 0
for filename, records in records_by_filename.items():
    if len(records) > 1:
        print(f"\n{filename}")
        for record in records:
            image_id = record.get("id")
            print(
                f"  image_id={image_id!r}, "
                f"annotations={annotation_counts[image_id]}"
            )
        shown += 1
        if shown >= 10:
            break

print("\nAudit complete. No files were modified.")


