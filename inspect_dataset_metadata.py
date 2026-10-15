import json
from pathlib import Path
from collections import Counter, defaultdict

root = Path.cwd()
json_path = root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
data = json.loads(json_path.read_text(encoding="utf-8"))

targets = {
    "20150825082034Lh",
    "20130305112734Ch",
    "20120206063134Lh",
    "20161231133734Ch",
    "20150720082014Th",
    "20130811140254Bh",
}

images = data["images"]
anns_by_image = defaultdict(list)
for a in data["annotations"]:
    anns_by_image[str(a["image_id"])].append(a)

print("Top-level JSON keys:", list(data.keys()))
print("\nImage-record metadata:\n")

for im in images:
    image_id = str(im["id"])
    filename = Path(str(im.get("file_name", ""))).stem
    if filename not in targets:
        continue

    anns = anns_by_image[image_id]
    categories = Counter(str(a.get("category_id", "unknown")) for a in anns)

    print(f"\nFilename: {im.get('file_name')}")
    print(f"  Source record ID: {image_id}")
    print(f"  Image dimensions: {im.get('width')} x {im.get('height')}")
    print(f"  Annotation count: {len(anns)}")
    print(f"  Category counts: {dict(categories)}")
    print(f"  Other image metadata: { {k:v for k,v in im.items() if k not in ['id','file_name','width','height']} }")

print("\nCategory definitions:")
for c in data.get("categories", []):
    print(c)
