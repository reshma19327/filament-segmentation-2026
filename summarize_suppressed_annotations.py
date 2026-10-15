import csv
from pathlib import Path
from collections import defaultdict

path = Path("pq_provisional_results_clean_cldice_best_512/suppressed_gt_audit.csv")
if not path.exists():
    raise FileNotFoundError(f"CSV not found: {path.resolve()}")

groups = {
    "High overlap (IoU >= 0.8)": lambda x: x >= 0.8,
    "Moderate overlap (0.5 <= IoU < 0.8)": lambda x: 0.5 <= x < 0.8,
    "Below threshold (IoU < 0.5)": lambda x: x < 0.5,
}
results = defaultdict(list)

with path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("CSV columns:", list(rows[0].keys()) if rows else "No rows found")

for row in rows:
    iou_key = next((k for k in row if k.lower() == "iou"), None)
    image_key = next((k for k in row if "image" in k.lower()), None)
    if iou_key is None or image_key is None:
        raise ValueError("Could not identify IoU and image columns. Check the printed column names.")
    iou = float(row[iou_key])
    for label, condition in groups.items():
        if condition(iou):
            results[label].append((row[image_key], iou))
            break

print(f"\nTotal rows: {len(rows)}")
for label, items in results.items():
    print(f"\n{label}")
    print(f"  Suppressed annotations: {len(items)}")
    print(f"  Unique images: {len(set(image for image, _ in items))}")
    if items:
        print(f"  Mean IoU: {sum(iou for _, iou in items) / len(items):.4f}")
