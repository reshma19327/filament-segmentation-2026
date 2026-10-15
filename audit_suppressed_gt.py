import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

ROOT = Path.cwd()
JSON_PATH = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SPLIT_PATH = ROOT / "benchmark_512" / "benchmark_split.json"
OUT = ROOT / "pq_provisional_results_clean_cldice_best_512" / "suppressed_gt_audit.csv"
SIZE = 512

def rasterize(seg, w, h):
    mask = np.zeros((h, w), np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    if mask.shape != (SIZE, SIZE):
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)

def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
images = defaultdict(list)
anns = defaultdict(list)

for im in data.get("images", []):
    images[Path(im.get("file_name", "")).name.lower()].append(im)
for ann in data.get("annotations", []):
    anns[ann.get("image_id")].append(ann)

rows = []
for item in split.get("validation", []):
    filename = Path(item["image"]).name
    candidates = []
    for im in images.get(filename.lower(), []):
        w, h = int(im.get("width", SIZE)), int(im.get("height", SIZE))
        for ann in anns.get(im.get("id"), []):
            mask = rasterize(ann.get("segmentation", []), w, h)
            if mask.any():
                candidates.append({
                    "ann_id": ann.get("id"),
                    "image_id": im.get("id"),
                    "mask": mask,
                    "area": int(mask.sum())
                })

    kept = []
    for cand in sorted(candidates, key=lambda x: x["area"], reverse=True):
        overlaps = [(iou(cand["mask"], old["mask"]), old) for old in kept]
        duplicates = [(score, old) for score, old in overlaps if score >= 0.5]
        if duplicates:
            score, old = max(duplicates, key=lambda x: x[0])
            rows.append({
                "image": filename,
                "suppressed_annotation_id": cand["ann_id"],
                "retained_annotation_id": old["ann_id"],
                "IoU": round(score, 6),
                "suppressed_area_pixels": cand["area"],
                "retained_area_pixels": old["area"],
                "suppressed_image_id": cand["image_id"],
                "retained_image_id": old["image_id"]
            })
        else:
            kept.append(cand)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image", "suppressed_annotation_id", "retained_annotation_id",
        "IoU", "suppressed_area_pixels", "retained_area_pixels",
        "suppressed_image_id", "retained_image_id"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("Suppressed annotations audited:", len(rows))
print("Unique affected images:", len(set(r["image"] for r in rows)))
print("Output:", OUT)
print("\nFirst 10 rows:")
for row in rows[:10]:
    print(row)
