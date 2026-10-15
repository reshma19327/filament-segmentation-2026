import json, csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

root = Path.cwd()
filename = "20150216171714Mh.jpeg"
audit = root / "pq_provisional_results_clean_cldice_best_512" / "suppressed_gt_audit.csv"
ann_path = root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
out = root / "pq_provisional_results_clean_cldice_best_512" / "annotation_pair_visualization.png"

with audit.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["image"] == filename]
rows.sort(key=lambda r: float(r["IoU"]), reverse=True)
rows = rows[:3]

data = json.loads(ann_path.read_text(encoding="utf-8"))
annotations = {str(a["id"]): a for a in data["annotations"]}
images = {str(i["id"]): i for i in data["images"]}

def make_mask(ann, im):
    h, w = int(im["height"]), int(im["width"])
    mask = np.zeros((h, w), np.uint8)
    seg = ann.get("segmentation", [])
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    return cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)

matches = []
for r in rows:
    a = annotations[r["suppressed_annotation_id"]]
    b = annotations[r["retained_annotation_id"]]
    ia, ib = images[str(a["image_id"])], images[str(b["image_id"])]
    ma, mb = make_mask(a, ia), make_mask(b, ib)
    matches.append((r, ma, mb))

dataset = root / "MAGFiLO_1.0_Kaggle_2026"
candidates = list(dataset.rglob(filename))
if not candidates:
    raise FileNotFoundError(f"Could not find {filename} inside {dataset}")
img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError(f"Could not read image: {candidates[0]}")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(len(matches), 4, figsize=(15, 4.2 * len(matches)))
if len(matches) == 1:
    axes = np.array([axes])

for i, (r, suppressed, retained) in enumerate(matches):
    overlap = suppressed & retained
    panels = [
        (img, "Original image", "gray"),
        (retained, f"Retained mask\nID: {r['retained_annotation_id'][:8]}", "gray"),
        (suppressed, f"Suppressed mask\nID: {r['suppressed_annotation_id'][:8]}", "gray"),
    ]
    for j, (arr, title, cmap) in enumerate(panels):
        axes[i, j].imshow(arr, cmap=cmap)
        axes[i, j].set_title(title)
        axes[i, j].axis("off")

    # White = overlap; red = suppressed-only; green = retained-only
    overlay = np.zeros((512, 512, 3), dtype=float)
    overlay[overlap] = [1, 1, 1]
    overlay[suppressed & ~retained] = [1, 0.15, 0.15]
    overlay[retained & ~suppressed] = [0.15, 1, 0.15]
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title(f"Overlap comparison\nIoU = {r['IoU']}")
    axes[i, 3].axis("off")

fig.suptitle(filename + " — highest-IoU suppressed annotation pairs", fontsize=13)
plt.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print("Image saved to:", out)
print("Pairs visualized:", len(matches))
plt.show()
