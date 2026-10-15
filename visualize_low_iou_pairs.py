import csv, json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

root = Path.cwd()
base = root / "pq_provisional_results_clean_cldice_best_512"
audit_path = base / "suppressed_gt_audit.csv"
json_path = root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
out = base / "low_iou_annotation_pairs.png"

with audit_path.open(encoding="utf-8-sig", newline="") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: float(r["IoU"]))[:6]

data = json.loads(json_path.read_text(encoding="utf-8"))
anns = {str(a["id"]): a for a in data["annotations"]}
imgs = {str(i["id"]): i for i in data["images"]}
dataset = root / "MAGFiLO_1.0_Kaggle_2026"

def mask_for(a):
    im = imgs[str(a["image_id"])]
    h, w = int(im["height"]), int(im["width"])
    m = np.zeros((h, w), np.uint8)
    seg = a.get("segmentation", [])
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                p[:, 0] = np.clip(p[:, 0], 0, w - 1)
                p[:, 1] = np.clip(p[:, 1], 0, h - 1)
                cv2.fillPoly(m, [np.rint(p).astype(np.int32)], 1)
    return cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)

fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.5 * len(rows)))

for i, r in enumerate(rows):
    a = anns[r["suppressed_annotation_id"]]
    b = anns[r["retained_annotation_id"]]
    ma, mb = mask_for(a), mask_for(b)
    filename = r["image"]
    candidates = list(dataset.rglob(filename))
    if not candidates:
        raise FileNotFoundError(filename)
    original = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (512, 512), interpolation=cv2.INTER_AREA)

    axes[i, 0].imshow(original, cmap="gray")
    axes[i, 0].set_title(f"{filename}\nIoU {float(r['IoU']):.3f}")
    axes[i, 1].imshow(mb, cmap="gray")
    axes[i, 1].set_title("Retained mask")
    axes[i, 2].imshow(ma, cmap="gray")
    axes[i, 2].set_title("Suppressed mask")

    overlap = np.zeros((512, 512, 3), float)
    overlap[ma & mb] = [1, 1, 1]
    overlap[ma & ~mb] = [1, 0.1, 0.1]
    overlap[mb & ~ma] = [0.1, 1, 0.1]
    axes[i, 3].imshow(overlap)
    axes[i, 3].set_title("White = overlap\nRed/green = differences")

    for ax in axes[i]:
        ax.axis("off")

fig.tight_layout()
fig.savefig(out, dpi=160, bbox_inches="tight")
print("Saved:", out)
plt.show()
