import csv
from pathlib import Path

p = Path("pq_provisional_results_clean_cldice_best_512/suppressed_gt_audit.csv")
with p.open(encoding="utf-8-sig", newline="") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: float(r["IoU"]))[:6]

for i, r in enumerate(rows, 1):
    print(f"\nPair {i}: {r['image']}")
    print(f"  IoU: {r['IoU']}")
    print(f"  Suppressed source image ID: {r['suppressed_image_id']}")
    print(f"  Retained source image ID:   {r['retained_image_id']}")
    print(f"  Same source image record:   {r['suppressed_image_id'] == r['retained_image_id']}")
