import csv
from pathlib import Path

path = Path("pq_provisional_results_clean_cldice_best_512/suppressed_gt_audit.csv")
with path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: float(r["IoU"]))

print("20 suppressed pairs with the lowest IoU:\n")
for i, r in enumerate(rows[:20], 1):
    print(
        f"{i:2}. IoU={float(r['IoU']):.4f} | "
        f"image={r['image']} | "
        f"suppressed area={r['suppressed_area_pixels']} | "
        f"retained area={r['retained_area_pixels']} | "
        f"suppressed ID={r['suppressed_annotation_id']} | "
        f"retained ID={r['retained_annotation_id']}"
    )
