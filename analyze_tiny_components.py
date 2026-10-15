from pathlib import Path
import csv
import cv2
import numpy as np

base = Path("benchmark_models_512/post_training_analysis_stronger")
pred_dir = base / "predictions" / "cldice_best"
out = base / "tiny_component_analysis.csv"

rows = []
for path in sorted(pred_dir.glob("*_filament.png")):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    binary = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(binary.sum())

    rows.append({
        "image": path.name.replace("_filament.png", ""),
        "foreground_pixels": total,
        "components": len(areas),
        "components_lt5": int(np.sum(areas < 5)),
        "components_lt10": int(np.sum(areas < 10)),
        "pixels_in_components_lt5": int(areas[areas < 5].sum()),
        "pixels_in_components_lt10": int(areas[areas < 10].sum()),
    })

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

total_pixels = sum(r["foreground_pixels"] for r in rows)
tiny5 = sum(r["pixels_in_components_lt5"] for r in rows)
tiny10 = sum(r["pixels_in_components_lt10"] for r in rows)

print(f"Images analyzed: {len(rows)}")
print(f"Saved: {out}")
print(f"Total predicted foreground pixels: {total_pixels}")
print(f"Pixels in components <5: {tiny5} ({100*tiny5/max(total_pixels,1):.3f}%)")
print(f"Pixels in components <10: {tiny10} ({100*tiny10/max(total_pixels,1):.3f}%)")
print("\nImages with most tiny components:")
for r in sorted(rows, key=lambda x: x["components_lt10"], reverse=True)[:10]:
    print(
        f'{r["image"]}: components={r["components"]}, '
        f'<5px={r["components_lt5"]}, <10px={r["components_lt10"]}'
    )
