


from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

DIRECTORIES = {
    "Original filament masks": ROOT / "data" / "full_masks",
    "Candidate filament masks": ROOT / "data" / "audited_masks_v1" / "filament_masks",
    "Original spine masks": ROOT / "data" / "full_spine_masks",
    "Candidate spine masks": ROOT / "data" / "audited_masks_v1" / "spine_masks",
}

for label, directory in DIRECTORIES.items():
    if not directory.is_dir():
        print(f"\n{label}: DIRECTORY MISSING — {directory}")
        continue

    files = sorted(directory.glob("*.png"))
    files_with_intermediate = 0
    all_values = Counter()
    examples = []

    for path in files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        values, counts = np.unique(mask, return_counts=True)
        all_values.update({
            int(value): int(count)
            for value, count in zip(values, counts)
        })

        intermediate = values[(values != 0) & (values != 255)]
        if len(intermediate):
            files_with_intermediate += 1
            if len(examples) < 5:
                examples.append(
                    (path.name, intermediate.tolist()[:20])
                )

    print(f"\n=== {label} ===")
    print("PNG files:", len(files))
    print("Distinct pixel values:", len(all_values))
    print("Pixel values present:", sorted(all_values))
    print("Files with intermediate values:", files_with_intermediate)
    print("Example files:", examples)

print("\nRead-only pixel-value inspection complete.")
