
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "provisional_pq_evaluator_512.py"
VARIANT = ROOT / "provisional_pq_evaluator_512_no_dedup.py"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Original evaluator not found: {SOURCE}")

code = SOURCE.read_text(encoding="utf-8")

replacements = [
    (
        'OUTPUT_DIR = ROOT / "pq_provisional_results_512"',
        'OUTPUT_DIR = ROOT / "pq_no_dedup_results_512"',
    ),
    (
        "gt_instances, suppressed = deduplicate_gt_instances(gt_candidates)",
        "gt_instances, suppressed = gt_candidates, 0",
    ),
    (
        '"Greedy suppression at mask IoU >= 0.5; provisional heuristic"',
        '"No duplicate-annotation suppression; sensitivity analysis only"',
    ),
]

for old, new in replacements:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of:\n{old}\nFound: {count}. "
            "No evaluator was generated or run."
        )
    code = code.replace(old, new, 1)

# Verify the original evaluation destination is not retained.
if 'OUTPUT_DIR = ROOT / "pq_provisional_results_512"' in code:
    raise RuntimeError("Output path isolation check failed.")

VARIANT.write_text(code, encoding="utf-8")
print("Created separate evaluator:", VARIANT.name)
print("Output directory: pq_no_dedup_results_512")
print("Original evaluator and results remain unchanged.")
print("Running the no-deduplication sensitivity evaluation...\n")

subprocess.run([sys.executable, str(VARIANT)], cwd=ROOT, check=True)


