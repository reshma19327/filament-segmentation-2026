from pathlib import Path

source = Path("provisional_pq_evaluator_clean_cldice_best_512_v2.py")
if not source.exists():
    raise FileNotFoundError(f"Cannot find {source.resolve()}")

code = source.read_text(encoding="utf-8")

old_function = """def deduplicate_gt_instances(instances):
    instances = sorted(instances, key=lambda x: int(x.sum()), reverse=True)
    kept, suppressed = [], 0
    for candidate in instances:
        if any(mask_iou(candidate, existing) >= DEDUP_IOU_THRESHOLD for existing in kept):
            suppressed += 1
        else:
            kept.append(candidate)
    return kept, suppressed"""

new_function = """def deduplicate_gt_instances(instances):
    # Sensitivity test only: retain every nonempty GT annotation.
    return list(instances), 0"""

if old_function not in code:
    raise RuntimeError(
        "The expected function was not found exactly. "
        "No file was changed; inspect the function before proceeding."
    )

code = code.replace(old_function, new_function, 1)
code = code.replace(
    'OUTPUT_DIR = ROOT / "pq_provisional_results_clean_cldice_best_512"',
    'OUTPUT_DIR = ROOT / "pq_sensitivity_no_dedup"',
    1
)

target = Path("pq_sensitivity_no_dedup_test.py")
target.write_text(code, encoding="utf-8")
print(f"Created test evaluator: {target}")
