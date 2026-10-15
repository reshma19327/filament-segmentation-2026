from pathlib import Path
import json
import csv
import re

root = Path(".")
base = root / "benchmark_models_512" / "post_training_analysis_stronger"
pq_script = root / "provisional_pq_evaluator_clean_cldice_best_512_v2.py"

print("=" * 65)
print("1. FINAL OUTPUTS AND METRIC CONSISTENCY")
print("=" * 65)

for f in [
    root / "benchmark_512" / "benchmark_split.json",
    root / "unified_evaluation_clean_512" / "unified_7_model_results.csv",
    base / "model_comparison.csv",
    base / "checkpoint_audit.json",
    base / "cldice_best_metrics.json",
    base / "cldice_best_per_image_metrics.csv",
    base / "morphology_measurements_verified.csv",
    base / "topology_diagnostic.csv",
    base / "tiny_component_analysis.csv",
    root / "pq_provisional_results_clean_cldice_best_512" / "provisional_pq_summary.json",
    base / "provisional_pq_summary.json",
]:
    print(f'{"OK  " if f.is_file() else "MISS"} {f}')

for f in [
    base / "cldice_best_metrics.json",
    base / "model_comparison.csv",
    root / "pq_provisional_results_clean_cldice_best_512" / "provisional_pq_summary.json",
    base / "provisional_pq_summary.json",
]:
    if f.is_file():
        print(f"\n--- {f.name} ---")
        try:
            if f.suffix == ".json":
                obj = json.loads(f.read_text(encoding="utf-8-sig"))
                print(json.dumps(obj, indent=2)[:3500])
            else:
                with f.open(newline="", encoding="utf-8-sig") as h:
                    for row in csv.DictReader(h):
                        print(row)
        except Exception as e:
            print("Could not read:", e)

print("\n" + "=" * 65)
print("2. PQ EVALUATOR: RELEVANT IMPLEMENTATION")
print("=" * 65)

if pq_script.is_file():
    lines = pq_script.read_text(encoding="utf-8-sig").splitlines()
    pat = re.compile(
        r"connectedComponents|Hungarian|linear_sum|IoU|iou|threshold|"
        r"duplicate|suppress|min_area|component|TP|FP|FN|def ",
        re.I
    )
    hits = [i for i, line in enumerate(lines) if pat.search(line)]
    chosen = set()
    for i in hits:
        chosen.update(range(max(0, i-1), min(len(lines), i+2)))
    last = -1
    for i in sorted(chosen):
        if i > last + 1:
            print("...")
        print(f"{i+1:4}: {lines[i]}")
        last = i
else:
    print("PQ evaluator script not found at expected path.")

print("\n" + "=" * 65)
print("3. TOPOLOGY DIAGNOSTIC STATUS")
print("=" * 65)
for name in ["topology_diagnostic.csv", "tiny_component_analysis.csv"]:
    f = base / name
    if f.is_file():
        with f.open(newline="", encoding="utf-8-sig") as h:
            rows = list(csv.DictReader(h))
        print(f"{name}: {len(rows)} rows")
    else:
        print(f"{name}: MISSING")

print("\n" + "=" * 65)
print("4. EXISTING UNCERTAINTY ANALYSIS CODE")
print("=" * 65)
terms = re.compile(r"uncertainty|mc.?dropout|entropy|variance|tta", re.I)
found = []
for f in root.glob("*.py"):
    try:
        lines = f.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        matches = [(i+1, line.strip()) for i, line in enumerate(lines)
                   if terms.search(line)]
        if matches:
            found.append((f.name, matches[:5]))
    except Exception:
        pass
if not found:
    print("No uncertainty-related code found in root Python scripts.")
else:
    for name, matches in found:
        print(f"\n{name}")
        for lineno, line in matches:
            print(f"  L{lineno}: {line[:180]}")

print("\n" + "=" * 65)
print("5-6. TARGET REASSESSMENT INPUTS")
print("=" * 65)
print("Use the verified clDice and PQ values above; do not infer a target")
print("is achievable from component counts alone.")
