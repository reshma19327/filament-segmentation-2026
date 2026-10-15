import json, csv
from pathlib import Path

root = Path.cwd()

def show_json(path):
    print(f"\n--- {path} ---")
    if not path.exists():
        print("MISSING")
        return
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            print(f"{k}: {v}")

files = [
    root / "benchmark_512" / "benchmark_split.json",
    root / "benchmark_models_512" / "post_training_analysis_stronger" / "baseline_metrics.json",
    root / "benchmark_models_512" / "post_training_analysis_stronger" / "cldice_best_metrics.json",
    root / "benchmark_models_512" / "post_training_analysis_stronger" / "cldice_stronger_best_metrics.json",
    root / "pq_provisional_results_clean_cldice_best_512" / "provisional_pq_summary.json",
]

for p in files:
    show_json(p)

print("\n--- Seven-model CSV row count ---")
p = root / "unified_evaluation_clean_512" / "unified_7_model_results.csv"
if p.exists():
    with p.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print("Rows:", len(rows))
    print("Columns:", list(rows[0].keys()) if rows else "No data rows")
else:
    print("MISSING:", p)

print("\n--- Checkpoint comparison CSV ---")
p = root / "benchmark_models_512" / "post_training_analysis_stronger" / "model_comparison.csv"
if p.exists():
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            print(row)
else:
    print("MISSING:", p)
