from pathlib import Path
import pandas as pd

d = Path("pq_provisional_results_clean_cldice_best_512")
per = pd.read_csv(d / "provisional_pq_per_image.csv")
matches = pd.read_csv(d / "provisional_pq_matches.csv")

print("=== PER-IMAGE PQ BOTTLENECK ===")
print("Images:", len(per))
for c in ["TP", "FP", "FN", "PQ", "SQ", "RQ"]:
    if c in per.columns:
        print(f"{c}: total/mean = {per[c].sum() if c in ['TP','FP','FN'] else per[c].mean():.4f}")

print("\n=== IMAGES WITH MOST FALSE POSITIVES ===")
print(per.sort_values("FP", ascending=False)[
    ["image", "TP", "FP", "FN", "PQ", "SQ", "RQ"]
].head(10).to_string(index=False))

print("\n=== IMAGES WITH MOST FALSE NEGATIVES ===")
print(per.sort_values("FN", ascending=False)[
    ["image", "TP", "FP", "FN", "PQ", "SQ", "RQ"]
].head(10).to_string(index=False))

print("\n=== MATCHED INSTANCE IoU DISTRIBUTION ===")
if len(matches):
    print(matches["IoU"].describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
    print("IoU <= 0.55:", int((matches["IoU"] <= .55).sum()))
    print("IoU 0.55–0.70:", int(((matches["IoU"] > .55) & (matches["IoU"] <= .70)).sum()))
    print("IoU > 0.70:", int((matches["IoU"] > .70).sum()))
else:
    print("No matched instances.")
