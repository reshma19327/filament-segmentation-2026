import pandas as pd
from pathlib import Path

p = Path("benchmark_models_512/post_training_analysis_stronger/topology_diagnostic.csv")
df = pd.read_csv(p)

print("=== VALIDATION-WIDE TOPOLOGY SUMMARY ===")
for col in [
    "pred_components", "gt_components",
    "pred_skeleton_components", "gt_skeleton_components",
    "pred_skeleton_endpoints", "gt_skeleton_endpoints",
]:
    print(
        f"{col}: median={df[col].median():.1f}, "
        f"mean={df[col].mean():.2f}, "
        f"max={df[col].max()}"
    )

print("\n=== CASES WITH MOST EXCESS PREDICTED COMPONENTS ===")
df["component_excess"] = df["pred_components"] - df["gt_components"]
print(
    df.sort_values("component_excess", ascending=False)[
        ["image", "pred_components", "gt_components", "component_excess",
         "pred_skeleton_components", "gt_skeleton_components"]
    ].head(10).to_string(index=False)
)

print("\n=== CASES WITH MOST SKELETON COMPONENT EXCESS ===")
df["skeleton_excess"] = (
    df["pred_skeleton_components"] - df["gt_skeleton_components"]
)
print(
    df.sort_values("skeleton_excess", ascending=False)[
        ["image", "pred_skeleton_components", "gt_skeleton_components",
         "skeleton_excess"]
    ].head(10).to_string(index=False)
)
