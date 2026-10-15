import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"unified_evaluation_512\unified_8_model_results_with_exp075_validation.csv"
output_path = r"unified_evaluation_512\unified_8_model_results_table.png"

df = pd.read_csv(csv_path).fillna("—")

fig, ax = plt.subplots(figsize=(22, max(4, len(df) * 0.75 + 1.5)))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)

ax.set_title("Unified Model Evaluation — 512 × 512", fontsize=15, pad=20)
plt.tight_layout()
plt.savefig(output_path, dpi=250, bbox_inches="tight")
plt.close()

print(f"Table image saved to: {output_path}")
