import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("unified_evaluation_512")
source = folder / "unified_7_model_results.csv"
csv_out = folder / "unified_7_model_results_final.csv"
png_out = folder / "unified_7_model_results_final.png"

df = pd.read_csv(source)
df = df[~df["Model"].astype(str).str.contains("exp075", case=False, na=False)].copy()

if "clDice" in df.columns:
    df.drop(columns=["clDice"], inplace=True)

df.insert(df.columns.get_loc("Status"), "clDice", "—")

mask = df["Model"].astype(str).str.contains(
    r"UNet\+\+.*Spine.*clDice.*Improved", case=False, regex=True, na=False
)

if mask.sum() != 1:
    raise ValueError(f"Expected one matching model row; found {mask.sum()}.")

df.loc[mask, "clDice"] = "0.799055"
df.loc[mask, "Dice"] = 0.724745

df.to_csv(csv_out, index=False)

shown = df.fillna("—").copy()
for col in ["Dice", "IoU", "Precision", "Recall", "MCC", "Cohen_Kappa", "clDice"]:
    if col in shown.columns:
        shown[col] = shown[col].map(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

fig, ax = plt.subplots(figsize=(22, 6))
ax.axis("off")
table = ax.table(
    cellText=shown.values,
    colLabels=shown.columns,
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)
ax.set_title("Unified Model Evaluation — 512 × 512", fontsize=15, pad=20)
plt.tight_layout()
plt.savefig(png_out, dpi=250, bbox_inches="tight")
plt.close()

print("CSV:", csv_out.resolve())
print("PNG:", png_out.resolve())
print("Rows:", len(df))
print("Updated model:", df.loc[mask, ["Model", "Dice", "clDice"]].to_string(index=False))
print("Note: Other first-row metrics are retained from the previous evaluation.")
