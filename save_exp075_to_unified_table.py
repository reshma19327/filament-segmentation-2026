from pathlib import Path
import pandas as pd

root = Path.cwd()
source = root / "unified_evaluation_512" / "unified_7_model_results.csv"
destination = root / "unified_evaluation_512" / "unified_8_model_results_with_exp075_validation.csv"

if not source.exists():
    raise FileNotFoundError(f"Unified table not found: {source}")
if destination.exists():
    raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

df = pd.read_csv(source)

row = {column: None for column in df.columns}
row.update({
    "Model": "UNet++ + Spine + clDice (exp075; validation-tuned)",
    "Dice": 0.724745,
    "Status": "VALIDATION ONLY; 142 images; threshold=0.0775; clDice=0.799055; not directly comparable to unified benchmark",
})

updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
updated.to_csv(destination, index=False)

print("Created:", destination)
print("Rows:", len(updated))
print("\nNew row:")
print(updated.tail(1).to_string(index=False))
print("\nExisting unified table was not modified.")
print("Note: clDice is recorded in Status because the original table has no clDice column.")
