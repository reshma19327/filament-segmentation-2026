import pandas as pd
from pathlib import Path

root = Path.cwd()
files = [
    root / "unified_evaluation_512" / "unified_7_model_results.csv",
    root / "unified_evaluation_512_new_cldice" / "unified_7_model_results.csv",
    root / "unified_evaluation_clean_512" / "unified_7_model_results.csv",
]

for path in files:
    print("\n" + "=" * 80)
    print(path)
    if not path.exists():
        print("NOT FOUND")
        continue
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    print(df.head(3).to_string(index=False))
