from pathlib import Path
import pandas as pd

root = Path.cwd()
src = root / "benchmark_models_512" / "threshold_sweep_all" / "threshold_sweep_per_image.csv"
out = root / "benchmark_models_512" / "threshold_sweep_all" / "hard_case_analysis"

if not src.exists():
    raise FileNotFoundError(f"Missing report: {src}")

df = pd.read_csv(src)
df = df[(df["threshold"] - 0.08).abs() < 1e-8]

pivot = df.pivot(index="image", columns="checkpoint", values="cldice")
dice = df.pivot(index="image", columns="checkpoint", values="dice")

pivot["exp075_change_vs_full10"] = pivot["exp075"] - pivot["full10"]
pivot["exp075_dice"] = dice["exp075"]
pivot["full10_dice"] = dice["full10"]

out.mkdir(parents=True, exist_ok=True)
pivot.sort_values("exp075").to_csv(out / "all_images_ranked.csv")

print("\n=== 20 HARDEST IMAGES: exp075 at threshold 0.08 ===")
print(pivot.sort_values("exp075").head(20).to_string(float_format=lambda x: f"{x:.5f}"))

print("\n=== 10 LARGEST DECLINES VS full10 ===")
print(pivot.sort_values("exp075_change_vs_full10").head(10).to_string(float_format=lambda x: f"{x:.5f}"))

print("\n=== SUMMARY ===")
print(f"Images: {len(pivot)}")
print(f"Mean exp075 clDice: {pivot['exp075'].mean():.6f}")
print(f"Mean full10 clDice at same threshold: {pivot['full10'].mean():.6f}")
print(f"exp075 improved: {(pivot['exp075_change_vs_full10'] > 1e-8).sum()}")
print(f"exp075 worsened: {(pivot['exp075_change_vs_full10'] < -1e-8).sum()}")
print(f"Saved complete ranking to: {out / 'all_images_ranked.csv'}")
