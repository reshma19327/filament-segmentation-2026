"""
Find clDice failure cases for the MAGFiLO solar-filament project.


Reads the verified per-image CSV created by verified_evaluation_7_models_512.py,
then creates a ranked CSV and visual panels for the lowest-clDice cases of the
current improved UNet++ + Spine + clDice model.


This is analysis only: it does not train models or modify existing results.
"""


from pathlib import Path
import csv
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "evaluation_verified_512" / "verified_per_image_metrics.csv"
PRED_DIR = ROOT / "unified_evaluation_512" / "predictions" / "unetpp_spine_cldice_improved"
IMAGE_DIR = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
UNION_MASK_DIR = ROOT / "data" / "audited_masks_v1" / "filament_masks"
OUT_DIR = ROOT / "cldice_failure_analysis"
TOP_N = 40
SIZE = 512




def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return img




def find_case_insensitive(folder, filename):
    p = folder / filename
    if p.exists():
        return p
    if not folder.exists():
        return None
    matches = [x for x in folder.iterdir() if x.is_file() and x.name.lower() == filename.lower()]
    return matches[0] if len(matches) == 1 else None




def colorize(image, original_gt, union_gt, pred):
    # BGR colors: original GT green; union-only additions blue;
    # prediction TP green, FN blue, FP red.
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    base = cv2.convertScaleAbs(base, alpha=0.8, beta=0)
    orig = original_gt > 0
    union = union_gt > 0
    p = pred > 127


    overlay = base.copy()
    overlay[orig] = (30, 190, 30)
    overlay[union & ~orig] = (255, 140, 0)
    overlay[p & union] = (30, 220, 30)
    overlay[union & ~p] = (255, 80, 0)
    overlay[p & ~union] = (0, 0, 255)
    return cv2.addWeighted(base, 0.28, overlay, 0.72, 0)




def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}\nRun verified_evaluation_7_models_512.py first."
        )
    if not PRED_DIR.is_dir():
        raise FileNotFoundError(f"Prediction directory missing: {PRED_DIR}")


    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))


    rows = [
        r for r in rows
        if r.get("Model_Key") == "unetpp_spine_cldice_improved"
        and r.get("Union_clDice", "").strip() != ""
    ]
    if not rows:
        raise ValueError("No improved-model Union_clDice rows found in verified CSV.")


    for row in rows:
        row["Union_clDice"] = float(row["Union_clDice"])
        row["Union_Dice"] = float(row["Union_Dice"])
        row["Union_IoU"] = float(row["Union_IoU"])
        row["Original_Dice"] = float(row["Original_Dice"])
        row["Original_IoU"] = float(row["Original_IoU"])


    rows.sort(key=lambda r: r["Union_clDice"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    ranked_path = OUT_DIR / "improved_model_cldice_ranked.csv"
    with ranked_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


    scores = np.array([r["Union_clDice"] for r in rows], dtype=float)
    print("=" * 76)
    print("CLDICE FAILURE-CASE ANALYSIS")
    print("=" * 76)
    print(f"Images ranked: {len(rows)}")
    print(f"Mean clDice  : {scores.mean():.6f}")
    print(f"Median       : {np.median(scores):.6f}")
    print(f"10th pct     : {np.percentile(scores, 10):.6f}")
    print(f"25th pct     : {np.percentile(scores, 25):.6f}")
    print(f"75th pct     : {np.percentile(scores, 75):.6f}")
    print(f"90th pct     : {np.percentile(scores, 90):.6f}")
    print(f"Ranked CSV   : {ranked_path}")


    panels_dir = OUT_DIR / "lowest_cldice_panels"
    panels_dir.mkdir(exist_ok=True)
    panel_index = []


    for rank, row in enumerate(rows[:TOP_N], 1):
        name = Path(row["Image"]).name
        stem = Path(name).stem
        pred_path = find_case_insensitive(PRED_DIR, f"{stem}.png")
        image_path = find_case_insensitive(IMAGE_DIR, name)
        union_path = find_case_insensitive(UNION_MASK_DIR, name)


        # Ground-truth masks are sometimes named with a .png extension even
        # when the source image uses .jpg; fall back to matching by stem.
        if union_path is None and UNION_MASK_DIR.exists():
            candidates = [p for p in UNION_MASK_DIR.iterdir()
                          if p.is_file() and p.stem.lower() == stem.lower()]
            union_path = candidates[0] if len(candidates) == 1 else None


        if pred_path is None or image_path is None or union_path is None:
            panel_index.append({
                "Rank": rank, "Image": name, "Union_clDice": row["Union_clDice"],
                "Status": "MISSING_FILE",
                "Prediction": str(pred_path), "Image_Path": str(image_path),
                "Union_Mask": str(union_path)
            })
            continue


        img = read_gray(image_path)
        union = read_gray(union_path)
        pred = read_gray(pred_path)
        if img is None or union is None or pred is None:
            panel_index.append({
                "Rank": rank, "Image": name, "Union_clDice": row["Union_clDice"],
                "Status": "UNREADABLE"
            })
            continue


        # Original masks are useful context but not required for clDice ranking.
        # Try to load the original GT with the same image stem.
        original_dir = ROOT / "data" / "full_masks"
        original_path = find_case_insensitive(original_dir, f"{stem}.png")
        if original_path is None and original_dir.exists():
            candidates = [p for p in original_dir.iterdir()
                          if p.is_file() and p.stem.lower() == stem.lower()]
            original_path = candidates[0] if len(candidates) == 1 else None
        original = read_gray(original_path) if original_path else np.zeros_like(union)


        panel = colorize(img, original, union, pred)
        label = f"Rank {rank} | clDice {row['Union_clDice']:.3f} | Dice {row['Union_Dice']:.3f}"
        cv2.rectangle(panel, (0, 0), (SIZE, 30), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out_path = panels_dir / f"{rank:02d}_{stem}_cldice_{row['Union_clDice']:.3f}.png"
        cv2.imwrite(str(out_path), panel)
        panel_index.append({
            "Rank": rank, "Image": name, "Union_clDice": row["Union_clDice"],
            "Union_Dice": row["Union_Dice"], "Union_IoU": row["Union_IoU"],
            "Status": "CREATED", "Panel": str(out_path)
        })


    with (OUT_DIR / "panel_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(dict.fromkeys(k for r in panel_index for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panel_index)


    print(f"Lowest-case panels: {panels_dir}")
    print(f"Panel index       : {OUT_DIR / 'panel_index.csv'}")
    print(f"Panels created    : {sum(r.get('Status') == 'CREATED' for r in panel_index)} / {min(TOP_N, len(rows))}")
    print("\nInterpretation note: low clDice cases are for diagnosis, not for tuning the model against validation labels.")
    print("Do not claim a score improvement until a new training run is evaluated with the same fixed protocol.")




if __name__ == "__main__":
    main()



