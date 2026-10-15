from pathlib import Path
import csv
import json
import time
import ast
import math

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import train_benchmark_unetpp_spine_only_512 as training

ROOT = Path("benchmark_models_512")
OUT = ROOT / "post_training_analysis"
SPLIT = Path("benchmark_512/benchmark_split.json")
BASELINE = ROOT / "unetpp_spine_only/best_unetpp_spine_only_512.pth"
CLDICE_BEST = ROOT / "unetpp_spine_cldice_finetune/best_unetpp_spine_topology_512.pth"
CLDICE_FINAL = ROOT / "unetpp_spine_cldice_finetune/final_unetpp_spine_topology_512.pth"
TRAINING_JSON = ROOT / "unetpp_spine_cldice_finetune/training_metrics_512.json"
THRESHOLD = 0.5
ITERATIONS = 10

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def json_dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def binary_metrics(tp, tn, fp, fn):
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def pil_mask(array):
    return Image.fromarray((array.astype(np.uint8) * 255), mode="L")

def safe_name(name):
    return Path(str(name)).name

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(training.DEVICE),
        "threshold": THRESHOLD,
        "cldice_iterations": ITERATIONS,
        "image_size": 512,
        "split_file": str(SPLIT),
        "validation_expected": 142,
        "checkpoints": {},
        "warnings": [],
    }

    required = [SPLIT, BASELINE, CLDICE_BEST, CLDICE_FINAL, TRAINING_JSON]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    history_data = json.loads(TRAINING_JSON.read_text(encoding="utf-8-sig"))
    history = history_data.get("history", [])
    audit["training_history_epochs"] = len(history)
    audit["training_history_reported_best_dice"] = history_data.get("best_validation_dice")
    audit["training_history_complete_20_epochs"] = len(history) == 20
    if len(history) != 20:
        audit["warnings"].append(f"Expected 20 history entries; found {len(history)}.")

    # Training curves, if matplotlib is installed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if history:
            epochs = [r["epoch"] for r in history]
            for field, title, filename in [
                ("train_loss", "Training loss", "training_loss.png"),
                ("validation_loss", "Validation loss", "validation_loss.png"),
                ("validation_dice", "Validation Dice", "validation_dice.png"),
                ("validation_iou", "Validation IoU", "validation_iou.png"),
            ]:
                if all(field in r for r in history):
                    plt.figure()
                    plt.plot(epochs, [r[field] for r in history], marker=".")
                    plt.xlabel("Epoch")
                    plt.ylabel(field.replace("_", " ").title())
                    plt.title(title)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(OUT / filename, dpi=150)
                    plt.close()
        audit["training_curves_created"] = True
    except ImportError:
        audit["training_curves_created"] = False
        audit["warnings"].append("matplotlib unavailable; training curves skipped.")

    # Optional skeletonization for pixel-based morphology.
    try:
        from skimage.morphology import skeletonize
        has_skeletonize = True
    except ImportError:
        skeletonize = None
        has_skeletonize = False
        audit["warnings"].append(
            "scikit-image unavailable; skeleton length and width estimates will be blank."
        )

    train_records, val_records = training.load_split()
    if len(train_records) != 565 or len(val_records) != 142:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_records)}, val={len(val_records)}"
        )

    dataset = training.FilamentSpineDataset(val_records)
    loader = DataLoader(
        dataset, batch_size=training.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(training.DEVICE.type == "cuda"),
        drop_last=False
    )

    specs = [
        ("baseline", BASELINE),
        ("cldice_best", CLDICE_BEST),
        ("cldice_final", CLDICE_FINAL),
    ]
    summary_rows = []
    all_image_rows = []
    prediction_dirs = {}
    reference_dir = OUT / "reference"
    reference_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("COMBINED POST-TRAINING ANALYSIS")
    print("=" * 70)
    print(f"Device: {training.DEVICE}")
    print(f"Train/validation: {len(train_records)}/{len(val_records)}")
    print(f"Output: {OUT.resolve()}")

    for label, checkpoint_path in specs:
        print(f"\nEvaluating {label}: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=training.DEVICE, weights_only=False
        )
        model = training.UNetPPSpine(in_channels=1, base_channels=16).to(training.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        pred_dir = OUT / "predictions" / label
        pred_dir.mkdir(parents=True, exist_ok=True)
        prediction_dirs[label] = pred_dir

        totals = {k: 0 for k in (
            "tp", "tn", "fp", "fn",
            "spine_tp", "spine_tn", "spine_fp", "spine_fn"
        )}
        image_rows = []
        start = time.time()
        processed = 0

        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(training.DEVICE)
                masks = batch["mask"].to(training.DEVICE)
                spines = batch["spine"].to(training.DEVICE)
                outputs = model(images)
                filament_logits, spine_logits = training.get_outputs(outputs)

                filament_pred = (torch.sigmoid(filament_logits) >= THRESHOLD).float()
                spine_pred = (torch.sigmoid(spine_logits) >= THRESHOLD).float()

                for i in range(images.shape[0]):
                    name = safe_name(batch["name"][i])
                    stem = Path(name).stem
                    target = masks[i:i+1]
                    pred = filament_pred[i:i+1]
                    target_spine = spines[i:i+1]
                    pred_spine = spine_pred[i:i+1]

                    fm = training.calculate_metrics(
                        filament_logits[i:i+1], target, THRESHOLD
                    )
                    sm = training.calculate_metrics(
                        spine_logits[i:i+1], target_spine, THRESHOLD
                    )
                    cld = float(training.cldice_score(
                        pred, target, iterations=ITERATIONS
                    ).item())

                    row = {
                        "image": name,
                        "model": label,
                        "dice": float(fm["dice"]),
                        "iou": float(fm["iou"]),
                        "precision": float(fm["precision"]),
                        "recall": float(fm["recall"]),
                        "cldice": cld,
                        "spine_dice": float(sm["dice"]),
                        "spine_iou": float(sm["iou"]),
                        "spine_precision": float(sm["precision"]),
                        "spine_recall": float(sm["recall"]),
                        "tp": int(fm["tp"]), "tn": int(fm["tn"]),
                        "fp": int(fm["fp"]), "fn": int(fm["fn"]),
                        "spine_tp": int(sm["tp"]), "spine_tn": int(sm["tn"]),
                        "spine_fp": int(sm["fp"]), "spine_fn": int(sm["fn"]),
                    }
                    image_rows.append(row)
                    all_image_rows.append(row)

                    for key in ("tp", "tn", "fp", "fn"):
                        totals[key] += int(fm[key])
                        totals["spine_" + key] += int(sm[key])

                    # Save every prediction, not just the first item in each batch.
                    pil_mask(pred[0, 0].cpu().numpy()).save(
                        pred_dir / f"{stem}_filament.png"
                    )
                    pil_mask(pred_spine[0, 0].cpu().numpy()).save(
                        pred_dir / f"{stem}_spine.png"
                    )

                    if label == "baseline":
                        input_array = (
                            images[i, 0].detach().float().cpu().numpy()
                        )
                        lo, hi = float(input_array.min()), float(input_array.max())
                        if hi > lo:
                            input_array = (input_array - lo) / (hi - lo)
                        else:
                            input_array = np.zeros_like(input_array)
                        Image.fromarray(
                            np.uint8(np.clip(input_array, 0, 1) * 255), mode="L"
                        ).save(reference_dir / f"{stem}_input.png")
                        pil_mask(target[0, 0].cpu().numpy()).save(
                            reference_dir / f"{stem}_ground_truth.png"
                        )
                        pil_mask(target_spine[0, 0].cpu().numpy()).save(
                            reference_dir / f"{stem}_ground_truth_spine.png"
                        )

                    processed += 1

                if processed % 20 < images.shape[0] or processed == len(dataset):
                    print(f"  {label}: {processed}/{len(dataset)}")

        if processed != len(dataset):
            raise RuntimeError(f"{label}: evaluated {processed}, expected {len(dataset)}")

        global_filament = binary_metrics(
            totals["tp"], totals["tn"], totals["fp"], totals["fn"]
        )
        global_spine = binary_metrics(
            totals["spine_tp"], totals["spine_tn"],
            totals["spine_fp"], totals["spine_fn"]
        )
        metric_names = [
            "dice", "iou", "precision", "recall", "cldice",
            "spine_dice", "spine_iou", "spine_precision", "spine_recall"
        ]
        mean_image = {
            key: sum(r[key] for r in image_rows) / len(image_rows)
            for key in metric_names
        }

        result = {
            "model": label,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_dice_metadata": checkpoint.get("best_dice"),
            "images_evaluated": processed,
            "threshold": THRESHOLD,
            "cldice_iterations": ITERATIONS,
            "evaluation_seconds": round(time.time() - start, 2),
            "global_pixel_filament_metrics": global_filament,
            "global_pixel_spine_metrics": global_spine,
            "mean_per_image_metrics": mean_image,
        }
        json_dump(OUT / f"{label}_metrics.json", result)
        write_csv(OUT / f"{label}_per_image_metrics.csv", image_rows)

        summary = {
            "model": label,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_dice_metadata": checkpoint.get("best_dice"),
            "images_evaluated": processed,
            **{f"global_{k}": v for k, v in global_filament.items()},
            **{f"mean_{k}": v for k, v in mean_image.items()},
            **{f"global_spine_{k}": v for k, v in global_spine.items()
               if k in ("dice", "iou", "precision", "recall")},
        }
        summary_rows.append(summary)
        audit["checkpoints"][label] = {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "metadata_best_dice": checkpoint.get("best_dice"),
            "images_evaluated": processed,
        }
        del model
        if training.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(OUT / "model_comparison.csv", summary_rows)
    write_csv(OUT / "per_image_all_models.csv", all_image_rows)

    # Build contact sheets comparing input, GT, baseline, best fine-tune, final epoch.
    panels_dir = OUT / "visualizations"
    panels_dir.mkdir(exist_ok=True)
    captions = ["Input", "Ground truth", "Baseline", "clDice best", "Epoch 20"]
    for rec in val_records:
        name = safe_name(rec["image"])
        stem = Path(name).stem
        paths = [
            reference_dir / f"{stem}_input.png",
            reference_dir / f"{stem}_ground_truth.png",
            prediction_dirs["baseline"] / f"{stem}_filament.png",
            prediction_dirs["cldice_best"] / f"{stem}_filament.png",
            prediction_dirs["cldice_final"] / f"{stem}_filament.png",
        ]
        if not all(p.exists() for p in paths):
            audit["warnings"].append(f"Visualization inputs missing for {name}")
            continue
        tile_w, tile_h, label_h = 256, 256, 26
        canvas = Image.new("RGB", (tile_w * len(paths), tile_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (path, caption) in enumerate(zip(paths, captions)):
            im = Image.open(path).convert("L").resize((tile_w, tile_h))
            canvas.paste(im.convert("RGB"), (j * tile_w, label_h))
            draw.text((j * tile_w + 5, 5), caption, fill="black")
        canvas.save(panels_dir / f"{stem}_comparison.png")

    # Morphology on the best fine-tuned binary masks; all units remain pixels.
    morphology_rows = []
    for rec in val_records:
        name = safe_name(rec["image"])
        stem = Path(name).stem
        gt_path = reference_dir / f"{stem}_ground_truth.png"
        pred_path = prediction_dirs["cldice_best"] / f"{stem}_filament.png"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = np.asarray(Image.open(gt_path).convert("L")) > 0
        pred = np.asarray(Image.open(pred_path).convert("L")) > 0

        row = {"image": name}
        for prefix, mask in [("ground_truth", gt), ("prediction", pred)]:
            row[f"{prefix}_area_pixels"] = int(mask.sum())
            ys, xs = np.where(mask)
            if len(xs):
                row[f"{prefix}_bbox_width_pixels"] = int(xs.max() - xs.min() + 1)
                row[f"{prefix}_bbox_height_pixels"] = int(ys.max() - ys.min() + 1)
            else:
                row[f"{prefix}_bbox_width_pixels"] = 0
                row[f"{prefix}_bbox_height_pixels"] = 0

            if has_skeletonize:
                skeleton = skeletonize(mask)
                length = int(skeleton.sum())
                row[f"{prefix}_skeleton_length_pixels_approx"] = length
                row[f"{prefix}_mean_width_pixels_approx"] = (
                    float(mask.sum() / length) if length else 0.0
                )
            else:
                row[f"{prefix}_skeleton_length_pixels_approx"] = ""
                row[f"{prefix}_mean_width_pixels_approx"] = ""
        morphology_rows.append(row)

    write_csv(OUT / "morphology_measurements.csv", morphology_rows)

    # A simple report with computed results and explicit limitations.
    baseline_row = next(r for r in summary_rows if r["model"] == "baseline")
    best_row = next(r for r in summary_rows if r["model"] == "cldice_best")
    final_row = next(r for r in summary_rows if r["model"] == "cldice_final")
    delta_dice = best_row["global_dice"] - baseline_row["global_dice"]
    delta_cldice = best_row["mean_cldice"] - baseline_row["mean_cldice"]

    report = [
        "# Solar Filament Segmentation — Post-Training Analysis",
        "",
        "## Evaluation setup",
        f"- Validation images evaluated per checkpoint: {len(val_records)}",
        f"- Training images in fixed split: {len(train_records)}",
        f"- Image size: 512 × 512; threshold: {THRESHOLD}",
        f"- Device: {training.DEVICE}",
        "- Ground truth: training module's configured full filament and spine masks.",
        "- Global metrics aggregate confusion counts across pixels; mean metrics weight images equally.",
        "",
        "## Checkpoint comparison",
        "| Model | Epoch | Global Dice | Global IoU | Mean-image Dice | Mean-image clDice | Mean spine Dice |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        report.append(
            f"| {r['model']} | {r['checkpoint_epoch']} | "
            f"{r['global_dice']:.6f} | {r['global_iou']:.6f} | "
            f"{r['mean_dice']:.6f} | {r['mean_cldice']:.6f} | "
            f"{r['mean_spine_dice']:.6f} |"
        )
    report += [
        "",
        f"- Best fine-tuned checkpoint minus baseline global Dice: {delta_dice:+.6f}.",
        f"- Best fine-tuned checkpoint minus baseline mean-image clDice: {delta_cldice:+.6f}.",
        "",
        "## Visual inspection",
        f"- Contact sheets saved in `{panels_dir}`.",
        "- Compare thin structures, fragmentation, missed filaments, and false positives.",
        "",
        "## Morphological measurements",
        f"- Rows with measurements: {len(morphology_rows)}.",
        "- Area, bounding-box dimensions, skeleton length, and area/skeleton-length width estimates are in pixels.",
        "- Skeleton length is an approximation; the width estimate is not a calibrated physical width.",
        "- Physical units require a validated image scale and appropriate solar-image calibration.",
        "",
        "## Limitations",
        "- Results are validation-set measurements, not an independent external test.",
        "- Checkpoint selection used validation Dice during training; differences should not be interpreted as proof of generalization.",
        "- Existing DeepLab and Mask R-CNN results are not automatically merged because their split/protocol comparability must be verified.",
        "- No official panoptic-quality score is calculated by this pipeline.",
        "- Visualizations are intended for qualitative review; inspect difficult cases before drawing scientific conclusions.",
        "",
        "## Artifacts",
        "- `checkpoint_audit.json`",
        "- `model_comparison.csv`",
        "- `per_image_all_models.csv`",
        "- `morphology_measurements.csv`",
        "- Per-model metrics JSON and CSV files",
        "- `predictions/`, `reference/`, `visualizations/`, and optional training plots",
    ]
    (OUT / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    json_dump(OUT / "checkpoint_audit.json", audit)

    print("\n" + "=" * 70)
    print("POST-TRAINING ANALYSIS COMPLETE")
    print("=" * 70)
    for r in summary_rows:
        print(
            f"{r['model']:12s} | global Dice {r['global_dice']:.6f}"
            f" | global IoU {r['global_iou']:.6f}"
            f" | mean clDice {r['mean_cldice']:.6f}"
        )
    print(f"\nReport: {(OUT / 'final_report.md').resolve()}")
    print(f"Comparison CSV: {(OUT / 'model_comparison.csv').resolve()}")
    print(f"Visualizations: {panels_dir.resolve()}")
    print(f"Audit: {(OUT / 'checkpoint_audit.json').resolve()}")

if __name__ == "__main__":
    main()
