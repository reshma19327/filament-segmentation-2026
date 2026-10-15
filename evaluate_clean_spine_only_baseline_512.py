from pathlib import Path
import csv
import json
import time

import torch
from torch.utils.data import DataLoader
import train_benchmark_unetpp_spine_only_512 as training

ROOT = Path("benchmark_models_512/unetpp_spine_only")
CHECKPOINT_PATH = ROOT / "best_unetpp_spine_only_512.pth"
SPLIT_PATH = Path("benchmark_512/benchmark_split.json")
RESULTS_DIR = Path("benchmark_models_512/baseline_evaluation_clean")
THRESHOLD = 0.5
CLDICE_ITERATIONS = 10

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in (CHECKPOINT_PATH, SPLIT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_records, val_records = training.load_split()
    dataset = training.FilamentSpineDataset(val_records)
    loader = DataLoader(
        dataset,
        batch_size=training.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(training.DEVICE.type == "cuda"),
        drop_last=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=training.DEVICE,
        weights_only=False,
    )

    model = training.UNetPPSpine(
        in_channels=1,
        base_channels=16,
    ).to(training.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 68)
    print("CLEAN BENCHMARK BASELINE EVALUATION")
    print("=" * 68)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Best validation Dice during training:", checkpoint.get("best_dice"))
    print("Training images:", len(train_records))
    print("Validation images:", len(dataset))
    print("Device:", training.DEVICE)
    print("Threshold:", THRESHOLD)
    print("clDice iterations:", CLDICE_ITERATIONS)

    per_image = []
    totals = {
        "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0,
        "spine_tp": 0.0, "spine_tn": 0.0,
        "spine_fp": 0.0, "spine_fn": 0.0,
    }
    start = time.time()
    processed = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(training.DEVICE)
            masks = batch["mask"].to(training.DEVICE)
            spines = batch["spine"].to(training.DEVICE)

            outputs = model(images)
            filament_logits, spine_logits = training.get_outputs(outputs)

            filament_pred = (
                torch.sigmoid(filament_logits) >= THRESHOLD
            ).float()
            spine_pred = (
                torch.sigmoid(spine_logits) >= THRESHOLD
            ).float()

            for i in range(images.shape[0]):
                name = batch["name"][i]
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

                # clDice is evaluated on binary predictions and binary GT.
                cld = training.cldice_score(
                    pred, target,
                    iterations=CLDICE_ITERATIONS
                ).item()

                row = {
                    "image": name,
                    "dice": float(fm["dice"]),
                    "iou": float(fm["iou"]),
                    "precision": float(fm["precision"]),
                    "recall": float(fm["recall"]),
                    "cldice": float(cld),
                    "spine_dice": float(sm["dice"]),
                    "spine_iou": float(sm["iou"]),
                    "spine_precision": float(sm["precision"]),
                    "spine_recall": float(sm["recall"]),
                    "tp": int(fm["tp"]),
                    "tn": int(fm["tn"]),
                    "fp": int(fm["fp"]),
                    "fn": int(fm["fn"]),
                    "spine_tp": int(sm["tp"]),
                    "spine_tn": int(sm["tn"]),
                    "spine_fp": int(sm["fp"]),
                    "spine_fn": int(sm["fn"]),
                }
                per_image.append(row)

                for key in ("tp", "tn", "fp", "fn"):
                    totals[key] += fm[key]
                    totals["spine_" + key] += sm[key]

                processed += 1

            if processed % 20 < images.shape[0] or processed == len(dataset):
                print(f"Evaluated {processed}/{len(dataset)} images")

    if processed != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} images, evaluated {processed}"
        )

    def scores(tp, tn, fp, fn):
        return {
            "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
        }

    global_filament = scores(
        totals["tp"], totals["tn"], totals["fp"], totals["fn"]
    )
    global_spine = scores(
        totals["spine_tp"], totals["spine_tn"],
        totals["spine_fp"], totals["spine_fn"]
    )

    metric_names = (
        "dice", "iou", "precision", "recall", "cldice",
        "spine_dice", "spine_iou", "spine_precision", "spine_recall"
    )
    mean_image_metrics = {
        key: sum(row[key] for row in per_image) / len(per_image)
        for key in metric_names
    }

    result = {
        "experiment": "Clean fixed-split UNet++ + Spine baseline; topology loss disabled",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_dice_during_training": checkpoint.get("best_dice"),
        "split_file": str(SPLIT_PATH),
        "train_images": len(train_records),
        "validation_images_evaluated": processed,
        "image_size": 512,
        "threshold": THRESHOLD,
        "cldice_iterations": CLDICE_ITERATIONS,
        "evaluation_seconds": round(time.time() - start, 2),
        "global_pixel_filament_metrics": global_filament,
        "global_pixel_spine_metrics": global_spine,
        "mean_per_image_metrics": mean_image_metrics,
        "notes": [
            "Global pixel metrics aggregate confusion counts across all validation pixels.",
            "Mean-per-image metrics average each image's metric equally.",
            "clDice uses thresholded binary predictions and binary ground-truth masks.",
            "Ground truth uses the training script's configured full_masks and full_spine_masks."
        ]
    }

    json_path = RESULTS_DIR / "baseline_metrics_512.json"
    csv_path = RESULTS_DIR / "baseline_per_image_metrics_512.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    print("\n" + "=" * 68)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 68)
    print("Images evaluated:", processed)
    print("\nGlobal pixel metrics:")
    for key in ("dice", "iou", "precision", "recall"):
        print(f"  {key:12s}: {global_filament[key]:.6f}")
    print("\nMean per-image metrics:")
    for key in metric_names:
        print(f"  {key:16s}: {mean_image_metrics[key]:.6f}")

    print("\nSaved:")
    print(json_path.resolve())
    print(csv_path.resolve())
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
