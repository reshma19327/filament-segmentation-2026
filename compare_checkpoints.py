from pathlib import Path
import random
import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataset_unetpp_spine import UNetPPSpineDataset
from unetpp_spine import UNetPPSpine


# ------------------------------------------------------------
# Configuration: mirrors the two supplied training scripts
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
MASK_DIR = PROJECT_ROOT / "data" / "full_masks"
SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"

BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_unetpp_spine.pth"
CLDICE_CHECKPOINT = PROJECT_ROOT / "checkpoints_cldice" / "best_unetpp_spine_cldice.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
TRAIN_RATIO = 0.80
RANDOM_SEED = 42
NUM_WORKERS = 0
OUTPUT_DIR = PROJECT_ROOT / "checkpoint_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_image_names():
    for directory in (IMAGE_DIR, MASK_DIR, SPINE_DIR):
        if not directory.exists():
            raise FileNotFoundError(f"Required directory does not exist:\n{directory}")

    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    image_names = []
    for image_path in image_files:
        stem = image_path.stem
        if (MASK_DIR / f"{stem}.png").exists() and (SPINE_DIR / f"{stem}.png").exists():
            image_names.append(image_path.name)

    if len(image_names) != 707:
        raise RuntimeError(
            f"Expected 707 usable images, found {len(image_names)}. "
            "This would not match the supplied training scripts."
        )
    return image_names


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n"
            "Check that the training runs saved their best checkpoints."
        )

    # Checkpoints produced by the supplied scripts are dictionaries containing
    # a 'model_state_dict' entry.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
    saved_dice = checkpoint.get("best_val_dice", None) if isinstance(checkpoint, dict) else None
    model.eval()
    return epoch, saved_dice


def new_confusion():
    return [0, 0, 0, 0]  # TP, TN, FP, FN


def accumulate_confusion(logits, target, counts):
    prediction = torch.sigmoid(logits) >= 0.5
    target = target >= 0.5
    counts[0] += int((prediction & target).sum().item())
    counts[1] += int((~prediction & ~target).sum().item())
    counts[2] += int((prediction & ~target).sum().item())
    counts[3] += int((~prediction & target).sum().item())


def calculate_metrics(counts):
    tp, tn, fp, fn = map(float, counts)
    eps = 1e-12
    return {
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def tensor_to_image(tensor):
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = np.transpose(arr, (1, 2, 0))
    # The dataset sanity check reported image tensors in [0, 1].
    arr = np.clip(arr, 0, 1)
    return arr


@torch.no_grad()
def evaluate_model(model, loader, device):
    mask_counts = new_confusion()
    spine_counts = new_confusion()
    for images, masks, spines, _names in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        spines = spines.to(device, non_blocking=True)
        mask_logits, spine_logits = model(images)
        accumulate_confusion(mask_logits, masks, mask_counts)
        accumulate_confusion(spine_logits, spines, spine_counts)
    return calculate_metrics(mask_counts), calculate_metrics(spine_counts)


@torch.no_grad()
def save_visual_examples(baseline, cldice, dataset, val_indices, device, max_examples=8):
    count = min(max_examples, len(val_indices))
    # Deterministic examples drawn from the validation subset.
    chosen_indices = val_indices[:count]
    fig, axes = plt.subplots(count, 7, figsize=(21, max(3, 3 * count)), squeeze=False)

    column_titles = [
        "Input image", "Ground-truth filament",
        "Baseline filament", "clDice filament",
        "Ground-truth spine", "Baseline spine", "clDice spine",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row, dataset_index in enumerate(chosen_indices):
        image, mask, spine, image_name = dataset[dataset_index]
        image_batch = image.unsqueeze(0).to(device)
        base_mask_logits, base_spine_logits = baseline(image_batch)
        cl_mask_logits, cl_spine_logits = cldice(image_batch)

        image_np = tensor_to_image(image)
        mask_np = tensor_to_image(mask)
        spine_np = tensor_to_image(spine)
        base_mask = (torch.sigmoid(base_mask_logits[0, 0]) >= 0.5).cpu().numpy()
        cl_mask = (torch.sigmoid(cl_mask_logits[0, 0]) >= 0.5).cpu().numpy()
        base_spine = (torch.sigmoid(base_spine_logits[0, 0]) >= 0.5).cpu().numpy()
        cl_spine = (torch.sigmoid(cl_spine_logits[0, 0]) >= 0.5).cpu().numpy()

        panels = [image_np, mask_np, base_mask, cl_mask, spine_np, base_spine, cl_spine]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel, cmap="gray" if np.asarray(panel).ndim == 2 else None)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(image_name[:18], fontsize=8)

    fig.suptitle("Validation predictions (threshold = 0.5)", fontsize=13)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "validation_visual_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisual examples saved to:\n{out_path}")


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    image_names = discover_image_names()
    dataset = UNetPPSpineDataset(
        image_dir=str(IMAGE_DIR),
        mask_dir=str(MASK_DIR),
        spine_dir=str(SPINE_DIR),
        image_names=image_names,
        image_size=IMAGE_SIZE,
    )

    # Must match random_split(full_dataset, [565, 142],
    # generator=torch.Generator().manual_seed(42)) in both training scripts.
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    _train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_indices = list(val_subset.indices)

    baseline = UNetPPSpine().to(device)
    cldice = UNetPPSpine().to(device)
    base_epoch, base_saved_dice = load_checkpoint(baseline, BASELINE_CHECKPOINT, device)
    cl_epoch, cl_saved_dice = load_checkpoint(cldice, CLDICE_CHECKPOINT, device)

    print(f"Validation images: {len(val_subset)}")
    print(f"Baseline checkpoint epoch: {base_epoch}; saved best Dice: {base_saved_dice}")
    print(f"clDice checkpoint epoch:   {cl_epoch}; saved best Dice: {cl_saved_dice}")
    print("Evaluating both models on the exact same validation images at threshold 0.5...")

    start = time.perf_counter()
    base_mask, base_spine = evaluate_model(baseline, val_loader, device)
    base_seconds = time.perf_counter() - start

    start = time.perf_counter()
    cl_mask, cl_spine = evaluate_model(cldice, val_loader, device)
    cl_seconds = time.perf_counter() - start

    rows = []
    for task, base_metrics, cl_metrics in (
        ("Filament", base_mask, cl_mask),
        ("Spine", base_spine, cl_spine),
    ):
        for metric in ("Dice", "IoU", "Precision", "Recall", "Accuracy"):
            rows.append({
                "Task": task, "Metric": metric,
                "Baseline": base_metrics[metric],
                "clDice": cl_metrics[metric],
                "clDice_minus_baseline": cl_metrics[metric] - base_metrics[metric],
            })
        for metric in ("TP", "TN", "FP", "FN"):
            rows.append({
                "Task": task, "Metric": metric,
                "Baseline": base_metrics[metric],
                "clDice": cl_metrics[metric],
                "clDice_minus_baseline": cl_metrics[metric] - base_metrics[metric],
            })

    csv_path = OUTPUT_DIR / "metrics_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def print_task(title, base, cl):
        print(f"\n{title.upper()}")
        print(f"{'Metric':<12}{'Baseline':>14}{'clDice':>14}{'Difference':>14}")
        for metric in ("Dice", "IoU", "Precision", "Recall", "Accuracy"):
            print(f"{metric:<12}{base[metric]:>14.6f}{cl[metric]:>14.6f}{cl[metric]-base[metric]:>14.6f}")
        print(f"TP          {base['TP']:>14}{cl['TP']:>14}{cl['TP']-base['TP']:>14}")
        print(f"TN          {base['TN']:>14}{cl['TN']:>14}{cl['TN']-base['TN']:>14}")
        print(f"FP          {base['FP']:>14}{cl['FP']:>14}{cl['FP']-base['FP']:>14}")
        print(f"FN          {base['FN']:>14}{cl['FN']:>14}{cl['FN']-base['FN']:>14}")

    print_task("Filament mask", base_mask, cl_mask)
    print_task("Spine mask", base_spine, cl_spine)
    print(f"\nEvaluation wall time (baseline): {base_seconds:.2f}s")
    print(f"Evaluation wall time (clDice):   {cl_seconds:.2f}s")
    print(f"CSV metrics saved to:\n{csv_path}")

    save_visual_examples(baseline, cldice, dataset, val_indices, device)


if __name__ == "__main__":
    main()
