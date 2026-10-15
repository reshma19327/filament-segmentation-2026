# ================================================================
# repair_maskrcnn_deeplabv3_512.py
#
# Repair + re-evaluate ONLY:
#   1) Mask R-CNN
#   2) DeepLabV3
#
# Uses the SAME fixed 142-image validation split used by the
# unified 7-model benchmark.
#
# IMPORTANT:
# - Mask R-CNN predictions are converted to a single binary
#   semantic mask by unioning all accepted instance masks.
# - Predictions are saved as true 0/255 PNG masks BEFORE evaluation.
# - DeepLabV3 predictions are converted from logits/probabilities
#   to true binary 0/255 PNG masks.
# - Existing broken Mask R-CNN / DeepLabV3 prediction PNGs are
#   overwritten.
# ================================================================

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
)

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"
SPLIT_FILE = PROJECT_ROOT / "benchmark_512" / "benchmark_split.json"

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_models_512"

MASKRCNN_DIR = BENCHMARK_DIR / "maskrcnn"
MASKRCNN_CKPT = MASKRCNN_DIR / "best_maskrcnn_512.pth"
MASKRCNN_PRED_DIR = MASKRCNN_DIR / "predictions"

DEEPLAB_DIR = BENCHMARK_DIR / "deeplabv3"
DEEPLAB_CKPT = DEEPLAB_DIR / "best_deeplabv3_512.pth"
DEEPLAB_PRED_DIR = DEEPLAB_DIR / "predictions"

OUTPUT_DIR = PROJECT_ROOT / "unified_evaluation_512"
REPAIR_REPORT = OUTPUT_DIR / "maskrcnn_deeplab_repair_report.json"
REPAIR_CSV = OUTPUT_DIR / "maskrcnn_deeplab_repaired_results.csv"

IMAGE_SIZE = 512
DEFAULT_PIXEL_THRESHOLD = 0.5
DEFAULT_MASK_SCORE_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------
# GENERAL HELPERS
# ----------------------------------------------------------------

def banner(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def normalize_name(value):
    return Path(str(value)).name


def find_file(directory, filename):
    filename = normalize_name(filename)
    direct = directory / filename
    if direct.exists():
        return direct

    target = filename.lower()
    for p in directory.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p

    return None


def get_record_filename(record, keys):
    for key in keys:
        value = record.get(key)
        if value:
            return normalize_name(value)
    return None


def load_split():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split file:\n{SPLIT_FILE}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("validation", "val", "valid", "validation_records"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            # Some split files use nested train/validation structures.
            records = None
            for value in data.values():
                if isinstance(value, list):
                    # Prefer a list whose records look like validation records.
                    if value and isinstance(value[0], dict):
                        records = value
                        break
            if records is None:
                raise ValueError(
                    "Could not locate validation records in benchmark_split.json"
                )
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported benchmark_split.json structure")

    return records


def resolve_validation_records():
    records = load_split()

    resolved = []

    for i, record in enumerate(records, 1):
        image_name = get_record_filename(
            record,
            [
                "image",
                "image_name",
                "filename",
                "file_name",
                "image_file",
                "image_path",
            ],
        )

        mask_name = get_record_filename(
            record,
            [
                "mask",
                "mask_name",
                "mask_file",
                "mask_filename",
                "mask_path",
            ],
        )

        if image_name is None:
            raise ValueError(f"Validation record {i} has no image filename:\n{record}")

        image_path = find_file(IMAGE_DIR, image_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Validation image not found for record {i}:\n{image_name}"
            )

        # If split contains an explicit mask name, use it.
        # Otherwise try the image stem in full_masks.
        if mask_name is not None:
            mask_path = find_file(MASK_DIR, mask_name)
        else:
            mask_path = None

        if mask_path is None:
            stem = Path(image_name).stem
            candidates = [
                stem + ".png",
                stem + ".jpg",
                stem + ".jpeg",
                stem + ".tif",
                stem + ".tiff",
                stem + "_mask.png",
                stem + "_mask.jpg",
            ]

            for candidate in candidates:
                mask_path = find_file(MASK_DIR, candidate)
                if mask_path is not None:
                    break

        if mask_path is None:
            raise FileNotFoundError(
                f"Validation mask not found for image:\n{image_name}"
            )

        resolved.append(
            {
                "image_name": image_path.name,
                "image_path": image_path,
                "mask_name": mask_path.name,
                "mask_path": mask_path,
            }
        )

    return resolved


def read_image_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Could not read image:\n{path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] >= 3:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    else:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32)

    if image.max() > 1.0:
        image /= 255.0

    return image


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask:\n{path}")

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask > 0


# ----------------------------------------------------------------
# CHECKPOINT HANDLING
# ----------------------------------------------------------------

def load_checkpoint_raw(path):
    print(f"Checkpoint: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict(), {"raw_module": True}

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, checkpoint

        # Plain state_dict checkpoint.
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if tensor_values:
                return checkpoint, checkpoint

    raise ValueError(
        f"Could not extract a state_dict from checkpoint:\n{path}"
    )


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        # Remove common wrappers repeatedly.
        changed = True
        while changed:
            changed = False
            for prefix in (
                "module.",
                "model.",
                "net.",
                "network.",
            ):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_state_dict_flexible(model, state_dict, checkpoint_path):
    state_dict = clean_state_dict(state_dict)

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Exact load first.
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully with strict=True.")
        return
    except RuntimeError as strict_error:
        pass

    # Shape-compatible partial load.
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_keys:
            skipped.append((key, "unexpected"))
            continue

        if not torch.is_tensor(value):
            skipped.append((key, "non-tensor"))
            continue

        if tuple(value.shape) != tuple(model.state_dict()[key].shape):
            skipped.append(
                (
                    key,
                    f"shape {tuple(value.shape)} != "
                    f"{tuple(model.state_dict()[key].shape)}",
                )
            )
            continue

        compatible[key] = value

    missing = sorted(model_keys - set(compatible.keys()))

    # We do NOT silently accept a badly mismatched model.
    coverage = len(compatible) / max(1, len(model_keys))

    print(f"Compatible checkpoint tensors: {len(compatible)}")
    print(f"Model tensors:                 {len(model_keys)}")
    print(f"Coverage:                      {coverage:.3f}")
    print(f"Skipped checkpoint tensors:    {len(skipped)}")
    print(f"Missing model tensors:          {len(missing)}")

    if coverage < 0.90:
        raise RuntimeError(
            f"\nCheckpoint architecture mismatch for:\n{checkpoint_path}\n"
            f"Only {coverage:.1%} of model tensors are compatible.\n"
            f"Do NOT continue with a partially initialized model."
        )

    model.load_state_dict(compatible, strict=False)
    print("Checkpoint loaded with strict=False using compatible tensors.")


# ----------------------------------------------------------------
# MASK R-CNN
# ----------------------------------------------------------------

def infer_maskrcnn_num_classes(state_dict):
    state_dict = clean_state_dict(state_dict)

    candidates = [
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
    ]

    for key in candidates:
        tensor = state_dict.get(key)
        if tensor is not None:
            if tensor.ndim == 2:
                return int(tensor.shape[0])
            if tensor.ndim == 1:
                return int(tensor.shape[0])

    return 2


def build_maskrcnn(checkpoint_path):
    state_dict, meta = load_checkpoint_raw(checkpoint_path)
    num_classes = infer_maskrcnn_num_classes(state_dict)

    print(f"Detected Mask R-CNN num_classes: {num_classes}")

    # No pretrained weights: the checkpoint supplies all learned weights.
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )

    load_state_dict_flexible(model, state_dict, checkpoint_path)
    model.to(DEVICE)
    model.eval()

    return model


@torch.inference_mode()
def predict_maskrcnn(model, image_tensor, score_threshold, pixel_threshold):
    output = model([image_tensor])[0]

    boxes = output.get("boxes")
    scores = output.get("scores")
    labels = output.get("labels")
    masks = output.get("masks")

    if masks is None or masks.numel() == 0:
        return (
            np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
            {
                "instances_total": 0,
                "instances_kept": 0,
                "max_score": 0.0,
                "positive_fraction": 0.0,
            },
        )

    total = int(masks.shape[0])

    if scores is None:
        keep = torch.ones(total, dtype=torch.bool, device=masks.device)
        max_score = 0.0
    else:
        scores = scores.detach()
        keep = scores >= float(score_threshold)
        max_score = float(scores.max().item()) if scores.numel() else 0.0

    # If class labels exist, use foreground classes only.
    if labels is not None:
        labels = labels.detach()
        keep = keep & (labels > 0)

    kept_count = int(keep.sum().item())

    if kept_count == 0:
        return (
            np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
            {
                "instances_total": total,
                "instances_kept": 0,
                "max_score": max_score,
                "positive_fraction": 0.0,
            },
        )

    selected = masks[keep, 0]

    # Torchvision Mask R-CNN inference returns mask probabilities.
    # Union all accepted instances.
    union_prob = selected.max(dim=0).values

    binary = union_prob >= float(pixel_threshold)

    # Final safety: always save a genuine binary mask.
    binary_np = binary.detach().cpu().numpy().astype(np.uint8) * 255

    return (
        binary_np,
        {
            "instances_total": total,
            "instances_kept": kept_count,
            "max_score": max_score,
            "positive_fraction": float(binary_np.mean() / 255.0),
        },
    )


def generate_maskrcnn_predictions(records, score_threshold, pixel_threshold):
    banner("REPAIRING MASK R-CNN PREDICTIONS")

    MASKRCNN_PRED_DIR.mkdir(parents=True, exist_ok=True)

    model = build_maskrcnn(MASKRCNN_CKPT)

    diagnostics = []

    start = time.time()

    for i, item in enumerate(records, 1):
        image = read_image_rgb(item["image_path"])

        tensor = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float().to(DEVICE)

        pred, info = predict_maskrcnn(
            model,
            tensor,
            score_threshold,
            pixel_threshold,
        )

        out_path = MASKRCNN_PRED_DIR / f"{Path(item['image_name']).stem}.png"
        cv2.imwrite(str(out_path), pred)

        diagnostics.append(
            {
                "image": item["image_name"],
                **info,
            }
        )

        if i == 1 or i % 20 == 0 or i == len(records):
            print(f"Generated {i:3d}/{len(records)}")

    elapsed = time.time() - start

    print(f"Prediction time: {elapsed:.1f}s")

    positive_fractions = [x["positive_fraction"] for x in diagnostics]
    kept_counts = [x["instances_kept"] for x in diagnostics]

    print(
        f"Average predicted foreground fraction: "
        f"{np.mean(positive_fractions):.6f}"
    )
    print(
        f"Images with >=1 accepted instance: "
        f"{sum(x > 0 for x in kept_counts)}/{len(records)}"
    )

    return diagnostics


# ----------------------------------------------------------------
# DEEPLABV3
# ----------------------------------------------------------------

def infer_deeplab_architecture(state_dict):
    state_dict = clean_state_dict(state_dict)

    keys = list(state_dict.keys())

    # Torchvision ResNet DeepLab has keys such as backbone.layer1...
    if any(k.startswith("backbone.layer1.") for k in keys):
        backbone = "resnet50"
    elif any(k.startswith("backbone.0.") for k in keys):
        backbone = "mobilenet_v3_large"
    else:
        # ResNet50 is the safest default for a standard torchvision
        # DeepLabV3 checkpoint.
        backbone = "resnet50"

    classifier_weight = state_dict.get("classifier.4.weight")
    classifier_bias = state_dict.get("classifier.4.bias")

    if classifier_weight is not None:
        num_classes = int(classifier_weight.shape[0])
    elif classifier_bias is not None:
        num_classes = int(classifier_bias.shape[0])
    else:
        num_classes = 1

    aux_loss = any(
        k.startswith("aux_classifier.")
        for k in keys
    )

    return backbone, num_classes, aux_loss


def build_deeplab(checkpoint_path):
    state_dict, meta = load_checkpoint_raw(checkpoint_path)

    backbone, num_classes, aux_loss = infer_deeplab_architecture(
        state_dict
    )

    print(f"Detected DeepLabV3 backbone: {backbone}")
    print(f"Detected DeepLabV3 num_classes: {num_classes}")
    print(f"Detected auxiliary classifier: {aux_loss}")

    if backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=aux_loss,
        )
    else:
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=aux_loss,
        )

    
    # Reconstruct the grayscale input layer used during training.
    old_conv = model.backbone.conv1
    model.backbone.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    load_state_dict_flexible(model, state_dict, checkpoint_path)




    model.to(DEVICE)
    model.eval()

    return model, num_classes


def extract_deeplab_logits(output):
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]

        # Fallback for unusual checkpoints.
        for value in output.values():
            if torch.is_tensor(value) and value.ndim == 4:
                return value

    if torch.is_tensor(output):
        return output

    raise TypeError(
        f"Unsupported DeepLab output type: {type(output)}"
    )


@torch.inference_mode()
def predict_deeplab(model, image_tensor, num_classes, threshold):
    output = model(image_tensor.unsqueeze(0))
    logits = extract_deeplab_logits(output)

    # Ensure 512x512 output.
    if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits[:, 0])
    else:
        probability = torch.softmax(logits, dim=1)[:, 1]

    binary = probability >= float(threshold)

    pred = (
        binary[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )

    info = {
        "probability_min": float(probability.min().item()),
        "probability_max": float(probability.max().item()),
        "probability_mean": float(probability.mean().item()),
        "positive_fraction": float(pred.mean() / 255.0),
    }

    return pred, info


def generate_deeplab_predictions(records, threshold):
    banner("REPAIRING DEEPLABV3 PREDICTIONS")

    DEEPLAB_PRED_DIR.mkdir(parents=True, exist_ok=True)

    model, num_classes = build_deeplab(DEEPLAB_CKPT)

    diagnostics = []
    start = time.time()

    for i, item in enumerate(records, 1):
        image = cv2.imread(
            str(item["image_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image:\n{item['image_path']}"
            )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(image).unsqueeze(0).float().to(DEVICE)


        pred, info = predict_deeplab(
            model,
            tensor,
            num_classes,
            threshold,
        )

        out_path = DEEPLAB_PRED_DIR / f"{Path(item['image_name']).stem}.png"
        cv2.imwrite(str(out_path), pred)

        diagnostics.append(
            {
                "image": item["image_name"],
                **info,
            }
        )

        if i == 1 or i % 20 == 0 or i == len(records):
            print(f"Generated {i:3d}/{len(records)}")

    elapsed = time.time() - start

    print(f"Prediction time: {elapsed:.1f}s")
    print(
        f"Average predicted foreground fraction: "
        f"{np.mean([x['positive_fraction'] for x in diagnostics]):.6f}"
    )

    return diagnostics


# ----------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------

def safe_div(a, b):
    return float(a / b) if b else 0.0


def compute_metrics(tp, tn, fp, fn):
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    denominator = np.sqrt(
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )

    mcc = safe_div(
        tp * tn - fp * fn,
        denominator,
    )

    total = tp + tn + fp + fn
    expected = safe_div(
        (tp + fp) * (tp + fn)
        + (tn + fn) * (tn + fp),
        total * total if total else 0,
    )

    observed = safe_div(tp + tn, total)

    kappa = safe_div(
        observed - expected,
        1.0 - expected,
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
        "Cohen_Kappa": kappa,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def evaluate_prediction_dir(records, prediction_dir):
    tp = tn = fp = fn = 0

    missing = []

    for i, item in enumerate(records, 1):
        stem = Path(item["image_name"]).stem
        pred_path = prediction_dir / f"{stem}.png"

        if not pred_path.exists():
            # Flexible fallback for extensions.
            candidates = list(prediction_dir.glob(stem + ".*"))
            if not candidates:
                missing.append(stem)
                continue
            pred_path = candidates[0]

        gt = read_mask(item["mask_path"])

        pred = cv2.imread(
            str(pred_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if pred is None:
            raise RuntimeError(f"Could not read prediction:\n{pred_path}")

        pred = cv2.resize(
            pred,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        # Predictions produced by this repair script are 0/255.
        pred_bool = pred >= 128

        tp += int(np.logical_and(pred_bool, gt).sum())
        tn += int(np.logical_and(~pred_bool, ~gt).sum())
        fp += int(np.logical_and(pred_bool, ~gt).sum())
        fn += int(np.logical_and(~pred_bool, gt).sum())

        if i == 1 or i % 20 == 0 or i == len(records):
            print(f"Evaluated {i:3d}/{len(records)}")

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} prediction files are missing in "
            f"{prediction_dir}.\nFirst missing:\n"
            + "\n".join(missing[:20])
        )

    return compute_metrics(tp, tn, fp, fn)


def save_csv(rows, path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "MCC",
        "Cohen_Kappa",
        "TP",
        "TN",
        "FP",
        "FN",
        "Status",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repair Mask R-CNN and DeepLabV3 512 benchmark predictions."
    )

    parser.add_argument(
        "--mask-score-threshold",
        type=float,
        default=DEFAULT_MASK_SCORE_THRESHOLD,
        help="Mask R-CNN instance confidence threshold.",
    )

    parser.add_argument(
        "--mask-pixel-threshold",
        type=float,
        default=DEFAULT_PIXEL_THRESHOLD,
        help="Mask R-CNN per-pixel mask threshold.",
    )

    parser.add_argument(
        "--deeplab-threshold",
        type=float,
        default=DEFAULT_PIXEL_THRESHOLD,
        help="DeepLabV3 pixel probability threshold.",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing prediction PNGs before repair.",
    )

    args = parser.parse_args()

    banner("MASK R-CNN + DEEPLABV3 512 REPAIR")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Device       : {DEVICE}")
    print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Mask score   : {args.mask_score_threshold}")
    print(f"Mask pixel   : {args.mask_pixel_threshold}")
    print(f"DeepLab      : {args.deeplab_threshold}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    banner("LOADING FIXED VALIDATION SPLIT")

    records = resolve_validation_records()
    print(f"Validation records: {len(records)}")

    if len(records) != 142:
        print(
            "WARNING: expected 142 validation records for this benchmark, "
            f"but found {len(records)}."
        )

    # Remove broken predictions so stale files cannot be mistaken for
    # repaired outputs.
    if not args.keep_existing:
        for directory in (MASKRCNN_PRED_DIR, DEEPLAB_PRED_DIR):
            if directory.exists():
                for p in directory.glob("*.png"):
                    p.unlink()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Mask R-CNN
    # ---------------------------

    mask_diag = generate_maskrcnn_predictions(
        records,
        args.mask_score_threshold,
        args.mask_pixel_threshold,
    )

    banner("EVALUATING REPAIRED MASK R-CNN")

    mask_metrics = evaluate_prediction_dir(
        records,
        MASKRCNN_PRED_DIR,
    )

    print_metrics("Mask R-CNN", mask_metrics)

    # ---------------------------
    # DeepLabV3
    # ---------------------------

    deeplab_diag = generate_deeplab_predictions(
        records,
        args.deeplab_threshold,
    )

    banner("EVALUATING REPAIRED DEEPLABV3")

    deeplab_metrics = evaluate_prediction_dir(
        records,
        DEEPLAB_PRED_DIR,
    )

    print_metrics("DeepLabV3", deeplab_metrics)

    rows = [
        {
            "Model": "Mask R-CNN",
            **mask_metrics,
            "Status": "SUCCESS",
        },
        {
            "Model": "DeepLabV3",
            **deeplab_metrics,
            "Status": "SUCCESS",
        },
    ]

    save_csv(rows, REPAIR_CSV)

    report = {
        "validation_records": len(records),
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "maskrcnn": {
            "checkpoint": str(MASKRCNN_CKPT),
            "score_threshold": args.mask_score_threshold,
            "pixel_threshold": args.mask_pixel_threshold,
            "metrics": mask_metrics,
            "diagnostics": mask_diag,
        },
        "deeplabv3": {
            "checkpoint": str(DEEPLAB_CKPT),
            "threshold": args.deeplab_threshold,
            "metrics": deeplab_metrics,
            "diagnostics": deeplab_diag,
        },
    }

    with open(REPAIR_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    banner("REPAIR COMPLETE")

    print(f"Mask R-CNN predictions : {MASKRCNN_PRED_DIR}")
    print(f"DeepLabV3 predictions  : {DEEPLAB_PRED_DIR}")
    print(f"Results CSV            : {REPAIR_CSV}")
    print(f"Repair report          : {REPAIR_REPORT}")

    print()
    print("IMPORTANT:")
    print("Run the unified 7-model evaluation again after this script.")
    print("The repaired PNGs are binary 0/255 masks, not probability PNGs.")


def print_metrics(name, metrics):
    print()
    print("-" * 80)
    print(f"{name}")
    print("-" * 80)

    print(f"Dice:          {metrics['Dice']:.6f}")
    print(f"IoU:           {metrics['IoU']:.6f}")
    print(f"Precision:     {metrics['Precision']:.6f}")
    print(f"Recall:        {metrics['Recall']:.6f}")
    print(f"MCC:            {metrics['MCC']:.6f}")
    print(f"Cohen Kappa:   {metrics['Cohen_Kappa']:.6f}")
    print(
        f"TP: {metrics['TP']:,} | "
        f"TN: {metrics['TN']:,} | "
        f"FP: {metrics['FP']:,} | "
        f"FN: {metrics['FN']:,}"
    )


if __name__ == "__main__":
    main()
