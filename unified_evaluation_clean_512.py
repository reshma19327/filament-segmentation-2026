# =============================================================================
# unified_evaluation_7_models_512.py
#
# UNIFIED 7-MODEL VALIDATION EVALUATION
#
# Models:
#   1. UNet++ + Spine + clDice (Improved)
#   2. UNet++ Baseline
#   3. UNet++ + Spine
#   4. Attention U-Net
#   5. UNeXt
#   6. Mask R-CNN
#   7. DeepLabV3
#
# Validation:
#   Fixed benchmark_512/benchmark_split.json
#   142 validation images
#   512 x 512
#
# Metrics:
#   Dice
#   IoU
#   Precision
#   Recall
#   MCC
#   Cohen Kappa
#
# IMPORTANT:
#   This evaluator NEVER substitutes a different architecture silently.
#   For custom Attention U-Net / UNeXt checkpoints, it searches project
#   Python modules for a compatible model class and validates checkpoint
#   compatibility before loading.
#
# Outputs:
#   unified_evaluation_512/
#       predictions/
#           <model_name>/
#       unified_7_model_results.csv
#       unified_7_model_results.json
#       unified_7_model_results.txt
# =============================================================================

import os
import sys
import json
import time
import math
import inspect
import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"

SPLIT_FILE = (
    PROJECT_ROOT
    / "benchmark_512"
    / "benchmark_split.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "unified_evaluation_clean_512"
)

PREDICTIONS_ROOT = OUTPUT_DIR / "predictions"

IMAGE_SIZE = 512
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Generate predictions even when prediction directory exists but is incomplete.
FORCE_REGENERATE = False

# Continue to next model when a model fails.
CONTINUE_ON_MODEL_ERROR = True


# =============================================================================
# MODEL PATHS
# =============================================================================

MODELS = [
    {
        "name": "UNet++ Baseline (clean benchmark)",
        "key": "unetpp_baseline",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "unetpp_baseline" / "best_unetpp_baseline_512.pth",
        "type": "unetpp_baseline",
    },
    {
        "name": "UNet++ Spine-only (clean benchmark)",
        "key": "unetpp_spine_only",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "unetpp_spine_only" / "best_unetpp_spine_only_512.pth",
        "type": "unetpp_spine",
    },
    {
        "name": "UNet++ Spine + clDice (clean fine-tune)",
        "key": "unetpp_spine_cldice_finetune",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "unetpp_spine_cldice_finetune" / "best_unetpp_spine_topology_512.pth",
        "type": "unetpp_spine",
    },
    {
        "name": "Attention U-Net",
        "key": "attention_unet",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "attention_unet" / "best_attention_unet_512.pth",
        "type": "attention_unet",
    },
    {
        "name": "UNeXt",
        "key": "unext",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "unext" / "best_unext_512.pth",
        "type": "unext",
    },
    {
        "name": "Mask R-CNN",
        "key": "maskrcnn",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "maskrcnn" / "best_maskrcnn_512.pth",
        "type": "maskrcnn",
    },
    {
        "name": "DeepLabV3",
        "key": "deeplabv3",
        "checkpoint": PROJECT_ROOT / "benchmark_models_512" / "deeplabv3" / "best_deeplabv3_512.pth",
        "type": "deeplabv3",
    },
]


# =============================================================================
# PRINTING
# =============================================================================

def banner(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def section(text):
    print()
    print("-" * 80)
    print(text)
    print("-" * 80)


# =============================================================================
# DEVICE
# =============================================================================

def print_device():
    print(f"Device       : {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(0)

        print(
            f"GPU memory   : "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )


# =============================================================================
# FILE HELPERS
# =============================================================================

def normalize_name(value):
    return Path(str(value)).name


def case_insensitive_file(directory: Path, filename: str):
    filename = normalize_name(filename)

    direct = directory / filename

    if direct.exists():
        return direct

    target = filename.lower()

    if not directory.exists():
        return None

    for p in directory.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p

    return None


def find_record_value(record, keys):
    for key in keys:
        if key in record and record[key]:
            return normalize_name(record[key])

    return None


# =============================================================================
# SPLIT LOADING
# =============================================================================

def load_validation_records():

    banner("LOADING FIXED VALIDATION SPLIT")

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark split not found:\n{SPLIT_FILE}"
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):

        if "validation" in data:
            records = data["validation"]

        elif "val" in data:
            records = data["val"]

        elif "valid" in data:
            records = data["valid"]

        else:
            raise ValueError(
                "Could not find validation records in split JSON."
            )

    elif isinstance(data, list):
        records = data

    else:
        raise ValueError(
            "Unsupported benchmark split format."
        )

    print(
        f"Validation records: {len(records)}"
    )

    return records


# =============================================================================
# RESOLVE VALIDATION FILES
# =============================================================================

def resolve_validation_files(records):

    banner(
        "RESOLVING VALIDATION IMAGES AND MASKS"
    )

    resolved = []

    for i, record in enumerate(records, 1):

        image_name = find_record_value(
            record,
            [
                "image",
                "image_path",
                "image_file",
                "filename",
                "file_name",
                "img",
            ]
        )

        mask_name = find_record_value(
            record,
            [
                "mask",
                "mask_path",
                "mask_file",
                "mask_filename",
            ]
        )

        if image_name is None:
            raise ValueError(
                f"Could not determine image filename "
                f"for validation record {i}:\n{record}"
            )

        if mask_name is None:
            mask_name = image_name

        image_path = case_insensitive_file(
            IMAGE_DIR,
            image_name
        )

        mask_path = case_insensitive_file(
            MASK_DIR,
            mask_name
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Validation image not found:\n"
                f"{image_name}\n"
                f"Directory: {IMAGE_DIR}"
            )

        if mask_path is None:

            # Try stem matching.
            stem = Path(mask_name).stem.lower()

            candidates = [
                p for p in MASK_DIR.iterdir()
                if p.is_file()
                and p.stem.lower() == stem
            ]

            if candidates:
                mask_path = candidates[0]

        if mask_path is None:
            raise FileNotFoundError(
                f"Validation mask not found:\n"
                f"{mask_name}\n"
                f"Directory: {MASK_DIR}"
            )

        resolved.append(
            {
                "image": image_path,
                "mask": mask_path,
                "image_name": image_path.name,
            }
        )

        if (
            i == 1
            or i % 20 == 0
            or i == len(records)
        ):
            print(
                f"Resolved {i}/{len(records)}"
            )

    print()
    print(
        f"Resolved validation images: "
        f"{len(resolved)}"
    )

    return resolved


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(path, model_type=None):
    """Load a grayscale image with the input shape expected by each model."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not read image:\n{path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)

    if model_type == "maskrcnn":
        # Torchvision detection models expect one CxHxW tensor per image.
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    else:
        # Segmentation models expect an NxCxHxW batch tensor.
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor



def load_mask(path):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask:\n{path}"
        )

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (
        mask > 0
    ).astype(np.uint8)

    return mask


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint_raw(path):

    print()
    print(
        f"Loading checkpoint:\n{path}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return checkpoint


def extract_state_dict(checkpoint):

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint object type: "
            f"{type(checkpoint)}"
        )

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "weights",
        "net",
    ]

    for key in possible_keys:

        value = checkpoint.get(key)

        if isinstance(value, dict):

            # Verify this looks like a state dictionary.
            if any(
                isinstance(k, str)
                for k in value.keys()
            ):
                return value

        if isinstance(value, nn.Module):
            return value.state_dict()

    # Some checkpoints are already state_dict.
    if checkpoint:
        tensor_values = sum(
            torch.is_tensor(v)
            for v in checkpoint.values()
        )

        if tensor_values > 0:
            return checkpoint

    raise RuntimeError(
        "Could not extract state_dict from checkpoint."
    )


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if not isinstance(key, str):
            continue

        new_key = key

        for prefix in [
            "module.",
            "model.",
            "net.",
        ]:

            if new_key.startswith(prefix):
                new_key = new_key[
                    len(prefix):
                ]

        cleaned[new_key] = value

    return cleaned


# =============================================================================
# UNET++ MODELS
# =============================================================================

def import_unetpp_module():

    try:
        import unetpp_spine
        return unetpp_spine

    except Exception as exc:
        raise RuntimeError(
            "Could not import unetpp_spine.py:\n"
            f"{exc}"
        )


def get_unetpp_class():

    module = import_unetpp_module()

    candidates = [
        "UNetPPSpine",
        "UNetPlusPlusSpine",
    ]

    for name in candidates:

        cls = getattr(
            module,
            name,
            None
        )

        if cls is not None:
            return cls

    raise RuntimeError(
        "Could not find UNet++ class in "
        "unetpp_spine.py"
    )


def build_unetpp_spine(base_channels=16):

    cls = get_unetpp_class()

    attempts = [
        {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": base_channels,
        },
        {
            "in_channels": 1,
            "out_channels": 1,
        },
        {
            "in_channels": 1,
            "num_classes": 1,
            "base_channels": base_channels,
        },
        {
            "in_channels": 1,
            "num_classes": 1,
        },
        {},
    ]

    last_error = None

    for kwargs in attempts:

        try:
            model = cls(**kwargs)

            return model

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not instantiate UNet++ model.\n"
        f"Last error: {last_error}"
    )


def infer_base_channels(state_dict):

    candidates = [
        "x00.block.0.weight",
        "conv0_0.block.0.weight",
    ]

    for key in candidates:

        if key in state_dict:

            shape = tuple(
                state_dict[key].shape
            )

            if len(shape) >= 1:
                return int(shape[0])

    # Fallback.
    return 16


def remap_legacy_unetpp_spine(
    state_dict
):

    """
    Legacy UNet++ + Spine checkpoint:

        conv0_0 -> x00
        conv1_0 -> x10
        conv2_0 -> x20
        conv3_0 -> x30
        conv4_0 -> x40

    etc.

    Also:
        filament_head -> mask_head
    """

    result = {}

    mapping = {
        "conv0_0": "x00",
        "conv1_0": "x10",
        "conv2_0": "x20",
        "conv3_0": "x30",
        "conv4_0": "x40",

        "conv0_1": "x01",
        "conv1_1": "x11",
        "conv2_1": "x21",
        "conv3_1": "x31",

        "conv0_2": "x02",
        "conv1_2": "x12",
        "conv2_2": "x22",

        "conv0_3": "x03",
        "conv1_3": "x13",

        "conv0_4": "x04",
    }

    for key, value in state_dict.items():

        new_key = key

        for old, new in mapping.items():

            if new_key.startswith(
                old + "."
            ):
                new_key = (
                    new
                    + new_key[len(old):]
                )
                break

        if new_key.startswith(
            "filament_head."
        ):
            new_key = (
                "mask_head."
                + new_key[
                    len("filament_head.") :
                ]
            )

        result[new_key] = value

    return result


def load_unetpp_model(
    checkpoint_path,
    legacy=False,
):

    checkpoint = load_checkpoint_raw(
        checkpoint_path
    )

    state_dict = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    if legacy:
        state_dict = (
            remap_legacy_unetpp_spine(
                state_dict
            )
        )

    base_channels = infer_base_channels(
        state_dict
    )

    print(
        f"Using UNet++ base channels: "
        f"{base_channels}"
    )

    model = build_unetpp_spine(
        base_channels=base_channels
    )

    # Handle old checkpoints with only
    # compatible keys.
    model_state = model.state_dict()

    compatible = {}

    for key, value in state_dict.items():

        if key not in model_state:
            continue

        if (
            tuple(value.shape)
            != tuple(model_state[key].shape)
        ):
            continue

        compatible[key] = value

    ratio = (
        len(compatible)
        / max(1, len(model_state))
    )

    print(
        f"Compatible checkpoint tensors: "
        f"{len(compatible)}/{len(model_state)} "
        f"({ratio * 100:.2f}%)"
    )

    if ratio < 0.90:
        raise RuntimeError(
            "Checkpoint is not sufficiently compatible with the reconstructed "
            "UNet++ architecture."
        )

    load_result = model.load_state_dict(compatible, strict=True)
    print(f"Checkpoint tensors loaded: {len(compatible)}")
    print(f"Missing model keys: {len(load_result.missing_keys)}")
    print(f"Unexpected checkpoint keys: {len(load_result.unexpected_keys)}")

    model.to(DEVICE)
    model.eval()
    print("Checkpoint loaded successfully.")
    return model


def extract_segmentation_tensor(output):

    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):

        preferred = [
            "out",
            "mask",
            "masks",
            "logits",
            "filament",
            "segmentation",
        ]

        for key in preferred:

            if key in output:

                value = output[key]

                if torch.is_tensor(value):
                    return value

        for value in output.values():

            if torch.is_tensor(value):
                return value

    if isinstance(output, (list, tuple)):

        for value in output:

            if torch.is_tensor(value):
                return value

            if isinstance(value, dict):

                try:
                    return extract_segmentation_tensor(
                        value
                    )
                except Exception:
                    pass

    raise RuntimeError(
        "Could not extract segmentation tensor "
        f"from output type {type(output)}"
    )


# =============================================================================
# CUSTOM MODEL DISCOVERY
# =============================================================================

def python_files():

    excluded = {
        ".venv",
        "__pycache__",
        "site-packages",
    }

    for path in PROJECT_ROOT.rglob("*.py"):

        parts = set(path.parts)

        if parts.intersection(excluded):
            continue

        if path.name.startswith(
            "unified_evaluation"
        ):
            continue

        yield path


def candidate_class_names(model_type):

    if model_type == "attention_unet":

        return [
            "AttentionUNet",
            "AttentionUnet",
            "AttentionUnet",
            "AttentionUNet2D",
            "AttUNet",
        ]

    if model_type == "unext":

        return [
            "UNeXt",
            "Unext",
            "UNext",
            "UNextSeg",
            "UNextModel",
        ]

    return []


def import_module_from_file(path):

    module_name = (
        "_auto_eval_"
        + path.stem
        + "_"
        + str(abs(hash(str(path))))
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path)
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(
        spec
    )

    try:
        spec.loader.exec_module(module)

    except Exception:
        return None

    return module


def instantiate_with_common_arguments(cls):

    attempts = [
        {
            "in_channels": 1,
            "out_channels": 1,
        },
        {
            "in_channels": 1,
            "num_classes": 1,
        },
        {
            "channels": 1,
            "classes": 1,
        },
        {
            "input_channels": 1,
            "num_classes": 1,
        },
        {
            "img_ch": 1,
            "output_ch": 1,
        },
        {},
    ]

    last_error = None

    for kwargs in attempts:

        try:
            signature = inspect.signature(
                cls
            )

            accepted = {}

            for key, value in kwargs.items():

                if key in signature.parameters:
                    accepted[key] = value

            model = cls(**accepted)

            if isinstance(model, nn.Module):
                return model

        except Exception as exc:
            last_error = exc

    return None


def checkpoint_compatibility_score(
    model,
    checkpoint_state,
):

    model_state = model.state_dict()

    matched = 0
    shape_matched = 0

    for key, value in checkpoint_state.items():

        if key in model_state:

            matched += 1

            if (
                tuple(value.shape)
                ==
                tuple(model_state[key].shape)
            ):
                shape_matched += 1

    denominator = max(
        1,
        len(model_state)
    )

    return (
        shape_matched / denominator,
        matched,
        shape_matched,
    )


def auto_discover_custom_model(
    checkpoint_path,
    model_type,
):

    checkpoint = load_checkpoint_raw(
        checkpoint_path
    )

    state_dict = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    desired_names = set(
        candidate_class_names(
            model_type
        )
    )

    candidates = []

    # Search project Python source.
    for path in python_files():

        try:
            source = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        lower = source.lower()

        if model_type == "attention_unet":

            if (
                "attention" not in lower
                or "unet" not in lower
            ):
                continue

        elif model_type == "unext":

            if "unext" not in lower:
                continue

        module = import_module_from_file(
            path
        )

        if module is None:
            continue

        for name in dir(module):

            obj = getattr(module, name)

            if not inspect.isclass(obj):
                continue

            if not issubclass(
                obj,
                nn.Module
            ):
                continue

            name_lower = name.lower()

            if desired_names:

                if (
                    name not in desired_names
                    and not (
                        model_type == "attention_unet"
                        and "attention" in name_lower
                        and "unet" in name_lower
                    )
                    and not (
                        model_type == "unext"
                        and "unext" in name_lower
                    )
                ):
                    continue

            model = instantiate_with_common_arguments(
                obj
            )

            if model is None:
                continue

            score, matched, shape_matched = (
                checkpoint_compatibility_score(
                    model,
                    state_dict
                )
            )

            candidates.append(
                (
                    score,
                    matched,
                    shape_matched,
                    path,
                    name,
                    model,
                )
            )

    if not candidates:

        raise RuntimeError(
            f"Could not automatically discover a "
            f"compatible {model_type} architecture.\n"
            f"Checkpoint:\n{checkpoint_path}\n\n"
            "The checkpoint exists, but the exact "
            "training architecture source is required "
            "to reconstruct it safely."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[2],
            x[1],
        ),
        reverse=True
    )

    best = candidates[0]

    score, matched, shape_matched, path, name, model = (
        best
    )

    print()
    print(
        f"Auto-discovered model class: {name}"
    )

    print(
        f"Source: {path}"
    )

    print(
        f"Compatible tensors: "
        f"{shape_matched}/{len(model.state_dict())} "
        f"({score * 100:.2f}%)"
    )

    if score < 0.90:

        raise RuntimeError(
            f"Best discovered architecture "
            f"matched only {score * 100:.2f}% "
            "of checkpoint tensors."
        )

    model_state = model.state_dict()

    compatible = {}

    for key, value in state_dict.items():

        if key not in model_state:
            continue

        if (
            tuple(value.shape)
            != tuple(model_state[key].shape)
        ):
            continue

        compatible[key] = value

    model.load_state_dict(
        compatible,
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    print(
        "Custom checkpoint loaded successfully."
    )

    return model


# =============================================================================
# TORCHVISION MODELS
# =============================================================================

def load_torchvision_model(
    checkpoint_path,
    model_type,
):

    try:
        import torchvision
        from torchvision.models.segmentation import (
            deeplabv3_resnet50,
        )
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn,
        )

    except Exception as exc:

        raise RuntimeError(
            "torchvision is required for "
            f"{model_type}.\n{exc}"
        )

    checkpoint = load_checkpoint_raw(
        checkpoint_path
    )

    state_dict = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    if model_type == "deeplabv3":
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=1,
        )

        # Match the grayscale architecture used during training.
        old_conv = model.backbone.conv1
        model.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        model.load_state_dict(state_dict, strict=True)

        print(
            "DeepLabV3 checkpoint loaded strictly; "
            f"input channels={model.backbone.conv1.in_channels}"
        )

    else:
        # The training checkpoint uses two classes: background and filament.
        # Load strictly so no parameters are silently left randomly initialized.
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            model_keys = set(model.state_dict())
            checkpoint_keys = set(state_dict)
            missing = sorted(model_keys - checkpoint_keys)
            unexpected = sorted(checkpoint_keys - model_keys)
            raise RuntimeError(
                "Mask R-CNN checkpoint did not strictly match the reconstructed "
                "two-class model. No predictions were generated.\n"
                f"Missing keys ({len(missing)}): {missing[:12]}\n"
                f"Unexpected keys ({len(unexpected)}): {unexpected[:12]}\n"
                f"Original error: {exc}"
            ) from exc

        print(
            f"Mask R-CNN checkpoint loaded strictly: "
            f"{len(state_dict)} tensors."
        )

    model.to(DEVICE)
    model.eval()

    print(
        f"{model_type} checkpoint loaded successfully."
    )

    return model


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(model_config):

    model_type = model_config["type"]

    checkpoint = model_config[
        "checkpoint"
    ]

    if model_type == "unetpp_improved":

        return load_unetpp_model(
            checkpoint,
            legacy=False,
        )

    if model_type == "unetpp_baseline":

        return load_unetpp_model(
            checkpoint,
            legacy=False,
        )

    if model_type in ["unetpp_spine", "unetpp_spine_only", "unetpp_spine_cldice"]:
        return load_unetpp_model(checkpoint, legacy=False)

    if model_type == "unetpp_spine_legacy":

        return load_unetpp_model(
            checkpoint,
            legacy=True,
        )

    if model_type in [
        "attention_unet",
        "unext",
    ]:

        return auto_discover_custom_model(
            checkpoint,
            model_type,
        )

    if model_type in [
        "maskrcnn",
        "deeplabv3",
    ]:

        return load_torchvision_model(
            checkpoint,
            model_type,
        )

    raise RuntimeError(
        f"Unknown model type: {model_type}"
    )


# =============================================================================
# FORWARD PASS
# =============================================================================

def model_prediction(
    model,
    image_tensor,
    model_type,
):

    # Torchvision Mask R-CNN expects a list of CxHxW image tensors.
    # The other models expect a batched NxCxHxW tensor.
    if model_type == "maskrcnn":
        output = model([image_tensor])
    else:
        output = model(image_tensor)

    # Mask R-CNN returns a list of detections.
    if model_type == "maskrcnn":

        if (
            not isinstance(output, list)
            or len(output) == 0
        ):
            return torch.zeros(
                (
                    1,
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                device=image_tensor.device,
            )

        detection = output[0]

        if (
            "masks" not in detection
            or detection["masks"].numel() == 0
        ):
            return torch.zeros(
                (
                    1,
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                device=image_tensor.device,
            )

        masks = detection["masks"]

        scores = detection.get(
            "scores",
            torch.ones(
                masks.shape[0],
                device=masks.device
            )
        )

        keep = scores >= THRESHOLD
        labels = detection.get("labels")

        if labels is not None:
            keep = keep & (labels > 0)



        if keep.sum() == 0:

            return torch.zeros(
                (
                    1,
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                device=image_tensor.device,
            )

        masks = masks[keep]

        combined = torch.max(
            masks,
            dim=0,
            keepdim=True
        ).values

        return combined

    tensor = extract_segmentation_tensor(
        output
    )

    # Normalize shape.
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    if tensor.ndim != 4:

        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{tuple(tensor.shape)}"
        )

    # If model outputs multiple channels,
    # use channel 1 for binary segmentation
    # when available.
    if tensor.shape[1] > 1:

        tensor = tensor[:, 1:2]

    if (
        tensor.shape[-2] != IMAGE_SIZE
        or tensor.shape[-1] != IMAGE_SIZE
    ):

        tensor = F.interpolate(
            tensor,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False,
        )

    return tensor


# =============================================================================
# PREDICTION DIRECTORY
# =============================================================================

def prediction_directory(
    model_config
):

    return (
        PREDICTIONS_ROOT
        / model_config["key"]
    )


def prediction_path(
    model_config,
    image_name,
):

    stem = Path(image_name).stem

    return (
        prediction_directory(
            model_config
        )
        / f"{stem}.png"
    )


# =============================================================================
# EXISTING PREDICTION CHECK
# =============================================================================

def missing_predictions(
    model_config,
    resolved_files,
):

    directory = prediction_directory(
        model_config
    )

    if not directory.exists():
        return list(resolved_files)

    missing = []

    for item in resolved_files:

        path = prediction_path(
            model_config,
            item["image_name"]
        )

        if not path.exists():
            missing.append(item)

    return missing


# =============================================================================
# PREDICTION GENERATION
# =============================================================================

@torch.no_grad()
def generate_predictions(
    model_config,
    resolved_files,
):

    section(
        "GENERATING PREDICTIONS: "
        + model_config["name"]
    )

    checkpoint = model_config[
        "checkpoint"
    ]

    print(
        f"Checkpoint:\n{checkpoint}"
    )

    directory = prediction_directory(
        model_config
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    missing = missing_predictions(
        model_config,
        resolved_files
    )

    if (
        not FORCE_REGENERATE
        and len(missing) == 0
    ):

        print(
            "All 142 predictions already exist."
        )

        return

    print(
        f"Predictions required: "
        f"{len(resolved_files)}"
    )

    print(
        f"Predictions missing: "
        f"{len(missing)}"
    )

    model = load_model(
        model_config
    )

    start = time.time()

    for index, item in enumerate(
        missing,
        1
    ):

        image_tensor = load_image(item["image"],model_config["type"])



        image_tensor = image_tensor.to(
            DEVICE
        )

        output = model_prediction(
            model,
            image_tensor,
            model_config["type"],
        )

        if model_config["type"] == "maskrcnn":
            # Mask R-CNN masks are already probabilities in [0, 1].
            probability = output
        else:
            # The other models return logits.
            probability = torch.sigmoid(output)


        binary = (
            probability >= THRESHOLD
        ).float()

        binary = (
            binary[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )

        out_path = prediction_path(
            model_config,
            item["image_name"]
        )

        cv2.imwrite(
            str(out_path),
            binary
        )

        if (
            index == 1
            or index % 20 == 0
            or index == len(missing)
        ):

            print(
                f"Generated "
                f"{index:3d}/{len(missing)}"
            )

    elapsed = time.time() - start

    print()
    print(
        f"Generated predictions: "
        f"{len(missing)}"
    )

    print(
        f"Prediction time: "
        f"{elapsed:.1f}s"
    )

    # Release GPU memory.
    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# METRICS
# =============================================================================

def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(
    tp,
    tn,
    fp,
    fn,
):

    dice = safe_divide(
        2.0 * tp,
        2.0 * tp + fp + fn
    )

    iou = safe_divide(
        tp,
        tp + fp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    denominator = math.sqrt(
        float(
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
        )
    )

    mcc = safe_divide(
        tp * tn - fp * fn,
        denominator
    )

    total = (
        tp + tn + fp + fn
    )

    po = safe_divide(
        tp + tn,
        total
    )

    actual_positive = tp + fn
    actual_negative = tn + fp

    predicted_positive = tp + fp
    predicted_negative = tn + fn

    pe = safe_divide(
        (
            actual_positive
            * predicted_positive
            +
            actual_negative
            * predicted_negative
        ),
        total * total
    )

    kappa = safe_divide(
        po - pe,
        1.0 - pe
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


# =============================================================================
# EVALUATE PREDICTION DIRECTORY
# =============================================================================

def evaluate_predictions(
    model_config,
    resolved_files,
):

    section(
        "EVALUATING: "
        + model_config["name"]
    )

    directory = prediction_directory(
        model_config
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Prediction directory does not exist:\n"
            f"{directory}"
        )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    missing = []

    processed = 0

    for index, item in enumerate(
        resolved_files,
        1
    ):

        pred_path = prediction_path(
            model_config,
            item["image_name"]
        )

        if not pred_path.exists():

            missing.append(
                item["image_name"]
            )

            continue

        prediction = cv2.imread(
            str(pred_path),
            cv2.IMREAD_GRAYSCALE
        )

        if prediction is None:

            missing.append(
                item["image_name"]
            )

            continue

        prediction = cv2.resize(
            prediction,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_NEAREST
        )

        prediction = prediction > 127

        ground_truth = load_mask(
            item["mask"]
        ) > 0

        tp += int(
            np.logical_and(
                prediction,
                ground_truth
            ).sum()
        )

        tn += int(
            np.logical_and(
                ~prediction,
                ~ground_truth
            ).sum()
        )

        fp += int(
            np.logical_and(
                prediction,
                ~ground_truth
            ).sum()
        )

        fn += int(
            np.logical_and(
                ~prediction,
                ground_truth
            ).sum()
        )

        processed += 1

        if (
            index == 1
            or index % 20 == 0
            or index == len(resolved_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(resolved_files)}"
            )

    if missing:

        raise FileNotFoundError(
            "\nPrediction directory is incomplete.\n"
            f"Model: {model_config['name']}\n"
            f"Directory: {directory}\n"
            f"Missing: {len(missing)}\n\n"
            "The prediction-generation stage should "
            "have created these files."
        )

    metrics = calculate_metrics(
        tp,
        tn,
        fp,
        fn
    )

    print()
    print(
        f"Processed images: "
        f"{processed}/{len(resolved_files)}"
    )

    print()
    print(
        f"Dice:          {metrics['Dice']:.6f}"
    )

    print(
        f"IoU:           {metrics['IoU']:.6f}"
    )

    print(
        f"Precision:     {metrics['Precision']:.6f}"
    )

    print(
        f"Recall:        {metrics['Recall']:.6f}"
    )

    print(
        f"MCC:           {metrics['MCC']:.6f}"
    )

    print(
        f"Cohen Kappa:   "
        f"{metrics['Cohen_Kappa']:.6f}"
    )

    print(
        f"TP: {metrics['TP']:,} | "
        f"TN: {metrics['TN']:,} | "
        f"FP: {metrics['FP']:,} | "
        f"FN: {metrics['FN']:,}"
    )

    return metrics


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "unified_7_model_results.csv"
    )

    fields = [
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

    with open(
        csv_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            ",".join(fields)
            + "\n"
        )

        for result in results:

            row = [
                result.get("Model", ""),
                f"{result.get('Dice', 0):.6f}"
                if result.get("Dice") is not None
                else "",
                f"{result.get('IoU', 0):.6f}"
                if result.get("IoU") is not None
                else "",
                f"{result.get('Precision', 0):.6f}"
                if result.get("Precision") is not None
                else "",
                f"{result.get('Recall', 0):.6f}"
                if result.get("Recall") is not None
                else "",
                f"{result.get('MCC', 0):.6f}"
                if result.get("MCC") is not None
                else "",
                f"{result.get('Cohen_Kappa', 0):.6f}"
                if result.get("Cohen_Kappa") is not None
                else "",
                str(result.get("TP", "")),
                str(result.get("TN", "")),
                str(result.get("FP", "")),
                str(result.get("FN", "")),
                result.get("Status", ""),
            ]

            f.write(
                ",".join(row)
                + "\n"
            )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "unified_7_model_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # -------------------------------------------------------------------------
    # TXT
    # -------------------------------------------------------------------------

    txt_path = (
        OUTPUT_DIR
        / "unified_7_model_results.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "UNIFIED 7-MODEL VALIDATION EVALUATION\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        for result in results:

            f.write(
                result["Model"]
                + "\n"
            )

            f.write(
                "-" * 60
                + "\n"
            )

            if result.get("Status") == "SUCCESS":

                f.write(
                    f"Dice:        "
                    f"{result['Dice']:.6f}\n"
                )

                f.write(
                    f"IoU:         "
                    f"{result['IoU']:.6f}\n"
                )

                f.write(
                    f"Precision:   "
                    f"{result['Precision']:.6f}\n"
                )

                f.write(
                    f"Recall:      "
                    f"{result['Recall']:.6f}\n"
                )

                f.write(
                    f"MCC:         "
                    f"{result['MCC']:.6f}\n"
                )

                f.write(
                    f"Cohen Kappa: "
                    f"{result['Cohen_Kappa']:.6f}\n"
                )

                f.write(
                    f"TP: {result['TP']:,}\n"
                )

                f.write(
                    f"TN: {result['TN']:,}\n"
                )

                f.write(
                    f"FP: {result['FP']:,}\n"
                )

                f.write(
                    f"FN: {result['FN']:,}\n"
                )

            else:

                f.write(
                    "STATUS: FAILED\n"
                )

                f.write(
                    result.get(
                        "Error",
                        "Unknown error"
                    )
                    + "\n"
                )

            f.write("\n")

    print()
    print(
        f"CSV saved : {csv_path}"
    )

    print(
        f"JSON saved: {json_path}"
    )

    print(
        f"TXT saved : {txt_path}"
    )


# =============================================================================
# FINAL TABLE
# =============================================================================

def print_final_table(results):

    banner(
        "FINAL UNIFIED 7-MODEL RESULTS"
    )

    header = (
        f"{'Model':42s}"
        f"{'Dice':>10s}"
        f"{'IoU':>10s}"
        f"{'Prec':>10s}"
        f"{'Recall':>10s}"
        f"{'MCC':>10s}"
        f"{'Kappa':>10s}"
        f"{'Status':>12s}"
    )

    print(header)
    print("-" * len(header))

    for result in results:

        name = result["Model"]

        if len(name) > 40:
            name = name[:40]

        status = result.get(
            "Status",
            "FAILED"
        )

        if status == "SUCCESS":

            print(
                f"{name:42s}"
                f"{result['Dice']:10.6f}"
                f"{result['IoU']:10.6f}"
                f"{result['Precision']:10.6f}"
                f"{result['Recall']:10.6f}"
                f"{result['MCC']:10.6f}"
                f"{result['Cohen_Kappa']:10.6f}"
                f"{status:>12s}"
            )

        else:

            print(
                f"{name:42s}"
                f"{'-':>10s}"
                f"{'-':>10s}"
                f"{'-':>10s}"
                f"{'-':>10s}"
                f"{'-':>10s}"
                f"{'-':>10s}"
                f"{status:>12s}"
            )

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "UNIFIED 7-MODEL VALIDATION EVALUATION"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Image dir    : {IMAGE_DIR}"
    )

    print(
        f"Mask dir     : {MASK_DIR}"
    )

    print(
        f"Split        : {SPLIT_FILE}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Image size   : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Threshold    : {THRESHOLD}"
    )

    print_device()

    # -------------------------------------------------------------------------
    # Validation split
    # -------------------------------------------------------------------------

    records = load_validation_records()

    resolved_files = (
        resolve_validation_files(
            records
        )
    )

    results = []

    # -------------------------------------------------------------------------
    # Seven models
    # -------------------------------------------------------------------------

    for model_config in MODELS:

        banner(
            "MODEL: "
            + model_config["name"]
        )

        result = {
            "Model": model_config["name"],
            "Status": "FAILED",
        }

        try:

            checkpoint = model_config[
                "checkpoint"
            ]

            if not checkpoint.exists():

                raise FileNotFoundError(
                    f"Checkpoint missing:\n"
                    f"{checkpoint}"
                )

            # -------------------------------------------------------------
            # Generate missing predictions.
            # -------------------------------------------------------------

            generate_predictions(
                model_config,
                resolved_files
            )

            # -------------------------------------------------------------
            # Evaluate.
            # -------------------------------------------------------------

            metrics = evaluate_predictions(
                model_config,
                resolved_files
            )

            result.update(
                metrics
            )

            result["Status"] = (
                "SUCCESS"
            )

        except Exception as exc:

            print()
            print(
                "!!! MODEL FAILED !!!"
            )

            print(
                f"Model: "
                f"{model_config['name']}"
            )

            print(
                f"Error:\n{exc}"
            )

            result["Error"] = (
                str(exc)
            )

            if not CONTINUE_ON_MODEL_ERROR:

                raise

            print()
            print(
                "Continuing to next model..."
            )

        results.append(result)

        # Free GPU memory.
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    save_results(
        results
    )

    print_final_table(
        results
    )

    banner(
        "EVALUATION COMPLETE"
    )

    successful = sum(
        r["Status"] == "SUCCESS"
        for r in results
    )

    print(
        f"Successful models: "
        f"{successful}/7"
    )

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()


