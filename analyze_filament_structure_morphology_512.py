"""
======================================================================
ANALYZE FILAMENT STRUCTURE + MORPHOLOGY
======================================================================

Model:
    UNet++ + Spine + clDice improved

Checkpoint:
    checkpoints_512_cldice_improved/
        best_unetpp_spine_512_cldice_improved.pth

Input:
    MAGFiLO 2048x2048 images

Processing:
    2048x2048 -> 512x512
    model segmentation
    probability thresholding
    morphological cleanup
    skeletonization
    connected-component analysis

Structural measurements:
    - filament area
    - filament length
    - skeleton length
    - width statistics
    - thickness statistics
    - number of components
    - endpoint count
    - junction count
    - branch count
    - branch density
    - tortuosity
    - orientation
    - density
    - coverage
    - largest component fraction

Outputs:
    analysis_512/
        morphology_metrics_512.csv
        morphology_summary_512.json
        masks/
        skeletons/
        overlays/
        visualizations/

======================================================================
"""

import os
import json
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from unetpp_spine import UNetPlusPlusSpine


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "MAGFiLO_1.0_Kaggle_2026"
    / "train"
    / "train_images"
)

MASK_DIR = PROJECT_ROOT / "data" / "full_masks"

SPINE_DIR = PROJECT_ROOT / "data" / "full_spine_masks"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints_512_cldice_improved"
    / "best_unetpp_spine_512_cldice_improved.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "analysis_512"

MASK_OUTPUT_DIR = OUTPUT_DIR / "masks"
SKELETON_OUTPUT_DIR = OUTPUT_DIR / "skeletons"
OVERLAY_OUTPUT_DIR = OUTPUT_DIR / "overlays"
VISUAL_OUTPUT_DIR = OUTPUT_DIR / "visualizations"

IMAGE_SIZE = 512

# Model threshold.
# Keep 0.50 initially so the morphology analysis is directly tied
# to the segmentation benchmark.
THRESHOLD = 0.50

# Small-object removal.
MIN_COMPONENT_PIXELS = 10

# Morphological cleanup.
MORPH_KERNEL_SIZE = 3

# Physical scale.
#
# IMPORTANT:
# The source images are 2048x2048.
# If you know the physical pixel scale, put it here.
#
# Example:
# PIXEL_SIZE = 1.0
# means one source pixel = one physical unit.
#
# Leave as None if physical calibration is not yet known.
PIXEL_SIZE = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# DIRECTORY SETUP
# =====================================================================

for directory in [
    OUTPUT_DIR,
    MASK_OUTPUT_DIR,
    SKELETON_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR,
    VISUAL_OUTPUT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# =====================================================================
# MODEL OUTPUT HANDLING
# =====================================================================

def get_filament_output(output):
    """
    Extract filament segmentation output from UNet++ + Spine.

    The training architecture may return:
        tuple
        list
        dict

    This function handles the common formats.
    """

    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise RuntimeError(
                "Model returned an empty tuple/list."
            )

        return output[0]

    if isinstance(output, dict):

        possible_keys = [
            "filament",
            "filament_logits",
            "mask",
            "out",
            "output",
            "segmentation",
        ]

        for key in possible_keys:
            if key in output:
                return output[key]

        # Fall back to first tensor.
        for value in output.values():
            if torch.is_tensor(value):
                return value

    raise RuntimeError(
        f"Unsupported model output type: {type(output)}"
    )


# =====================================================================
# LOAD MODEL
# =====================================================================

def load_model():

    print("=" * 70)
    print("LOADING BEST UNET++ + SPINE + clDICE MODEL")
    print("=" * 70)

    print(
        f"Checkpoint:\n{CHECKPOINT}"
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    model = UNetPlusPlusSpine(
        in_channels=1,
        out_channels=1
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint Dice  : "
        f"{checkpoint.get('best_dice', 'unknown')}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    if DEVICE.type == "cuda":
        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)

    return model


# =====================================================================
# IMAGE LOADING
# =====================================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    original_height, original_width = image.shape

    image_512 = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image_float = (
        image_512.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        image_float
    ).unsqueeze(0).unsqueeze(0)

    return (
        image,
        image_512,
        tensor,
        original_width,
        original_height
    )


# =====================================================================
# MODEL PREDICTION
# =====================================================================

@torch.no_grad()
def predict(model, image_tensor):

    image_tensor = image_tensor.to(
        DEVICE,
        non_blocking=True
    )

    output = model(image_tensor)

    filament_logits = get_filament_output(
        output
    )

    if filament_logits.shape[-2:] != (
        IMAGE_SIZE,
        IMAGE_SIZE
    ):
        filament_logits = F.interpolate(
            filament_logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

    probability = torch.sigmoid(
        filament_logits
    )

    probability = probability[0, 0].cpu().numpy()

    mask = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    return probability, mask


# =====================================================================
# MORPHOLOGICAL CLEANUP
# =====================================================================

def clean_mask(mask):

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    # Remove tiny noise.
    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill very small gaps.
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    binary = (
        closed > 0
    ).astype(np.uint8)

    # Remove tiny connected components.
    labels, num_labels = cv2.connectedComponents(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(
        binary,
        dtype=np.uint8
    )

    for component_id in range(
        1,
        num_labels
    ):

        component_pixels = np.sum(
            labels == component_id
        )

        if component_pixels >= MIN_COMPONENT_PIXELS:
            cleaned[
                labels == component_id
            ] = 1

    return cleaned


# =====================================================================
# SKELETON
# =====================================================================

def make_skeleton(mask):

    skeleton = skeletonize(
        mask.astype(bool)
    )

    return skeleton.astype(
        np.uint8
    )


# =====================================================================
# CONNECTED COMPONENT ANALYSIS
# =====================================================================

def connected_component_metrics(mask):

    labeled = label(
        mask,
        connectivity=2
    )

    regions = regionprops(
        labeled
    )

    component_count = len(regions)

    if component_count == 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction": 0.0,
            "mean_component_area": 0.0,
            "median_component_area": 0.0,
            "max_component_area": 0.0,
        }

    areas = np.array(
        [
            region.area
            for region in regions
        ],
        dtype=np.float64
    )

    return {
        "component_count":
            int(component_count),

        "largest_component_pixels":
            int(areas.max()),

        "largest_component_fraction":
            float(
                areas.max()
                / max(areas.sum(), 1.0)
            ),

        "mean_component_area":
            float(areas.mean()),

        "median_component_area":
            float(np.median(areas)),

        "max_component_area":
            float(areas.max()),
    }


# =====================================================================
# ENDPOINT / JUNCTION ANALYSIS
# =====================================================================

def skeleton_node_metrics(skeleton):

    ys, xs = np.where(
        skeleton > 0
    )

    if len(xs) == 0:
        return {
            "skeleton_pixels": 0,
            "endpoint_count": 0,
            "junction_count": 0,
            "branch_count": 0,
        }

    padded = np.pad(
        skeleton,
        1,
        mode="constant"
    )

    neighbor_kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    neighbor_count = cv2.filter2D(
        padded,
        -1,
        neighbor_kernel
    )

    # filter2D count includes the center pixel.
    neighbor_count = (
        neighbor_count[1:-1, 1:-1]
        - skeleton
    )

    endpoint_mask = (
        (skeleton == 1)
        &
        (neighbor_count == 1)
    )

    junction_mask = (
        (skeleton == 1)
        &
        (neighbor_count >= 3)
    )

    endpoint_count = int(
        endpoint_mask.sum()
    )

    junction_count = int(
        junction_mask.sum()
    )

    # A practical branch-count estimate.
    branch_count = int(
        max(
            endpoint_count
            + junction_count,
            0
        )
    )

    return {
        "skeleton_pixels":
            int(skeleton.sum()),

        "endpoint_count":
            endpoint_count,

        "junction_count":
            junction_count,

        "branch_count":
            branch_count,
    }


# =====================================================================
# ORIENTATION ANALYSIS
# =====================================================================

def orientation_metrics(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 2:
        return {
            "orientation_mean_degrees": 0.0,
            "orientation_std_degrees": 0.0,
        }

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    x -= x.mean()
    y -= y.mean()

    covariance = np.cov(
        np.vstack([x, y])
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    principal_vector = (
        eigenvectors[:, np.argmax(eigenvalues)]
    )

    angle = math.degrees(
        math.atan2(
            principal_vector[1],
            principal_vector[0]
        )
    )

    # Normalize to [-90, 90].
    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return {
        "orientation_mean_degrees":
            float(angle),

        "orientation_std_degrees":
            0.0,
    }


# =====================================================================
# WIDTH / THICKNESS
# =====================================================================

def thickness_metrics(mask, skeleton):

    area_pixels = int(
        mask.sum()
    )

    skeleton_pixels = int(
        skeleton.sum()
    )

    if skeleton_pixels == 0:
        return {
            "mean_width_pixels": 0.0,
            "median_width_pixels": 0.0,
            "std_width_pixels": 0.0,
            "min_width_pixels": 0.0,
            "max_width_pixels": 0.0,
        }

    distance = cv2.distanceTransform(
        mask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    radii = distance[
        skeleton > 0
    ]

    widths = (
        2.0 * radii
    )

    widths = widths[
        widths > 0
    ]

    if len(widths) == 0:
        return {
            "mean_width_pixels": 0.0,
            "median_width_pixels": 0.0,
            "std_width_pixels": 0.0,
            "min_width_pixels": 0.0,
            "max_width_pixels": 0.0,
        }

    return {
        "mean_width_pixels":
            float(np.mean(widths)),

        "median_width_pixels":
            float(np.median(widths)),

        "std_width_pixels":
            float(np.std(widths)),

        "min_width_pixels":
            float(np.min(widths)),

        "max_width_pixels":
            float(np.max(widths)),
    }


# =====================================================================
# TORTUOSITY
# =====================================================================

def tortuosity_metrics(skeleton):

    ys, xs = np.where(
        skeleton > 0
    )

    if len(xs) < 2:
        return {
            "skeleton_euclidean_span_pixels": 0.0,
            "tortuosity": 0.0,
        }

    points = np.column_stack(
        [
            xs.astype(np.float64),
            ys.astype(np.float64)
        ]
    )

    # Approximate skeleton path length using
    # 8-connected neighboring skeleton pixels.
    path_length = 0.0

    skeleton_bool = skeleton.astype(bool)

    horizontal_vertical = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    diagonal = [
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    visited_edges = set()

    height, width = skeleton.shape

    for y, x in zip(ys, xs):

        for dy, dx in (
            horizontal_vertical
            + diagonal
        ):

            ny = int(y + dy)
            nx = int(x + dx)

            if (
                ny < 0
                or ny >= height
                or nx < 0
                or nx >= width
            ):
                continue

            if not skeleton_bool[ny, nx]:
                continue

            edge_a = (
                int(y),
                int(x)
            )

            edge_b = (
                ny,
                nx
            )

            edge = tuple(
                sorted(
                    [edge_a, edge_b]
                )
            )

            if edge in visited_edges:
                continue

            visited_edges.add(edge)

            if abs(dx) == 1 and abs(dy) == 1:
                path_length += math.sqrt(2.0)
            else:
                path_length += 1.0

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    euclidean_span = math.sqrt(
        (max_x - min_x) ** 2
        +
        (max_y - min_y) ** 2
    )

    if euclidean_span > 0:
        tortuosity = (
            path_length
            / euclidean_span
        )
    else:
        tortuosity = 0.0

    return {
        "skeleton_path_length_pixels":
            float(path_length),

        "skeleton_euclidean_span_pixels":
            float(euclidean_span),

        "tortuosity":
            float(tortuosity),
    }


# =====================================================================
# DENSITY / COVERAGE
# =====================================================================

def density_metrics(mask, skeleton):

    total_pixels = (
        mask.shape[0]
        * mask.shape[1]
    )

    filament_pixels = int(
        mask.sum()
    )

    skeleton_pixels = int(
        skeleton.sum()
    )

    coverage = (
        filament_pixels
        / max(total_pixels, 1)
    )

    skeleton_density = (
        skeleton_pixels
        / max(total_pixels, 1)
    )

    return {
        "filament_area_pixels":
            filament_pixels,

        "image_area_pixels":
            int(total_pixels),

        "filament_coverage":
            float(coverage),

        "skeleton_density":
            float(skeleton_density),
    }


# =====================================================================
# PHYSICAL UNITS
# =====================================================================

def physical_metrics(metrics):

    if PIXEL_SIZE is None:
        metrics[
            "filament_area_physical"
        ] = None

        metrics[
            "skeleton_length_physical"
        ] = None

        metrics[
            "mean_width_physical"
        ] = None

        metrics[
            "median_width_physical"
        ] = None

        return metrics

    scale = float(
        PIXEL_SIZE
    )

    metrics[
        "filament_area_physical"
    ] = (
        metrics["filament_area_pixels"]
        * scale
        * scale
    )

    metrics[
        "skeleton_length_physical"
    ] = (
        metrics[
            "skeleton_path_length_pixels"
        ]
        * scale
    )

    metrics[
        "mean_width_physical"
    ] = (
        metrics[
            "mean_width_pixels"
        ]
        * scale
    )

    metrics[
        "median_width_physical"
    ] = (
        metrics[
            "median_width_pixels"
        ]
        * scale
    )

    return metrics


# =====================================================================
# IMAGE SAVING
# =====================================================================

def save_mask(
    mask,
    path
):

    cv2.imwrite(
        str(path),
        (
            mask.astype(np.uint8)
            * 255
        )
    )


def save_skeleton(
    skeleton,
    path
):

    cv2.imwrite(
        str(path),
        (
            skeleton.astype(np.uint8)
            * 255
        )
    )


def save_overlay(
    image,
    mask,
    skeleton,
    path
):

    if image.ndim == 2:
        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )
    else:
        overlay = image.copy()

    # Filament pixels.
    filament = mask > 0

    # Skeleton pixels.
    spine = skeleton > 0

    # Overlay filament in red.
    overlay[filament] = (
        0,
        0,
        255
    )

    # Skeleton in green.
    overlay[spine] = (
        0,
        255,
        0
    )

    cv2.imwrite(
        str(path),
        overlay
    )


# =====================================================================
# SINGLE IMAGE ANALYSIS
# =====================================================================

def analyze_image(
    model,
    image_path,
    index,
    total
):

    print(
        f"[{index:4d}/{total:4d}] "
        f"{image_path.name}"
    )

    (
        original,
        image_512,
        tensor,
        original_width,
        original_height
    ) = load_image(
        image_path
    )

    probability, raw_mask = predict(
        model,
        tensor
    )

    mask = clean_mask(
        raw_mask
    )

    skeleton = make_skeleton(
        mask
    )

    component_metrics = (
        connected_component_metrics(
            mask
        )
    )

    node_metrics = (
        skeleton_node_metrics(
            skeleton
        )
    )

    orientation = (
        orientation_metrics(
            mask
        )
    )

    thickness = (
        thickness_metrics(
            mask,
            skeleton
        )
    )

    tortuosity = (
        tortuosity_metrics(
            skeleton
        )
    )

    density = (
        density_metrics(
            mask,
            skeleton
        )
    )

    metrics = {
        "image_name":
            image_path.name,

        "image_stem":
            image_path.stem,

        "original_width":
            int(original_width),

        "original_height":
            int(original_height),

        "analysis_width":
            IMAGE_SIZE,

        "analysis_height":
            IMAGE_SIZE,

        "threshold":
            THRESHOLD,
    }

    metrics.update(
        component_metrics
    )

    metrics.update(
        node_metrics
    )

    metrics.update(
        orientation
    )

    metrics.update(
        thickness
    )

    metrics.update(
        tortuosity
    )

    metrics.update(
        density
    )

    metrics = physical_metrics(
        metrics
    )

    stem = image_path.stem

    save_mask(
        mask,
        MASK_OUTPUT_DIR
        / f"{stem}_mask.png"
    )

    save_skeleton(
        skeleton,
        SKELETON_OUTPUT_DIR
        / f"{stem}_skeleton.png"
    )

    save_overlay(
        image_512,
        mask,
        skeleton,
        OVERLAY_OUTPUT_DIR
        / f"{stem}_overlay.png"
    )

    # Save probability map.
    probability_image = (
        np.clip(
            probability,
            0.0,
            1.0
        )
        * 255.0
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            VISUAL_OUTPUT_DIR
            / f"{stem}_probability.png"
        ),
        probability_image
    )

    return metrics


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def make_summary(df):

    numeric_columns = [
        column
        for column in df.columns
        if (
            pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    summary = {
        "number_of_images":
            int(len(df)),

        "threshold":
            THRESHOLD,

        "pixel_size":
            PIXEL_SIZE,

        "metrics": {}
    }

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        summary["metrics"][column] = {
            "mean":
                float(values.mean()),

            "std":
                float(values.std()),

            "median":
                float(values.median()),

            "minimum":
                float(values.min()),

            "maximum":
                float(values.max()),
        }

    return summary


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 70)
    print("FILAMENT STRUCTURAL + MORPHOLOGICAL ANALYSIS")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Image dir    : {IMAGE_DIR}"
    )

    print(
        f"Checkpoint   : {CHECKPOINT}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Threshold    : {THRESHOLD}"
    )

    print("=" * 70)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    image_paths = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.suffix.lower()
            in [
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp"
            ]
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            "No images found."
        )

    print(
        f"Images found: {len(image_paths)}"
    )

    model = load_model()

    results = []

    print()
    print("=" * 70)
    print("RUNNING STRUCTURAL ANALYSIS")
    print("=" * 70)

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            metrics = analyze_image(
                model,
                image_path,
                index,
                len(image_paths)
            )

            results.append(
                metrics
            )

        except Exception as exc:

            warnings.warn(
                f"Failed on "
                f"{image_path.name}: "
                f"{exc}"
            )

    if len(results) == 0:
        raise RuntimeError(
            "No images were successfully analyzed."
        )

    df = pd.DataFrame(
        results
    )

    # Sort by image name.
    df = df.sort_values(
        "image_name"
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "morphology_metrics_512.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # -------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------

    summary = make_summary(
        df
    )

    summary_path = (
        OUTPUT_DIR
        / "morphology_summary_512.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # -------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STRUCTURAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Images analyzed       : "
        f"{len(df)}"
    )

    print(
        f"Mean filament area    : "
        f"{df['filament_area_pixels'].mean():.2f} px"
    )

    print(
        f"Mean skeleton length  : "
        f"{df['skeleton_path_length_pixels'].mean():.2f} px"
    )

    print(
        f"Mean width            : "
        f"{df['mean_width_pixels'].mean():.2f} px"
    )

    print(
        f"Mean components       : "
        f"{df['component_count'].mean():.2f}"
    )

    print(
        f"Mean endpoints        : "
        f"{df['endpoint_count'].mean():.2f}"
    )

    print(
        f"Mean junctions        : "
        f"{df['junction_count'].mean():.2f}"
    )

    print(
        f"Mean tortuosity       : "
        f"{df['tortuosity'].mean():.4f}"
    )

    print(
        f"Mean filament coverage: "
        f"{df['filament_coverage'].mean():.6f}"
    )

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("Summary:")
    print(summary_path)

    print()
    print("Masks:")
    print(MASK_OUTPUT_DIR)

    print()
    print("Skeletons:")
    print(SKELETON_OUTPUT_DIR)

    print()
    print("Overlays:")
    print(OVERLAY_OUTPUT_DIR)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()