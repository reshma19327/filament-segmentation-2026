# ==============================================================
# evaluate_unetpp_spine.py
#
# Evaluate the BEST UNet++ + Spine checkpoint
#
# Uses:
#   - Same 707-image dataset
#   - Same deterministic seed = 42
#   - Same 80/20 split = 565 train / 142 validation
#   - Same image size = 256
#   - Same model architecture
#
# Outputs:
#   - Dice
#   - IoU
#   - Precision
#   - Recall
#   - Accuracy
#   - MCC
#   - Cohen's Kappa
#   - TP / TN / FP / FN
#   - Spine Dice / IoU
#   - Inference time
#   - Qualitative prediction images
#   - evaluation_metrics.txt
# ==============================================================

import os
import time
import random

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from dataset_unetpp_spine import UNetPPSpineDataset
from unetpp_spine import UNetPPSpine


# ==============================================================
# CONFIGURATION
# ==============================================================

IMAGE_DIR = r".\MAGFiLO_1.0_Kaggle_2026\train\train_images"

MASK_DIR = r".\data\full_masks"

SPINE_DIR = r".\data\full_spine_masks"

CHECKPOINT = r".\checkpoints\best_unetpp_spine.pth"

OUTPUT_DIR = r".\evaluation_results"

IMAGE_SIZE = 256

BATCH_SIZE = 8

SEED = 42

THRESHOLD = 0.5

NUM_VISUALIZATIONS = 12


# ==============================================================
# REPRODUCIBILITY
# ==============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==============================================================
# METRICS FROM CONFUSION MATRIX
# ==============================================================

def metrics_from_counts(tp, tn, fp, fn):

    eps = 1e-8

    # ----------------------------------------------------------
    # Dice
    # ----------------------------------------------------------

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + eps)
    )

    # ----------------------------------------------------------
    # IoU
    # ----------------------------------------------------------

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    # ----------------------------------------------------------
    # Precision
    # ----------------------------------------------------------

    precision = (
        tp
        / (tp + fp + eps)
    )

    # ----------------------------------------------------------
    # Recall
    # ----------------------------------------------------------

    recall = (
        tp
        / (tp + fn + eps)
    )

    # ----------------------------------------------------------
    # Accuracy
    # ----------------------------------------------------------

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn)
        / (total + eps)
    )

    # ----------------------------------------------------------
    # Matthews Correlation Coefficient
    # ----------------------------------------------------------

    denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator > 0:

        mcc = (
            (tp * tn - fp * fn)
            / (denominator ** 0.5)
        )

    else:

        mcc = 0.0

    # ----------------------------------------------------------
    # Cohen's Kappa
    # ----------------------------------------------------------

    if total > 0:

        observed_agreement = (
            (tp + tn)
            / total
        )

        actual_positive = tp + fn
        actual_negative = tn + fp

        predicted_positive = tp + fp
        predicted_negative = tn + fn

        expected_agreement = (
            (
                actual_positive
                * predicted_positive
            )
            +
            (
                actual_negative
                * predicted_negative
            )
        ) / (total * total)

        if abs(1.0 - expected_agreement) > eps:

            kappa = (
                observed_agreement
                - expected_agreement
            ) / (
                1.0
                - expected_agreement
            )

        else:

            kappa = 0.0

    else:

        kappa = 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ==============================================================
# FIND VALID IMAGES
# ==============================================================

def get_image_names():

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_names = []

    if not os.path.isdir(IMAGE_DIR):

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    for filename in sorted(
        os.listdir(IMAGE_DIR)
    ):

        extension = (
            os.path.splitext(filename)[1]
            .lower()
        )

        if extension not in valid_extensions:
            continue

        stem = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            MASK_DIR,
            stem + ".png"
        )

        spine_path = os.path.join(
            SPINE_DIR,
            stem + ".png"
        )

        if (
            os.path.isfile(mask_path)
            and os.path.isfile(spine_path)
        ):

            image_names.append(filename)

    return image_names


# ==============================================================
# SAVE QUALITATIVE VISUALIZATION
# ==============================================================

def save_visualization(
    image,
    gt_mask,
    pred_mask,
    gt_spine,
    pred_spine,
    image_name,
    output_dir,
):

    # ----------------------------------------------------------
    # Convert image
    # ----------------------------------------------------------

    image = np.asarray(image)

    if image.ndim == 3:

        image = image[0]

    image = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # ----------------------------------------------------------
    # Convert masks
    # ----------------------------------------------------------

    gt_mask = (
        np.asarray(gt_mask) > 0.5
    ).astype(np.uint8) * 255

    pred_mask = (
        np.asarray(pred_mask) > 0.5
    ).astype(np.uint8) * 255

    gt_spine = (
        np.asarray(gt_spine) > 0.5
    ).astype(np.uint8) * 255

    pred_spine = (
        np.asarray(pred_spine) > 0.5
    ).astype(np.uint8) * 255

    # ----------------------------------------------------------
    # Convert grayscale to BGR
    # ----------------------------------------------------------

    original = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    gt_mask_color = cv2.cvtColor(
        gt_mask,
        cv2.COLOR_GRAY2BGR,
    )

    pred_mask_color = cv2.cvtColor(
        pred_mask,
        cv2.COLOR_GRAY2BGR,
    )

    gt_spine_color = cv2.cvtColor(
        gt_spine,
        cv2.COLOR_GRAY2BGR,
    )

    pred_spine_color = cv2.cvtColor(
        pred_spine,
        cv2.COLOR_GRAY2BGR,
    )

    # ----------------------------------------------------------
    # Overlay
    #
    # Green = GT filament
    # Red   = predicted filament
    # Blue  = GT spine
    # ----------------------------------------------------------

    overlay = original.copy()

    overlay[gt_mask > 0] = (
        0,
        255,
        0,
    )

    overlay[pred_mask > 0] = (
        0,
        0,
        255,
    )

    overlay[gt_spine > 0] = (
        255,
        0,
        0,
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        overlay,
        0.45,
        0,
    )

    # ----------------------------------------------------------
    # Panels
    # ----------------------------------------------------------

    panels = [
        ("Original", original),
        ("GT Filament", gt_mask_color),
        ("Pred Filament", pred_mask_color),
        ("GT Spine", gt_spine_color),
        ("Pred Spine", pred_spine_color),
        ("Overlay", overlay),
    ]

    labeled_panels = []

    for title, panel in panels:

        panel = panel.copy()

        cv2.rectangle(
            panel,
            (0, 0),
            (255, 28),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            panel,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        labeled_panels.append(panel)

    # ----------------------------------------------------------
    # 2 x 3 grid
    # ----------------------------------------------------------

    row1 = np.hstack(
        labeled_panels[0:3]
    )

    row2 = np.hstack(
        labeled_panels[3:6]
    )

    final_image = np.vstack(
        [
            row1,
            row2,
        ]
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    safe_name = os.path.splitext(
        image_name
    )[0]

    output_path = os.path.join(
        output_dir,
        safe_name + "_evaluation.png",
    )

    cv2.imwrite(
        output_path,
        final_image,
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("UNET++ + SPINE CHECKPOINT EVALUATION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------

    set_seed(SEED)

    # ----------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # ----------------------------------------------------------
    # Check required paths
    # ----------------------------------------------------------

    required_paths = [
        IMAGE_DIR,
        MASK_DIR,
        SPINE_DIR,
        CHECKPOINT,
    ]

    for path in required_paths:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired path not found:\n{path}"
            )

    # ----------------------------------------------------------
    # Find images
    # ----------------------------------------------------------

    image_names = get_image_names()

    print()
    print(
        "Usable images:",
        len(image_names),
    )

    if len(image_names) != 707:

        print(
            "WARNING:"
            " Expected 707 usable images."
        )

    # ----------------------------------------------------------
    # Recreate EXACT training split
    #
    # Important:
    # The training script used:
    #
    #   random seed = 42
    #   80% train
    #   20% validation
    #
    # ----------------------------------------------------------

    split_rng = random.Random(SEED)

    split_rng.shuffle(
        image_names
    )

    split_index = int(
        0.8 * len(image_names)
    )

    train_names = image_names[
        :split_index
    ]

    val_names = image_names[
        split_index:
    ]

    print(
        "Training images:",
        len(train_names),
    )

    print(
        "Validation images:",
        len(val_names),
    )

    # ----------------------------------------------------------
    # Validation dataset
    # ----------------------------------------------------------

    val_dataset = UNetPPSpineDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        spine_dir=SPINE_DIR,
        image_names=val_names,
        image_size=IMAGE_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # ----------------------------------------------------------
    # Create model
    #
    # IMPORTANT:
    # Your actual UNetPPSpine constructor
    # accepts in_channels but NOT out_channels.
    # ----------------------------------------------------------

    model = UNetPPSpine(
        in_channels=1
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    # ----------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    # ----------------------------------------------------------
    # Support different checkpoint formats
    # ----------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            # The checkpoint itself may
            # already be the state dict.
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ----------------------------------------------------------
    # Remove "module." prefix if checkpoint
    # was saved using DataParallel.
    # ----------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            new_key = key[
                len("module.") :
            ]

        else:

            new_key = key

        cleaned_state_dict[
            new_key
        ] = value

    # ----------------------------------------------------------
    # Load model weights
    # ----------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    print(
        "Checkpoint loaded successfully."
    )

    # ----------------------------------------------------------
    # Evaluation mode
    # ----------------------------------------------------------

    model.eval()

    # ==========================================================
    # CONFUSION MATRIX ACCUMULATORS
    # ==========================================================

    mask_tp = 0
    mask_tn = 0
    mask_fp = 0
    mask_fn = 0

    spine_tp = 0
    spine_tn = 0
    spine_fp = 0
    spine_fn = 0

    # ==========================================================
    # INFERENCE TIME
    # ==========================================================

    total_inference_time = 0.0

    total_images = 0

    # ==========================================================
    # VISUALIZATION COUNT
    # ==========================================================

    saved_visualizations = 0

    # ==========================================================
    # EVALUATION
    # ==========================================================

    print()
    print("Evaluating...")
    print()

    with torch.no_grad():

        for batch_index, batch in enumerate(
            val_loader
        ):

            # --------------------------------------------------
            # Dataset returns:
            #
            # image
            # mask
            # spine
            # image_name
            # --------------------------------------------------

            images = batch[0]

            masks = batch[1]

            spines = batch[2]

            image_names_batch = batch[3]

            # --------------------------------------------------
            # Move tensors to GPU
            # --------------------------------------------------

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            spines = spines.to(
                device,
                non_blocking=True,
            )

            # --------------------------------------------------
            # GPU synchronization before timing
            # --------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            mask_logits, spine_logits = model(
                images
            )

            # --------------------------------------------------
            # GPU synchronization after timing
            # --------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            current_batch_size = (
                images.shape[0]
            )

            total_inference_time += (
                elapsed
            )

            total_images += (
                current_batch_size
            )

            # --------------------------------------------------
            # Convert logits -> probabilities
            # --------------------------------------------------

            mask_probabilities = torch.sigmoid(
                mask_logits
            )

            spine_probabilities = torch.sigmoid(
                spine_logits
            )

            # --------------------------------------------------
            # Threshold
            # --------------------------------------------------

            predicted_masks = (
                mask_probabilities
                >= THRESHOLD
            ).cpu().numpy()

            predicted_spines = (
                spine_probabilities
                >= THRESHOLD
            ).cpu().numpy()

            true_masks = (
                masks >= 0.5
            ).cpu().numpy()

            true_spines = (
                spines >= 0.5
            ).cpu().numpy()

            # ==================================================
            # PROCESS EACH IMAGE
            # ==================================================

            for i in range(
                current_batch_size
            ):

                predicted_mask = (
                    predicted_masks[
                        i,
                        0,
                    ]
                )

                true_mask = (
                    true_masks[
                        i,
                        0,
                    ]
                )

                predicted_spine = (
                    predicted_spines[
                        i,
                        0,
                    ]
                )

                true_spine = (
                    true_spines[
                        i,
                        0,
                    ]
                )

                # --------------------------------------------------
                # Filament confusion matrix
                # --------------------------------------------------

                mask_tp += int(
                    np.logical_and(
                        predicted_mask,
                        true_mask,
                    ).sum()
                )

                mask_tn += int(
                    np.logical_and(
                        ~predicted_mask,
                        ~true_mask,
                    ).sum()
                )

                mask_fp += int(
                    np.logical_and(
                        predicted_mask,
                        ~true_mask,
                    ).sum()
                )

                mask_fn += int(
                    np.logical_and(
                        ~predicted_mask,
                        true_mask,
                    ).sum()
                )

                # --------------------------------------------------
                # Spine confusion matrix
                # --------------------------------------------------

                spine_tp += int(
                    np.logical_and(
                        predicted_spine,
                        true_spine,
                    ).sum()
                )

                spine_tn += int(
                    np.logical_and(
                        ~predicted_spine,
                        ~true_spine,
                    ).sum()
                )

                spine_fp += int(
                    np.logical_and(
                        predicted_spine,
                        ~true_spine,
                    ).sum()
                )

                spine_fn += int(
                    np.logical_and(
                        ~predicted_spine,
                        true_spine,
                    ).sum()
                )

                # --------------------------------------------------
                # Save qualitative visualization
                # --------------------------------------------------

                if (
                    saved_visualizations
                    < NUM_VISUALIZATIONS
                ):

                    save_visualization(
                        image=images[
                            i
                        ].cpu().numpy(),

                        gt_mask=true_mask,

                        pred_mask=predicted_mask,

                        gt_spine=true_spine,

                        pred_spine=predicted_spine,

                        image_name=str(
                            image_names_batch[
                                i
                            ]
                        ),

                        output_dir=OUTPUT_DIR,
                    )

                    saved_visualizations += 1

            # --------------------------------------------------
            # Progress
            # --------------------------------------------------

            processed = min(
                total_images,
                len(val_dataset),
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 5 == 0
                or processed
                == len(val_dataset)
            ):

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(val_dataset)}"
                )

    # ==========================================================
    # CALCULATE FINAL METRICS
    # ==========================================================

    mask_metrics = metrics_from_counts(
        mask_tp,
        mask_tn,
        mask_fp,
        mask_fn,
    )

    spine_metrics = metrics_from_counts(
        spine_tp,
        spine_tn,
        spine_fp,
        spine_fn,
    )

    average_inference_time = (
        total_inference_time
        / max(total_images, 1)
    )

    fps = (
        1.0
        / average_inference_time
        if average_inference_time > 0
        else 0.0
    )

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(
        os.path.abspath(CHECKPOINT)
    )

    print()
    print("DATASET")
    print("-" * 40)

    print(
        f"Total images:      {len(image_names)}"
    )

    print(
        f"Training images:   {len(train_names)}"
    )

    print(
        f"Validation images: {len(val_names)}"
    )

    print(
        f"Threshold:         {THRESHOLD}"
    )

    # ----------------------------------------------------------
    # Filament
    # ----------------------------------------------------------

    print()
    print("FILAMENT MASK")
    print("-" * 40)

    print(
        f"Dice:       "
        f"{mask_metrics['dice']:.6f}"
    )

    print(
        f"IoU:        "
        f"{mask_metrics['iou']:.6f}"
    )

    print(
        f"Precision:  "
        f"{mask_metrics['precision']:.6f}"
    )

    print(
        f"Recall:     "
        f"{mask_metrics['recall']:.6f}"
    )

    print(
        f"Accuracy:   "
        f"{mask_metrics['accuracy']:.6f}"
    )

    print(
        f"MCC:        "
        f"{mask_metrics['mcc']:.6f}"
    )

    print(
        f"Kappa:      "
        f"{mask_metrics['kappa']:.6f}"
    )

    print()
    print("Pixels")
    print(
        f"TP = {mask_tp}"
    )

    print(
        f"TN = {mask_tn}"
    )

    print(
        f"FP = {mask_fp}"
    )

    print(
        f"FN = {mask_fn}"
    )

    # ----------------------------------------------------------
    # Spine
    # ----------------------------------------------------------

    print()
    print("SPINE")
    print("-" * 40)

    print(
        f"Dice:       "
        f"{spine_metrics['dice']:.6f}"
    )

    print(
        f"IoU:        "
        f"{spine_metrics['iou']:.6f}"
    )

    print()
    print("Spine Pixels")

    print(
        f"TP = {spine_tp}"
    )

    print(
        f"TN = {spine_tn}"
    )

    print(
        f"FP = {spine_fp}"
    )

    print(
        f"FN = {spine_fn}"
    )

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    print()
    print("INFERENCE PERFORMANCE")
    print("-" * 40)

    print(
        f"Total inference time: "
        f"{total_inference_time:.4f} s"
    )

    print(
        f"Average inference:    "
        f"{average_inference_time:.6f} s/image"
    )

    print(
        f"FPS:                  "
        f"{fps:.2f}"
    )

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------

    print()
    print("QUALITATIVE RESULTS")
    print("-" * 40)

    print(
        f"Images saved: "
        f"{saved_visualizations}"
    )

    print(
        "Directory:"
    )

    print(
        os.path.abspath(OUTPUT_DIR)
    )

    # ----------------------------------------------------------
    # Complete
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    # ==========================================================
    # SAVE REPORT
    # ==========================================================

    report_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "UNET++ + SPINE EVALUATION REPORT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{os.path.abspath(CHECKPOINT)}\n"
        )

        f.write(
            f"Total images: "
            f"{len(image_names)}\n"
        )

        f.write(
            f"Training images: "
            f"{len(train_names)}\n"
        )

        f.write(
            f"Validation images: "
            f"{len(val_names)}\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"Image size: "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Threshold: "
            f"{THRESHOLD}\n\n"
        )

        # ------------------------------------------------------
        # Filament
        # ------------------------------------------------------

        f.write(
            "FILAMENT MASK\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Dice:       "
            f"{mask_metrics['dice']:.6f}\n"
        )

        f.write(
            f"IoU:        "
            f"{mask_metrics['iou']:.6f}\n"
        )

        f.write(
            f"Precision:  "
            f"{mask_metrics['precision']:.6f}\n"
        )

        f.write(
            f"Recall:     "
            f"{mask_metrics['recall']:.6f}\n"
        )

        f.write(
            f"Accuracy:   "
            f"{mask_metrics['accuracy']:.6f}\n"
        )

        f.write(
            f"MCC:        "
            f"{mask_metrics['mcc']:.6f}\n"
        )

        f.write(
            f"Kappa:      "
            f"{mask_metrics['kappa']:.6f}\n"
        )

        f.write(
            f"TP:         {mask_tp}\n"
        )

        f.write(
            f"TN:         {mask_tn}\n"
        )

        f.write(
            f"FP:         {mask_fp}\n"
        )

        f.write(
            f"FN:         {mask_fn}\n\n"
        )

        # ------------------------------------------------------
        # Spine
        # ------------------------------------------------------

        f.write(
            "SPINE\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Dice:       "
            f"{spine_metrics['dice']:.6f}\n"
        )

        f.write(
            f"IoU:        "
            f"{spine_metrics['iou']:.6f}\n"
        )

        f.write(
            f"TP:         {spine_tp}\n"
        )

        f.write(
            f"TN:         {spine_tn}\n"
        )

        f.write(
            f"FP:         {spine_fp}\n"
        )

        f.write(
            f"FN:         {spine_fn}\n\n"
        )

        # ------------------------------------------------------
        # Inference
        # ------------------------------------------------------

        f.write(
            "INFERENCE PERFORMANCE\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Total inference time: "
            f"{total_inference_time:.6f} s\n"
        )

        f.write(
            f"Average inference: "
            f"{average_inference_time:.6f} s/image\n"
        )

        f.write(
            f"FPS: "
            f"{fps:.4f}\n"
        )

    print()
    print(
        "Report saved:"
    )

    print(
        os.path.abspath(report_path)
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()