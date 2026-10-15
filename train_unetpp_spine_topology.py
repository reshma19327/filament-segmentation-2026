# ==============================================================
# train_unetpp_spine_topology.py
#
# UNet++ + Spine + Topology-Aware Loss
# ==============================================================

import os
import random
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset_unetpp_spine import UNetPPSpineDataset
from unetpp_spine import UNetPPSpine

from loss_unetpp_spine_topology import (
    UNetPPSpineTopologyLoss,
)


# ==============================================================
# CONFIG
# ==============================================================

IMAGE_DIR = r".\MAGFiLO_1.0_Kaggle_2026\train\train_images"

MASK_DIR = r".\data\full_masks"

SPINE_DIR = r".\data\full_spine_masks"

CHECKPOINT_DIR = r".\checkpoints_topology"

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 50

LEARNING_RATE = 1e-3

SEED = 42

MASK_WEIGHT = 1.0

SPINE_WEIGHT = 0.5

TOPOLOGY_WEIGHT = 0.1


# ==============================================================
# SEED
# ==============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==============================================================
# DISCOVER IMAGES
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

    names = []

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
            stem + ".png",
        )

        spine_path = os.path.join(
            SPINE_DIR,
            stem + ".png",
        )

        if (
            os.path.isfile(mask_path)
            and os.path.isfile(spine_path)
        ):

            names.append(filename)

    return names


# ==============================================================
# METRICS
# ==============================================================

def calculate_metrics(
    tp,
    tn,
    fp,
    fn,
):

    eps = 1e-8

    dice = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
            + eps
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + eps
        )
    )

    total = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = (
        tp
        + tn
    ) / (
        total
        + eps
    )

    denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator > 0:

        mcc = (
            tp * tn
            - fp * fn
        ) / (
            denominator ** 0.5
        )

    else:

        mcc = 0.0

    if total > 0:

        observed = (
            tp + tn
        ) / total

        actual_positive = (
            tp + fn
        )

        actual_negative = (
            tn + fp
        )

        predicted_positive = (
            tp + fp
        )

        predicted_negative = (
            tn + fn
        )

        expected = (
            actual_positive
            * predicted_positive
            +
            actual_negative
            * predicted_negative
        ) / (
            total * total
        )

        if abs(
            1.0 - expected
        ) > eps:

            kappa = (
                observed - expected
            ) / (
                1.0 - expected
            )

        else:

            kappa = 0.0

    else:

        kappa = 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "mcc": mcc,
        "kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ==============================================================
# RUN ONE EPOCH
# ==============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training=True,
):

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    total_mask_loss = 0.0

    total_spine_loss = 0.0

    total_topology_loss = 0.0

    num_batches = 0

    mask_tp = 0
    mask_tn = 0
    mask_fp = 0
    mask_fn = 0

    spine_tp = 0
    spine_tn = 0
    spine_fp = 0
    spine_fn = 0

    start_time = time.perf_counter()

    for batch in loader:

        images = batch[0]

        masks = batch[1]

        spines = batch[2]

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

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            mask_prediction, spine_prediction = model(
                images
            )

            (
                loss,
                mask_loss,
                spine_loss,
                topology_loss,
            ) = criterion(
                mask_prediction,
                spine_prediction,
                masks,
                spines,
            )

            if training:

                loss.backward()

                optimizer.step()

        total_loss += (
            loss.item()
        )

        total_mask_loss += (
            mask_loss.item()
        )

        total_spine_loss += (
            spine_loss.item()
        )

        total_topology_loss += (
            topology_loss.item()
        )

        num_batches += 1

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        mask_probability = torch.sigmoid(
            mask_prediction
        )

        spine_probability = torch.sigmoid(
            spine_prediction
        )

        predicted_mask = (
            mask_probability >= 0.5
        )

        predicted_spine = (
            spine_probability >= 0.5
        )

        true_mask = (
            masks >= 0.5
        )

        true_spine = (
            spines >= 0.5
        )

        mask_tp += int(
            torch.logical_and(
                predicted_mask,
                true_mask,
            ).sum().item()
        )

        mask_tn += int(
            torch.logical_and(
                ~predicted_mask,
                ~true_mask,
            ).sum().item()
        )

        mask_fp += int(
            torch.logical_and(
                predicted_mask,
                ~true_mask,
            ).sum().item()
        )

        mask_fn += int(
            torch.logical_and(
                ~predicted_mask,
                true_mask,
            ).sum().item()
        )

        spine_tp += int(
            torch.logical_and(
                predicted_spine,
                true_spine,
            ).sum().item()
        )

        spine_tn += int(
            torch.logical_and(
                ~predicted_spine,
                ~true_spine,
            ).sum().item()
        )

        spine_fp += int(
            torch.logical_and(
                predicted_spine,
                ~true_spine,
            ).sum().item()
        )

        spine_fn += int(
            torch.logical_and(
                ~predicted_spine,
                true_spine,
            ).sum().item()
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    mask_metrics = calculate_metrics(
        mask_tp,
        mask_tn,
        mask_fp,
        mask_fn,
    )

    spine_metrics = calculate_metrics(
        spine_tp,
        spine_tn,
        spine_fp,
        spine_fn,
    )

    return {
        "loss": (
            total_loss / num_batches
        ),

        "mask_loss": (
            total_mask_loss
            / num_batches
        ),

        "spine_loss": (
            total_spine_loss
            / num_batches
        ),

        "topology_loss": (
            total_topology_loss
            / num_batches
        ),

        "mask": mask_metrics,

        "spine": spine_metrics,

        "time": elapsed,
    }


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print(
        "UNET++ + SPINE + TOPOLOGY-AWARE TRAINING"
    )
    print("=" * 70)

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
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

    print(
        "Device:",
        device,
    )

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
    # Dataset
    # ----------------------------------------------------------

    image_names = get_image_names()

    print()
    print(
        "Images found:",
        len(image_names),
    )

    if len(image_names) != 707:

        raise RuntimeError(
            "Expected exactly 707 usable images."
        )

    # ----------------------------------------------------------
    # EXACT SAME SPLIT AS BASELINE
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
    # Datasets
    # ----------------------------------------------------------

    train_dataset = UNetPPSpineDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        spine_dir=SPINE_DIR,
        image_names=train_names,
        image_size=IMAGE_SIZE,
    )

    val_dataset = UNetPPSpineDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        spine_dir=SPINE_DIR,
        image_names=val_names,
        image_size=IMAGE_SIZE,
    )

    # ----------------------------------------------------------
    # Data loaders
    # ----------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
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

    # ----------------------------------------------------------
    # Model
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
    # Loss
    # ----------------------------------------------------------

    criterion = UNetPPSpineTopologyLoss(
        mask_weight=MASK_WEIGHT,
        spine_weight=SPINE_WEIGHT,
        topology_weight=TOPOLOGY_WEIGHT,
    ).to(device)

    print()
    print(
        "Loss configuration:"
    )

    print(
        "Mask weight:",
        MASK_WEIGHT,
    )

    print(
        "Spine weight:",
        SPINE_WEIGHT,
    )

    print(
        "Topology weight:",
        TOPOLOGY_WEIGHT,
    )

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ----------------------------------------------------------
    # Sanity check
    # ----------------------------------------------------------

    print()
    print(
        "Running sanity check..."
    )

    first_batch = next(
        iter(train_loader)
    )

    sanity_images = first_batch[0].to(
        device
    )

    sanity_masks = first_batch[1].to(
        device
    )

    sanity_spines = first_batch[2].to(
        device
    )

    model.eval()

    with torch.no_grad():

        sanity_mask_prediction, sanity_spine_prediction = model(
            sanity_images
        )

    (
        sanity_total,
        sanity_mask,
        sanity_spine,
        sanity_topology,
    ) = criterion(
        sanity_mask_prediction,
        sanity_spine_prediction,
        sanity_masks,
        sanity_spines,
    )

    print(
        "Input:",
        tuple(sanity_images.shape),
    )

    print(
        "Mask output:",
        tuple(
            sanity_mask_prediction.shape
        ),
    )

    print(
        "Spine output:",
        tuple(
            sanity_spine_prediction.shape
        ),
    )

    print(
        f"Total loss:    "
        f"{sanity_total.item():.6f}"
    )

    print(
        f"Mask loss:     "
        f"{sanity_mask.item():.6f}"
    )

    print(
        f"Spine loss:    "
        f"{sanity_spine.item():.6f}"
    )

    print(
        f"Topology loss: "
        f"{sanity_topology.item():.6f}"
    )

    print(
        "SANITY CHECK PASSED"
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    best_dice = -1.0

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    history = []

    print()
    print("=" * 70)
    print(
        "STARTING TOPOLOGY-AWARE TRAINING"
    )
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("-" * 70)

        # ------------------------------------------------------
        # Train
        # ------------------------------------------------------

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        val_result = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )

        # ------------------------------------------------------
        # Current learning rate
        # ------------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ------------------------------------------------------
        # Print
        # ------------------------------------------------------

        print(
            f"Train | "
            f"total={train_result['loss']:.6f} "
            f"mask={train_result['mask_loss']:.6f} "
            f"spine={train_result['spine_loss']:.6f} "
            f"topology={train_result['topology_loss']:.6f}"
        )

        print(
            f"Val   | "
            f"total={val_result['loss']:.6f} "
            f"mask={val_result['mask_loss']:.6f} "
            f"spine={val_result['spine_loss']:.6f} "
            f"topology={val_result['topology_loss']:.6f}"
        )

        print(
            f"Mask  | "
            f"Dice={val_result['mask']['dice']:.4f} "
            f"IoU={val_result['mask']['iou']:.4f} "
            f"Precision={val_result['mask']['precision']:.4f} "
            f"Recall={val_result['mask']['recall']:.4f}"
        )

        print(
            f"Mask  | "
            f"Accuracy={val_result['mask']['accuracy']:.4f} "
            f"MCC={val_result['mask']['mcc']:.4f} "
            f"Kappa={val_result['mask']['kappa']:.4f}"
        )

        print(
            f"Spine | "
            f"Dice={val_result['spine']['dice']:.4f} "
            f"IoU={val_result['spine']['iou']:.4f}"
        )

        print(
            f"LR={current_lr:.6f} "
            f"Time={val_result['time']:.1f}s"
        )

        # ------------------------------------------------------
        # History
        # ------------------------------------------------------

        history.append(
            {
                "epoch": epoch,

                "train_loss":
                    train_result["loss"],

                "train_mask_loss":
                    train_result["mask_loss"],

                "train_spine_loss":
                    train_result["spine_loss"],

                "train_topology_loss":
                    train_result["topology_loss"],

                "val_loss":
                    val_result["loss"],

                "val_mask_loss":
                    val_result["mask_loss"],

                "val_spine_loss":
                    val_result["spine_loss"],

                "val_topology_loss":
                    val_result["topology_loss"],

                "val_dice":
                    val_result["mask"]["dice"],

                "val_iou":
                    val_result["mask"]["iou"],

                "val_precision":
                    val_result["mask"]["precision"],

                "val_recall":
                    val_result["mask"]["recall"],

                "val_accuracy":
                    val_result["mask"]["accuracy"],

                "val_mcc":
                    val_result["mask"]["mcc"],

                "val_kappa":
                    val_result["mask"]["kappa"],

                "val_spine_dice":
                    val_result["spine"]["dice"],

                "val_spine_iou":
                    val_result["spine"]["iou"],

                "lr":
                    current_lr,
            }
        )

        # ------------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------------

        current_dice = (
            val_result["mask"]["dice"]
        )

        if current_dice > best_dice:

            best_dice = current_dice

            best_val_loss = (
                val_result["loss"]
            )

            best_epoch = epoch

            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best_unetpp_spine_topology.pth",
            )

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_dice":
                        best_dice,

                    "best_val_loss":
                        best_val_loss,

                    "topology_weight":
                        TOPOLOGY_WEIGHT,

                    "mask_weight":
                        MASK_WEIGHT,

                    "spine_weight":
                        SPINE_WEIGHT,

                    "seed":
                        SEED,

                    "train_images":
                        len(train_names),

                    "validation_images":
                        len(val_names),

                    "history":
                        history,
                },
                best_path,
            )

            print()
            print(
                "NEW BEST CHECKPOINT"
            )

            print(
                f"Best Dice: "
                f"{best_dice:.6f}"
            )

            print(
                "Saved:",
                best_path,
            )

    # ==========================================================
    # LAST CHECKPOINT
    # ==========================================================

    last_path = os.path.join(
        CHECKPOINT_DIR,
        "last_unetpp_spine_topology.pth",
    )

    torch.save(
        {
            "epoch": EPOCHS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_dice":
                best_dice,

            "best_val_loss":
                best_val_loss,

            "topology_weight":
                TOPOLOGY_WEIGHT,

            "mask_weight":
                MASK_WEIGHT,

            "spine_weight":
                SPINE_WEIGHT,

            "seed":
                SEED,

            "train_images":
                len(train_names),

            "validation_images":
                len(val_names),

            "history":
                history,
        },
        last_path,
    )

    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================

    final_result = val_result

    print()
    print("=" * 70)
    print(
        "TOPOLOGY-AWARE TRAINING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    print(
        f"Best validation loss:  "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best epoch:            "
        f"{best_epoch}"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        os.path.abspath(
            os.path.join(
                CHECKPOINT_DIR,
                "best_unetpp_spine_topology.pth",
            )
        )
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        os.path.abspath(
            last_path
        )
    )

    print()
    print(
        "Final validation metrics:"
    )

    print(
        f"Dice:       "
        f"{final_result['mask']['dice']:.6f}"
    )

    print(
        f"IoU:        "
        f"{final_result['mask']['iou']:.6f}"
    )

    print(
        f"Precision:  "
        f"{final_result['mask']['precision']:.6f}"
    )

    print(
        f"Recall:     "
        f"{final_result['mask']['recall']:.6f}"
    )

    print(
        f"Accuracy:   "
        f"{final_result['mask']['accuracy']:.6f}"
    )

    print(
        f"MCC:        "
        f"{final_result['mask']['mcc']:.6f}"
    )

    print(
        f"Kappa:      "
        f"{final_result['mask']['kappa']:.6f}"
    )

    print(
        f"Spine Dice: "
        f"{final_result['spine']['dice']:.6f}"
    )

    print(
        f"Spine IoU:  "
        f"{final_result['spine']['iou']:.6f}"
    )

    print()
    print("=" * 70)


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":

    main()