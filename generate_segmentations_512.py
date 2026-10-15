"""
======================================================================
GENERATE SEGMENTATION RESULTS
MODEL 2: UNET++ + SPINE
======================================================================

Uses the EXACT architecture used to train:

    benchmark_models_512/
        unetpp_spine/
            best_unetpp_spine_512.pth

Dataset:
    707 images
    565 training
    142 validation

Resolution:
    512x512

Outputs:
    benchmark_models_512/
        unetpp_spine/
            segmentation_results/
                input/
                ground_truth/
                filament/
                spine/
                overlay/
======================================================================
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "full_masks"
)

SPINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "full_spine_masks"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "benchmark_512"
    / "benchmark_split.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "benchmark_models_512"
    / "unetpp_spine"
)

CHECKPOINT_FILE = (
    MODEL_DIR
    / "best_unetpp_spine_512.pth"
)

RESULTS_DIR = (
    MODEL_DIR
    / "segmentation_results"
)

INPUT_DIR = RESULTS_DIR / "input"
GROUND_TRUTH_DIR = RESULTS_DIR / "ground_truth"
FILAMENT_DIR = RESULTS_DIR / "filament"
SPINE_RESULT_DIR = RESULTS_DIR / "spine"
OVERLAY_DIR = RESULTS_DIR / "overlay"

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# CREATE DIRECTORIES
# =====================================================================

def create_directories():

    for directory in [
        INPUT_DIR,
        GROUND_TRUTH_DIR,
        FILAMENT_DIR,
        SPINE_RESULT_DIR,
        OVERLAY_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# =====================================================================
# DATASET
# =====================================================================

class FilamentSpineDataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):

        return len(self.records)

    # -----------------------------------------------------------------
    # Resolve filename
    # -----------------------------------------------------------------

    def get_value(self, record, keys):

        for key in keys:

            if key in record:

                return record[key]

        return None

    def resolve_path(self, value, directory):

        if value is None:

            raise RuntimeError(
                "Could not find filename in benchmark record."
            )

        path = Path(value)

        # Absolute path
        if path.is_absolute():

            if path.exists():

                return path

        # Normal filename
        candidate = directory / path.name

        if candidate.exists():

            return candidate

        # Case-insensitive search
        target = path.name.lower()

        for file in directory.iterdir():

            if file.name.lower() == target:

                return file

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{path.name}\n\n"
            f"Directory:\n"
            f"{directory}"
        )

    # -----------------------------------------------------------------
    # Read image
    # -----------------------------------------------------------------

    def read_gray(self, path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read:\n{path}"
            )

        return image

    # -----------------------------------------------------------------
    # Get item
    # -----------------------------------------------------------------

    def __getitem__(self, index):

        record = self.records[index]

        image_name = self.get_value(
            record,
            [
                "image",
                "image_file",
                "image_filename",
                "image_path",
                "filename",
            ]
        )

        mask_name = self.get_value(
            record,
            [
                "mask",
                "mask_file",
                "mask_filename",
                "mask_path",
            ]
        )

        spine_name = self.get_value(
            record,
            [
                "spine",
                "spine_file",
                "spine_filename",
                "spine_path",
            ]
        )

        image_path = self.resolve_path(
            image_name,
            IMAGE_DIR
        )

        mask_path = self.resolve_path(
            mask_name,
            MASK_DIR
        )

        spine_path = self.resolve_path(
            spine_name,
            SPINE_DIR
        )

        image = self.read_gray(
            image_path
        )

        mask = self.read_gray(
            mask_path
        )

        spine = self.read_gray(
            spine_path
        )

        # -------------------------------------------------------------
        # Resize
        # -------------------------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        spine = cv2.resize(
            spine,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------------------
        # Normalize
        # -------------------------------------------------------------

        image_float = (
            image.astype(np.float32)
            / 255.0
        )

        mask_float = (
            mask.astype(np.float32)
            / 255.0
        )

        spine_float = (
            spine.astype(np.float32)
            / 255.0
        )

        mask_float = (
            mask_float >= 0.5
        ).astype(np.float32)

        spine_float = (
            spine_float >= 0.5
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Tensor
        # -------------------------------------------------------------

        image_tensor = torch.from_numpy(
            image_float
        ).unsqueeze(0).float()

        mask_tensor = torch.from_numpy(
            mask_float
        ).unsqueeze(0).float()

        spine_tensor = torch.from_numpy(
            spine_float
        ).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "spine": spine_tensor,
            "name": image_path.stem,
        }


# =====================================================================
# EXACT TRAINING MODEL
# =====================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):

        return self.block(x)


class UNetPPSpine(nn.Module):

    """
    EXACT architecture used during the
    UNet++ + Spine benchmark training.

    Filters:
        32
        64
        128
        256
        512

    Outputs:
        filament
        spine
    """

    def __init__(self):

        super().__init__()

        filters = [
            32,
            64,
            128,
            256,
            512
        ]

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        self.conv0_0 = ConvBlock(
            1,
            filters[0]
        )

        self.conv1_0 = ConvBlock(
            filters[0],
            filters[1]
        )

        self.conv2_0 = ConvBlock(
            filters[1],
            filters[2]
        )

        self.conv3_0 = ConvBlock(
            filters[2],
            filters[3]
        )

        self.conv4_0 = ConvBlock(
            filters[3],
            filters[4]
        )

        # -------------------------------------------------------------
        # UNet++ decoder
        # -------------------------------------------------------------

        self.conv0_1 = ConvBlock(
            filters[0] + filters[1],
            filters[0]
        )

        self.conv1_1 = ConvBlock(
            filters[1] + filters[2],
            filters[1]
        )

        self.conv2_1 = ConvBlock(
            filters[2] + filters[3],
            filters[2]
        )

        self.conv3_1 = ConvBlock(
            filters[3] + filters[4],
            filters[3]
        )

        self.conv0_2 = ConvBlock(
            filters[0] * 2 + filters[1],
            filters[0]
        )

        self.conv1_2 = ConvBlock(
            filters[1] * 2 + filters[2],
            filters[1]
        )

        self.conv2_2 = ConvBlock(
            filters[2] * 2 + filters[3],
            filters[2]
        )

        self.conv0_3 = ConvBlock(
            filters[0] * 3 + filters[1],
            filters[0]
        )

        self.conv1_3 = ConvBlock(
            filters[1] * 3 + filters[2],
            filters[1]
        )

        self.conv0_4 = ConvBlock(
            filters[0] * 4 + filters[1],
            filters[0]
        )

        # -------------------------------------------------------------
        # Output heads
        # -------------------------------------------------------------

        self.filament_head = nn.Conv2d(
            filters[0],
            1,
            kernel_size=1
        )

        self.spine_head = nn.Conv2d(
            filters[0],
            1,
            kernel_size=1
        )

    # -----------------------------------------------------------------
    # Upsampling
    # -----------------------------------------------------------------

    def up(
        self,
        x,
        target
    ):

        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, x):

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(
            self.pool(x0_0)
        )

        x2_0 = self.conv2_0(
            self.pool(x1_0)
        )

        x3_0 = self.conv3_0(
            self.pool(x2_0)
        )

        x4_0 = self.conv4_0(
            self.pool(x3_0)
        )

        # -------------------------------------------------------------
        # First nested level
        # -------------------------------------------------------------

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    self.up(
                        x1_0,
                        x0_0
                    ),
                ],
                dim=1
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    self.up(
                        x2_0,
                        x1_0
                    ),
                ],
                dim=1
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    self.up(
                        x3_0,
                        x2_0
                    ),
                ],
                dim=1
            )
        )

        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    self.up(
                        x4_0,
                        x3_0
                    ),
                ],
                dim=1
            )
        )

        # -------------------------------------------------------------
        # Second nested level
        # -------------------------------------------------------------

        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    self.up(
                        x1_1,
                        x0_0
                    ),
                ],
                dim=1
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    self.up(
                        x2_1,
                        x1_0
                    ),
                ],
                dim=1
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    self.up(
                        x3_1,
                        x2_0
                    ),
                ],
                dim=1
            )
        )

        # -------------------------------------------------------------
        # Third nested level
        # -------------------------------------------------------------

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    self.up(
                        x1_2,
                        x0_0
                    ),
                ],
                dim=1
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    self.up(
                        x2_2,
                        x1_0
                    ),
                ],
                dim=1
            )
        )

        # -------------------------------------------------------------
        # Fourth nested level
        # -------------------------------------------------------------

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    self.up(
                        x1_3,
                        x0_0
                    ),
                ],
                dim=1
            )
        )

        # -------------------------------------------------------------
        # Outputs
        # -------------------------------------------------------------

        filament = self.filament_head(
            x0_4
        )

        spine = self.spine_head(
            x0_4
        )

        return filament, spine


# =====================================================================
# LOAD SPLIT
# =====================================================================

def load_validation_records():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "validation" in data:

        records = data["validation"]

    elif "val" in data:

        records = data["val"]

    else:

        raise KeyError(
            "No validation split found."
        )

    print(
        f"Validation images: {len(records)}"
    )

    if len(records) != 142:

        print(
            "WARNING: expected 142 validation images."
        )

    return records


# =====================================================================
# LOAD CHECKPOINT
# =====================================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)

    print(
        f"Checkpoint:\n{CHECKPOINT_FILE}"
    )

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    model = UNetPPSpine().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    # Training script saved:
    # {
    #     "model_state_dict": ...
    # }

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        best_dice = checkpoint.get(
            "best_dice",
            "unknown"
        )

        print(
            f"Checkpoint epoch: {epoch}"
        )

        print(
            f"Checkpoint Dice:  {best_dice}"
        )

    else:

        # Also support raw state_dict
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print()
    print(
        "CHECKPOINT LOADED SUCCESSFULLY"
    )

    return model


# =====================================================================
# SAVE ONE PREDICTION
# =====================================================================

def save_prediction(
    name,
    image,
    ground_truth,
    filament_probability,
    spine_probability
):

    # -------------------------------------------------------------
    # Convert arrays
    # -------------------------------------------------------------

    image_uint8 = (
        image * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    ground_truth_uint8 = (
        ground_truth >= 0.5
    ).astype(
        np.uint8
    ) * 255

    filament_uint8 = (
        filament_probability >= THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    spine_uint8 = (
        spine_probability >= THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    # -------------------------------------------------------------
    # Input
    # -------------------------------------------------------------

    Image.fromarray(
        image_uint8
    ).save(
        INPUT_DIR / f"{name}.png"
    )

    # -------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------

    Image.fromarray(
        ground_truth_uint8
    ).save(
        GROUND_TRUTH_DIR
        / f"{name}.png"
    )

    # -------------------------------------------------------------
    # Filament segmentation
    # -------------------------------------------------------------

    Image.fromarray(
        filament_uint8
    ).save(
        FILAMENT_DIR
        / f"{name}.png"
    )

    # -------------------------------------------------------------
    # Spine segmentation
    # -------------------------------------------------------------

    Image.fromarray(
        spine_uint8
    ).save(
        SPINE_RESULT_DIR
        / f"{name}.png"
    )

    # -------------------------------------------------------------
    # Overlay
    #
    # Green = filament
    # Red   = spine
    # -------------------------------------------------------------

    overlay = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_GRAY2BGR
    )

    filament_pixels = (
        filament_uint8 > 0
    )

    spine_pixels = (
        spine_uint8 > 0
    )

    overlay[
        filament_pixels
    ] = (
        0,
        255,
        0
    )

    overlay[
        spine_pixels
    ] = (
        0,
        0,
        255
    )

    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    Image.fromarray(
        overlay_rgb
    ).save(
        OVERLAY_DIR
        / f"{name}.png"
    )


# =====================================================================
# GENERATE SEGMENTATIONS
# =====================================================================

@torch.no_grad()
def generate_segmentations(
    model,
    loader
):

    model.eval()

    total = len(loader.dataset)

    completed = 0

    print()
    print("=" * 70)
    print("GENERATING SEGMENTATION RESULTS")
    print("=" * 70)

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True
        )

        masks = batch[
            "mask"
        ]

        names = batch[
            "name"
        ]

        # ---------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------

        filament_logits, spine_logits = model(
            images
        )

        filament_probability = torch.sigmoid(
            filament_logits
        ).cpu().numpy()

        spine_probability = torch.sigmoid(
            spine_logits
        ).cpu().numpy()

        images_np = (
            images.cpu().numpy()
        )

        masks_np = (
            masks.numpy()
        )

        # ---------------------------------------------------------
        # Save every image in batch
        # ---------------------------------------------------------

        for i in range(
            images.shape[0]
        ):

            name = names[i]

            save_prediction(
                name=name,
                image=images_np[
                    i,
                    0
                ],
                ground_truth=masks_np[
                    i,
                    0
                ],
                filament_probability=
                    filament_probability[
                        i,
                        0
                    ],
                spine_probability=
                    spine_probability[
                        i,
                        0
                    ]
            )

            completed += 1

        print(
            f"Processed "
            f"{completed:3d}/{total:3d}"
        )

    print()
    print(
        f"Finished: {completed} images"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("GENERATING SEGMENTATION RESULTS")
    print("MODEL: UNET++ + SPINE")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT_FILE}"
    )

    print(
        f"Results:\n"
        f"{RESULTS_DIR}"
    )

    # ---------------------------------------------------------------
    # Directories
    # ---------------------------------------------------------------

    create_directories()

    # ---------------------------------------------------------------
    # Validation split
    # ---------------------------------------------------------------

    validation_records = (
        load_validation_records()
    )

    validation_dataset = (
        FilamentSpineDataset(
            validation_records
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = load_model()

    # ---------------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    test_batch = next(
        iter(validation_loader)
    )

    test_image = test_batch[
        "image"
    ].to(DEVICE)

    with torch.no_grad():

        test_filament, test_spine = model(
            test_image
        )

    print(
        f"Input:     {tuple(test_image.shape)}"
    )

    print(
        f"Filament:  {tuple(test_filament.shape)}"
    )

    print(
        f"Spine:     {tuple(test_spine.shape)}"
    )

    assert test_filament.shape == (
        test_image.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert test_spine.shape == (
        test_image.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print(
        "MODEL SANITY CHECK PASSED"
    )

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------

    generate_segmentations(
        model,
        validation_loader
    )

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("SEGMENTATION GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Input images:")
    print(INPUT_DIR)

    print()
    print("Ground truth:")
    print(GROUND_TRUTH_DIR)

    print()
    print("Filament segmentations:")
    print(FILAMENT_DIR)

    print()
    print("Spine segmentations:")
    print(SPINE_RESULT_DIR)

    print()
    print("Overlays:")
    print(OVERLAY_DIR)

    print()
    print("Total validation images:")
    print(len(validation_dataset))

    print("=" * 70)


if __name__ == "__main__":
    main()