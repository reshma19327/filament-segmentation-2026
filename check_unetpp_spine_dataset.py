import os
from dataset_unetpp_spine import UNetPPSpineDataset


BASE_DIR = r"C:\Users\Kevin\Downloads\filament-segmentation-2026"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "MAGFiLO_1.0_Kaggle_2026",
    "train",
    "train_images"
)

MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "full_masks"
)

SPINE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "full_spine_masks"
)


# ---------------------------------------
# Get all physical images
# ---------------------------------------

image_names = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])


print("=" * 60)
print("CHECKING UNET++ + SPINE DATASET")
print("=" * 60)

print("Images:", len(image_names))

# ---------------------------------------
# Create dataset
# ---------------------------------------

dataset = UNetPPSpineDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    spine_dir=SPINE_DIR,
    image_names=image_names,
    image_size=256
)

print("Dataset length:", len(dataset))

# ---------------------------------------
# Read first sample
# ---------------------------------------

image, mask, spine, name = dataset[0]

print("-" * 60)
print("FIRST SAMPLE")
print("-" * 60)

print("Filename:", name)

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Spine shape:", spine.shape)

print("Image min:", image.min().item())
print("Image max:", image.max().item())

print("Mask min:", mask.min().item())
print("Mask max:", mask.max().item())

print("Spine min:", spine.min().item())
print("Spine max:", spine.max().item())

print("=" * 60)
print("DATASET CHECK PASSED")
print("=" * 60)