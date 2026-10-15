import os
import cv2
import torch
from torch.utils.data import Dataset


class UNetPPSpineDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        spine_dir,
        image_names,
        image_size=256
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.spine_dir = spine_dir
        self.image_names = image_names
        self.image_size = image_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(image_name)[0] + ".png"
        )

        spine_path = os.path.join(
            self.spine_dir,
            os.path.splitext(image_name)[0] + ".png"
        )

        # ---------------------------------------
        # Read image
        # ---------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ---------------------------------------
        # Read filament mask
        # ---------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ---------------------------------------
        # Read spine mask
        # ---------------------------------------

        spine = cv2.imread(
            spine_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        if spine is None:
            raise FileNotFoundError(
                f"Spine mask not found: {spine_path}"
            )

        # ---------------------------------------
        # Resize
        # ---------------------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        spine = cv2.resize(
            spine,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # ---------------------------------------
        # Normalize image
        # ---------------------------------------

        image = image.astype("float32") / 255.0

        # Binary masks
        mask = (mask > 0).astype("float32")
        spine = (spine > 0).astype("float32")

        # ---------------------------------------
        # Convert to tensors
        # ---------------------------------------

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        spine = torch.from_numpy(spine).unsqueeze(0)

        return image, mask, spine, image_name