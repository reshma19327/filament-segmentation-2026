import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):

        prediction = torch.sigmoid(prediction)

        prediction = prediction.contiguous().view(
            prediction.size(0), -1
        )

        target = target.contiguous().view(
            target.size(0), -1
        )

        intersection = (
            prediction * target
        ).sum(dim=1)

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                prediction.sum(dim=1)
                + target.sum(dim=1)
                + self.smooth
            )
        )

        return 1.0 - dice.mean()


# ============================================================
# BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, prediction, target):

        bce_loss = self.bce(
            prediction,
            target
        )

        dice_loss = self.dice(
            prediction,
            target
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


# ============================================================
# UNET++ + SPINE LOSS
# ============================================================

class UNetPPSpineLoss(nn.Module):

    def __init__(
        self,
        mask_weight=1.0,
        spine_weight=0.5
    ):

        super().__init__()

        self.mask_weight = mask_weight
        self.spine_weight = spine_weight

        self.segmentation_loss = BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5
        )

    def forward(
        self,
        mask_prediction,
        spine_prediction,
        mask_target,
        spine_target
    ):

        # ----------------------------------------------------
        # Filament segmentation loss
        # ----------------------------------------------------

        mask_loss = self.segmentation_loss(
            mask_prediction,
            mask_target
        )

        # ----------------------------------------------------
        # Spine segmentation loss
        # ----------------------------------------------------

        spine_loss = self.segmentation_loss(
            spine_prediction,
            spine_target
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            self.mask_weight * mask_loss
            +
            self.spine_weight * spine_loss
        )

        return (
            total_loss,
            mask_loss,
            spine_loss
        )


# ============================================================
# TEST LOSS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING UNET++ + SPINE LOSS")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # Fake model outputs
    # --------------------------------------------------------

    mask_prediction = torch.randn(
        1,
        1,
        256,
        256,
        device=device
    )

    spine_prediction = torch.randn(
        1,
        1,
        256,
        256,
        device=device
    )

    # --------------------------------------------------------
    # Fake ground truth
    # --------------------------------------------------------

    mask_target = torch.randint(
        0,
        2,
        (
            1,
            1,
            256,
            256
        ),
        device=device
    ).float()

    spine_target = torch.randint(
        0,
        2,
        (
            1,
            1,
            256,
            256
        ),
        device=device
    ).float()

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    criterion = UNetPPSpineLoss(
        mask_weight=1.0,
        spine_weight=0.5
    ).to(device)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    total_loss, mask_loss, spine_loss = criterion(
        mask_prediction,
        spine_prediction,
        mask_target,
        spine_target
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("-" * 60)

    print(
        "Mask Loss  :",
        mask_loss.item()
    )

    print(
        "Spine Loss :",
        spine_loss.item()
    )

    print(
        "Total Loss :",
        total_loss.item()
    )

    print("-" * 60)

    print("UNET++ + SPINE LOSS WORKS SUCCESSFULLY")

    print("=" * 60)