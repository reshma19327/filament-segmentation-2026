"""
UNet++ + Spine + clDice Loss

Loss design:
    Mask loss  = BCE + clDice
    Spine loss = BCE + clDice

    Total loss = Mask loss + 0.5 * Spine loss

clDice weight = 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SOFT MORPHOLOGICAL OPERATIONS
# ============================================================

def soft_erode(img):
    """
    Differentiable approximation of morphological erosion.
    """
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(
            -img,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        )

        p2 = -F.max_pool2d(
            -img,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1)
        )

        return torch.minimum(p1, p2)

    raise ValueError("Input must be a 4D tensor [B, C, H, W]")


def soft_dilate(img):
    """
    Differentiable approximation of morphological dilation.
    """
    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1
    )


def soft_open(img):
    """
    Differentiable morphological opening.
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iterations=10):
    """
    Differentiable soft skeletonization.

    This repeatedly removes soft boundary pixels while
    preserving the centerline/topological structure.
    """

    img = img.clamp(0.0, 1.0)

    skeleton = torch.zeros_like(img)

    for _ in range(iterations):
        eroded = soft_erode(img)

        opened = soft_open(eroded)

        delta = F.relu(eroded - opened)

        skeleton = torch.maximum(skeleton, delta)

        img = eroded

    return skeleton


# ============================================================
# clDice
# ============================================================

def soft_cldice(
    prediction,
    target,
    iterations=10,
    smooth=1e-6
):
    """
    Compute differentiable clDice.

    clDice is designed for tubular/filamentary structures
    and explicitly considers skeleton overlap.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted probabilities, shape [B, 1, H, W]

    target : torch.Tensor
        Binary target mask, shape [B, 1, H, W]

    iterations : int
        Number of soft skeletonization iterations.

    smooth : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        clDice loss = 1 - clDice score
    """

    prediction = prediction.clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    # Soft skeletons
    prediction_skeleton = soft_skeletonize(
        prediction,
        iterations=iterations
    )

    target_skeleton = soft_skeletonize(
        target,
        iterations=iterations
    )

    # --------------------------------------------------------
    # Topological precision
    # How much of the predicted skeleton lies inside target
    # --------------------------------------------------------

    tprec = (
        (prediction_skeleton * target).sum(dim=(1, 2, 3))
        + smooth
    ) / (
        prediction_skeleton.sum(dim=(1, 2, 3))
        + smooth
    )

    # --------------------------------------------------------
    # Topological sensitivity
    # How much of the target skeleton is covered by prediction
    # --------------------------------------------------------

    tsens = (
        (target_skeleton * prediction).sum(dim=(1, 2, 3))
        + smooth
    ) / (
        target_skeleton.sum(dim=(1, 2, 3))
        + smooth
    )

    # --------------------------------------------------------
    # clDice
    # --------------------------------------------------------

    cldice_score = (
        2.0 * tprec * tsens
    ) / (
        tprec + tsens + smooth
    )

    cldice_score = cldice_score.mean()

    # Convert score to loss
    cldice_loss = 1.0 - cldice_score

    return cldice_loss


# ============================================================
# BCE + clDice
# ============================================================

class BCECLDiceLoss(nn.Module):
    """
    Combined BCE + clDice loss.

    loss = BCE_WEIGHT * BCE
         + CLDICE_WEIGHT * clDice
    """

    def __init__(
        self,
        bce_weight=1.0,
        cldice_weight=1.0,
        skeleton_iterations=10
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.skeleton_iterations = skeleton_iterations

    def forward(self, logits, target):
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw model output.

        target : torch.Tensor
            Binary target mask.

        Returns
        -------
        total_loss
        bce_loss
        cldice_loss
        """

        target = target.float()

        # BCE with logits is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            target
        )

        # Convert logits to probabilities for clDice
        prediction = torch.sigmoid(logits)

        cldice_loss = soft_cldice(
            prediction,
            target,
            iterations=self.skeleton_iterations
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.cldice_weight * cldice_loss
        )

        return (
            total_loss,
            bce_loss,
            cldice_loss
        )


# ============================================================
# UNET++ + SPINE + clDice LOSS
# ============================================================

class UNetPPSpineCLDiceLoss(nn.Module):
    """
    Complete loss for UNet++ + Spine model.

    Mask:
        BCE + clDice

    Spine:
        BCE + clDice

    Total:
        mask_loss + 0.5 * spine_loss
    """

    def __init__(
        self,
        mask_weight=1.0,
        spine_weight=0.5,
        bce_weight=1.0,
        cldice_weight=1.0,
        skeleton_iterations=10
    ):
        super().__init__()

        self.mask_weight = mask_weight
        self.spine_weight = spine_weight

        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.skeleton_iterations = skeleton_iterations

        self.mask_loss_fn = BCECLDiceLoss(
            bce_weight=bce_weight,
            cldice_weight=cldice_weight,
            skeleton_iterations=skeleton_iterations
        )

        self.spine_loss_fn = BCECLDiceLoss(
            bce_weight=bce_weight,
            cldice_weight=cldice_weight,
            skeleton_iterations=skeleton_iterations
        )

    def forward(
        self,
        mask_logits,
        spine_logits,
        mask_target,
        spine_target
    ):
        """
        Returns
        -------
        total_loss
        mask_loss
        spine_loss
        mask_bce
        mask_cldice
        spine_bce
        spine_cldice
        """

        # ----------------------------------------------------
        # Filament mask
        # ----------------------------------------------------

        (
            mask_loss,
            mask_bce,
            mask_cldice
        ) = self.mask_loss_fn(
            mask_logits,
            mask_target
        )

        # ----------------------------------------------------
        # Spine
        # ----------------------------------------------------

        (
            spine_loss,
            spine_bce,
            spine_cldice
        ) = self.spine_loss_fn(
            spine_logits,
            spine_target
        )

        # ----------------------------------------------------
        # Weighted total
        # ----------------------------------------------------

        total_loss = (
            self.mask_weight * mask_loss
            + self.spine_weight * spine_loss
        )

        return (
            total_loss,
            mask_loss,
            spine_loss,
            mask_bce,
            mask_cldice,
            spine_bce,
            spine_cldice
        )


# ============================================================
# COMPATIBILITY WRAPPER
# ============================================================

def combined_loss(
    mask_logits,
    spine_logits,
    mask_target,
    spine_target,
    mask_weight=1.0,
    spine_weight=0.5,
    bce_weight=1.0,
    cldice_weight=1.0,
    skeleton_iterations=10
):
    """
    Convenience wrapper.
    """

    criterion = UNetPPSpineCLDiceLoss(
        mask_weight=mask_weight,
        spine_weight=spine_weight,
        bce_weight=bce_weight,
        cldice_weight=cldice_weight,
        skeleton_iterations=skeleton_iterations
    )

    return criterion(
        mask_logits,
        spine_logits,
        mask_target,
        spine_target
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING UNET++ + SPINE + clDice LOSS")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    BATCH_SIZE = 2
    IMAGE_SIZE = 256

    MASK_WEIGHT = 1.0
    SPINE_WEIGHT = 0.5

    BCE_WEIGHT = 1.0
    CLDICE_WEIGHT = 1.0

    SKELETON_ITERATIONS = 10

    # --------------------------------------------------------
    # Fake model outputs
    # --------------------------------------------------------

    mask_logits = torch.randn(
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        requires_grad=True
    )

    spine_logits = torch.randn(
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        requires_grad=True
    )

    # --------------------------------------------------------
    # Fake binary targets
    # --------------------------------------------------------

    mask_target = (
        torch.rand(
            BATCH_SIZE,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device
        ) > 0.8
    ).float()

    spine_target = (
        torch.rand(
            BATCH_SIZE,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device
        ) > 0.9
    ).float()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = UNetPPSpineCLDiceLoss(
        mask_weight=MASK_WEIGHT,
        spine_weight=SPINE_WEIGHT,
        bce_weight=BCE_WEIGHT,
        cldice_weight=CLDICE_WEIGHT,
        skeleton_iterations=SKELETON_ITERATIONS
    ).to(device)

    (
        total_loss,
        mask_loss,
        spine_loss,
        mask_bce,
        mask_cldice,
        spine_bce,
        spine_cldice
    ) = criterion(
        mask_logits,
        spine_logits,
        mask_target,
        spine_target
    )

    print()
    print(f"Mask BCE Loss     : {mask_bce.item():.6f}")
    print(f"Mask clDice Loss  : {mask_cldice.item():.6f}")
    print(f"Mask Total Loss   : {mask_loss.item():.6f}")

    print()

    print(f"Spine BCE Loss    : {spine_bce.item():.6f}")
    print(f"Spine clDice Loss : {spine_cldice.item():.6f}")
    print(f"Spine Total Loss  : {spine_loss.item():.6f}")

    print()

    print(f"Total Loss        : {total_loss.item():.6f}")

    # --------------------------------------------------------
    # Gradient test
    # --------------------------------------------------------

    total_loss.backward()

    print()
    print(
        "Mask gradient OK :",
        mask_logits.grad is not None
        and torch.isfinite(mask_logits.grad).all().item()
    )

    print(
        "Spine gradient OK:",
        spine_logits.grad is not None
        and torch.isfinite(spine_logits.grad).all().item()
    )

    print()

    print("=" * 60)
    print("UNET++ + SPINE + clDice LOSS WORKS SUCCESSFULLY")
    print("=" * 60)