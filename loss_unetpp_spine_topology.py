# ==============================================================
# loss_unetpp_spine_topology.py
#
# UNet++ + Spine + Topology-Aware Loss
#
# Total loss:
#
#   L = mask_weight * (BCE + Dice)
#     + spine_weight * (BCE + Dice)
#     + topology_weight * topology_loss
#
# The topology term encourages continuity of predicted
# filament structures by penalizing differences in local
# connectivity.
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# DICE LOSS
# ==============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, prediction, target):

        prediction = torch.sigmoid(prediction)

        prediction = prediction.contiguous().view(
            prediction.size(0),
            -1,
        )

        target = target.contiguous().view(
            target.size(0),
            -1,
        )

        intersection = (
            prediction * target
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            prediction.sum(dim=1)
            + target.sum(dim=1)
            + self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ==============================================================
# BCE + DICE
# ==============================================================

class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):

        super().__init__()

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.dice = DiceLoss()

    def forward(
        self,
        prediction,
        target,
    ):

        bce = F.binary_cross_entropy_with_logits(
            prediction,
            target,
        )

        dice = self.dice(
            prediction,
            target,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )


# ==============================================================
# DIFFERENTIABLE LOCAL CONNECTIVITY
# ==============================================================

class LocalConnectivityLoss(nn.Module):

    """
    Measures local connectivity agreement between
    prediction and target.

    For every foreground pixel, neighboring foreground
    pixels contribute to a local connectivity map.

    The loss compares:
        - horizontal connectivity
        - vertical connectivity
        - diagonal connectivity

    This is differentiable and therefore suitable for
    backpropagation.
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        prediction,
        target,
    ):

        prediction = torch.sigmoid(
            prediction
        )

        # ------------------------------------------------------
        # Horizontal neighbors
        # ------------------------------------------------------

        pred_horizontal = (
            prediction[:, :, :, 1:]
            *
            prediction[:, :, :, :-1]
        )

        target_horizontal = (
            target[:, :, :, 1:]
            *
            target[:, :, :, :-1]
        )

        horizontal_loss = F.l1_loss(
            pred_horizontal,
            target_horizontal,
        )

        # ------------------------------------------------------
        # Vertical neighbors
        # ------------------------------------------------------

        pred_vertical = (
            prediction[:, :, 1:, :]
            *
            prediction[:, :, :-1, :]
        )

        target_vertical = (
            target[:, :, 1:, :]
            *
            target[:, :, :-1, :]
        )

        vertical_loss = F.l1_loss(
            pred_vertical,
            target_vertical,
        )

        # ------------------------------------------------------
        # Main diagonal
        # ------------------------------------------------------

        pred_diag1 = (
            prediction[:, :, 1:, 1:]
            *
            prediction[:, :, :-1, :-1]
        )

        target_diag1 = (
            target[:, :, 1:, 1:]
            *
            target[:, :, :-1, :-1]
        )

        diagonal1_loss = F.l1_loss(
            pred_diag1,
            target_diag1,
        )

        # ------------------------------------------------------
        # Other diagonal
        # ------------------------------------------------------

        pred_diag2 = (
            prediction[:, :, 1:, :-1]
            *
            prediction[:, :, :-1, 1:]
        )

        target_diag2 = (
            target[:, :, 1:, :-1]
            *
            target[:, :, :-1, 1:]
        )

        diagonal2_loss = F.l1_loss(
            pred_diag2,
            target_diag2,
        )

        # ------------------------------------------------------
        # Average connectivity discrepancy
        # ------------------------------------------------------

        connectivity_loss = (
            horizontal_loss
            + vertical_loss
            + diagonal1_loss
            + diagonal2_loss
        ) / 4.0

        return connectivity_loss


# ==============================================================
# TOPOLOGY-AWARE LOSS
# ==============================================================

class TopologyAwareLoss(nn.Module):

    """
    Combines:

        BCE + Dice

    with:

        differentiable local connectivity loss.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        topology_weight=0.1,
    ):

        super().__init__()

        self.segmentation_loss = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        self.topology_loss = LocalConnectivityLoss()

        self.topology_weight = topology_weight

    def forward(
        self,
        prediction,
        target,
    ):

        segmentation = self.segmentation_loss(
            prediction,
            target,
        )

        topology = self.topology_loss(
            prediction,
            target,
        )

        total = (
            segmentation
            + self.topology_weight * topology
        )

        return (
            total,
            segmentation,
            topology,
        )


# ==============================================================
# UNET++ + SPINE + TOPOLOGY LOSS
# ==============================================================

class UNetPPSpineTopologyLoss(nn.Module):

    def __init__(
        self,
        mask_weight=1.0,
        spine_weight=0.5,
        topology_weight=0.1,
    ):

        super().__init__()

        self.mask_weight = mask_weight

        self.spine_weight = spine_weight

        self.topology_weight = topology_weight

        self.mask_loss = BCEDiceLoss()

        self.spine_loss = BCEDiceLoss()

        self.topology_loss = LocalConnectivityLoss()

    def forward(
        self,
        mask_prediction,
        spine_prediction,
        mask_target,
        spine_target,
    ):

        # ------------------------------------------------------
        # Standard mask segmentation loss
        # ------------------------------------------------------

        mask_loss = self.mask_loss(
            mask_prediction,
            mask_target,
        )

        # ------------------------------------------------------
        # Spine segmentation loss
        # ------------------------------------------------------

        spine_loss = self.spine_loss(
            spine_prediction,
            spine_target,
        )

        # ------------------------------------------------------
        # Topology applied to filament mask
        #
        # The topology term is applied to the main filament
        # prediction because filament connectivity is the
        # primary topology objective.
        # ------------------------------------------------------

        topology_loss = self.topology_loss(
            mask_prediction,
            mask_target,
        )

        # ------------------------------------------------------
        # Total loss
        # ------------------------------------------------------

        total_loss = (
            self.mask_weight * mask_loss
            +
            self.spine_weight * spine_loss
            +
            self.topology_weight * topology_loss
        )

        return (
            total_loss,
            mask_loss,
            spine_loss,
            topology_loss,
        )


# ==============================================================
# COMPATIBILITY FUNCTION
# ==============================================================

def combined_loss(
    mask_prediction,
    spine_prediction,
    mask_target,
    spine_target,
    mask_weight=1.0,
    spine_weight=0.5,
    topology_weight=0.1,
):

    criterion = UNetPPSpineTopologyLoss(
        mask_weight=mask_weight,
        spine_weight=spine_weight,
        topology_weight=topology_weight,
    )

    return criterion(
        mask_prediction,
        spine_prediction,
        mask_target,
        spine_target,
    )


# ==============================================================
# SELF TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING UNET++ + SPINE + TOPOLOGY-AWARE LOSS")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    batch_size = 2

    height = 256

    width = 256

    mask_prediction = torch.randn(
        batch_size,
        1,
        height,
        width,
        device=device,
        requires_grad=True,
    )

    spine_prediction = torch.randn(
        batch_size,
        1,
        height,
        width,
        device=device,
        requires_grad=True,
    )

    mask_target = (
        torch.rand(
            batch_size,
            1,
            height,
            width,
            device=device,
        )
        > 0.8
    ).float()

    spine_target = (
        torch.rand(
            batch_size,
            1,
            height,
            width,
            device=device,
        )
        > 0.9
    ).float()

    criterion = UNetPPSpineTopologyLoss(
        mask_weight=1.0,
        spine_weight=0.5,
        topology_weight=0.1,
    ).to(device)

    total, mask, spine, topology = criterion(
        mask_prediction,
        spine_prediction,
        mask_target,
        spine_target,
    )

    print()
    print(
        f"Mask Loss:     {mask.item():.6f}"
    )

    print(
        f"Spine Loss:    {spine.item():.6f}"
    )

    print(
        f"Topology Loss: {topology.item():.6f}"
    )

    print(
        f"Total Loss:    {total.item():.6f}"
    )

    total.backward()

    print()
    print(
        "Gradient check:"
    )

    print(
        "Mask gradient:",
        mask_prediction.grad is not None,
    )

    print(
        "Spine gradient:",
        spine_prediction.grad is not None,
    )

    print()
    print(
        "UNET++ + SPINE + TOPOLOGY LOSS "
        "WORKS SUCCESSFULLY"
    )

    print("=" * 70)