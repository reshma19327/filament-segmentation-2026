import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC CONVOLUTION BLOCK
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# LIGHTWEIGHT UNET++ + SPINE
# ============================================================

class UNetPPSpine(nn.Module):

    def __init__(self, in_channels=1, base_channels=16):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.x00 = ConvBlock(
            in_channels,
            c1
        )

        self.x10 = ConvBlock(
            c1,
            c2
        )

        self.x20 = ConvBlock(
            c2,
            c3
        )

        self.x30 = ConvBlock(
            c3,
            c4
        )

        # ----------------------------------------------------
        # UNet++ nested decoder
        # ----------------------------------------------------

        self.x01 = ConvBlock(
            c1 + c2,
            c1
        )

        self.x11 = ConvBlock(
            c2 + c3,
            c2
        )

        self.x21 = ConvBlock(
            c3 + c4,
            c3
        )

        self.x02 = ConvBlock(
            c1 + c1 + c2,
            c1
        )

        self.x12 = ConvBlock(
            c2 + c2 + c3,
            c2
        )

        self.x03 = ConvBlock(
            c1 + c1 + c1 + c2,
            c1
        )

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Filament output
        # ----------------------------------------------------

        self.mask_head = nn.Conv2d(
            c1,
            1,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Spine output
        # ----------------------------------------------------

        self.spine_head = nn.Conv2d(
            c1,
            1,
            kernel_size=1
        )

    # ========================================================
    # UPSAMPLE
    # ========================================================

    def upsample_to(self, x, target):

        return F.interpolate(
            x,
            size=target.shape[2:],
            mode="bilinear",
            align_corners=False
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x00 = self.x00(x)

        x10 = self.x10(
            self.pool(x00)
        )

        x20 = self.x20(
            self.pool(x10)
        )

        x30 = self.x30(
            self.pool(x20)
        )

        # ----------------------------------------------------
        # UNet++ node x01
        # ----------------------------------------------------

        x01_input = torch.cat(
            [
                x00,
                self.upsample_to(x10, x00)
            ],
            dim=1
        )

        x01 = self.x01(
            x01_input
        )

        # ----------------------------------------------------
        # UNet++ node x11
        # ----------------------------------------------------

        x11_input = torch.cat(
            [
                x10,
                self.upsample_to(x20, x10)
            ],
            dim=1
        )

        x11 = self.x11(
            x11_input
        )

        # ----------------------------------------------------
        # UNet++ node x21
        # ----------------------------------------------------

        x21_input = torch.cat(
            [
                x20,
                self.upsample_to(x30, x20)
            ],
            dim=1
        )

        x21 = self.x21(
            x21_input
        )

        # ----------------------------------------------------
        # UNet++ node x02
        # ----------------------------------------------------

        x02_input = torch.cat(
            [
                x00,
                x01,
                self.upsample_to(x11, x00)
            ],
            dim=1
        )

        x02 = self.x02(
            x02_input
        )

        # ----------------------------------------------------
        # UNet++ node x12
        # ----------------------------------------------------

        x12_input = torch.cat(
            [
                x10,
                x11,
                self.upsample_to(x21, x10)
            ],
            dim=1
        )

        x12 = self.x12(
            x12_input
        )

        # ----------------------------------------------------
        # UNet++ node x03
        # ----------------------------------------------------

        x03_input = torch.cat(
            [
                x00,
                x01,
                x02,
                self.upsample_to(x12, x00)
            ],
            dim=1
        )

        x03 = self.x03(
            x03_input
        )

        # ----------------------------------------------------
        # Outputs
        # ----------------------------------------------------

        mask_output = self.mask_head(
            x03
        )

        spine_output = self.spine_head(
            x03
        )

        # Ensure exact input spatial size
        mask_output = F.interpolate(
            mask_output,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        spine_output = F.interpolate(
            spine_output,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return mask_output, spine_output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNET++ + SPINE MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = UNetPPSpine(
        in_channels=1,
        base_channels=16
    ).to(device)

    # Small test batch
    x = torch.randn(
        1,
        1,
        256,
        256,
        device=device
    )

    with torch.no_grad():

        mask_output, spine_output = model(x)

    print("-" * 60)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Mask output:",
        mask_output.shape
    )

    print(
        "Spine output:",
        spine_output.shape
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("-" * 60)

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print("-" * 60)
    print("UNET++ + SPINE MODEL CREATED SUCCESSFULLY")
    print("=" * 60)