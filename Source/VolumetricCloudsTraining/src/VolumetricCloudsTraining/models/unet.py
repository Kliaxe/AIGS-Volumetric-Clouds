from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# What is U-Net?
# - An encoder–decoder CNN with skip connections between matching
#   resolution levels. The encoder compresses spatial information
#   into deep features; the decoder restores resolution while reusing
#   fine details from the encoder via skip connections.
#
# Why use it for super-resolution/refinement?
# - We want global context (from deep features) and crisp local detail
#   (from early layers). U-Net fuses both through the skips.
#
# Tensor layout
# - PyTorch convention is channels-first: [B, C, H, W].
# - Convs here use kernel_size=3, padding=1 to keep H and W unchanged.
#
# Residual learning
# - If learn_residual=True, the network predicts a residual that is
#   added to the input and clamped to [0,1]. This is helpful when the
#   target is a refined version of the input (identity shortcut).
# ================================================================


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    Rationale:
    - Two 3x3 layers provide more non-linearity than a single 5x5,
      often at comparable parameter cost.
    - BatchNorm stabilizes training and helps with scale.
    - ReLU introduces non-linearity after each conv.
    Output: same spatial size (padding=1), channels -> out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """
    Downsampling block = MaxPool(2x2) + DoubleConv.
    - MaxPool halves H and W (keeps channels).
    - DoubleConv increases representational capacity at the new scale.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsampling block = Upsample/ConvTranspose + skip concat + DoubleConv.
    - Upsample brings features to higher resolution (2x in H and W).
    - We concatenate with encoder features of the same spatial size
      (the "skip connection") to re-inject fine details.
    - DoubleConv then fuses and refines these features.

    bilinear=True:
      Use non-learned bilinear upsampling followed by DoubleConv.
      This avoids checkerboard artifacts and is inexpensive.
    bilinear=False:
      Use transposed convolution (learned upsampling).
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: decoder feature to be upsampled
        x2: encoder feature (skip) at target resolution
        Returns: fused feature map at the higher resolution.
        """
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # Pad x1 to same size as x2
        # (Handles odd spatial sizes so concatenation succeeds.)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # Channel-wise concatenation of skip (x2) and upsampled decoder (x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution to collapse channels to the desired output.
    - Spatial size is preserved, only channels are mixed/reshaped.
    - For RGB image output, out_channels=3.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@dataclass
class UNetConfig:
    """
    Configuration for UNet:
    - in_channels: number of channels in the input image (e.g., 3 for RGB)
    - out_channels: number of channels to predict (e.g., 3 for RGB)
    - base_channels: width (capacity) of the network. Higher -> more params.
    - bilinear: choose bilinear upsampling vs transposed convolution.
    - learn_residual: if True, return clamp(input + prediction, 0..1).
    """
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    bilinear: bool = True
    learn_residual: bool = True


class UNet(nn.Module):
    """
    Standard U-Net. If learn_residual=True, the network learns a residual
    added to the input (useful for refining upsampled inputs).
    """

    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        self._config = config or UNetConfig()
        b = self._config.base_channels
        bilinear = self._config.bilinear

        # Encoder (down path): progressively reduce spatial resolution,
        # increase channels to capture global context.
        self.inc = DoubleConv(self._config.in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(b * 8, b * 16 // factor)

        # Decoder (up path): progressively restore spatial resolution
        # and fuse with encoder features (skip connections) to recover details.
        self.up1 = Up(b * 16, b * 8 // factor, bilinear)
        self.up2 = Up(b * 8, b * 4 // factor, bilinear)
        self.up3 = Up(b * 4, b * 2 // factor, bilinear)
        self.up4 = Up(b * 2, b, bilinear)

        # Output head maps final features to desired channel count.
        self.outc = OutConv(b, self._config.out_channels)

    # ----------------- Getters / Setters -----------------
    @property
    def config(self) -> UNetConfig:
        return self._config

    @config.setter
    def config(self, new_config: UNetConfig) -> None:
        self._config = new_config

    # ----------------- Forward -----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward overview:
        1) Encode input to multi-scale features (x1..x5).
        2) Decode with skip connections to combine context + detail.
        3) Produce an output image with OutConv.
        4) Optionally add residual (identity shortcut) and clamp to [0,1].

        Shape sketch (ignoring batch dimension B):
        - Input x:  [C,  H,    W   ]
        - x1:       [b,  H,    W   ]
        - x2:       [2b, H/2,  W/2 ]
        - x3:       [4b, H/4,  W/4 ]
        - x4:       [8b, H/8,  W/8 ]
        - x5:       [16b/f, H/16, W/16]  (f = 2 if bilinear upsampling)
        After ups and concats we return to [b, H, W] then map to out_channels.
        """
        x_in = x
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred = self.outc(x)

        if self._config.learn_residual:
            # Residual refinement: output = clamp(input + residual, [0,1])
            return torch.clamp(pred + x_in, 0.0, 1.0)
        return pred



