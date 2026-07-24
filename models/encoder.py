"""
=========================================================
3D Encoder
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""

import torch.nn as nn

from models.blocks import (
    DoubleConv3D,
    ResidualBlock3D,
    DownBlock3D
)
import torch

class Encoder3D(nn.Module):

    """
    3D Encoder
    """

    def __init__(self, in_channels=1):

        super().__init__()

        # Initial feature extraction

        self.initial = nn.Sequential(

            DoubleConv3D(
                in_channels,
                32
            ),

            ResidualBlock3D(
                32
            )
        )

        # Encoder stages

        self.down1 = DownBlock3D(
            32,
            64
        )

        self.down2 = DownBlock3D(
            64,
            128
        )

        self.down3 = DownBlock3D(
            128,
            256
        )

        self.down4 = DownBlock3D(
            256,
            512
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        # Initial feature extraction

        x1 = self.initial(x)

        # Encoder Level 1

        x2 = self.down1(x1)

        # Encoder Level 2

        x3 = self.down2(x2)

        # Encoder Level 3

        x4 = self.down3(x3)

        # Encoder Level 4

        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5