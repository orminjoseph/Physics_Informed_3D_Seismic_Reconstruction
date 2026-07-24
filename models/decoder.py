"""
=========================================================
3D Decoder
=========================================================
"""

import torch
import torch.nn as nn

from models.blocks import UpBlock3D


class Decoder3D(nn.Module):
    """
    Attention-guided 3D Decoder
    """

    def __init__(self):

        super().__init__()

        self.up4 = UpBlock3D(
            512,
            256,
            256
        )

        self.up3 = UpBlock3D(
            256,
            128,
            128
        )

        self.up2 = UpBlock3D(
            128,
            64,
            64
        )

        self.up1 = UpBlock3D(
            64,
            32,
            32
        )

    def forward(
        self,
        x1,
        x2,
        x3,
        x4,
        bottleneck
    ):

        d4 = self.up4(
            bottleneck,
            x4
        )

        d3 = self.up3(
            d4,
            x3
        )

        d2 = self.up2(
            d3,
            x2
        )

        d1 = self.up1(
            d2,
            x1
        )

        return d1