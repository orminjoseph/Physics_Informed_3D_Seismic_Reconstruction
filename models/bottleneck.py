"""
=========================================================
Research Bottleneck
=========================================================
"""

import torch.nn as nn

from models.blocks import (
    DoubleConv3D,
    ResidualBlock3D
)


class Bottleneck3D(nn.Module):

    def __init__(self, channels=512, dropout=0.3):

        super().__init__()

        self.double_conv = DoubleConv3D(
            channels,
            channels
        )

        self.residual1 = ResidualBlock3D(
            channels
        )

        self.dilated = nn.Sequential(

            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),

            nn.BatchNorm3d(channels),

            nn.ReLU(inplace=True)

        )

        self.residual2 = ResidualBlock3D(
            channels
        )

        self.dropout = nn.Dropout3d(
            p=dropout
        )

    def forward(self, x):

        x = self.double_conv(x)

        x = self.residual1(x)

        x = self.dilated(x)

        x = self.residual2(x)

        x = self.dropout(x)

        return x