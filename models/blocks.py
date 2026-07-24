"""
=========================================================
3D Building Blocks
=========================================================

Project:
Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Author:
Ormin Joseph

Description:
Reusable building blocks for the 3D neural network.
=========================================================
"""


import torch
import torch.nn as nn
from models.attention import AttentionGate3D


class DoubleConv3D(nn.Module):
    """
    Double 3D Convolution Block

    Conv3D
        ↓
    BatchNorm3D
        ↓
    ReLU
        ↓
    Conv3D
        ↓
    BatchNorm3D
        ↓
    ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        """
        Forward propagation.
        """
        return self.double_conv(x)


class ResidualBlock3D(nn.Module):
        """
        =====================================================
        Residual Block for 3D Seismic Feature Learning

        Input
            │
        Conv3D
            │
        BatchNorm
            │
        ReLU
            │
        Conv3D
            │
        BatchNorm
            │
            +  Identity Connection
            │
        ReLU
            │
          Output
        =====================================================
        """

        def __init__(self, channels):
            super().__init__()

            self.conv1 = nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            )

            self.bn1 = nn.BatchNorm3d(channels)

            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            )

            self.bn2 = nn.BatchNorm3d(channels)

        def forward(self, x):
            identity = x

            out = self.conv1(x)

            out = self.bn1(out)

            out = self.relu(out)

            out = self.conv2(out)

            out = self.bn2(out)


            # Identity (skip) connection

            out += identity

            out = self.relu(out)

            return out


class DownBlock3D(nn.Module):
    """
    =====================================================
    Down-sampling Block

    MaxPool3D
         ↓
    DoubleConv3D
         ↓
    ResidualBlock3D
    =====================================================
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv3D(
            in_channels,
            out_channels
        )

        self.residual = ResidualBlock3D(
            out_channels
        )

    def forward(self, x):

        x = self.pool(x)

        x = self.conv(x)

        x = self.residual(x)

        return x




class UpBlock3D(nn.Module):
    """
    =========================================================
    Attention-guided Up-sampling Block
    =========================================================

    Decoder Feature
            │
    Transposed Convolution
            │
    Attention Gate
            │
    Concatenate
            │
    DoubleConv3D
            │
    ResidualBlock3D

    =========================================================
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.attention = AttentionGate3D(
            encoder_channels=skip_channels,
            decoder_channels=out_channels,
            inter_channels=out_channels // 2
        )

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

        self.residual = ResidualBlock3D(
            out_channels
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        encoder_feature: torch.Tensor
    ) -> torch.Tensor:

        decoder_feature = self.up(decoder_feature)

        encoder_feature = self.attention(
            encoder_feature,
            decoder_feature
        )

        x = torch.cat(
            [encoder_feature, decoder_feature],
            dim=1
        )

        x = self.conv(x)

        x = self.residual(x)

        return x