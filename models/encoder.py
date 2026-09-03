"""
=========================================================
3D Encoder
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Tensor convention:

    [B, C, D, H, W]

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from models.blocks import (
    DoubleConv3D,
    ResidualBlock3D,
    DownBlock3D
)


class Encoder3D(nn.Module):
    """
    3D Encoder for seismic-volume feature extraction.

    Architecture:

        Input
          ↓
        Initial Feature Extraction
          ↓
        Down 1
          ↓
        Down 2
          ↓
        Down 3
          ↓
        Down 4

    For an input:

        [B, 1, 64, 128, 128]

    the expected feature hierarchy is:

        x1 = [B,  32, 64, 128, 128]
        x2 = [B,  64, 32,  64,  64]
        x3 = [B, 128, 16,  32,  32]
        x4 = [B, 256,  8,  16,  16]
        x5 = [B, 512,  4,   8,   8]
    """

    def __init__(
        self,
        in_channels=1,
        use_residual=True
    ):

        super().__init__()

        self.use_residual = use_residual

        # =================================================
        # INITIAL FEATURE EXTRACTION
        # =================================================

        if self.use_residual:

            self.initial = nn.Sequential(

                DoubleConv3D(
                    in_channels,
                    32
                ),

                ResidualBlock3D(
                    32
                )
            )

        else:

            self.initial = nn.Sequential(

                DoubleConv3D(
                    in_channels,
                    32
                )
            )

        # =================================================
        # ENCODER STAGE 1
        # =================================================

        self.down1 = DownBlock3D(
            in_channels=32,
            out_channels=64,
            use_residual=self.use_residual
        )

        # =================================================
        # ENCODER STAGE 2
        # =================================================

        self.down2 = DownBlock3D(
            in_channels=64,
            out_channels=128,
            use_residual=self.use_residual
        )

        # =================================================
        # ENCODER STAGE 3
        # =================================================

        self.down3 = DownBlock3D(
            in_channels=128,
            out_channels=256,
            use_residual=self.use_residual
        )

        # =================================================
        # ENCODER STAGE 4
        # =================================================

        self.down4 = DownBlock3D(
            in_channels=256,
            out_channels=512,
            use_residual=self.use_residual
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        x: torch.Tensor
    ):

        # -------------------------------------------------
        # Initial features
        # -------------------------------------------------

        x1 = self.initial(x)

        # -------------------------------------------------
        # Encoder level 1
        # -------------------------------------------------

        x2 = self.down1(x1)

        # -------------------------------------------------
        # Encoder level 2
        # -------------------------------------------------

        x3 = self.down2(x2)

        # -------------------------------------------------
        # Encoder level 3
        # -------------------------------------------------

        x4 = self.down3(x3)

        # -------------------------------------------------
        # Encoder level 4
        # -------------------------------------------------

        x5 = self.down4(x4)

        return (
            x1,
            x2,
            x3,
            x4,
            x5
        )