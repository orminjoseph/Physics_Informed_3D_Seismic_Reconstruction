"""
=========================================================
3D Building Blocks
=========================================================

Project:
Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Reusable 3D convolutional building blocks for the
Physics-Informed 3D Encoder–Decoder framework.

The architecture contains:

    1. DoubleConv3D
    2. ResidualBlock3D
    3. DownBlock3D
    4. UpBlock3D
    5. Attention-guided skip connections

Predictive uncertainty
----------------------

Dropout3D is retained inside the residual blocks.

This is intentional because the framework will later
use Monte Carlo Dropout to estimate epistemic uncertainty.

Aleatoric uncertainty is handled separately by the
network's log-variance output head.

Tensor convention
-----------------

All tensors use:

    [B, C, D, H, W]

where:

    B = batch
    C = channel
    D = depth
    H = crossline
    W = inline

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from models.attention import AttentionGate3D


# =========================================================
# 1. DOUBLE CONVOLUTION BLOCK
# =========================================================

class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolution operations.

    Structure:

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

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.

    out_channels : int
        Number of output feature channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.double_conv = nn.Sequential(

            # -------------------------------------------------
            # First convolution
            # -------------------------------------------------

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            # -------------------------------------------------
            # Second convolution
            # -------------------------------------------------

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.double_conv(x)


# =========================================================
# 2. RESIDUAL BLOCK
# =========================================================

class ResidualBlock3D(nn.Module):
    """
    Residual 3D convolutional block.

    Structure:

        Input
          │
          ├──────────── Identity ────────────┐
          │                                  │
          ↓                                  │
        Conv3D                               │
          ↓                                  │
        BatchNorm                            │
          ↓                                  │
        ReLU                                 │
          ↓                                  │
        Dropout3D                             │
          ↓                                  │
        Conv3D                               │
          ↓                                  │
        BatchNorm                            │
          │                                  │
          └──────────── Addition ─────────────┘
                         ↓
                       ReLU

    Important
    ---------

    Dropout3D is deliberately retained.

    It will later support Monte Carlo Dropout for
    epistemic uncertainty estimation.

    During normal training:
        Dropout = active

    During ordinary evaluation:
        Dropout = inactive

    During MC inference:
        Dropout = deliberately activated.
    """

    def __init__(
        self,
        channels,
        dropout_probability=0.20
    ):

        super().__init__()

        if channels < 1:
            raise ValueError(
                "channels must be greater than zero."
            )

        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "dropout_probability must be "
                "in the range [0,1)."
            )

        # -------------------------------------------------
        # First convolution
        # -------------------------------------------------

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(
            channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        # -------------------------------------------------
        # Dropout
        # -------------------------------------------------

        self.dropout = nn.Dropout3d(
            p=dropout_probability
        )

        # -------------------------------------------------
        # Second convolution
        # -------------------------------------------------

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(
            channels
        )

    def forward(self, x):

        # -------------------------------------------------
        # Identity branch
        # -------------------------------------------------

        identity = x

        # -------------------------------------------------
        # Residual branch
        # -------------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)

        out = self.bn2(out)

        # -------------------------------------------------
        # Residual addition
        # -------------------------------------------------

        out = out + identity

        out = self.relu(out)

        return out


# =========================================================
# 3. DOWN-SAMPLING BLOCK
# =========================================================

class DownBlock3D(nn.Module):
    """
    3D encoder down-sampling block.

    Structure:

        MaxPool3D
            ↓
        DoubleConv3D
            ↓
        ResidualBlock3D (optional)

    Parameters
    ----------
    in_channels : int
        Input feature channels.

    out_channels : int
        Output feature channels.

    use_residual : bool
        Whether to use the residual block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_residual=True
    ):

        super().__init__()

        self.use_residual = use_residual

        # -------------------------------------------------
        # Spatial down-sampling
        # -------------------------------------------------

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -------------------------------------------------
        # Feature extraction
        # -------------------------------------------------

        self.conv = DoubleConv3D(
            in_channels,
            out_channels
        )

        # -------------------------------------------------
        # Optional residual refinement
        # -------------------------------------------------

        if self.use_residual:

            self.residual = ResidualBlock3D(
                out_channels
            )

    def forward(self, x):

        x = self.pool(x)

        x = self.conv(x)

        if self.use_residual:

            x = self.residual(x)

        return x


# =========================================================
# 4. UP-SAMPLING BLOCK
# =========================================================

class UpBlock3D(nn.Module):
    """
    Attention-guided 3D decoder up-sampling block.

    Structure:

        Decoder feature
              ↓
        Transposed Conv3D
              ↓
        Attention Gate ← Encoder skip feature
              ↓
        Concatenation
              ↓
        DoubleConv3D
              ↓
        ResidualBlock3D (optional)

    Parameters
    ----------
    in_channels : int
        Number of decoder input channels.

    skip_channels : int
        Number of channels in the encoder skip feature.

    out_channels : int
        Number of output channels.

    use_attention : bool
        Enable attention-gated skip connections.

    use_residual : bool
        Enable residual refinement.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_attention=True,
        use_residual=True
    ):

        super().__init__()

        self.use_attention = use_attention
        self.use_residual = use_residual

        # -------------------------------------------------
        # Transposed convolution
        # -------------------------------------------------

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # -------------------------------------------------
        # Attention gate
        # -------------------------------------------------

        if self.use_attention:

            self.attention = AttentionGate3D(
                encoder_channels=skip_channels,
                decoder_channels=out_channels,
                inter_channels=max(
                    out_channels // 2,
                    1
                )
            )

        # -------------------------------------------------
        # Feature fusion
        # -------------------------------------------------

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

        # -------------------------------------------------
        # Optional residual refinement
        # -------------------------------------------------

        if self.use_residual:

            self.residual = ResidualBlock3D(
                out_channels
            )

    def forward(
        self,
        decoder_feature,
        encoder_feature
    ):

        # -------------------------------------------------
        # Upsample decoder feature
        # -------------------------------------------------

        decoder_feature = self.up(
            decoder_feature
        )

        # -------------------------------------------------
        # Attention-guided skip connection
        # -------------------------------------------------

        if self.use_attention:

            encoder_feature = self.attention(
                encoder_feature,
                decoder_feature
            )

        # -------------------------------------------------
        # Concatenate decoder and encoder features
        # -------------------------------------------------

        x = torch.cat(
            [
                encoder_feature,
                decoder_feature
            ],
            dim=1
        )

        # -------------------------------------------------
        # Feature refinement
        # -------------------------------------------------

        x = self.conv(x)

        # -------------------------------------------------
        # Optional residual refinement
        # -------------------------------------------------

        if self.use_residual:

            x = self.residual(x)

        return x