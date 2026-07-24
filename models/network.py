"""
=========================================================
Physics-Informed 3D Encoder–Decoder Framework
=========================================================

Complete Network
=========================================================
"""

import torch
import torch.nn as nn

from models.encoder import Encoder3D
from models.bottleneck import Bottleneck3D
from models.decoder import Decoder3D


class PhysicsInformed3DUNet(nn.Module):
    """
    Physics-Informed 3D Encoder–Decoder Framework

    Outputs
    -------
    reconstruction
    log_variance
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1
    ):

        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels
        )

        self.bottleneck = Bottleneck3D()

        self.decoder = Decoder3D()

        # --------------------------------------------------
        # Reconstruction Head
        # --------------------------------------------------

        self.reconstruction_head = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

        # --------------------------------------------------
        # Predictive Uncertainty Head
        # --------------------------------------------------

        self.uncertainty_head = nn.Sequential(

            nn.Conv3d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                16,
                out_channels,
                kernel_size=1
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ):

        x1, x2, x3, x4, x5 = self.encoder(x)

        latent = self.bottleneck(x5)

        features = self.decoder(
            x1,
            x2,
            x3,
            x4,
            latent
        )

        # -------------------------------
        # Reconstruction
        # -------------------------------

        reconstruction = self.reconstruction_head(
            features
        )

        # -------------------------------
        # Predictive Uncertainty
        # -------------------------------

        raw_log_variance = self.uncertainty_head(
            features
        )

        # Smoothly bound log-variance
        log_variance = 5.0 * torch.tanh(
            raw_log_variance / 5.0
        )

        return reconstruction, log_variance