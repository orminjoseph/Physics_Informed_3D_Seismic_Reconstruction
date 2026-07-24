"""
=========================================================
Complete Physics-Informed 3D Encoder–Decoder Network
=========================================================

Project:
Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from models.encoder import Encoder3D
from models.bottleneck import Bottleneck3D
from models.decoder import Decoder3D


class Network3D(nn.Module):
    """
    Complete 3D Encoder–Decoder Network
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=1
    ):

        super().__init__()

        # ------------------------------------------
        # Encoder
        # ------------------------------------------

        self.encoder = Encoder3D(
            in_channels=in_channels
        )

        # ------------------------------------------
        # Bottleneck
        # ------------------------------------------

        self.bottleneck = Bottleneck3D(
            channels=512
        )

        # ------------------------------------------
        # Decoder
        # ------------------------------------------

        self.decoder = Decoder3D()

        # ------------------------------------------
        # Reconstruction Head
        # ------------------------------------------

        self.reconstruction_head = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )
        # ------------------------------------------
        # Predictive uncertainty head
        # ------------------------------------------

        self.uncertainty_head = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(
            self,
            x
    ):

        # ------------------------------------------
        # Encoder
        # ------------------------------------------

        x1, x2, x3, x4, x5 = self.encoder(x)

        # ------------------------------------------
        # Bottleneck
        # ------------------------------------------

        bottleneck_output = self.bottleneck(x5)

        # ------------------------------------------
        # Decoder
        # ------------------------------------------

        decoder_output = self.decoder(

            x1,

            x2,

            x3,

            x4,

            bottleneck_output

        )

        # ------------------------------------------
        # Reconstruction
        # ------------------------------------------

        reconstructed_cube = self.reconstruction_head(
            decoder_output
        )

        # ------------------------------------------
        # Predictive uncertainty
        # ------------------------------------------

        log_variance = self.uncertainty_head(
            decoder_output
        )

        return reconstructed_cube, log_variance
