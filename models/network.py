"""
Complete Physics-Informed 3D Encoder–Decoder Network

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction
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
            out_channels=1,
            use_uncertainty=True,
            use_residual=True,
            use_attention=True
    ):

        super().__init__()

        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty

        # Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            use_residual=use_residual
        )

        # Bottleneck
        self.bottleneck = Bottleneck3D(
            channels=512,
            use_residual=use_residual
        )

        # Decoder
        self.decoder = Decoder3D(
            use_attention=use_attention,
            use_residual=use_residual
        )

        # Reconstruction head
        self.reconstruction_head = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

        # Uncertainty head
        if self.use_uncertainty:

            self.uncertainty_head = nn.Conv3d(
                32,
                out_channels,
                kernel_size=1
            )

    def forward(self, x):

        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)

        # Bottleneck
        bottleneck_output = self.bottleneck(x5)

        # Decoder
        decoder_output = self.decoder(
            x1,
            x2,
            x3,
            x4,
            bottleneck_output
        )

        # Reconstruction
        reconstructed_cube = self.reconstruction_head(
            decoder_output
        )

        # Predictive uncertainty
        if self.use_uncertainty:

            log_variance = self.uncertainty_head(
                decoder_output
            )

        else:

            log_variance = torch.zeros_like(
                reconstructed_cube
            )

        return reconstructed_cube, log_variance