"""
=========================================================
3D Attention Gate
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""

import torch
import torch.nn as nn


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate
    """

    def __init__(self, encoder_channels, decoder_channels, inter_channels):

        super().__init__()

        self.theta_x = nn.Sequential(
            nn.Conv3d(
                encoder_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm3d(inter_channels)
        )

        self.phi_g = nn.Sequential(
            nn.Conv3d(
                decoder_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(
                inter_channels,
                1,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
            self,
            encoder_feature,
            decoder_feature
    ):
        theta = self.theta_x(
            encoder_feature
        )

        phi = self.phi_g(
            decoder_feature
        )

        attention = self.relu(
            theta + phi
        )

        # Attention coefficients

        attention = self.psi(
            attention
        )

        # Gate encoder features

        output = encoder_feature * attention

        return output
