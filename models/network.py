"""
=========================================================
Complete Physics-Informed 3D Encoder–Decoder Network
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Network outputs
---------------

1. reconstructed_cube
      Reconstructed seismic volume.

2. travel_time
      Predicted seismic travel-time field T(x,y,z)
      used by the 3D Eikonal physics loss.

3. log_variance
      Predicted logarithmic variance used for
      predictive uncertainty estimation.

Governing physics
-----------------

The Eikonal equation is

    |∇T|² = 1 / V²

where

    T = seismic travel time [s]
    V = P-wave velocity [m/s]

Tensor convention
-----------------

Input:

    [B, C, D, H, W]

Outputs:

    reconstructed_cube:
        [B, C, D, H, W]

    travel_time:
        [B, C, D, H, W]

    log_variance:
        [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from models.encoder import Encoder3D
from models.bottleneck import Bottleneck3D
from models.decoder import Decoder3D

from utils.config import TRAVEL_TIME_SCALE


class Network3D(nn.Module):
    """
    Physics-Informed 3D Encoder-Decoder Network.

    Architecture
    ------------

        Input seismic volume
                |
                v
        +---------------+
        | 3D Encoder    |
        +---------------+
                |
                v
        +---------------+
        | Bottleneck    |
        +---------------+
                |
                v
        +---------------+
        | 3D Decoder    |
        +---------------+
                |
        +-------+-------+-------+
        |               |       |
        v               v       v
    Reconstruction   Travel-T  Uncertainty
       Head            Head       Head
        |               |         |
        v               v         v
    Seismic volume   T(x,y,z)  log(sigma^2)

    The physical P-wave velocity model is supplied
    externally by the dataset and is NOT predicted
    by this network.
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

        # =================================================
        # STORE OPTIONS
        # =================================================

        self.use_uncertainty = use_uncertainty
        self.use_residual = use_residual
        self.use_attention = use_attention

        # =================================================
        # ENCODER
        # =================================================

        self.encoder = Encoder3D(
            in_channels=in_channels,
            use_residual=use_residual
        )

        # =================================================
        # BOTTLENECK
        # =================================================

        self.bottleneck = Bottleneck3D(
            channels=512,
            use_residual=use_residual
        )

        # =================================================
        # DECODER
        # =================================================

        self.decoder = Decoder3D(
            use_attention=use_attention,
            use_residual=use_residual
        )

        # =================================================
        # RECONSTRUCTION HEAD
        # =================================================
        #
        # Decoder output:
        #
        #     [B, 32, D, H, W]
        #
        # Reconstruction output:
        #
        #     [B, 1, D, H, W]
        #
        # No activation is used because seismic amplitudes
        # may contain both positive and negative values.
        # =================================================

        self.reconstruction_head = nn.Conv3d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # =================================================
        # TRAVEL-TIME HEAD
        # =================================================
        #
        # Predicts:
        #
        #     T(x,y,z)
        #
        # The Eikonal equation differentiates T spatially:
        #
        #     |∇T|² = 1 / V²
        #
        # Therefore, the initial travel-time prediction must
        # not contain excessively large spatial variations.
        #
        # A deliberately small initialization is used here
        # to prevent the randomly initialized travel-time head
        # from producing unrealistically large gradients.
        # =================================================

        self.travel_time_head = nn.Conv3d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # -------------------------------------------------
        # Physics-aware initialization
        # -------------------------------------------------
        #
        # Standard Conv3D initialization produced travel-time
        # gradients that were much larger than the physical
        # Eikonal scale during the numerical audit.
        #
        # A small initialization reduces this initial
        # gradient magnitude while keeping the parameters
        # fully trainable.
        # -------------------------------------------------

        nn.init.normal_(
            self.travel_time_head.weight,
            mean=0.0,
            std=1.0e-3
        )

        nn.init.constant_(
            self.travel_time_head.bias,
            0.0
        )

        # =================================================
        # TRAVEL-TIME ACTIVATION
        # =================================================
        #
        # Softplus provides a smooth positive-valued
        # travel-time representation.
        #
        # Softplus is preferable to ReLU here because its
        # derivative is smooth, which is important because
        # the Eikonal loss computes spatial derivatives of T.
        # =================================================

        self.travel_time_activation = nn.Softplus(
            beta=1.0,
            threshold=20.0
        )

        # =================================================
        # UNCERTAINTY HEAD
        # =================================================
        #
        # Predicts:
        #
        #     log(sigma^2)
        #
        # Log variance must remain unrestricted.
        #
        # Therefore, no Softplus, ReLU or Sigmoid is applied.
        # =================================================

        if self.use_uncertainty:

            self.uncertainty_head = nn.Conv3d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _validate_input(x):
        """
        Validate the input seismic tensor.
        """

        if not isinstance(x, torch.Tensor):

            raise TypeError(
                "Network3D input must be a torch.Tensor."
            )

        if x.ndim != 5:

            raise ValueError(
                "Network3D expects input with shape "
                "[B, C, D, H, W]. "
                f"Received: {tuple(x.shape)}"
            )

        if not torch.isfinite(x).all():

            raise ValueError(
                "Network3D input contains NaN or Inf values."
            )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor

            Incomplete seismic volume.

            Shape:
                [B, C, D, H, W]

        Returns
        -------
        reconstructed_cube : torch.Tensor

            Reconstructed seismic volume.

        travel_time : torch.Tensor

            Predicted non-negative travel-time field.

        log_variance : torch.Tensor

            Predicted logarithmic variance.
        """

        # =================================================
        # VALIDATE INPUT
        # =================================================

        self._validate_input(x)

        # =================================================
        # ENCODER
        # =================================================

        x1, x2, x3, x4, x5 = self.encoder(x)

        # =================================================
        # BOTTLENECK
        # =================================================

        bottleneck_output = self.bottleneck(
            x5
        )

        # =================================================
        # DECODER
        # =================================================

        decoder_output = self.decoder(
            x1,
            x2,
            x3,
            x4,
            bottleneck_output
        )

        # =================================================
        # RECONSTRUCTION HEAD
        # =================================================

        reconstructed_cube = (
            self.reconstruction_head(
                decoder_output
            )
        )

        # =================================================
        # TRAVEL-TIME HEAD
        # =================================================

        raw_travel_time = (
            self.travel_time_head(
                decoder_output
            )
        )

        # =================================================
        # POSITIVE TRAVEL-TIME REPRESENTATION
        # =================================================

        normalized_travel_time = (
            self.travel_time_activation(
                raw_travel_time
            )
        )

        # =================================================
        # PHYSICAL TRAVEL-TIME SCALING
        # =================================================
        #
        # The network predicts a dimensionless positive
        # quantity which is converted into seconds using
        # TRAVEL_TIME_SCALE.
        # =================================================

        travel_time = (
            TRAVEL_TIME_SCALE
            *
            normalized_travel_time
        )

        # =================================================
        # UNCERTAINTY HEAD
        # =================================================

        if self.use_uncertainty:

            log_variance = (
                self.uncertainty_head(
                    decoder_output
                )
            )

        else:

            log_variance = torch.zeros_like(
                reconstructed_cube
            )

        # =================================================
        # OUTPUT VALIDATION
        # =================================================

        if reconstructed_cube.shape != x.shape:

            raise RuntimeError(
                "Reconstruction output shape does not "
                "match input shape. "
                f"Input: {tuple(x.shape)}, "
                f"Output: {tuple(reconstructed_cube.shape)}"
            )

        if travel_time.shape != x.shape:

            raise RuntimeError(
                "Travel-time output shape does not "
                "match input shape. "
                f"Input: {tuple(x.shape)}, "
                f"Output: {tuple(travel_time.shape)}"
            )

        if log_variance.shape != x.shape:

            raise RuntimeError(
                "Uncertainty output shape does not "
                "match input shape. "
                f"Input: {tuple(x.shape)}, "
                f"Output: {tuple(log_variance.shape)}"
            )

        # =================================================
        # RETURN THREE NETWORK OUTPUTS
        # =================================================

        return (
            reconstructed_cube,
            travel_time,
            log_variance
        )