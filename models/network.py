"""
=========================================================
Network3D
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Architecture
------------

    Input Seismic Cube
          |
          v
      3D Encoder
          |
          +---- x1
          +---- x2
          +---- x3
          +---- x4
          +---- x5
                    |
                    v
              3D Bottleneck
                    |
                    v
              3D Decoder
                    |
                    v
             Decoder Features
                    |
          +---------+---------+
          |         |         |
          v         v         v
    Reconstruction  Travel-   Aleatoric
        Head        Time Head  Uncertainty
                                  Head
          |           |             |
          v           v             v
      Seismic       Positive     log(sigma_a^2)
    Reconstruction Travel Time

Predictive uncertainty is estimated during inference:

    Predictive Variance
        =
    Aleatoric Variance
        +
    Epistemic Variance

where:

    Aleatoric Variance
        = mean(exp(log_variance_samples))

    Epistemic Variance
        = variance of MC-Dropout reconstruction samples.

Tensor convention
-----------------

    Input:
        [B, C, D, H, W]

    Outputs:
        Reconstruction:
            [B, 1, D, H, W]

        Travel time:
            [B, 1, D, H, W]

        Log variance:
            [B, 1, D, H, W]

=========================================================
Author:
Ormin Joseph
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
    Complete Physics-Informed 3D Encoder–Decoder Network.

    Parameters
    ----------
    in_channels : int
        Number of input seismic channels.

    out_channels : int
        Number of reconstructed seismic channels.

    use_attention : bool
        Enables attention gates in the decoder.

    use_residual : bool
        Enables residual connections.

    use_uncertainty : bool
        Enables the aleatoric uncertainty head.

    Notes
    -----
    The network always returns three outputs so that the
    training and inference interfaces remain consistent:

        reconstruction
        travel_time
        log_variance
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    ):

        super().__init__()

        # =================================================
        # Validate constructor arguments
        # =================================================

        if not isinstance(in_channels, int):
            raise TypeError(
                "in_channels must be an integer."
            )

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than zero."
            )

        if not isinstance(out_channels, int):
            raise TypeError(
                "out_channels must be an integer."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be greater than zero."
            )

        if not isinstance(use_attention, bool):
            raise TypeError(
                "use_attention must be a boolean."
            )

        if not isinstance(use_residual, bool):
            raise TypeError(
                "use_residual must be a boolean."
            )

        if not isinstance(use_uncertainty, bool):
            raise TypeError(
                "use_uncertainty must be a boolean."
            )

        # =================================================
        # Validate travel-time scaling
        # =================================================

        if not isinstance(
            TRAVEL_TIME_SCALE,
            (int, float)
        ):
            raise TypeError(
                "TRAVEL_TIME_SCALE must be numeric."
            )

        if TRAVEL_TIME_SCALE <= 0:
            raise ValueError(
                "TRAVEL_TIME_SCALE must be greater than zero."
            )

        # =================================================
        # Store configuration
        # =================================================

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_uncertainty = use_uncertainty

        # =================================================
        # 3D ENCODER
        # =================================================
        #
        # IMPORTANT:
        # Encoder3D does NOT accept use_attention.
        #
        # Attention is handled inside the decoder through
        # UpBlock3D.
        #
        # The encoder returns:
        #
        #     x1, x2, x3, x4, x5
        #
        # =================================================

        self.encoder = Encoder3D(
            in_channels=in_channels,
            use_residual=use_residual
        )

        # =================================================
        # 3D BOTTLENECK
        # =================================================
        #
        # The encoder's deepest feature x5 contains
        # 512 channels.
        #
        # Bottleneck3D therefore operates on 512 channels.
        #
        # dropout_probability defaults to 0.20 in the
        # existing Bottleneck3D implementation.
        #
        # =================================================

        self.bottleneck = Bottleneck3D(
            channels=512,
            use_residual=use_residual
        )

        # =================================================
        # 3D DECODER
        # =================================================
        #
        # Decoder3D expects:
        #
        #     x1
        #     x2
        #     x3
        #     x4
        #     bottleneck_output
        #
        # Attention and residual settings are handled
        # internally by the decoder's UpBlock3D modules.
        #
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
        # A 1x1x1 convolution maps these features to
        # the reconstructed seismic channel.
        #
        # No sigmoid or tanh activation is used because
        # seismic amplitudes are signed.
        #
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
        # Produces a scalar travel-time field at each voxel.
        #
        # Softplus is applied so that:
        #
        #     T(x,y,z) > 0
        #
        # =================================================

        self.travel_time_head = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Small initialization helps avoid excessively
        # large initial physics gradients.

        nn.init.normal_(
            self.travel_time_head.weight,
            mean=0.0,
            std=1e-3
        )

        nn.init.zeros_(
            self.travel_time_head.bias
        )

        # Positive travel-time activation.

        self.travel_time_activation = nn.Softplus(
            beta=1.0,
            threshold=20.0
        )

        # =================================================
        # ALEATORIC UNCERTAINTY HEAD
        # =================================================
        #
        # The head predicts:
        #
        #     log(sigma_a^2)
        #
        # rather than sigma_a^2 directly.
        #
        # The exponential transformation is performed in
        # the heteroscedastic uncertainty loss/evaluation.
        #
        # No activation is applied here.
        #
        # =================================================

        self.uncertainty_head = nn.Conv3d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # =================================================
        # UNCERTAINTY HEAD INITIALIZATION
        # =================================================
        #
        # Start with:
        #
        #     log(sigma_a^2) = 0
        #
        # Therefore:
        #
        #     sigma_a^2 = exp(0) = 1
        #
        # This gives a neutral deterministic initial
        # uncertainty state.
        #
        # It does NOT impose a fixed final uncertainty.
        # The uncertainty is learned during training.
        #
        # =================================================

        nn.init.zeros_(
            self.uncertainty_head.weight
        )

        nn.init.zeros_(
            self.uncertainty_head.bias
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(
        self,
        x: torch.Tensor
    ):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input seismic volume.

            Shape:

                [B, C, D, H, W]

        Returns
        -------
        reconstructed_cube : torch.Tensor
            Reconstructed seismic volume.

        travel_time : torch.Tensor
            Positive travel-time field.

        log_variance : torch.Tensor
            Log aleatoric variance.
        """

        # =================================================
        # Validate input
        # =================================================

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Network input must be a torch.Tensor."
            )

        if x.ndim != 5:
            raise ValueError(
                "Network input must have shape "
                "[B, C, D, H, W]. "
                f"Received: {tuple(x.shape)}"
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Network input contains NaN or Inf values."
            )

        # =================================================
        # ENCODER
        # =================================================
        #
        # Encoder3D returns:
        #
        #     x1
        #     x2
        #     x3
        #     x4
        #     x5
        #
        # x5 is the deepest feature representation.
        #
        # =================================================

        (
            x1,
            x2,
            x3,
            x4,
            x5
        ) = self.encoder(x)

        # =================================================
        # BOTTLENECK
        # =================================================

        bottleneck_output = self.bottleneck(
            x5
        )

        # =================================================
        # DECODER
        # =================================================
        #
        # The decoder receives the required skip
        # connections explicitly.
        #
        # =================================================

        decoder_output = self.decoder(
            x1,
            x2,
            x3,
            x4,
            bottleneck_output
        )

        # =================================================
        # RECONSTRUCTION
        # =================================================

        reconstructed_cube = (
            self.reconstruction_head(
                decoder_output
            )
        )

        # =================================================
        # TRAVEL-TIME FIELD
        # =================================================

        raw_travel_time = (
            self.travel_time_head(
                decoder_output
            )
        )

        # Enforce positivity.

        normalized_travel_time = (
            self.travel_time_activation(
                raw_travel_time
            )
        )

        # Apply configured travel-time scale.

        travel_time = (
            TRAVEL_TIME_SCALE
            * normalized_travel_time
        )

        # =================================================
        # ALEATORIC UNCERTAINTY
        # =================================================

        if self.use_uncertainty:

            # Predict log(sigma_a^2).

            log_variance = (
                self.uncertainty_head(
                    decoder_output
                )
            )

        else:

            # Preserve the three-output interface even
            # when uncertainty is disabled.

            log_variance = torch.zeros_like(
                reconstructed_cube
            )

        # =================================================
        # OUTPUT VALIDATION
        # =================================================

        reconstruction_shape = (
            reconstructed_cube.shape
        )

        # -------------------------------------------------
        # Travel-time shape
        # -------------------------------------------------

        if travel_time.shape != reconstruction_shape:

            raise RuntimeError(
                "Travel-time output shape does not match "
                "the reconstruction output shape.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction_shape)}\n"
                f"Travel time: "
                f"{tuple(travel_time.shape)}"
            )

        # -------------------------------------------------
        # Log-variance shape
        # -------------------------------------------------

        if log_variance.shape != reconstruction_shape:

            raise RuntimeError(
                "Log-variance output shape does not match "
                "the reconstruction output shape.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction_shape)}\n"
                f"Log variance: "
                f"{tuple(log_variance.shape)}"
            )

        # -------------------------------------------------
        # Reconstruction finite check
        # -------------------------------------------------

        if not torch.isfinite(
            reconstructed_cube
        ).all():

            raise RuntimeError(
                "Reconstruction output contains "
                "NaN or Inf values."
            )

        # -------------------------------------------------
        # Travel-time finite check
        # -------------------------------------------------

        if not torch.isfinite(
            travel_time
        ).all():

            raise RuntimeError(
                "Travel-time output contains "
                "NaN or Inf values."
            )

        # -------------------------------------------------
        # Log-variance finite check
        # -------------------------------------------------

        if not torch.isfinite(
            log_variance
        ).all():

            raise RuntimeError(
                "Log-variance output contains "
                "NaN or Inf values."
            )

        # =================================================
        # RETURN
        # =================================================
        #
        # The three-output interface is deliberately kept
        # consistent throughout training and inference.
        #
        # =================================================

        return (
            reconstructed_cube,
            travel_time,
            log_variance
        )