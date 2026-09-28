"""
=========================================================
Structural Similarity Loss
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

SSIM Loss for Seismic Reconstruction
---------------------------------------------------------

Structural Similarity Index (SSIM) measures similarity
between a reconstructed seismic volume and its reference
volume in terms of:

    1. Luminance
    2. Contrast
    3. Structural information

The SSIM index is expressed conceptually as:

    SSIM = L * C * S

where:

    L = luminance similarity
    C = contrast similarity
    S = structural similarity

The corresponding loss is defined as:

    L_SSIM = 1 - SSIM

Therefore, minimizing this loss encourages the reconstructed
seismic volume to preserve the structural characteristics
of the target volume.

For the current normalized seismic-data convention:

    amplitude range = [-1, 1]
    data range      = 2.0

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from metrics.reconstruction_metrics import ssim


class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss.

    The loss is defined as:

        L_SSIM = 1 - SSIM(prediction, target)

    A perfect reconstruction has:

        SSIM = 1

    and therefore:

        L_SSIM = 0

    Parameters
    ----------
    data_range : float
        Dynamic range of the seismic amplitudes.

        For the current normalized seismic-data convention
        of [-1, 1], this value is 2.0.
    """

    def __init__(self, data_range=2.0):
        super().__init__()

        # -------------------------------------------------
        # Validate SSIM data range
        # -------------------------------------------------

        if data_range <= 0:
            raise ValueError(
                "SSIM data_range must be greater than zero. "
                f"Received: {data_range}"
            )

        self.data_range = float(data_range)

    def forward(self, prediction, target):
        """
        Compute the SSIM loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.

        target : torch.Tensor
            Ground-truth seismic volume.

        Returns
        -------
        torch.Tensor
            Scalar SSIM loss.
        """

        # -------------------------------------------------
        # Validate tensor shapes
        # -------------------------------------------------

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have identical "
                "shapes for SSIM loss.\n"
                f"Prediction shape: {tuple(prediction.shape)}\n"
                f"Target shape: {tuple(target.shape)}"
            )

        # -------------------------------------------------
        # Validate tensor dimensionality
        #
        # The framework operates on 3D seismic volumes:
        #
        # [B, C, D, H, W]
        # -------------------------------------------------

        if prediction.ndim != 5:
            raise ValueError(
                "SSIMLoss expects 5D tensors with shape "
                "[B, C, D, H, W].\n"
                f"Received shape: {tuple(prediction.shape)}"
            )

        # -------------------------------------------------
        # Validate numerical values
        # -------------------------------------------------

        if not torch.isfinite(prediction).all():
            raise ValueError(
                "Prediction contains NaN or Inf values."
            )

        if not torch.isfinite(target).all():
            raise ValueError(
                "Target contains NaN or Inf values."
            )

        # -------------------------------------------------
        # Calculate structural similarity
        # -------------------------------------------------

        score = ssim(
            prediction,
            target,
            data_range=self.data_range
        )

        # -------------------------------------------------
        # Convert similarity into a minimization loss
        # -------------------------------------------------

        loss = 1.0 - score

        # -------------------------------------------------
        # Ensure a scalar loss is returned
        # -------------------------------------------------

        if loss.ndim != 0:
            loss = loss.mean()

        return loss