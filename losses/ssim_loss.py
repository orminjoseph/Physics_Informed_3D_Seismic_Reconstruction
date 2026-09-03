"""
=========================================================
Structural Similarity Loss
=========================================================

SSIM Loss for Seismic Reconstruction

SSIM is composed of:

    1. Luminance similarity
    2. Contrast similarity
    3. Structural similarity

The complete SSIM formulation is:

    SSIM = L * C * S

The loss is defined as:

    SSIM Loss = 1 - SSIM

Therefore, minimizing the loss
maximizes structural similarity.

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from metrics.reconstruction_metrics import ssim


class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss.

    Loss formulation:

        L_SSIM = 1 - SSIM

    A perfect reconstruction has:

        SSIM = 1

    and therefore:

        L_SSIM = 0
    """

    def __init__(
            self,
            data_range=2.0
    ):
        super().__init__()

        self.data_range = data_range

    def forward(
            self,
            prediction,
            target
    ):

        # -------------------------------------------------
        # Shape validation
        # -------------------------------------------------

        if prediction.shape != target.shape:

            raise ValueError(
                "Prediction and target must have "
                "identical shapes for SSIM loss.\n"
                f"Prediction shape: "
                f"{tuple(prediction.shape)}\n"
                f"Target shape: "
                f"{tuple(target.shape)}"
            )

        # -------------------------------------------------
        # Calculate SSIM
        # -------------------------------------------------

        score = ssim(
            prediction,
            target,
            data_range=self.data_range
        )

        # -------------------------------------------------
        # Convert similarity to loss
        # -------------------------------------------------

        loss = 1.0 - score

        return loss