"""
=========================================================
Structural Similarity Loss
=========================================================

SSIM Loss for Seismic Reconstruction

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss

    Returns

        1 - SSIM

    so that minimizing the loss
    maximizes structural similarity.
    """

    def __init__(
        self,
        data_range=1.0
    ):

        super().__init__()

        self.data_range = data_range

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ):

        score = ssim(
            prediction,
            target,
            data_range=self.data_range,
            size_average=True
        )

        loss = 1.0 - score

        return loss