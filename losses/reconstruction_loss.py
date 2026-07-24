"""
=========================================================
Reconstruction Loss
=========================================================

Primary reconstruction objective using MAE.

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Mean Absolute Error (L1 Loss)
    """

    def __init__(self):

        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        return self.loss(
            prediction,
            target
        )