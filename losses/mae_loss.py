"""
=========================================================
MAE Loss
=========================================================

Mean Absolute Error (MAE) Loss for seismic data
reconstruction.

MAE is used as the primary reconstruction loss because it
is robust to outliers and preserves seismic amplitudes
better than MSE.

Author: Ormin Joseph
=========================================================
"""

import torch.nn as nn


class MAELoss(nn.Module):
    """
    Mean Absolute Error (L1) Loss.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
            self,
            prediction,
            target
    ):
        """
        Compute the Mean Absolute Error.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted seismic cube.

        target : torch.Tensor
            Ground-truth seismic cube.
        """

        return self.loss(
            prediction,
            target
        )