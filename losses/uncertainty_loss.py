"""
=========================================================
Predictive Uncertainty Loss
=========================================================

Gaussian Negative Log-Likelihood Loss

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        log_variance: torch.Tensor
    ) -> torch.Tensor:

        squared_error = (prediction - target) ** 2

        print(
            f"log_variance: "
            f"min={log_variance.min().item():.4f}, "
            f"max={log_variance.max().item():.4f}, "
            f"has_nan={torch.isnan(log_variance).any().item()}, "
            f"has_inf={torch.isinf(log_variance).any().item()}"
        )
        log_variance = torch.clamp(
            log_variance,
            min=-10,
            max=10
        )

        precision = torch.exp(
            -log_variance
        )


        loss = 0.5 * (
            precision * squared_error
            + log_variance
        )

        return loss.mean()