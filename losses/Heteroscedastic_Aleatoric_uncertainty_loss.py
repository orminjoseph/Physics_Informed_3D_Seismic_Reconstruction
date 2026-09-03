"""
=========================================================
Predictive Uncertainty Loss
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """
    Predictive uncertainty loss
    """

    def __init__(self):

        super().__init__()

    def forward(

        self,

        prediction,

        target,

        log_variance

    ):

        squared_error = (

            prediction - target

        ) ** 2

        log_variance = torch.clamp(

            log_variance,

            min=-10.0,

            max=10.0

        )

        precision = torch.exp(

            -log_variance

        )


        loss = (

            0.5 * precision * squared_error

            +

            0.5 * log_variance

        )

        return loss.mean()