"""
=========================================================
Composite Total Loss
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""

import torch
import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.ssim_loss import SSIMLoss
from losses.uncertainty_loss import UncertaintyLoss

from utils.config import LOSS_WEIGHTS


class TotalLoss(nn.Module):

    """
    Composite loss function.
    """

    def __init__(self):

        super().__init__()

        self.mae_loss = MAELoss()

        self.physics_loss = PhysicsLoss()

        self.ssim_loss = SSIMLoss()

        self.uncertainty_loss = UncertaintyLoss()

    def forward(
            self,
            prediction,
            target,
            velocity_model,
            log_variance
    ):


        mae = self.mae_loss(

            prediction,

            target

        )

        physics = self.physics_loss(

            prediction,

            target,

            velocity_model

        )


        ssim = self.ssim_loss(

            prediction,

            target

        )

        uncertainty = self.uncertainty_loss(

            prediction,

            target,

            log_variance

        )

        total = (

            LOSS_WEIGHTS["mae"] * mae

            +

            LOSS_WEIGHTS["physics"] * physics

            +

            LOSS_WEIGHTS["uncertainty"] * uncertainty

            +

            LOSS_WEIGHTS["ssim"] * ssim

        )

        return {

            "mae": mae,

            "physics": physics,

            "uncertainty": uncertainty,

            "ssim": ssim,

            "total": total

        }