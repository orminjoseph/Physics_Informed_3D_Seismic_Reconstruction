"""
=========================================================
Total Loss
=========================================================

Composite loss function for the Physics-Informed
3D Encoder–Decoder Framework.

The total loss consists of:

    1. Reconstruction Loss (MAE)
    2. Physics Loss
    3. Predictive Uncertainty Loss
    4. Structural Similarity Loss

Author: Ormin Joseph
=========================================================
"""

import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.uncertainty_loss import UncertaintyLoss
from losses.ssim_loss import SSIMLoss


class TotalLoss(nn.Module):
    """
    Composite loss function.

    L_total =

        λ₁ L_MAE
      + λ₂ L_Physics
      + λ₃ L_Uncertainty
      + λ₄ L_SSIM
    """

    def __init__(
            self,
            mae_weight=1.0,
            physics_weight=1e-4,
            uncertainty_weight=0.10,
            ssim_weight=0.10
    ):

        super().__init__()

        # ------------------------------------------
        # Loss Weights
        # ------------------------------------------

        self.mae_weight = mae_weight
        self.physics_weight = physics_weight
        self.uncertainty_weight = uncertainty_weight
        self.ssim_weight = ssim_weight

        # ------------------------------------------
        # Individual Losses
        # ------------------------------------------

        self.mae_loss = MAELoss()

        self.physics_loss = PhysicsLoss()

        self.uncertainty_loss = UncertaintyLoss()

        self.ssim_loss = SSIMLoss()

    # --------------------------------------------------

    def forward(
            self,
            reconstruction,
            target,
            log_variance,
            velocity_model=None
    ):
        """
        Compute the composite loss.

        Parameters
        ----------
        reconstruction : torch.Tensor
            Predicted seismic cube.

        target : torch.Tensor
            Ground-truth seismic cube.

        log_variance : torch.Tensor
            Predicted uncertainty.

        velocity_model : torch.Tensor, optional
            Spatial velocity model.
            If None, PhysicsLoss uses a constant velocity.
        """

        # ------------------------------------------
        # Reconstruction Loss
        # ------------------------------------------

        mae = self.mae_loss(
            reconstruction,
            target
        )

        # ------------------------------------------
        # Physics Loss
        # ------------------------------------------

        physics = self.physics_loss(
            prediction=reconstruction,
            target=target,
            velocity_model=velocity_model
        )

        # ------------------------------------------
        # Predictive Uncertainty Loss
        # ------------------------------------------

        uncertainty = self.uncertainty_loss(
            reconstruction,
            target,
            log_variance
        )

        # ------------------------------------------
        # Structural Similarity Loss
        # ------------------------------------------

        ssim = self.ssim_loss(
            reconstruction,
            target
        )

        # ------------------------------------------
        # Composite Loss
        # ------------------------------------------

        total = (

            self.mae_weight * mae

            +

            self.physics_weight * physics

            +

            self.uncertainty_weight * uncertainty

            +

            self.ssim_weight * ssim

        )

        # ------------------------------------------
        # Logging Dictionary
        # ------------------------------------------

        losses = {

            "mae": float(mae.detach()),

            "physics": float(physics.detach()),

            "uncertainty": float(uncertainty.detach()),

            "ssim": float(ssim.detach()),

            "total": float(total.detach())

        }

        return total, losses