"""
=========================================================
Composite Total Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Total loss:

    L_total =
        λ_mae L_mae
        +
        λ_physics L_physics
        +
        λ_uncertainty L_uncertainty
        +
        λ_ssim L_ssim

Physics loss:

    L_physics =
        λ_eikonal L_eikonal
        +
        λ_source L_source
        +
        λ_travel_time L_travel_time

Tensor convention:

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.ssim_loss import SSIMLoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import UncertaintyLoss

from utils.config import (
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS
)


class TotalLoss(nn.Module):
    """
    Complete composite loss for the
    Physics-Informed 3D Encoder-Decoder framework.
    """

    def __init__(
        self,
        dx,
        dy,
        dz
    ):
        super().__init__()

        # -------------------------------------------------
        # Reconstruction loss
        # -------------------------------------------------

        self.mae_loss = MAELoss()

        # -------------------------------------------------
        # Physics-informed loss
        # -------------------------------------------------

        self.physics_loss = PhysicsLoss(
            dx=dx,
            dy=dy,
            dz=dz,

            eikonal_weight=(
                PHYSICS_LOSS_WEIGHTS["eikonal"]
            ),

            source_weight=(
                PHYSICS_LOSS_WEIGHTS["source"]
            ),

            travel_time_weight=(
                PHYSICS_LOSS_WEIGHTS["travel_time"]
            )
        )

        # -------------------------------------------------
        # Structural similarity loss
        # -------------------------------------------------

        self.ssim_loss = SSIMLoss(
            data_range=2.0
        )

        # -------------------------------------------------
        # Predictive uncertainty loss
        # -------------------------------------------------

        self.uncertainty_loss = UncertaintyLoss()

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        prediction,
        target,
        travel_time,
        velocity_model,
        log_variance,
        source_indices=None,
        travel_time_target=None
    ):
        """
        Calculate the complete loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.
            Shape: [B,C,D,H,W]

        target : torch.Tensor
            Ground-truth seismic volume.
            Shape: [B,C,D,H,W]

        travel_time : torch.Tensor
            Predicted travel-time field.
            Shape: [B,C,D,H,W]

        velocity_model : torch.Tensor
            P-wave velocity model.
            Shape: [B,C,D,H,W]

        log_variance : torch.Tensor
            Predicted logarithmic variance.
            Shape: [B,C,D,H,W]

        source_indices : torch.Tensor, optional
            Source coordinates.
            Shape: [B,3]

            Coordinate order:
                [depth, crossline, inline]

        travel_time_target : torch.Tensor, optional
            Independently valid travel-time target.

        Returns
        -------
        dict
            Individual and weighted loss components.
        """

        # =================================================
        # 1. Validate main tensors
        # =================================================

        if prediction.ndim != 5:
            raise ValueError(
                "prediction must have shape [B,C,D,H,W]."
            )

        if target.shape != prediction.shape:
            raise ValueError(
                "prediction and target must have identical shapes."
            )

        if travel_time.shape != prediction.shape:
            raise ValueError(
                "travel_time and prediction must have identical shapes."
            )

        if velocity_model.shape != prediction.shape:
            raise ValueError(
                "velocity_model and prediction must have identical shapes."
            )

        if log_variance.shape != prediction.shape:
            raise ValueError(
                "log_variance and prediction must have identical shapes."
            )

        # =================================================
        # 2. MAE reconstruction loss
        # =================================================

        mae = self.mae_loss(
            prediction,
            target
        )

        # =================================================
        # 3. Physics-informed loss
        # =================================================

        physics_components = self.physics_loss(
            travel_time=travel_time,
            velocity=velocity_model,
            source_indices=source_indices,
            travel_time_target=travel_time_target
        )

        physics = physics_components["total"]

        eikonal = physics_components["eikonal"]

        source = physics_components["source"]

        travel_time_loss = (
            physics_components["travel_time"]
        )

        # =================================================
        # 4. SSIM loss
        # =================================================

        ssim = self.ssim_loss(
            prediction,
            target
        )

        # =================================================
        # 5. Predictive uncertainty loss
        # =================================================

        uncertainty = self.uncertainty_loss(
            prediction,
            target,
            log_variance
        )

        # =================================================
        # 6. Apply global loss weights
        # =================================================

        weighted_mae = (
            LOSS_WEIGHTS["mae"]
            * mae
        )

        weighted_physics = (
            LOSS_WEIGHTS["physics"]
            * physics
        )

        weighted_uncertainty = (
            LOSS_WEIGHTS["uncertainty"]
            * uncertainty
        )

        weighted_ssim = (
            LOSS_WEIGHTS["ssim"]
            * ssim
        )

        # =================================================
        # 7. Total loss
        # =================================================

        total = (
            weighted_mae
            +
            weighted_physics
            +
            weighted_uncertainty
            +
            weighted_ssim
        )

        # =================================================
        # 8. Return all components
        # =================================================

        return {

            # Raw losses
            "mae": mae,
            "physics": physics,
            "uncertainty": uncertainty,
            "ssim": ssim,

            # Physics sub-losses
            "eikonal": eikonal,
            "source": source,
            "travel_time": travel_time_loss,

            # Weighted losses
            "weighted_mae": weighted_mae,
            "weighted_physics": weighted_physics,
            "weighted_uncertainty": weighted_uncertainty,
            "weighted_ssim": weighted_ssim,

            # Final composite loss
            "total": total
        }