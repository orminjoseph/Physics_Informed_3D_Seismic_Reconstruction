"""
=========================================================
Composite Total Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Training objective:

    L_total =
        λ_mae L_mae
        +
        λ_physics L_physics
        +
        λ_aleatoric L_aleatoric
        +
        λ_ssim L_ssim

where:

    L_aleatoric =
        1/2 exp(-s)(y - y_hat)^2
        +
        1/2 s

and:

    s = log(sigma_a^2)

Epistemic uncertainty is NOT included directly in the
training loss. It is estimated during inference using
Monte Carlo Dropout.

Predictive uncertainty:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

Physics loss:

    L_physics =
        λ_eikonal L_eikonal
        +
        λ_source L_source
        +
        λ_travel_time L_travel_time

Tensor convention:

    [B, C, D, H, W]

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.ssim_loss import SSIMLoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    UncertaintyLoss
)

from utils.config import (
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    SEISMIC_DATA_RANGE
)


class TotalLoss(nn.Module):
    """
    Composite training loss for the Physics-Informed
    3D Encoder-Decoder framework.

    Components:

        1. MAE reconstruction loss
        2. Physics-informed loss
        3. Heteroscedastic aleatoric uncertainty loss
        4. SSIM structural loss

    Epistemic uncertainty is intentionally excluded from
    training and estimated separately using MC Dropout.
    """

    def __init__(
        self,
        dx,
        dy,
        dz,
        use_uncertainty=True
    ):
        """
        Parameters
        ----------
        dx : float
            Grid spacing along the x/crossline direction.

        dy : float
            Grid spacing along the y/inline direction.

        dz : float
            Grid spacing along the depth direction.

        use_uncertainty : bool
            Enables or disables the aleatoric uncertainty
            training loss.
        """

        super().__init__()

        # =================================================
        # 1. VALIDATE GRID SPACING
        # =================================================

        for value, name in (
            (dx, "dx"),
            (dy, "dy"),
            (dz, "dz")
        ):

            if not isinstance(
                value,
                (int, float)
            ):
                raise TypeError(
                    f"{name} must be a numeric value."
                )

            if not torch.isfinite(
                torch.tensor(float(value))
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        # =================================================
        # 2. VALIDATE UNCERTAINTY SWITCH
        # =================================================

        if not isinstance(
            use_uncertainty,
            bool
        ):
            raise TypeError(
                "use_uncertainty must be a boolean value."
            )

        self.use_uncertainty = use_uncertainty

        # =================================================
        # 3. VALIDATE GLOBAL LOSS WEIGHTS
        # =================================================

        required_loss_weights = (
            "mae",
            "physics",
            "uncertainty",
            "ssim"
        )

        for name in required_loss_weights:

            if name not in LOSS_WEIGHTS:
                raise KeyError(
                    f"Missing loss weight '{name}' "
                    "in LOSS_WEIGHTS."
                )

            weight = LOSS_WEIGHTS[name]

            if not isinstance(
                weight,
                (int, float)
            ):
                raise TypeError(
                    f"LOSS_WEIGHTS['{name}'] "
                    "must be numeric."
                )

            if not torch.isfinite(
                torch.tensor(float(weight))
            ):
                raise ValueError(
                    f"LOSS_WEIGHTS['{name}'] "
                    "must be finite."
                )

            if weight < 0.0:
                raise ValueError(
                    f"LOSS_WEIGHTS['{name}'] "
                    "cannot be negative."
                )

        # =================================================
        # 4. VALIDATE PHYSICS LOSS WEIGHTS
        # =================================================

        required_physics_weights = (
            "eikonal",
            "source",
            "travel_time"
        )

        for name in required_physics_weights:

            if name not in PHYSICS_LOSS_WEIGHTS:
                raise KeyError(
                    f"Missing physics loss weight "
                    f"'{name}' in PHYSICS_LOSS_WEIGHTS."
                )

            weight = PHYSICS_LOSS_WEIGHTS[name]

            if not isinstance(
                weight,
                (int, float)
            ):
                raise TypeError(
                    f"PHYSICS_LOSS_WEIGHTS['{name}'] "
                    "must be numeric."
                )

            if not torch.isfinite(
                torch.tensor(float(weight))
            ):
                raise ValueError(
                    f"PHYSICS_LOSS_WEIGHTS['{name}'] "
                    "must be finite."
                )

            if weight < 0.0:
                raise ValueError(
                    f"PHYSICS_LOSS_WEIGHTS['{name}'] "
                    "cannot be negative."
                )

        # =================================================
        # 5. VALIDATE SEISMIC DATA RANGE
        # =================================================

        if not isinstance(
            SEISMIC_DATA_RANGE,
            (int, float)
        ):
            raise TypeError(
                "SEISMIC_DATA_RANGE must be numeric."
            )

        if not torch.isfinite(
            torch.tensor(float(SEISMIC_DATA_RANGE))
        ):
            raise ValueError(
                "SEISMIC_DATA_RANGE must be finite."
            )

        if SEISMIC_DATA_RANGE <= 0.0:
            raise ValueError(
                "SEISMIC_DATA_RANGE must be greater "
                "than zero."
            )

        # =================================================
        # 6. STORE CONFIGURATION
        # =================================================

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        # =================================================
        # 7. MAE LOSS
        # =================================================

        self.mae_loss = MAELoss()

        # =================================================
        # 8. PHYSICS LOSS
        # =================================================

        self.physics_loss = PhysicsLoss(
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,

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

        # =================================================
        # 9. SSIM LOSS
        # =================================================
        #
        # Use the configuration-defined seismic data range
        # instead of hard-coding 2.0.
        #
        # Current convention:
        #
        #     amplitude range = [-1, 1]
        #     data range      = 2.0
        #
        # =================================================

        self.ssim_loss = SSIMLoss(
            data_range=SEISMIC_DATA_RANGE
        )

        # =================================================
        # 10. HETEROSCEDASTIC ALEATORIC LOSS
        # =================================================

        self.aleatoric_loss = UncertaintyLoss()

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _validate_tensor(
        tensor,
        name
    ):
        """
        Validate a five-dimensional seismic tensor.

        Required shape:

            [B,C,D,H,W]
        """

        if not isinstance(
            tensor,
            torch.Tensor
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B,C,D,H,W]. "
                f"Received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

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
        Calculate the complete composite training loss.
        """

        # =================================================
        # 1. VALIDATE MAIN TENSORS
        # =================================================

        tensors = {
            "prediction": prediction,
            "target": target,
            "travel_time": travel_time,
            "velocity_model": velocity_model,
            "log_variance": log_variance
        }

        for name, tensor in tensors.items():

            self._validate_tensor(
                tensor,
                name
            )

        # =================================================
        # 2. VERIFY SHAPE COMPATIBILITY
        # =================================================

        expected_shape = prediction.shape

        for name, tensor in tensors.items():

            if tensor.shape != expected_shape:

                raise ValueError(
                    f"{name} and prediction must have "
                    f"identical shapes.\n"
                    f"Prediction: {tuple(expected_shape)}\n"
                    f"{name}: {tuple(tensor.shape)}"
                )

        # =================================================
        # 3. VALIDATE OPTIONAL SOURCE INDICES
        # =================================================

        if source_indices is not None:

            if not isinstance(
                source_indices,
                torch.Tensor
            ):
                raise TypeError(
                    "source_indices must be a "
                    "torch.Tensor."
                )

            if source_indices.ndim != 2:
                raise ValueError(
                    "source_indices must have shape "
                    "[B,3]."
                )

            if source_indices.shape[0] != prediction.shape[0]:
                raise ValueError(
                    "source_indices batch dimension must "
                    "match prediction."
                )

            if source_indices.shape[1] != 3:
                raise ValueError(
                    "source_indices must have shape [B,3]."
                )

        # =================================================
        # 4. VALIDATE OPTIONAL TRAVEL-TIME TARGET
        # =================================================

        if travel_time_target is not None:

            self._validate_tensor(
                travel_time_target,
                "travel_time_target"
            )

            if travel_time_target.shape != expected_shape:
                raise ValueError(
                    "travel_time_target must have the "
                    "same shape as prediction."
                )

        # =================================================
        # 5. MAE
        # =================================================

        mae = self.mae_loss(
            prediction,
            target
        )

        # =================================================
        # 6. PHYSICS
        # =================================================

        physics_components = self.physics_loss(
            travel_time=travel_time,
            velocity=velocity_model,
            source_indices=source_indices,
            travel_time_target=travel_time_target
        )

        if not isinstance(
            physics_components,
            dict
        ):
            raise TypeError(
                "PhysicsLoss must return a dictionary."
            )

        required_physics_outputs = (
            "total",
            "eikonal",
            "source",
            "travel_time"
        )

        for name in required_physics_outputs:

            if name not in physics_components:
                raise KeyError(
                    f"PhysicsLoss output is missing "
                    f"'{name}'."
                )

        physics = physics_components["total"]

        eikonal = physics_components["eikonal"]

        source = physics_components["source"]

        travel_time_loss = (
            physics_components["travel_time"]
        )

        # =================================================
        # 7. SSIM
        # =================================================

        ssim = self.ssim_loss(
            prediction,
            target
        )

        # =================================================
        # 8. ALEATORIC UNCERTAINTY
        # =================================================

        if self.use_uncertainty:

            aleatoric_nll = self.aleatoric_loss(
                prediction,
                target,
                log_variance
            )

        else:

            aleatoric_nll = torch.zeros(
                (),
                device=prediction.device,
                dtype=prediction.dtype
            )

        # =================================================
        # 9. VALIDATE RAW LOSS COMPONENTS
        # =================================================

        loss_components = {
            "mae": mae,
            "physics": physics,
            "eikonal": eikonal,
            "source": source,
            "travel_time": travel_time_loss,
            "ssim": ssim,
            "aleatoric_nll": aleatoric_nll
        }

        for name, value in loss_components.items():

            if not isinstance(
                value,
                torch.Tensor
            ):
                raise TypeError(
                    f"Loss component '{name}' must "
                    "be a torch.Tensor."
                )

            if value.ndim != 0:
                raise ValueError(
                    f"Loss component '{name}' must "
                    "be scalar."
                )

            if not torch.isfinite(
                value
            ).all():
                raise ValueError(
                    f"Loss component '{name}' contains "
                    "NaN or Inf values."
                )

        # =================================================
        # 10. APPLY GLOBAL WEIGHTS
        # =================================================

        weighted_mae = (
            LOSS_WEIGHTS["mae"]
            * mae
        )

        weighted_physics = (
            LOSS_WEIGHTS["physics"]
            * physics
        )

        weighted_aleatoric = (
            LOSS_WEIGHTS["uncertainty"]
            * aleatoric_nll
        )

        weighted_ssim = (
            LOSS_WEIGHTS["ssim"]
            * ssim
        )

        # =================================================
        # 11. TOTAL LOSS
        # =================================================

        total = (
            weighted_mae
            +
            weighted_physics
            +
            weighted_aleatoric
            +
            weighted_ssim
        )

        # =================================================
        # 12. VALIDATE TOTAL
        # =================================================

        if not torch.isfinite(
            total
        ).all():

            raise ValueError(
                "Total loss contains NaN or Inf values."
            )

        # =================================================
        # 13. RETURN COMPLETE LOSS BREAKDOWN
        # =================================================

        return {

            # Raw components
            "mae": mae,

            "physics": physics,

            "eikonal": eikonal,

            "source": source,

            "travel_time": travel_time_loss,

            "aleatoric_nll": aleatoric_nll,

            "ssim": ssim,

            # Backward-compatible aliases
            "uncertainty": aleatoric_nll,

            # Weighted components
            "weighted_mae": weighted_mae,

            "weighted_physics": weighted_physics,

            "weighted_aleatoric": weighted_aleatoric,

            "weighted_uncertainty": weighted_aleatoric,

            "weighted_ssim": weighted_ssim,

            # Final objective
            "total": total
        }