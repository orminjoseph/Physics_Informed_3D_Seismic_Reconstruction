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

    L_aleatoric

is the heteroscedastic Gaussian negative log-likelihood
associated with the network-predicted log variance.

IMPORTANT UNCERTAINTY DESIGN
----------------------------

Aleatoric uncertainty is learned during training:

    s = log(sigma_a^2)

    L_aleatoric
        =
    1/2 exp(-s)(y - y_hat)^2
        +
    1/2 s

Epistemic uncertainty is NOT included directly in this
training loss.

Epistemic uncertainty is estimated after training/inference
using Monte Carlo Dropout:

    sigma_e^2
        =
    Var_MC(y_hat)

Total predictive uncertainty is then:

    sigma_predictive^2
        =
    sigma_a^2
        +
    sigma_e^2

Therefore:

    TotalLoss
        !=
    predictive uncertainty

TotalLoss is the optimization objective, whereas predictive
uncertainty is an inference/evaluation quantity.

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
    PHYSICS_LOSS_WEIGHTS
)


class TotalLoss(nn.Module):
    """
    Composite training loss for the Physics-Informed
    3D Encoder-Decoder framework.

    The loss contains four optimization components:

        1. MAE reconstruction loss
        2. Physics-informed loss
        3. Heteroscedastic aleatoric uncertainty loss
        4. SSIM structural loss

    Epistemic uncertainty is intentionally excluded from
    the training loss because it is estimated from multiple
    stochastic MC-Dropout forward passes during inference.

    Final predictive uncertainty is:

        predictive_variance
            =
        aleatoric_variance
            +
        epistemic_variance
    """

    def __init__(
        self,
        dx,
        dy,
        dz
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
        """

        super().__init__()

        # =================================================
        # VALIDATE GRID SPACING
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

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        # =================================================
        # VALIDATE LOSS WEIGHTS
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
                    f"Missing loss weight: '{name}' "
                    "in LOSS_WEIGHTS."
                )

            weight = LOSS_WEIGHTS[name]

            if not isinstance(
                weight,
                (int, float)
            ):
                raise TypeError(
                    f"LOSS_WEIGHTS['{name}'] must be numeric."
                )

            if weight < 0.0:
                raise ValueError(
                    f"LOSS_WEIGHTS['{name}'] cannot be negative."
                )

        # =================================================
        # VALIDATE PHYSICS LOSS WEIGHTS
        # =================================================

        required_physics_weights = (
            "eikonal",
            "source",
            "travel_time"
        )

        for name in required_physics_weights:

            if name not in PHYSICS_LOSS_WEIGHTS:
                raise KeyError(
                    f"Missing physics loss weight: '{name}' "
                    "in PHYSICS_LOSS_WEIGHTS."
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

            if weight < 0.0:
                raise ValueError(
                    f"PHYSICS_LOSS_WEIGHTS['{name}'] "
                    "cannot be negative."
                )

        # =================================================
        # STORE GRID SPACING
        # =================================================

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        # =================================================
        # RECONSTRUCTION LOSS
        # =================================================

        self.mae_loss = MAELoss()

        # =================================================
        # PHYSICS-INFORMED LOSS
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
        # STRUCTURAL SIMILARITY LOSS
        # =================================================
        #
        # The current seismic normalization is assumed to
        # be approximately [-1, 1].
        #
        # Therefore:
        #
        #     data_range = 2.0
        #
        # =================================================

        self.ssim_loss = SSIMLoss(
            data_range=2.0
        )

        # =================================================
        # HETEROSCEDASTIC ALEATORIC UNCERTAINTY LOSS
        # =================================================
        #
        # This learns the voxel-wise conditional variance:
        #
        #     log_variance = log(sigma_a^2)
        #
        # It is NOT an epistemic uncertainty estimator.
        #
        # Epistemic uncertainty is calculated later by
        # MC Dropout.
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

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.

            Shape:

                [B,C,D,H,W]

        target : torch.Tensor
            Ground-truth seismic volume.

            Shape:

                [B,C,D,H,W]

        travel_time : torch.Tensor
            Predicted travel-time field.

            Shape:

                [B,C,D,H,W]

        velocity_model : torch.Tensor
            P-wave velocity model.

            Shape:

                [B,C,D,H,W]

        log_variance : torch.Tensor
            Predicted logarithmic aleatoric variance.

            Shape:

                [B,C,D,H,W]

        source_indices : torch.Tensor, optional
            Source coordinates.

            Expected shape:

                [B,3]

        travel_time_target : torch.Tensor, optional
            Independently valid travel-time target.

        Returns
        -------
        dict
            Dictionary containing raw, weighted, and total
            loss components.

        IMPORTANT
        ---------

        This function calculates the TRAINING OBJECTIVE.

        It does not calculate MC-Dropout epistemic uncertainty.

        Predictive uncertainty is calculated separately using:

            models/mc_dropout.py

        and:

            models/predictive_uncertainty.py
        """

        # =================================================
        # 1. VALIDATE MAIN TENSORS
        # =================================================

        self._validate_tensor(
            prediction,
            "prediction"
        )

        self._validate_tensor(
            target,
            "target"
        )

        self._validate_tensor(
            travel_time,
            "travel_time"
        )

        self._validate_tensor(
            velocity_model,
            "velocity_model"
        )

        self._validate_tensor(
            log_variance,
            "log_variance"
        )

        # =================================================
        # 2. VERIFY SHAPE COMPATIBILITY
        # =================================================

        expected_shape = prediction.shape

        tensors_to_compare = {
            "target": target,
            "travel_time": travel_time,
            "velocity_model": velocity_model,
            "log_variance": log_variance
        }

        for name, tensor in tensors_to_compare.items():

            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} and prediction must have "
                    f"identical shapes. "
                    f"Prediction: {tuple(expected_shape)}, "
                    f"{name}: {tuple(tensor.shape)}."
                )

        # =================================================
        # 3. MAE RECONSTRUCTION LOSS
        # =================================================

        mae = self.mae_loss(
            prediction,
            target
        )

        # =================================================
        # 4. PHYSICS-INFORMED LOSS
        # =================================================

        physics_components = self.physics_loss(
            travel_time=travel_time,
            velocity=velocity_model,
            source_indices=source_indices,
            travel_time_target=travel_time_target
        )

        # -------------------------------------------------
        # Validate physics-loss output.
        # -------------------------------------------------

        if not isinstance(
            physics_components,
            dict
        ):
            raise TypeError(
                "PhysicsLoss must return a dictionary "
                "containing the physics loss components."
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
        # 5. SSIM STRUCTURAL LOSS
        # =================================================

        ssim = self.ssim_loss(
            prediction,
            target
        )

        # =================================================
        # 6. HETEROSCEDASTIC ALEATORIC NLL
        # =================================================
        #
        # The uncertainty loss learns:
        #
        #     log(sigma_a^2)
        #
        # through the Gaussian negative log-likelihood:
        #
        #     1/2 exp(-s)(y-y_hat)^2 + 1/2 s
        #
        # This is ALEATORIC uncertainty only.
        #
        # No MC-Dropout samples are used here.
        #
        # No epistemic uncertainty is included here.
        # =================================================

        aleatoric_nll = self.aleatoric_loss(
            prediction,
            target,
            log_variance
        )

        # =================================================
        # 7. VALIDATE LOSS VALUES
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
                    f"Loss component '{name}' must be "
                    "a torch.Tensor."
                )

            if not torch.isfinite(
                value
            ).all():
                raise ValueError(
                    f"Loss component '{name}' contains "
                    "NaN or Inf values."
                )

        # =================================================
        # 8. APPLY GLOBAL LOSS WEIGHTS
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
        # 9. TOTAL TRAINING LOSS
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
        # 10. FINAL TOTAL VALIDATION
        # =================================================

        if not torch.isfinite(
            total
        ).all():
            raise ValueError(
                "Total loss contains NaN or Inf values."
            )

        # =================================================
        # 11. RETURN COMPLETE LOSS BREAKDOWN
        # =================================================

        return {

            # -------------------------------------------------
            # Raw reconstruction loss
            # -------------------------------------------------

            "mae": mae,

            # -------------------------------------------------
            # Raw physics loss
            # -------------------------------------------------

            "physics": physics,

            # -------------------------------------------------
            # Physics sub-components
            # -------------------------------------------------

            "eikonal": eikonal,

            "source": source,

            "travel_time": travel_time_loss,

            # -------------------------------------------------
            # Raw heteroscedastic aleatoric loss
            # -------------------------------------------------

            "aleatoric_nll": aleatoric_nll,

            # -------------------------------------------------
            # Backward-compatible uncertainty name
            # -------------------------------------------------
            #
            # Existing Trainer/tests may still access:
            #
            #     losses["uncertainty"]
            #
            # Keep this alias while making the terminology
            # explicit through "aleatoric_nll".
            # -------------------------------------------------

            "uncertainty": aleatoric_nll,

            # -------------------------------------------------
            # Raw SSIM loss
            # -------------------------------------------------

            "ssim": ssim,

            # -------------------------------------------------
            # Weighted reconstruction loss
            # -------------------------------------------------

            "weighted_mae": weighted_mae,

            # -------------------------------------------------
            # Weighted physics loss
            # -------------------------------------------------

            "weighted_physics": weighted_physics,

            # -------------------------------------------------
            # Weighted aleatoric uncertainty loss
            # -------------------------------------------------

            "weighted_aleatoric": weighted_aleatoric,

            # -------------------------------------------------
            # Backward-compatible weighted uncertainty name
            # -------------------------------------------------

            "weighted_uncertainty": weighted_aleatoric,

            # -------------------------------------------------
            # Weighted SSIM loss
            # -------------------------------------------------

            "weighted_ssim": weighted_ssim,

            # -------------------------------------------------
            # Final composite training loss
            # -------------------------------------------------

            "total": total
        }