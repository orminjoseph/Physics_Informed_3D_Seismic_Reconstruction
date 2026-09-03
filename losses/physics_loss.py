"""
=========================================================
Physics-Informed 3D Eikonal Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Phase B
-------
Numerically stabilized Eikonal physics loss.

Governing equation
------------------

For seismic travel time T(x, y, z):

    |grad T| = 1 / V

where:

    T = seismic travel-time field [s]
    V = P-wave velocity field [m/s]

Equivalent classical form:

    |grad T|^2 = 1 / V^2

For numerical stability during neural-network training,
the positive-velocity first-order form is used:

    V |grad T| = 1

Therefore, the optimized Eikonal residual is:

    R_eikonal = V |grad T| - 1

The Eikonal loss is:

    L_eikonal = mean(R_eikonal^2)

This formulation avoids the explicit V^2 factor used in:

    V^2 |grad T|^2 - 1

which can produce unnecessarily large numerical gradients
when physical velocity values are represented in m/s.

Source condition
----------------

When valid source coordinates are supplied:

    T(x_s, y_s, z_s) = 0

Source loss:

    L_source = mean(T_source^2)

Optional travel-time supervision
--------------------------------

When an independently valid travel-time target is supplied:

    L_travel_time =
        mean((T_pred - T_target)^2)

Total physics loss
------------------

    L_physics =
        lambda_eikonal L_eikonal
        +
        lambda_source L_source
        +
        lambda_travel_time L_travel_time

Tensor convention
-----------------

All tensors:

    [B, C, D, H, W]

where:

    D = depth
    H = crossline
    W = inline

Spatial derivatives:

    dT/dz -> dimension 2
    dT/dy -> dimension 3
    dT/dx -> dimension 4

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Numerically stabilized physics-informed 3D Eikonal loss.

    The Eikonal residual is formulated as:

        R = V * |grad(T)| - 1

    rather than:

        R = V^2 * |grad(T)|^2 - 1

    to reduce numerical scaling problems associated with
    physical velocity values expressed in m/s.
    """

    def __init__(
        self,
        dx,
        dy,
        dz,
        eikonal_weight=1.0,
        source_weight=1.0,
        travel_time_weight=1.0,
        eps=1.0e-12
    ):
        super().__init__()

        # =================================================
        # VALIDATE SPATIAL SAMPLING
        # =================================================

        if dx <= 0.0:
            raise ValueError(
                "dx must be greater than zero."
            )

        if dy <= 0.0:
            raise ValueError(
                "dy must be greater than zero."
            )

        if dz <= 0.0:
            raise ValueError(
                "dz must be greater than zero."
            )

        # =================================================
        # VALIDATE LOSS WEIGHTS
        # =================================================

        if eikonal_weight < 0.0:
            raise ValueError(
                "eikonal_weight must be non-negative."
            )

        if source_weight < 0.0:
            raise ValueError(
                "source_weight must be non-negative."
            )

        if travel_time_weight < 0.0:
            raise ValueError(
                "travel_time_weight must be non-negative."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be greater than zero."
            )

        # =================================================
        # STORE PARAMETERS
        # =================================================

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        self.eikonal_weight = float(
            eikonal_weight
        )

        self.source_weight = float(
            source_weight
        )

        self.travel_time_weight = float(
            travel_time_weight
        )

        self.eps = float(eps)

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _validate_field(
        field,
        name
    ):
        """
        Validate a 3D tensor field.

        Required shape:

            [B, C, D, H, W]
        """

        if not isinstance(
            field,
            torch.Tensor
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if field.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B, C, D, H, W]. "
                f"Received: {tuple(field.shape)}."
            )

        if not torch.isfinite(field).all():
            raise ValueError(
                f"{name} contains NaN or "
                "infinite values."
            )

    # =====================================================
    # SPATIAL DERIVATIVE
    # =====================================================

    @staticmethod
    def _derivative(
        field,
        spacing,
        dimension
    ):
        """
        Calculate the first spatial derivative.

        Central difference in the interior:

            df/dx =
                [f(x+h) - f(x-h)] / (2h)

        Forward difference at the first boundary:

            df/dx =
                [f(x+h) - f(x)] / h

        Backward difference at the final boundary:

            df/dx =
                [f(x) - f(x-h)] / h
        """

        if field.ndim != 5:
            raise ValueError(
                "field must have shape "
                "[B, C, D, H, W]."
            )

        if spacing <= 0.0:
            raise ValueError(
                "spacing must be greater than zero."
            )

        size = field.shape[dimension]

        if size < 2:
            raise ValueError(
                "The selected spatial dimension "
                "must contain at least two samples."
            )

        derivative = torch.empty_like(field)

        # =================================================
        # INTERIOR: CENTRAL DIFFERENCE
        # =================================================

        if size > 2:

            center = [slice(None)] * 5
            forward = [slice(None)] * 5
            backward = [slice(None)] * 5

            center[dimension] = slice(1, -1)
            forward[dimension] = slice(2, None)
            backward[dimension] = slice(None, -2)

            derivative[tuple(center)] = (
                field[tuple(forward)]
                -
                field[tuple(backward)]
            ) / (
                2.0 * spacing
            )

        # =================================================
        # FIRST BOUNDARY: FORWARD DIFFERENCE
        # =================================================

        first = [slice(None)] * 5
        first_forward = [slice(None)] * 5

        first[dimension] = 0
        first_forward[dimension] = 1

        derivative[tuple(first)] = (
            field[tuple(first_forward)]
            -
            field[tuple(first)]
        ) / spacing

        # =================================================
        # FINAL BOUNDARY: BACKWARD DIFFERENCE
        # =================================================

        last = [slice(None)] * 5
        last_backward = [slice(None)] * 5

        last[dimension] = -1
        last_backward[dimension] = -2

        derivative[tuple(last)] = (
            field[tuple(last)]
            -
            field[tuple(last_backward)]
        ) / spacing

        return derivative

    # =====================================================
    # TRAVEL-TIME GRADIENT
    # =====================================================

    def travel_time_gradient(
        self,
        travel_time
    ):
        """
        Calculate the 3D spatial gradient of travel time.

        Returns
        -------

        dT_dz
        dT_dy
        dT_dx
        gradient_squared
        gradient_magnitude
        """

        self._validate_field(
            travel_time,
            "travel_time"
        )

        # -------------------------------------------------
        # Depth derivative
        # -------------------------------------------------

        dT_dz = self._derivative(
            travel_time,
            spacing=self.dz,
            dimension=2
        )

        # -------------------------------------------------
        # Crossline derivative
        # -------------------------------------------------

        dT_dy = self._derivative(
            travel_time,
            spacing=self.dy,
            dimension=3
        )

        # -------------------------------------------------
        # Inline derivative
        # -------------------------------------------------

        dT_dx = self._derivative(
            travel_time,
            spacing=self.dx,
            dimension=4
        )

        # -------------------------------------------------
        # Squared gradient magnitude
        # -------------------------------------------------

        gradient_squared = (
            dT_dx.pow(2)
            +
            dT_dy.pow(2)
            +
            dT_dz.pow(2)
        )

        # -------------------------------------------------
        # Gradient magnitude
        #
        # eps prevents sqrt(0) numerical problems.
        # -------------------------------------------------

        gradient_magnitude = torch.sqrt(
            gradient_squared
            +
            self.eps
        )

        return (
            dT_dz,
            dT_dy,
            dT_dx,
            gradient_squared,
            gradient_magnitude
        )

    # =====================================================
    # EIKONAL RESIDUAL
    # =====================================================

    def eikonal_residual(
        self,
        travel_time,
        velocity
    ):
        """
        Calculate the stabilized dimensionless
        3D Eikonal residual.

        Governing equation:

            |grad T| = 1 / V

        Equivalent form:

            V |grad T| = 1

        Residual:

            R_eikonal =
                V |grad T| - 1

        Parameters
        ----------
        travel_time : torch.Tensor
            Predicted travel-time field.

            Shape:
                [B,C,D,H,W]

            Units:
                seconds [s]

        velocity : torch.Tensor
            Physical P-wave velocity field.

            Shape:
                [B,C,D,H,W]

            Units:
                metres/second [m/s]

        Returns
        -------
        torch.Tensor
            Dimensionless Eikonal residual.
        """

        # =================================================
        # 1. VALIDATE INPUTS
        # =================================================

        self._validate_field(
            travel_time,
            "travel_time"
        )

        self._validate_field(
            velocity,
            "velocity"
        )

        # =================================================
        # 2. VALIDATE SHAPES
        # =================================================

        if travel_time.shape != velocity.shape:
            raise ValueError(
                "travel_time and velocity must have "
                "identical shapes. "
                f"Travel time: "
                f"{tuple(travel_time.shape)}, "
                f"Velocity: "
                f"{tuple(velocity.shape)}."
            )

        # =================================================
        # 3. VALIDATE VELOCITY
        # =================================================

        if torch.any(velocity <= 0.0):
            raise ValueError(
                "Velocity must contain strictly positive "
                "P-wave velocity values."
            )

        # =================================================
        # 4. COMPUTE GRADIENT
        # =================================================

        (
            _,
            _,
            _,
            _,
            gradient_magnitude
        ) = self.travel_time_gradient(
            travel_time
        )

        # =================================================
        # 5. STABILIZED EIKONAL RESIDUAL
        # =================================================

        residual = (
            velocity
            *
            gradient_magnitude
            -
            1.0
        )

        # =================================================
        # 6. NUMERICAL VALIDATION
        # =================================================

        if not torch.isfinite(residual).all():
            raise ValueError(
                "Eikonal residual contains NaN or "
                "infinite values."
            )

        return residual

    # =====================================================
    # EIKONAL LOSS
    # =====================================================

    def eikonal_loss(
        self,
        travel_time,
        velocity
    ):
        """
        Compute the mean squared stabilized
        Eikonal residual.
        """

        residual = self.eikonal_residual(
            travel_time,
            velocity
        )

        loss = residual.pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Eikonal loss contains NaN or "
                "infinite values."
            )

        return loss

    # =====================================================
    # SOURCE CONDITION LOSS
    # =====================================================

    def source_condition_loss(
        self,
        travel_time,
        source_indices
    ):
        """
        Enforce the source condition:

            T(x_s, y_s, z_s) = 0

        Parameters
        ----------
        travel_time : torch.Tensor
            Travel-time field [B,C,D,H,W].

        source_indices : torch.Tensor or None
            Source coordinates:

                [B,3]

            ordered as:

                [depth, crossline, inline]
        """

        self._validate_field(
            travel_time,
            "travel_time"
        )

        if source_indices is None:
            return travel_time.new_zeros(())

        if not isinstance(
            source_indices,
            torch.Tensor
        ):
            raise TypeError(
                "source_indices must be "
                "a torch.Tensor."
            )

        if source_indices.ndim != 2:
            raise ValueError(
                "source_indices must have "
                "shape [B,3]."
            )

        if source_indices.shape[1] != 3:
            raise ValueError(
                "source_indices must contain "
                "[depth, crossline, inline]."
            )

        batch_size = travel_time.shape[0]

        if source_indices.shape[0] != batch_size:
            raise ValueError(
                "Number of source locations "
                "must match batch size."
            )

        source_indices = source_indices.to(
            device=travel_time.device,
            dtype=torch.long
        )

        source_values = []

        for batch_index in range(batch_size):

            z = int(
                source_indices[
                    batch_index,
                    0
                ].item()
            )

            y = int(
                source_indices[
                    batch_index,
                    1
                ].item()
            )

            x = int(
                source_indices[
                    batch_index,
                    2
                ].item()
            )

            if not (
                0 <= z < travel_time.shape[2]
                and
                0 <= y < travel_time.shape[3]
                and
                0 <= x < travel_time.shape[4]
            ):
                raise ValueError(
                    f"Source index {(z,y,x)} "
                    "lies outside the "
                    "travel-time volume."
                )

            source_values.append(
                travel_time[
                    batch_index,
                    :,
                    z,
                    y,
                    x
                ]
            )

        source_time = torch.stack(
            source_values,
            dim=0
        )

        loss = source_time.pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Source-condition loss contains "
                "NaN or infinite values."
            )

        return loss

    # =====================================================
    # TRAVEL-TIME SUPERVISION LOSS
    # =====================================================

    def travel_time_supervision_loss(
        self,
        predicted,
        target
    ):
        """
        Compute optional supervised travel-time loss.
        """

        self._validate_field(
            predicted,
            "predicted travel_time"
        )

        if target is None:
            return predicted.new_zeros(())

        self._validate_field(
            target,
            "travel_time_target"
        )

        if predicted.shape != target.shape:
            raise ValueError(
                "Predicted and target travel-time "
                "fields must have identical shapes."
            )

        loss = (
            predicted
            -
            target
        ).pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Travel-time supervision loss contains "
                "NaN or infinite values."
            )

        return loss

    # =====================================================
    # COMPLETE PHYSICS LOSS
    # =====================================================

    def forward(
        self,
        travel_time,
        velocity,
        source_indices=None,
        travel_time_target=None
    ):
        """
        Calculate the complete physics-informed loss.

        Returns
        -------

        Dictionary containing:

            total

            eikonal
            source
            travel_time

            weighted_eikonal
            weighted_source
            weighted_travel_time
        """

        # =================================================
        # 1. EIKONAL LOSS
        # =================================================

        eikonal = self.eikonal_loss(
            travel_time,
            velocity
        )

        # =================================================
        # 2. SOURCE CONDITION
        # =================================================

        source = self.source_condition_loss(
            travel_time,
            source_indices
        )

        # =================================================
        # 3. OPTIONAL TRAVEL-TIME SUPERVISION
        # =================================================

        travel_time_supervision = (
            self.travel_time_supervision_loss(
                travel_time,
                travel_time_target
            )
        )

        # =================================================
        # 4. APPLY WEIGHTS
        # =================================================

        weighted_eikonal = (
            self.eikonal_weight
            *
            eikonal
        )

        weighted_source = (
            self.source_weight
            *
            source
        )

        weighted_travel_time = (
            self.travel_time_weight
            *
            travel_time_supervision
        )

        # =================================================
        # 5. TOTAL PHYSICS LOSS
        # =================================================

        total = (
            weighted_eikonal
            +
            weighted_source
            +
            weighted_travel_time
        )

        # =================================================
        # 6. NUMERICAL VALIDATION
        # =================================================

        if not torch.isfinite(total):
            raise ValueError(
                "Total physics loss contains "
                "NaN or infinite values."
            )

        # =================================================
        # 7. RETURN COMPONENTS
        # =================================================

        return {
            "total": total,

            "eikonal": eikonal,

            "source": source,

            "travel_time":
                travel_time_supervision,

            "weighted_eikonal":
                weighted_eikonal,

            "weighted_source":
                weighted_source,

            "weighted_travel_time":
                weighted_travel_time
        }