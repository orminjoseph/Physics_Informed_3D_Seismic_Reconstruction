"""
=========================================================
Physics-Informed Loss
=========================================================

Physics regularization for 3D seismic reconstruction.

This implementation is designed for reconstructed
3D seismic image volumes rather than full wavefield
simulation.

Physics Components
------------------
1. Velocity-weighted Laplacian
2. Gradient consistency
3. Smoothness regularization

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Physics-informed regularization for seismic
    reconstruction.

    Total Physics Loss

    Lphysics

    =
    λlap Llap
    +
    λgrad Lgrad
    +
    λsmooth Lsmooth
    """

    def __init__(
            self,
            laplacian_weight=1.0,
            gradient_weight=0.30,
            smoothness_weight=0.05
    ):

        super().__init__()

        self.laplacian_weight = laplacian_weight
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight

    # --------------------------------------------------
    # First-order gradients
    # --------------------------------------------------

    def gradient_x(self, x):

        return x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

    def gradient_y(self, x):

        return x[:, :, :, 1:, :] - x[:, :, :, :-1, :]

    def gradient_z(self, x):

        return x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    # --------------------------------------------------
    # 3D Laplacian
    # --------------------------------------------------

    def laplacian3d(self, x):

        center = x[:, :, 1:-1, 1:-1, 1:-1]

        dxx = (

            x[:, :, 2:, 1:-1, 1:-1]

            - 2.0 * center

            + x[:, :, :-2, 1:-1, 1:-1]

        )

        dyy = (

            x[:, :, 1:-1, 2:, 1:-1]

            - 2.0 * center

            + x[:, :, 1:-1, :-2, 1:-1]

        )

        dzz = (

            x[:, :, 1:-1, 1:-1, 2:]

            - 2.0 * center

            + x[:, :, 1:-1, 1:-1, :-2]

        )

        return dxx + dyy + dzz

    # --------------------------------------------------
    # Velocity-weighted Laplacian
    # --------------------------------------------------

    def laplacian_loss(
            self,
            prediction,
            velocity_model
    ):
        laplace = self.laplacian3d(prediction)

        velocity = velocity_model[
            :,
            :,
            1:-1,
            1:-1,
            1:-1
        ]

        velocity = velocity / 3500.0

        residual = velocity.pow(2) * laplace

        return torch.mean(residual ** 2)

    # --------------------------------------------------
    # Gradient consistency
    # --------------------------------------------------

    def gradient_loss(
            self,
            prediction,
            target
    ):

        gx1 = self.gradient_x(prediction)
        gx2 = self.gradient_x(target)

        gy1 = self.gradient_y(prediction)
        gy2 = self.gradient_y(target)

        gz1 = self.gradient_z(prediction)
        gz2 = self.gradient_z(target)

        loss = (

            torch.mean((gx1 - gx2) ** 2)

            +

            torch.mean((gy1 - gy2) ** 2)

            +

            torch.mean((gz1 - gz2) ** 2)

        )

        return loss

    # --------------------------------------------------
    # Smoothness
    # --------------------------------------------------

    def smoothness_loss(
            self,
            prediction
    ):

        laplace = self.laplacian3d(prediction)

        return torch.mean(laplace ** 2)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
            self,
            prediction,
            target,
            velocity_model
    ):

        lap_loss = self.laplacian_loss(

            prediction,

            velocity_model

        )

        grad_loss = self.gradient_loss(

            prediction,

            target

        )

        smooth_loss = self.smoothness_loss(

            prediction

        )

        total_loss = (

            self.laplacian_weight * lap_loss

            +

            self.gradient_weight * grad_loss

            +

            self.smoothness_weight * smooth_loss

        )

        return total_loss