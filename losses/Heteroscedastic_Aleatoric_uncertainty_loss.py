"""
=========================================================
Heteroscedastic Aleatoric Uncertainty Loss
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
This module implements the heteroscedastic Gaussian negative
log-likelihood used to learn aleatoric uncertainty during
seismic data reconstruction.

The model predicts the logarithm of the aleatoric variance:

    s = log(sigma_a^2)

where:

    sigma_a^2 = exp(s)

The uncertainty loss is:

    L_aleatoric =
        0.5 * exp(-s) * (y_hat - y)^2
        + 0.5 * s

Aleatoric uncertainty represents uncertainty associated with
the data or observation process.

Epistemic uncertainty is NOT learned by this loss. It is
estimated separately using Monte Carlo Dropout during
inference.

Predictive variance is subsequently obtained as:

    sigma_predictive^2 =
        sigma_aleatoric^2 + sigma_epistemic^2

Input convention
----------------
Prediction, target, and log_variance are expected to have
the same 5-D tensor shape:

    [B, C, D, H, W]

where:

    B = batch size
    C = number of channels
    D = depth / time-sample dimension
    H = crossline dimension
    W = inline dimension

=========================================================
"""

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """
    Heteroscedastic Gaussian negative log-likelihood loss
    for learning aleatoric seismic reconstruction uncertainty.

    The network predicts log_variance = log(sigma_a^2).

    Epistemic uncertainty is estimated separately using
    Monte Carlo Dropout and is not included in this loss.
    """

    def __init__(
        self,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
    ):
        """
        Initialize the heteroscedastic aleatoric uncertainty loss.

        Parameters
        ----------
        min_log_variance : float
            Lower numerical bound for predicted log variance.

        max_log_variance : float
            Upper numerical bound for predicted log variance.
        """

        super().__init__()

        if min_log_variance >= max_log_variance:
            raise ValueError(
                "min_log_variance must be smaller than "
                "max_log_variance."
            )

        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the heteroscedastic aleatoric uncertainty loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.

        target : torch.Tensor
            Ground-truth seismic volume.

        log_variance : torch.Tensor
            Predicted logarithm of aleatoric variance.

        Returns
        -------
        torch.Tensor
            Scalar aleatoric uncertainty loss.
        """

        # -------------------------------------------------
        # Validate tensor types
        # -------------------------------------------------

        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a torch.Tensor.")

        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a torch.Tensor.")

        if not isinstance(log_variance, torch.Tensor):
            raise TypeError("log_variance must be a torch.Tensor.")

        # -------------------------------------------------
        # Validate dimensionality
        # -------------------------------------------------

        if prediction.ndim != 5:
            raise ValueError(
                "prediction must have shape [B, C, D, H, W]. "
                f"Received shape: {tuple(prediction.shape)}"
            )

        if target.ndim != 5:
            raise ValueError(
                "target must have shape [B, C, D, H, W]. "
                f"Received shape: {tuple(target.shape)}"
            )

        if log_variance.ndim != 5:
            raise ValueError(
                "log_variance must have shape [B, C, D, H, W]. "
                f"Received shape: {tuple(log_variance.shape)}"
            )

        # -------------------------------------------------
        # Validate matching shapes
        # -------------------------------------------------

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes. "
                f"Received {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        if prediction.shape != log_variance.shape:
            raise ValueError(
                "prediction and log_variance must have identical "
                "shapes. "
                f"Received {tuple(prediction.shape)} and "
                f"{tuple(log_variance.shape)}."
            )

        # -------------------------------------------------
        # Validate finite input values
        # -------------------------------------------------

        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "prediction contains NaN or Inf values."
            )

        if not torch.isfinite(target).all():
            raise FloatingPointError(
                "target contains NaN or Inf values."
            )

        if not torch.isfinite(log_variance).all():
            raise FloatingPointError(
                "log_variance contains NaN or Inf values."
            )

        # -------------------------------------------------
        # Compute squared reconstruction error
        # -------------------------------------------------

        squared_error = (
            prediction - target
        ) ** 2

        # -------------------------------------------------
        # Stabilize the predicted log variance
        # -------------------------------------------------

        log_variance = torch.clamp(
            log_variance,
            min=self.min_log_variance,
            max=self.max_log_variance,
        )

        # -------------------------------------------------
        # Convert log variance to precision
        #
        # precision = 1 / variance
        #            = exp(-log_variance)
        # -------------------------------------------------

        precision = torch.exp(
            -log_variance
        )

        # -------------------------------------------------
        # Heteroscedastic Gaussian negative log-likelihood
        #
        # L =
        # 0.5 * precision * squared_error
        # +
        # 0.5 * log_variance
        # -------------------------------------------------

        loss = (
            0.5 * precision * squared_error
            +
            0.5 * log_variance
        )

        # -------------------------------------------------
        # Reduce to a scalar
        # -------------------------------------------------

        loss = loss.mean()

        # -------------------------------------------------
        # Final numerical stability check
        # -------------------------------------------------

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Heteroscedastic aleatoric uncertainty loss "
                "became NaN or Inf."
            )

        return loss