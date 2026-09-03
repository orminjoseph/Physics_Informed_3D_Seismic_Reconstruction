"""
=========================================================
Monte Carlo Dropout for Epistemic Uncertainty
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Monte Carlo (MC) Dropout provides an approximation of
epistemic (model) uncertainty.

Epistemic uncertainty represents uncertainty associated
with limited knowledge of the learned model parameters.

During inference, dropout layers are intentionally kept
active and the network is evaluated multiple times.

For N stochastic forward passes:

    y_1, y_2, ..., y_N

the predictive mean is:

    μ(x) = 1/N Σ y_i

and the epistemic variance is:

    σ²_epistemic(x)
        = 1/N Σ (y_i - μ(x))²

This module does NOT replace the heteroscedastic
aleatoric uncertainty predicted by Network3D.

Instead:

    Aleatoric uncertainty
        -> predicted through log_variance

    Epistemic uncertainty
        -> estimated through MC Dropout

Together they provide a decomposition of predictive
uncertainty.

Tensor convention
-----------------

Input:

    [B, C, D, H, W]

Reconstruction samples:

    [N, B, C, D, H, W]

Predictive mean:

    [B, C, D, H, W]

Epistemic variance:

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class MCDropout3D:
    """
    Monte Carlo Dropout estimator for epistemic uncertainty.

    Parameters
    ----------
    model : nn.Module
        Trained Physics-Informed 3D Encoder-Decoder network.

    num_samples : int
        Number of stochastic forward passes.

    Notes
    -----
    Dropout is activated during inference while the remaining
    network layers retain their evaluation behavior.
    """

    def __init__(
        self,
        model,
        num_samples=20
    ):

        # =================================================
        # VALIDATE MODEL
        # =================================================

        if not isinstance(model, nn.Module):

            raise TypeError(
                "model must be an instance of "
                "torch.nn.Module."
            )

        # =================================================
        # VALIDATE NUMBER OF MC SAMPLES
        # =================================================

        if not isinstance(
            num_samples,
            int
        ):

            raise TypeError(
                "num_samples must be an integer."
            )

        if num_samples < 2:

            raise ValueError(
                "num_samples must be at least 2."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.model = model

        self.num_samples = num_samples

    # =====================================================
    # ENABLE MC DROPOUT
    # =====================================================

    def _enable_dropout(self):
        """
        Enable only dropout layers.

        The complete model remains in evaluation mode,
        while Dropout3d layers are switched to training mode.

        This prevents BatchNorm3d statistics from changing
        during MC inference.
        """

        self.model.eval()

        for module in self.model.modules():

            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d
                )
            ):

                module.train()

    # =====================================================
    # STOCHASTIC FORWARD PASSES
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        x
    ):
        """
        Perform multiple stochastic forward passes.

        Parameters
        ----------
        x : torch.Tensor
            Input seismic volume.

            Shape:
                [B,C,D,H,W]

        Returns
        -------
        dictionary containing:

            reconstruction_samples
            travel_time_samples
            log_variance_samples

            reconstruction_mean
            travel_time_mean
            log_variance_mean

            reconstruction_epistemic_variance
            travel_time_epistemic_variance
            log_variance_epistemic_variance
        """

        # =================================================
        # VALIDATE INPUT
        # =================================================

        if not isinstance(
            x,
            torch.Tensor
        ):

            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 5:

            raise ValueError(
                "x must have shape "
                "[B,C,D,H,W]. "
                f"Received {tuple(x.shape)}."
            )

        if not torch.isfinite(x).all():

            raise ValueError(
                "x contains NaN or infinite values."
            )

        # =================================================
        # ENABLE MC DROPOUT
        # =================================================

        self._enable_dropout()

        # =================================================
        # COLLECT STOCHASTIC PREDICTIONS
        # =================================================

        reconstruction_samples = []

        travel_time_samples = []

        log_variance_samples = []

        for _ in range(
            self.num_samples
        ):

            (
                reconstructed_cube,
                travel_time,
                log_variance
            ) = self.model(x)

            reconstruction_samples.append(
                reconstructed_cube
            )

            travel_time_samples.append(
                travel_time
            )

            log_variance_samples.append(
                log_variance
            )

        # =================================================
        # STACK MC SAMPLES
        # =================================================

        reconstruction_samples = torch.stack(
            reconstruction_samples,
            dim=0
        )

        travel_time_samples = torch.stack(
            travel_time_samples,
            dim=0
        )

        log_variance_samples = torch.stack(
            log_variance_samples,
            dim=0
        )

        # =================================================
        # PREDICTIVE MEANS
        # =================================================

        reconstruction_mean = (
            reconstruction_samples.mean(
                dim=0
            )
        )

        travel_time_mean = (
            travel_time_samples.mean(
                dim=0
            )
        )

        log_variance_mean = (
            log_variance_samples.mean(
                dim=0
            )
        )

        # =================================================
        # EPISTEMIC VARIANCE
        # =================================================
        #
        # Population variance:
        #
        #     σ² = mean((y - μ)²)
        #
        # We use unbiased=False because these MC samples
        # represent stochastic samples from the predictive
        # distribution rather than a conventional finite
        # statistical sample for estimating a population
        # parameter.
        # =================================================

        reconstruction_epistemic_variance = (
            reconstruction_samples.var(
                dim=0,
                unbiased=False
            )
        )

        travel_time_epistemic_variance = (
            travel_time_samples.var(
                dim=0,
                unbiased=False
            )
        )

        log_variance_epistemic_variance = (
            log_variance_samples.var(
                dim=0,
                unbiased=False
            )
        )

        # =================================================
        # RETURN RESULTS
        # =================================================

        return {

            "reconstruction_samples":
                reconstruction_samples,

            "travel_time_samples":
                travel_time_samples,

            "log_variance_samples":
                log_variance_samples,

            "reconstruction_mean":
                reconstruction_mean,

            "travel_time_mean":
                travel_time_mean,

            "log_variance_mean":
                log_variance_mean,

            "reconstruction_epistemic_variance":
                reconstruction_epistemic_variance,

            "travel_time_epistemic_variance":
                travel_time_epistemic_variance,

            "log_variance_epistemic_variance":
                log_variance_epistemic_variance
        }