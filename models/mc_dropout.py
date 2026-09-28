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

This module estimates epistemic uncertainty for the
reconstructed seismic volume.

Aleatoric uncertainty is predicted separately by the
network through log_variance and is therefore not replaced
by this module.

Predictive variance is subsequently obtained from:

    σ²_predictive =
        σ²_aleatoric + σ²_epistemic

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
        Trained Physics-Informed 3D Encoder–Decoder network.

    num_samples : int
        Number of stochastic forward passes.

    Notes
    -----
    The complete model is placed in evaluation mode while
    dropout layers are selectively activated.

    This prevents BatchNorm layers, if present, from updating
    their running statistics during MC inference.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 20,
    ):
        """
        Initialize the MC Dropout estimator.
        """

        # -------------------------------------------------
        # Validate model
        # -------------------------------------------------

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module."
            )

        # -------------------------------------------------
        # Validate MC sample count
        # -------------------------------------------------

        if not isinstance(num_samples, int):
            raise TypeError(
                "num_samples must be an integer."
            )

        if num_samples < 2:
            raise ValueError(
                "num_samples must be at least 2."
            )

        # -------------------------------------------------
        # Store configuration
        # -------------------------------------------------

        self.model = model
        self.num_samples = num_samples

    # =====================================================
    # ENABLE MC DROPOUT
    # =====================================================

    def _enable_dropout(self):
        """
        Put the model into evaluation mode and activate only
        dropout layers.

        Returns
        -------
        dict
            Original training/evaluation state of every module.
        """

        # -------------------------------------------------
        # Save original state of every module
        # -------------------------------------------------

        original_states = {
            module: module.training
            for module in self.model.modules()
        }

        # -------------------------------------------------
        # Set complete model to evaluation mode
        # -------------------------------------------------

        self.model.eval()

        # -------------------------------------------------
        # Activate only dropout layers
        # -------------------------------------------------

        for module in self.model.modules():

            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d,
                ),
            ):
                module.train()

        return original_states

    # =====================================================
    # RESTORE MODEL STATE
    # =====================================================

    @staticmethod
    def _restore_model_state(
        original_states
    ):
        """
        Restore the training/evaluation state of every module.
        """

        for module, training_state in original_states.items():

            module.training = training_state

    # =====================================================
    # STOCHASTIC FORWARD PASSES
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ):
        """
        Perform multiple stochastic forward passes.

        Parameters
        ----------
        x : torch.Tensor
            Input seismic volume with shape:

                [B, C, D, H, W]

        Returns
        -------
        dict
            MC predictions and epistemic uncertainty estimates.
        """

        # -------------------------------------------------
        # Validate input type
        # -------------------------------------------------

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        # -------------------------------------------------
        # Validate input dimensionality
        # -------------------------------------------------

        if x.ndim != 5:
            raise ValueError(
                "x must have shape [B, C, D, H, W]. "
                f"Received {tuple(x.shape)}."
            )

        # -------------------------------------------------
        # Validate input values
        # -------------------------------------------------

        if not torch.isfinite(x).all():
            raise FloatingPointError(
                "x contains NaN or infinite values."
            )

        # -------------------------------------------------
        # Save and modify model states
        # -------------------------------------------------

        original_states = self._enable_dropout()

        reconstruction_samples = []
        travel_time_samples = []
        log_variance_samples = []

        try:

            # ---------------------------------------------
            # Perform stochastic forward passes
            # ---------------------------------------------

            for _ in range(self.num_samples):

                outputs = self.model(x)

                # -----------------------------------------
                # Validate model output
                # -----------------------------------------

                if not isinstance(
                    outputs,
                    (tuple, list)
                ):
                    raise TypeError(
                        "The model must return a tuple or list "
                        "containing reconstruction, travel_time, "
                        "and log_variance."
                    )

                if len(outputs) != 3:
                    raise ValueError(
                        "The model must return exactly three outputs: "
                        "reconstruction, travel_time, and log_variance."
                    )

                (
                    reconstructed_cube,
                    travel_time,
                    log_variance,
                ) = outputs

                # -----------------------------------------
                # Validate output tensors
                # -----------------------------------------

                for name, tensor in (
                    ("reconstruction", reconstructed_cube),
                    ("travel_time", travel_time),
                    ("log_variance", log_variance),
                ):

                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(
                            f"{name} must be a torch.Tensor."
                        )

                    if not torch.isfinite(tensor).all():
                        raise FloatingPointError(
                            f"{name} contains NaN or infinite values."
                        )

                # -----------------------------------------
                # Validate reconstruction shape
                # -----------------------------------------

                if reconstructed_cube.shape != x.shape:
                    raise ValueError(
                        "Reconstruction shape must match input shape. "
                        f"Input: {tuple(x.shape)}, "
                        f"reconstruction: "
                        f"{tuple(reconstructed_cube.shape)}."
                    )

                # -----------------------------------------
                # Validate log-variance shape
                # -----------------------------------------

                if log_variance.shape != reconstructed_cube.shape:
                    raise ValueError(
                        "log_variance must have the same shape as "
                        "the reconstruction. "
                        f"Received {tuple(log_variance.shape)}."
                    )

                # -----------------------------------------
                # Store stochastic outputs
                # -----------------------------------------

                reconstruction_samples.append(
                    reconstructed_cube
                )

                travel_time_samples.append(
                    travel_time
                )

                log_variance_samples.append(
                    log_variance
                )

        finally:

            # ---------------------------------------------
            # ALWAYS restore original model state
            # ---------------------------------------------

            self._restore_model_state(
                original_states
            )

        # =================================================
        # STACK MC SAMPLES
        # =================================================

        reconstruction_samples = torch.stack(
            reconstruction_samples,
            dim=0,
        )

        travel_time_samples = torch.stack(
            travel_time_samples,
            dim=0,
        )

        log_variance_samples = torch.stack(
            log_variance_samples,
            dim=0,
        )

        # =================================================
        # PREDICTIVE MEANS
        # =================================================

        reconstruction_mean = (
            reconstruction_samples.mean(dim=0)
        )

        travel_time_mean = (
            travel_time_samples.mean(dim=0)
        )

        log_variance_mean = (
            log_variance_samples.mean(dim=0)
        )

        # =================================================
        # EPISTEMIC VARIANCE
        # =================================================
        #
        # Population variance across MC predictions:
        #
        #     σ² =
        #         mean((y_i - μ)²)
        #
        # unbiased=False is used because the MC samples
        # approximate a predictive distribution.
        # =================================================

        reconstruction_epistemic_variance = (
            reconstruction_samples.var(
                dim=0,
                unbiased=False,
            )
        )

        travel_time_epistemic_variance = (
            travel_time_samples.var(
                dim=0,
                unbiased=False,
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
        }