"""
=========================================================
3D Seismic Model Predictor
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Purpose
-------
This class provides deterministic inference for the final
Physics-Informed 3D Encoder-Decoder network.

The network produces three outputs:

    1. reconstruction
    2. travel_time
    3. log_variance

The predictor exposes:

    reconstruction
    travel_time
    log_variance
    aleatoric_std

Important
---------
This predictor performs ONE deterministic forward pass.

Therefore:

    aleatoric_std
        = sqrt(aleatoric_variance)
        = exp(0.5 * log_variance)

It does NOT calculate epistemic uncertainty.

Epistemic uncertainty requires multiple stochastic
forward passes, which are handled by the MC Dropout
component.

Likewise, total predictive uncertainty is NOT calculated
here. It is obtained from:

    predictive_variance
        = aleatoric_variance
        + epistemic_variance

Tensor convention
-----------------
Input:

    [C, D, H, W]

or:

    [B, C, D, H, W]

Reconstruction:

    [B, C, D, H, W]

Travel time:

    [B, C, D, H, W]

Log variance:

    [B, C, D, H, W]

Aleatoric standard deviation:

    [B, C, D, H, W]

=========================================================
"""

import os

import torch


class Predictor:
    """
    Deterministic inference wrapper for the final
    Physics-Informed 3D Encoder-Decoder network.

    This class performs a single forward pass and exposes
    the raw log-variance prediction together with its
    corresponding aleatoric standard deviation.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        model,
        checkpoint,
        device
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Physics-Informed 3D Encoder-Decoder model.

        checkpoint : str
            Path to the trained model checkpoint.

        device : torch.device
            CPU or CUDA device.
        """

        self.model = model

        self.checkpoint = checkpoint

        self.device = device

        # -------------------------------------------------
        # Validate checkpoint path
        # -------------------------------------------------

        if not os.path.isfile(self.checkpoint):

            raise FileNotFoundError(
                "\nCheckpoint not found:\n"
                f"{self.checkpoint}"
            )

        # -------------------------------------------------
        # Move model to selected device
        # -------------------------------------------------

        self.model = self.model.to(
            self.device
        )

        # -------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------

        self._load_checkpoint()

        # -------------------------------------------------
        # Use evaluation mode for deterministic inference.
        #
        # This keeps Dropout disabled.
        #
        # MC Dropout inference is handled separately by
        # the MC Dropout component.
        # -------------------------------------------------

        self.model.eval()

    # =====================================================
    # CHECKPOINT LOADING
    # =====================================================

    def _load_checkpoint(self):
        """
        Load model parameters from the checkpoint.

        Supported checkpoint formats:

            1. Direct state_dict

            2. Dictionary containing:
                   model_state_dict

            3. Dictionary containing:
                   state_dict

            4. Dictionary containing:
                   model
               where model is itself a state_dict.
        """

        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.device
        )

        # -------------------------------------------------
        # Validate checkpoint type
        # -------------------------------------------------

        if not isinstance(
            checkpoint,
            dict
        ):

            raise TypeError(
                "Unsupported checkpoint format. "
                "Expected a dictionary containing a "
                "model state_dict."
            )

        # -------------------------------------------------
        # Extract model state dictionary.
        # -------------------------------------------------

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif (
            "model"
            in checkpoint
            and isinstance(
                checkpoint["model"],
                dict
            )
        ):

            state_dict = (
                checkpoint["model"]
            )

        else:

            # -------------------------------------------------
            # Assume the checkpoint itself is a state_dict.
            # -------------------------------------------------

            state_dict = checkpoint

        # -------------------------------------------------
        # Load model parameters.
        # -------------------------------------------------

        self.model.load_state_dict(
            state_dict
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        corrupted_cube
    ):
        """
        Perform one deterministic forward pass.

        Parameters
        ----------
        corrupted_cube : torch.Tensor

            Accepted shapes:

                [C, D, H, W]

            or:

                [B, C, D, H, W]

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed seismic volume.

        travel_time : torch.Tensor
            Predicted seismic travel-time field.

        log_variance : torch.Tensor
            Predicted logarithmic aleatoric variance:

                log_variance = log(sigma_a^2)

        aleatoric_std : torch.Tensor
            Aleatoric standard deviation:

                sigma_a = exp(0.5 * log_variance)

        Notes
        -----
        This method does NOT estimate epistemic uncertainty.

        It also does NOT calculate total predictive
        uncertainty.

        Epistemic uncertainty requires MC Dropout samples.
        """

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if not isinstance(
            corrupted_cube,
            torch.Tensor
        ):

            raise TypeError(
                "corrupted_cube must be a "
                "torch.Tensor."
            )

        # -------------------------------------------------
        # Validate finite input values.
        # -------------------------------------------------

        if not torch.isfinite(
            corrupted_cube
        ).all():

            raise ValueError(
                "corrupted_cube contains "
                "NaN or infinite values."
            )

        # =================================================
        # PREPARE INPUT
        # =================================================

        # -------------------------------------------------
        # Add batch dimension when input is:
        #
        #     [C,D,H,W]
        #
        # Network expects:
        #
        #     [B,C,D,H,W]
        # -------------------------------------------------

        if corrupted_cube.dim() == 4:

            corrupted_cube = (
                corrupted_cube.unsqueeze(0)
            )

        # -------------------------------------------------
        # Validate final input dimension.
        # -------------------------------------------------

        if corrupted_cube.dim() != 5:

            raise ValueError(
                "Predictor expects input with shape "
                "[C,D,H,W] or [B,C,D,H,W]. "
                f"Received: "
                f"{tuple(corrupted_cube.shape)}"
            )

        # -------------------------------------------------
        # Move input to selected device.
        # -------------------------------------------------

        corrupted_cube = (
            corrupted_cube.to(
                self.device
            )
        )

        # =================================================
        # DETERMINISTIC FORWARD PASS
        # =================================================

        # -------------------------------------------------
        # Ensure Dropout is disabled.
        #
        # MC Dropout must NOT be performed through this
        # deterministic Predictor.
        # -------------------------------------------------

        self.model.eval()

        # -------------------------------------------------
        # Disable gradient computation because this is
        # inference rather than training.
        # -------------------------------------------------

        with torch.inference_mode():

            # -------------------------------------------------
            # Final network interface:
            #
            #     reconstruction
            #     travel_time
            #     log_variance
            # -------------------------------------------------

            (
                reconstruction,
                travel_time,
                log_variance
            ) = self.model(
                corrupted_cube
            )

        # =================================================
        # OUTPUT VALIDATION
        # =================================================

        # -------------------------------------------------
        # All three network outputs must be finite.
        # -------------------------------------------------

        if not torch.isfinite(
            reconstruction
        ).all():

            raise RuntimeError(
                "Reconstruction contains "
                "NaN or infinite values."
            )

        if not torch.isfinite(
            travel_time
        ).all():

            raise RuntimeError(
                "Travel-time prediction contains "
                "NaN or infinite values."
            )

        if not torch.isfinite(
            log_variance
        ).all():

            raise RuntimeError(
                "Log-variance prediction contains "
                "NaN or infinite values."
            )

        # -------------------------------------------------
        # Validate output shapes.
        # -------------------------------------------------

        if reconstruction.shape != (
            corrupted_cube.shape
        ):

            raise RuntimeError(
                "Reconstruction shape does not match "
                "input shape.\n"
                f"Input: "
                f"{tuple(corrupted_cube.shape)}\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}"
            )

        if travel_time.shape != (
            reconstruction.shape
        ):

            raise RuntimeError(
                "Travel-time output shape does not "
                "match reconstruction shape.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Travel time: "
                f"{tuple(travel_time.shape)}"
            )

        if log_variance.shape != (
            reconstruction.shape
        ):

            raise RuntimeError(
                "Log-variance output shape does not "
                "match reconstruction shape.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Log variance: "
                f"{tuple(log_variance.shape)}"
            )

        # =================================================
        # ALEATORIC UNCERTAINTY
        # =================================================

        # -------------------------------------------------
        # The network predicts:
        #
        #     s = log(sigma_a^2)
        #
        # Therefore:
        #
        #     sigma_a^2 = exp(s)
        #
        # and:
        #
        #     sigma_a = exp(0.5s)
        #
        # Clamp s to prevent numerical overflow or
        # underflow.
        # -------------------------------------------------

        log_variance = torch.clamp(
            log_variance,
            min=-10.0,
            max=10.0
        )

        # -------------------------------------------------
        # Convert log variance to aleatoric standard
        # deviation.
        # -------------------------------------------------

        aleatoric_std = torch.exp(
            0.5 * log_variance
        )

        # -------------------------------------------------
        # Final numerical validation.
        # -------------------------------------------------

        if not torch.isfinite(
            aleatoric_std
        ).all():

            raise RuntimeError(
                "Aleatoric standard deviation "
                "contains NaN or infinite values."
            )

        if (
            aleatoric_std < 0
        ).any():

            raise RuntimeError(
                "Aleatoric standard deviation "
                "contains negative values."
            )

        # =================================================
        # RETURN RESULTS
        # =================================================

        return (
            reconstruction.cpu(),
            travel_time.cpu(),
            log_variance.cpu(),
            aleatoric_std.cpu()
        )