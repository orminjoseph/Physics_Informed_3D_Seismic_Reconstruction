"""
=========================================================
3D Seismic Model Predictor
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

The final Network3D produces three outputs:

    1. reconstructed_cube
    2. travel_time
    3. log_variance

The Predictor exposes:

    reconstruction
    travel_time
    uncertainty

where:

    uncertainty = exp(0.5 * log_variance)

Tensor convention:

    Input:
        [B, C, D, H, W]

    Reconstruction:
        [B, C, D, H, W]

    Travel time:
        [B, C, D, H, W]

    Uncertainty:
        [B, C, D, H, W]

=========================================================
"""

import os

import torch


class Predictor:
    """
    Inference wrapper for the final Physics-Informed
    3D Encoder-Decoder network.
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
            Network3D model.

        checkpoint : str
            Path to trained checkpoint.

        device : torch.device
            CPU or CUDA device.
        """

        self.model = model

        self.checkpoint = checkpoint

        self.device = device

        # -------------------------------------------------
        # Validate checkpoint path
        # -------------------------------------------------

        if not os.path.exists(self.checkpoint):

            raise FileNotFoundError(
                "\nCheckpoint not found:\n"
                f"{self.checkpoint}"
            )

        # -------------------------------------------------
        # Move model to device
        # -------------------------------------------------

        self.model = self.model.to(
            self.device
        )

        # -------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------

        self._load_checkpoint()

        # -------------------------------------------------
        # Evaluation mode
        # -------------------------------------------------

        self.model.eval()

    # =====================================================
    # CHECKPOINT LOADING
    # =====================================================

    def _load_checkpoint(self):
        """
        Load the trained model parameters.

        Supports checkpoints stored either as:

            state_dict

        or as dictionaries containing:

            model_state_dict
            state_dict
            model
        """

        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.device
        )

        # -------------------------------------------------
        # Case 1:
        # checkpoint is already a state dictionary
        # -------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ):

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
                # Assume the dictionary itself is a state_dict.
                # -------------------------------------------------

                state_dict = checkpoint

        else:

            raise TypeError(
                "Unsupported checkpoint format."
            )

        # -------------------------------------------------
        # Load model parameters
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
        Perform inference on one corrupted seismic cube.

        Parameters
        ----------
        corrupted_cube : torch.Tensor

            Shape:

                [C, D, H, W]

            or:

                [B, C, D, H, W]

        Returns
        -------
        reconstruction : torch.Tensor

            Reconstructed seismic volume.

        travel_time : torch.Tensor

            Predicted seismic travel-time field.

        uncertainty : torch.Tensor

            Predictive uncertainty represented as
            standard deviation:

                sigma = exp(0.5 * log_variance)
        """

        # =================================================
        # EVALUATION MODE
        # =================================================

        self.model.eval()

        # =================================================
        # DISABLE GRADIENT COMPUTATION
        # =================================================

        with torch.no_grad():

            # -------------------------------------------------
            # Add batch dimension if necessary.
            #
            # Dataset sample:
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
            # Validate input dimension.
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
            # FINAL NETWORK INTERFACE
            # =================================================
            #
            # Network3D returns:
            #
            #     reconstruction
            #     travel_time
            #     log_variance
            # =================================================

            (
                reconstruction,
                travel_time,
                log_variance
            ) = self.model(
                corrupted_cube
            )

            # =================================================
            # UNCERTAINTY
            # =================================================

            # -------------------------------------------------
            # Prevent numerical overflow/underflow.
            # -------------------------------------------------

            log_variance = torch.clamp(
                log_variance,
                min=-10.0,
                max=10.0
            )

            # -------------------------------------------------
            # Convert logarithmic variance to standard
            # deviation:
            #
            #     sigma = exp(0.5 * log(sigma²))
            #
            # Therefore:
            #
            #     sigma = exp(0.5 * log_variance)
            # -------------------------------------------------

            uncertainty = torch.exp(
                0.5 * log_variance
            )

        # =====================================================
        # RETURN RESULTS
        # =====================================================

        return (
            reconstruction.cpu(),
            travel_time.cpu(),
            uncertainty.cpu()
        )