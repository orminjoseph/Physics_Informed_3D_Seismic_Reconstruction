"""
=========================================================
Trainer
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Training pipeline:

    Input seismic cube
          |
          v
    Physics-Informed 3D Network
          |
          +---- Reconstruction
          |
          +---- Travel-Time Field
          |
          +---- Log Variance
          |
          v
    Composite Total Loss
          |
          +---- MAE
          +---- Eikonal Physics
          +---- Uncertainty
          +---- SSIM

Dataset batch convention:

    inputs
    targets
    mask
    velocity_model

Tensor convention:

    [B, C, D, H, W]

Important numerical-stability policy:

    Large finite gradients
        -> WARNING
        -> gradient clipping
        -> optimizer step

    NaN / Inf gradients
        -> FATAL ERROR

The Trainer does NOT modify the physics equation.
Large Eikonal gradients are diagnosed and controlled
through gradient clipping.

Author: Ormin Joseph
=========================================================
"""

import csv
import json
import os
import time

import matplotlib.pyplot as plt
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.experiment_manager import ExperimentManager


# =========================================================
# DEBUG OPTIONS
# =========================================================

DEBUG_VALIDATION = False

# ---------------------------------------------------------
# Gradient warning threshold.
#
# This is NOT a failure threshold.
#
# A finite gradient larger than this value generates a
# warning but is subsequently controlled by clipping.
# ---------------------------------------------------------

GRADIENT_WARNING_THRESHOLD = 1.0e6

# ---------------------------------------------------------
# Maximum gradient norm used by clipping.
# ---------------------------------------------------------

MAX_GRAD_NORM = 1.0


class Trainer:
    """
    Trainer for the Physics-Informed 3D Encoder-Decoder
    seismic reconstruction framework.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        experiment_manager=None
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Physics-informed 3D encoder-decoder network.

        criterion : torch.nn.Module
            Composite TotalLoss.

        optimizer : torch.optim.Optimizer
            Optimizer used for training.

        device : torch.device or str
            Training device.

        experiment_manager : ExperimentManager, optional
            Experiment directory manager.
        """

        # -------------------------------------------------
        # Validate required objects
        # -------------------------------------------------

        if model is None:
            raise ValueError(
                "model cannot be None."
            )

        if criterion is None:
            raise ValueError(
                "criterion cannot be None."
            )

        if optimizer is None:
            raise ValueError(
                "optimizer cannot be None."
            )

        # -------------------------------------------------
        # Store core components
        # -------------------------------------------------

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = torch.device(device)

        # -------------------------------------------------
        # Experiment manager
        # -------------------------------------------------

        self.experiment = (
            experiment_manager
            if experiment_manager is not None
            else ExperimentManager()
        )

        # -------------------------------------------------
        # Move model to selected device
        # -------------------------------------------------

        self.model.to(
            self.device
        )

        # =================================================
        # TENSORBOARD
        # =================================================

        self.writer = SummaryWriter(
            log_dir=self.experiment.tensorboard
        )

        # =================================================
        # CHECKPOINT DIRECTORY
        # =================================================

        self.checkpoint_directory = (
            self.experiment.global_checkpoints
        )

        os.makedirs(
            self.checkpoint_directory,
            exist_ok=True
        )

        # =================================================
        # TRAINING LOG
        # =================================================

        self.log_file = os.path.join(
            self.experiment.logs,
            "training_history.csv"
        )

        os.makedirs(
            self.experiment.logs,
            exist_ok=True
        )

        # -------------------------------------------------
        # Create CSV header only when the file does not
        # already exist.
        # -------------------------------------------------

        if not os.path.exists(
            self.log_file
        ):

            with open(
                self.log_file,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Epoch",
                    "Train_Total",
                    "Validation_Total",
                    "Train_MAE",
                    "Validation_MAE",
                    "Train_Physics",
                    "Validation_Physics",
                    "Train_Uncertainty",
                    "Validation_Uncertainty",
                    "Train_SSIM",
                    "Validation_SSIM",
                    "RMSE",
                    "PSNR",
                    "SNR",
                    "Metric_SSIM",
                    "Learning_Rate",
                    "Train_Gradient_Norm",
                    "Train_Maximum_Gradient"
                ])

        # =================================================
        # EARLY STOPPING
        # =================================================

        self.best_validation_loss = float(
            "inf"
        )

        self.best_epoch = 0

        self.patience = 10

        self.wait = 0

        # =================================================
        # LEARNING-RATE SCHEDULER
        # =================================================

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        # =================================================
        # BEST RECONSTRUCTION METRICS
        # =================================================

        self.best_metrics = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "PSNR": 0.0,
            "SNR": 0.0,
            "SSIM": 0.0
        }

        # =================================================
        # CURRENT EPOCH
        # =================================================

        self.current_epoch = 0

        # =================================================
        # TRAINING HISTORY
        # =================================================

        self.history = {

            "total": [],
            "mae": [],
            "physics": [],
            "uncertainty": [],
            "ssim": [],
            "learning_rate": [],

            "gradient_norm": [],
            "maximum_gradient": []
        }

        # =================================================
        # VALIDATION HISTORY
        # =================================================

        self.validation_history = {

            "total": [],
            "mae": [],
            "physics": [],
            "uncertainty": [],
            "ssim": [],

            "metric_mae": [],
            "metric_rmse": [],
            "metric_psnr": [],
            "metric_snr": [],
            "metric_ssim": []
        }

    # =====================================================
    # VALIDATE DATA BATCH
    # =====================================================

    @staticmethod
    def _validate_batch(
        batch,
        batch_name="training"
    ):
        """
        Validate dataset batch structure.

        Expected:

            inputs,
            targets,
            mask,
            velocity_model
        """

        if not isinstance(
            batch,
            (tuple, list)
        ):

            raise TypeError(
                f"{batch_name} batch must be "
                "a tuple or list."
            )

        if len(batch) != 4:

            raise ValueError(
                f"Expected {batch_name} batch to contain "
                "four tensors: "
                "(inputs, targets, mask, velocity_model)."
            )

        (
            inputs,
            targets,
            mask,
            velocity_model
        ) = batch

        # -------------------------------------------------
        # Validate tensors
        # -------------------------------------------------

        tensors = {
            "inputs": inputs,
            "targets": targets,
            "mask": mask,
            "velocity_model": velocity_model
        }

        for name, tensor in tensors.items():

            if not isinstance(
                tensor,
                torch.Tensor
            ):

                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

        # -------------------------------------------------
        # Main tensors must be five-dimensional.
        # -------------------------------------------------

        for name in (
            "inputs",
            "targets",
            "mask",
            "velocity_model"
        ):

            if tensors[name].ndim != 5:

                raise ValueError(
                    f"{name} must have shape "
                    "[B,C,D,H,W]. "
                    f"Received: "
                    f"{tuple(tensors[name].shape)}."
                )

        # -------------------------------------------------
        # Inputs and targets must have identical shapes.
        # -------------------------------------------------

        if inputs.shape != targets.shape:

            raise ValueError(
                f"{batch_name}: inputs and targets "
                "must have identical shapes.\n"
                f"Inputs : {tuple(inputs.shape)}\n"
                f"Targets: {tuple(targets.shape)}"
            )

        # -------------------------------------------------
        # Mask must match seismic data shape.
        # -------------------------------------------------

        if mask.shape != targets.shape:

            raise ValueError(
                f"{batch_name}: mask and targets "
                "must have identical shapes.\n"
                f"Mask   : {tuple(mask.shape)}\n"
                f"Targets: {tuple(targets.shape)}"
            )

        # -------------------------------------------------
        # Velocity model must have the same spatial shape.
        # -------------------------------------------------

        if velocity_model.shape != targets.shape:

            raise ValueError(
                f"{batch_name}: velocity_model and targets "
                "must have identical shapes.\n"
                f"Velocity: {tuple(velocity_model.shape)}\n"
                f"Targets : {tuple(targets.shape)}"
            )

        # -------------------------------------------------
        # Check numerical validity.
        # -------------------------------------------------

        for name, tensor in tensors.items():

            if not torch.isfinite(
                tensor
            ).all():

                raise ValueError(
                    f"{batch_name}: {name} contains "
                    "NaN or infinite values."
                )

        return (
            inputs,
            targets,
            mask,
            velocity_model
        )

    # =====================================================
    # GRADIENT DIAGNOSTICS
    # =====================================================

    def _inspect_gradients(
        self,
        batch_index
    ):
        """
        Inspect model gradients after backward propagation.

        Returns
        -------
        tuple
            raw_gradient_norm,
            maximum_gradient
        """

        maximum_gradient = 0.0

        total_gradient_norm_squared = 0.0

        # -------------------------------------------------
        # Examine every trainable parameter.
        # -------------------------------------------------

        for parameter in self.model.parameters():

            if not parameter.requires_grad:

                continue

            if parameter.grad is None:

                continue

            # -------------------------------------------------
            # NaN / Inf gradients are fatal.
            # -------------------------------------------------

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "Non-finite gradient detected at "
                    f"batch {batch_index + 1}."
                )

            # -------------------------------------------------
            # Maximum absolute gradient.
            # -------------------------------------------------

            parameter_max_gradient = (
                parameter.grad.detach()
                .abs()
                .max()
                .item()
            )

            maximum_gradient = max(
                maximum_gradient,
                parameter_max_gradient
            )

            # -------------------------------------------------
            # Parameter L2 gradient norm.
            # -------------------------------------------------

            parameter_norm = (
                parameter.grad.detach()
                .norm(2)
                .item()
            )

            total_gradient_norm_squared += (
                parameter_norm ** 2
            )

        # -------------------------------------------------
        # Total gradient norm.
        # -------------------------------------------------

        raw_gradient_norm = (
            total_gradient_norm_squared ** 0.5
        )

        # -------------------------------------------------
        # Large finite gradient is a warning, not an error.
        # -------------------------------------------------

        if (
            maximum_gradient
            >
            GRADIENT_WARNING_THRESHOLD
        ):

            print(
                f"WARNING: Large gradient detected "
                f"at batch {batch_index + 1}: "
                f"{maximum_gradient:.6e}",
                flush=True
            )

        return (
            raw_gradient_norm,
            maximum_gradient
        )

    # =====================================================
    # CLIP GRADIENTS
    # =====================================================

    def _clip_gradients(
        self,
        raw_gradient_norm,
        batch_index
    ):
        """
        Clip gradients and return the resulting norm.

        Large finite gradients are expected to be handled
        here rather than treated as fatal errors.
        """

        clipped_gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=MAX_GRAD_NORM
            )
        )

        # -------------------------------------------------
        # clip_grad_norm_ returns the norm BEFORE clipping.
        # -------------------------------------------------

        clipped_gradient_norm_value = float(
            clipped_gradient_norm
        )

        # -------------------------------------------------
        # Ensure the returned norm itself is finite.
        # -------------------------------------------------

        if not torch.isfinite(
            torch.tensor(
                clipped_gradient_norm_value,
                device=self.device
            )
        ):

            raise RuntimeError(
                "Gradient clipping produced a "
                "non-finite gradient norm at "
                f"batch {batch_index + 1}."
            )

        # -------------------------------------------------
        # Optional diagnostic.
        # -------------------------------------------------

        if (
            raw_gradient_norm
            >
            MAX_GRAD_NORM
            and batch_index % 5 == 0
        ):

            print(
                f"Gradient clipping applied: "
                f"{raw_gradient_norm:.6e} "
                f"-> maximum norm {MAX_GRAD_NORM:.6e}",
                flush=True
            )

        return clipped_gradient_norm_value

    # =====================================================
    # TRAINING EPOCH
    # =====================================================

    def train_epoch(
        self,
        dataloader
    ):
        """
        Train the network for one epoch.

        Dataset batch:

            inputs,
            targets,
            mask,
            velocity_model

        Network output:

            reconstruction,
            travel_time,
            log_variance
        """

        self.model.train()

        # =================================================
        # LOSS ACCUMULATORS
        # =================================================

        running_total = 0.0

        running_mae = 0.0

        running_physics = 0.0

        running_uncertainty = 0.0

        running_ssim = 0.0

        # =================================================
        # GRADIENT ACCUMULATORS
        # =================================================

        running_gradient_norm = 0.0

        maximum_gradient_seen = 0.0

        # =================================================
        # VALIDATE DATALOADER
        # =================================================

        num_batches = len(
            dataloader
        )

        if num_batches == 0:

            raise RuntimeError(
                "Training DataLoader contains no batches."
            )

        # =================================================
        # BATCH LOOP
        # =================================================

        for batch_index, batch in enumerate(
            dataloader
        ):

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if batch_index % 5 == 0:

                print(
                    f"Training batch "
                    f"{batch_index + 1}/{num_batches}",
                    flush=True
                )

            # =================================================
            # VALIDATE DATASET BATCH
            # =================================================

            (
                inputs,
                targets,
                mask,
                velocity_model
            ) = self._validate_batch(
                batch,
                batch_name="training"
            )

            # =================================================
            # MOVE DATA TO DEVICE
            # =================================================

            inputs = inputs.to(
                self.device,
                non_blocking=True
            )

            targets = targets.to(
                self.device,
                non_blocking=True
            )

            mask = mask.to(
                self.device,
                non_blocking=True
            )

            velocity_model = velocity_model.to(
                self.device,
                non_blocking=True
            )

            # -------------------------------------------------
            # Mask is deliberately not applied here.
            #
            # The current TotalLoss API does not accept mask.
            #
            # This prevents the Trainer from silently applying
            # an incorrect masking operation.
            #
            # Mask-aware reconstruction loss can be introduced
            # later as a controlled change to TotalLoss.
            # -------------------------------------------------

            # =================================================
            # CLEAR GRADIENTS
            # =================================================

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # =================================================
            # FORWARD PASS
            # =================================================

            (
                reconstruction,
                travel_time,
                log_variance
            ) = self.model(
                inputs
            )

            # =================================================
            # OUTPUT SHAPE VALIDATION
            # =================================================

            if reconstruction.shape != targets.shape:

                raise RuntimeError(
                    "Reconstruction and target shapes "
                    "do not match.\n"
                    f"Reconstruction: "
                    f"{tuple(reconstruction.shape)}\n"
                    f"Target: "
                    f"{tuple(targets.shape)}"
                )

            if travel_time.shape != velocity_model.shape:

                raise RuntimeError(
                    "Travel-time and velocity-model shapes "
                    "do not match.\n"
                    f"Travel-time: "
                    f"{tuple(travel_time.shape)}\n"
                    f"Velocity: "
                    f"{tuple(velocity_model.shape)}"
                )

            if log_variance.shape != reconstruction.shape:

                raise RuntimeError(
                    "Log-variance and reconstruction shapes "
                    "do not match.\n"
                    f"Log-variance: "
                    f"{tuple(log_variance.shape)}\n"
                    f"Reconstruction: "
                    f"{tuple(reconstruction.shape)}"
                )

            # =================================================
            # OUTPUT FINITENESS
            # =================================================

            outputs = {
                "reconstruction": reconstruction,
                "travel_time": travel_time,
                "log_variance": log_variance
            }

            for name, tensor in outputs.items():

                if not torch.isfinite(
                    tensor
                ).all():

                    raise RuntimeError(
                        f"Non-finite {name} detected "
                        f"at training batch "
                        f"{batch_index + 1}."
                    )

            # =================================================
            # COMPOSITE LOSS
            # =================================================

            #
            # The current dataset does not provide:
            #
            #     source_indices
            #     travel_time_target
            #
            # Therefore they are intentionally omitted.
            #
            # The current TotalLoss then uses:
            #
            #     MAE
            #     Eikonal physics
            #     uncertainty
            #     SSIM
            #

            losses = self.criterion(
                reconstruction,
                targets,
                travel_time,
                velocity_model,
                log_variance
            )

            # =================================================
            # TOTAL LOSS
            # =================================================

            loss = losses["total"]

            # -------------------------------------------------
            # Loss must be finite.
            # -------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite training loss detected "
                    f"at batch {batch_index + 1}."
                )

            # =================================================
            # BACKPROPAGATION
            # =================================================

            loss.backward()

            # =================================================
            # GRADIENT INSPECTION
            # =================================================

            (
                raw_gradient_norm,
                maximum_gradient
            ) = self._inspect_gradients(
                batch_index
            )

            # -------------------------------------------------
            # Accumulate gradient diagnostics.
            # -------------------------------------------------

            running_gradient_norm += (
                raw_gradient_norm
            )

            maximum_gradient_seen = max(
                maximum_gradient_seen,
                maximum_gradient
            )

            # =================================================
            # GRADIENT CLIPPING
            # =================================================

            self._clip_gradients(
                raw_gradient_norm,
                batch_index
            )

            # =================================================
            # OPTIMIZER UPDATE
            # =================================================

            self.optimizer.step()

            # =================================================
            # ACCUMULATE LOSSES
            # =================================================

            running_total += (
                losses["total"]
                .detach()
                .item()
            )

            running_mae += (
                losses["mae"]
                .detach()
                .item()
            )

            running_physics += (
                losses["physics"]
                .detach()
                .item()
            )

            running_uncertainty += (
                losses["uncertainty"]
                .detach()
                .item()
            )

            running_ssim += (
                losses["ssim"]
                .detach()
                .item()
            )

        # =================================================
        # RETURN AVERAGE TRAINING RESULTS
        # =================================================

        return {

            "total":
                running_total / num_batches,

            "mae":
                running_mae / num_batches,

            "physics":
                running_physics / num_batches,

            "uncertainty":
                running_uncertainty / num_batches,

            "ssim":
                running_ssim / num_batches,

            "gradient_norm":
                running_gradient_norm / num_batches,

            "maximum_gradient":
                maximum_gradient_seen
        }

    # =====================================================
    # VALIDATION EPOCH
    # =====================================================

    def validate_epoch(
        self,
        dataloader
    ):
        """
        Validate the model for one epoch.
        """

        self.model.eval()

        # =================================================
        # LOSS ACCUMULATORS
        # =================================================

        running_total = 0.0

        running_mae = 0.0

        running_physics = 0.0

        running_uncertainty = 0.0

        running_ssim = 0.0

        # =================================================
        # METRIC ACCUMULATORS
        # =================================================

        running_metric_mae = 0.0

        running_metric_rmse = 0.0

        running_metric_psnr = 0.0

        running_metric_snr = 0.0

        running_metric_ssim = 0.0

        # =================================================
        # VALIDATE DATALOADER
        # =================================================

        num_batches = len(
            dataloader
        )

        if num_batches == 0:

            raise RuntimeError(
                "Validation DataLoader contains no batches."
            )

        # =================================================
        # VALIDATION LOOP
        # =================================================

        with torch.no_grad():

            for batch_index, batch in enumerate(
                dataloader
            ):

                # =================================================
                # VALIDATE BATCH
                # =================================================

                (
                    inputs,
                    targets,
                    mask,
                    velocity_model
                ) = self._validate_batch(
                    batch,
                    batch_name="validation"
                )

                # =================================================
                # MOVE TO DEVICE
                # =================================================

                inputs = inputs.to(
                    self.device,
                    non_blocking=True
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True
                )

                mask = mask.to(
                    self.device,
                    non_blocking=True
                )

                velocity_model = velocity_model.to(
                    self.device,
                    non_blocking=True
                )

                # =================================================
                # FORWARD PASS
                # =================================================

                (
                    reconstruction,
                    travel_time,
                    log_variance
                ) = self.model(
                    inputs
                )

                # =================================================
                # OUTPUT SHAPE VALIDATION
                # =================================================

                if reconstruction.shape != targets.shape:

                    raise RuntimeError(
                        "Validation reconstruction and target "
                        "shapes do not match.\n"
                        f"Reconstruction: "
                        f"{tuple(reconstruction.shape)}\n"
                        f"Target: "
                        f"{tuple(targets.shape)}"
                    )

                if travel_time.shape != velocity_model.shape:

                    raise RuntimeError(
                        "Validation travel-time and "
                        "velocity-model shapes do not match.\n"
                        f"Travel-time: "
                        f"{tuple(travel_time.shape)}\n"
                        f"Velocity: "
                        f"{tuple(velocity_model.shape)}"
                    )

                if log_variance.shape != reconstruction.shape:

                    raise RuntimeError(
                        "Validation log-variance and "
                        "reconstruction shapes do not match."
                    )

                # =================================================
                # OUTPUT FINITENESS
                # =================================================

                for name, tensor in {
                    "reconstruction": reconstruction,
                    "travel_time": travel_time,
                    "log_variance": log_variance
                }.items():

                    if not torch.isfinite(
                        tensor
                    ).all():

                        raise RuntimeError(
                            f"Non-finite validation "
                            f"{name} detected at "
                            f"batch {batch_index + 1}."
                        )

                # =================================================
                # DEBUG INFORMATION
                # =================================================

                if (
                    DEBUG_VALIDATION
                    and batch_index % 20 == 0
                ):

                    print()
                    print(
                        "Validation Batch Statistics"
                    )
                    print(
                        "---------------------------"
                    )

                    print(
                        f"Prediction : "
                        f"min={reconstruction.min().item():.6f}, "
                        f"max={reconstruction.max().item():.6f}, "
                        f"mean={reconstruction.mean().item():.6f}"
                    )

                    print(
                        f"Target : "
                        f"min={targets.min().item():.6f}, "
                        f"max={targets.max().item():.6f}, "
                        f"mean={targets.mean().item():.6f}"
                    )

                    print(
                        f"Travel Time : "
                        f"min={travel_time.min().item():.6e}, "
                        f"max={travel_time.max().item():.6e}, "
                        f"mean={travel_time.mean().item():.6e}"
                    )

                    print(
                        f"Velocity : "
                        f"min={velocity_model.min().item():.6f}, "
                        f"max={velocity_model.max().item():.6f}, "
                        f"mean={velocity_model.mean().item():.6f}"
                    )

                    print(
                        f"Log Variance : "
                        f"min={log_variance.min().item():.6f}, "
                        f"max={log_variance.max().item():.6f}, "
                        f"mean={log_variance.mean().item():.6f}"
                    )

                # =================================================
                # COMPOSITE LOSS
                # =================================================

                losses = self.criterion(
                    reconstruction,
                    targets,
                    travel_time,
                    velocity_model,
                    log_variance
                )

                # =================================================
                # VALIDATION LOSS FINITENESS
                # =================================================

                if not torch.isfinite(
                    losses["total"]
                ):

                    raise RuntimeError(
                        "Non-finite validation loss detected "
                        f"at batch {batch_index + 1}."
                    )

                # =================================================
                # RECONSTRUCTION METRICS
                # =================================================

                metric_mae = mae(
                    reconstruction,
                    targets
                )

                metric_rmse = rmse(
                    reconstruction,
                    targets
                )

                metric_psnr = psnr(
                    reconstruction,
                    targets
                )

                metric_snr = snr(
                    reconstruction,
                    targets
                )

                metric_ssim = ssim(
                    reconstruction,
                    targets
                )

                # =================================================
                # METRIC FINITENESS
                # =================================================

                metrics = {
                    "MAE": metric_mae,
                    "RMSE": metric_rmse,
                    "PSNR": metric_psnr,
                    "SNR": metric_snr,
                    "SSIM": metric_ssim
                }

                for name, metric in metrics.items():

                    if not torch.isfinite(
                        metric
                    ):

                        raise RuntimeError(
                            f"Non-finite validation "
                            f"{name} metric detected."
                        )

                # =================================================
                # ACCUMULATE LOSSES
                # =================================================

                running_total += (
                    losses["total"]
                    .item()
                )

                running_mae += (
                    losses["mae"]
                    .item()
                )

                running_physics += (
                    losses["physics"]
                    .item()
                )

                running_uncertainty += (
                    losses["uncertainty"]
                    .item()
                )

                running_ssim += (
                    losses["ssim"]
                    .item()
                )

                # =================================================
                # ACCUMULATE METRICS
                # =================================================

                running_metric_mae += (
                    metric_mae.item()
                )

                running_metric_rmse += (
                    metric_rmse.item()
                )

                running_metric_psnr += (
                    metric_psnr.item()
                )

                running_metric_snr += (
                    metric_snr.item()
                )

                running_metric_ssim += (
                    metric_ssim.item()
                )

                # =================================================
                # SAVE FIRST VALIDATION VISUALIZATION
                # =================================================

                if batch_index == 0:

                    uncertainty = torch.exp(
                        0.5 * log_variance
                    )

                    self.save_validation_visualization(
                        inputs,
                        targets,
                        reconstruction,
                        travel_time,
                        uncertainty,
                        self.current_epoch
                    )

        # =================================================
        # RETURN AVERAGE VALIDATION RESULTS
        # =================================================

        return {

            "total":
                running_total / num_batches,

            "mae":
                running_mae / num_batches,

            "physics":
                running_physics / num_batches,

            "uncertainty":
                running_uncertainty / num_batches,

            "ssim":
                running_ssim / num_batches,

            "metric_mae":
                running_metric_mae / num_batches,

            "metric_rmse":
                running_metric_rmse / num_batches,

            "metric_psnr":
                running_metric_psnr / num_batches,

            "metric_snr":
                running_metric_snr / num_batches,

            "metric_ssim":
                running_metric_ssim / num_batches
        }

    # =====================================================
    # VALIDATION VISUALIZATION
    # =====================================================

    def save_validation_visualization(
        self,
        inputs,
        targets,
        reconstruction,
        travel_time,
        uncertainty,
        epoch
    ):
        """
        Save a validation visualization.

        Figure contents:

            1. Incomplete input
            2. Ground truth
            3. Reconstruction
            4. Travel-time field
            5. Predictive uncertainty
            6. Absolute reconstruction error
        """

        output_directory = (
            self.experiment.training_progress
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        # =================================================
        # SELECT FIRST SAMPLE AND FIRST CHANNEL
        # =================================================

        inputs_np = (
            inputs[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        targets_np = (
            targets[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        reconstruction_np = (
            reconstruction[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        travel_time_np = (
            travel_time[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        uncertainty_np = (
            uncertainty[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # =================================================
        # MIDDLE DEPTH SLICE
        # =================================================

        middle = (
            inputs_np.shape[0] // 2
        )

        # =================================================
        # CREATE FIGURE
        # =================================================

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9)
        )

        # =================================================
        # INPUT
        # =================================================

        axes[0, 0].imshow(
            inputs_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 0].set_title(
            "Incomplete Input"
        )

        # =================================================
        # TARGET
        # =================================================

        axes[0, 1].imshow(
            targets_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 1].set_title(
            "Ground Truth"
        )

        # =================================================
        # RECONSTRUCTION
        # =================================================

        axes[0, 2].imshow(
            reconstruction_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 2].set_title(
            "Reconstruction"
        )

        # =================================================
        # TRAVEL TIME
        # =================================================

        travel_image = axes[1, 0].imshow(
            travel_time_np[middle],
            cmap="viridis",
            aspect="auto"
        )

        axes[1, 0].set_title(
            "Predicted Travel Time"
        )

        fig.colorbar(
            travel_image,
            ax=axes[1, 0],
            fraction=0.046,
            pad=0.04
        )

        # =================================================
        # UNCERTAINTY
        # =================================================

        uncertainty_image = axes[1, 1].imshow(
            uncertainty_np[middle],
            cmap="hot",
            aspect="auto"
        )

        axes[1, 1].set_title(
            "Predictive Uncertainty"
        )

        fig.colorbar(
            uncertainty_image,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04
        )

        # =================================================
        # ABSOLUTE ERROR
        # =================================================

        absolute_error = (
            abs(
                reconstruction_np
                -
                targets_np
            )
        )

        error_image = axes[1, 2].imshow(
            absolute_error[middle],
            cmap="magma",
            aspect="auto"
        )

        axes[1, 2].set_title(
            "Absolute Reconstruction Error"
        )

        fig.colorbar(
            error_image,
            ax=axes[1, 2],
            fraction=0.046,
            pad=0.04
        )

        # =================================================
        # REMOVE AXES
        # =================================================

        for axis in axes.flat:

            axis.axis("off")

        plt.tight_layout()

        # =================================================
        # SAVE FIGURE
        # =================================================

        output_file = os.path.join(
            output_directory,
            f"epoch_{epoch:03d}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        loss
    ):
        """
        Save the latest checkpoint and historical epoch
        checkpoint.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.

        loss : float
            Validation loss associated with the checkpoint.
        """

        # =================================================
        # CHECKPOINT DATA
        # =================================================

        checkpoint = {

            "epoch":
                int(epoch),

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_validation_loss":
                float(
                    self.best_validation_loss
                ),

            "best_epoch":
                int(
                    self.best_epoch
                ),

            "best_metrics":
                dict(
                    self.best_metrics
                ),

            "early_stopping_wait":
                int(
                    self.wait
                ),

            "loss":
                float(
                    loss
                )
        }

        # =================================================
        # SAVE TRAINING STATE JSON
        # =================================================

        state_file = os.path.join(
            self.checkpoint_directory,
            "training_state.json"
        )

        with open(
            state_file,
            "w"
        ) as file:

            json.dump(
                {
                    "current_epoch":
                        int(epoch),

                    "completed_epoch":
                        int(epoch) + 1,

                    "best_epoch":
                        int(
                            self.best_epoch
                        ),

                    "best_epoch_number":
                        int(
                            self.best_epoch
                        ) + 1,

                    "best_validation_loss":
                        float(
                            self.best_validation_loss
                        ),

                    "best_metrics":
                        self.best_metrics,

                    "early_stopping_wait":
                        int(
                            self.wait
                        )
                },
                file,
                indent=4
            )

        # =================================================
        # SAVE LATEST CHECKPOINT
        # =================================================

        latest_file = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth"
        )

        torch.save(
            checkpoint,
            latest_file
        )

        # =================================================
        # HISTORICAL CHECKPOINT DIRECTORY
        # =================================================

        epoch_directory = os.path.join(
            self.checkpoint_directory,
            "epoch_sensitivity"
        )

        os.makedirs(
            epoch_directory,
            exist_ok=True
        )

        # =================================================
        # HUMAN-READABLE EPOCH NUMBER
        # =================================================

        epoch_number = (
            int(epoch) + 1
        )

        epoch_file = os.path.join(
            epoch_directory,
            f"epoch_{epoch_number:04d}.pth"
        )

        torch.save(
            checkpoint,
            epoch_file
        )

        # =================================================
        # CONSOLE REPORT
        # =================================================

        print(
            f"Checkpoint saved: "
            f"epoch_{epoch_number:04d}.pth"
        )

    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path=None
    ):
        """
        Load a previously saved checkpoint.

        Returns
        -------
        int
            Next zero-based epoch index.
        """

        # =================================================
        # DEFAULT CHECKPOINT
        # =================================================

        if checkpoint_path is None:

            checkpoint_path = os.path.join(
                self.checkpoint_directory,
                "latest_checkpoint.pth"
            )

        # =================================================
        # CHECK FILE
        # =================================================

        if not os.path.exists(
            checkpoint_path
        ):

            print()
            print("=" * 60)
            print(
                "No checkpoint found."
            )
            print(
                "Starting training from scratch."
            )
            print("=" * 60)

            self.wait = 0

            return 0

        # =================================================
        # LOAD CHECKPOINT
        # =================================================

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # =================================================
        # STORED EPOCH
        # =================================================

        stored_epoch = int(
            checkpoint.get(
                "epoch",
                -1
            )
        )

        stored_best_epoch = int(
            checkpoint.get(
                "best_epoch",
                stored_epoch
            )
        )

        # =================================================
        # DISPLAY CONTENTS
        # =================================================

        print()
        print("=" * 60)
        print(
            "CHECKPOINT CONTENTS"
        )
        print("=" * 60)

        print(
            "Stored Epoch:",
            stored_epoch
        )

        print(
            "Completed Epoch:",
            stored_epoch + 1
        )

        print(
            "Stored Best Epoch:",
            stored_best_epoch
        )

        print(
            "Best Epoch:",
            stored_best_epoch + 1
        )

        print("=" * 60)

        # =================================================
        # RESTORE MODEL
        # =================================================

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # =================================================
        # RESTORE OPTIMIZER
        # =================================================

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        # =================================================
        # RESTORE SCHEDULER
        # =================================================

        if (
            "scheduler_state_dict"
            in checkpoint
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        # =================================================
        # RESTORE BEST VALIDATION LOSS
        # =================================================

        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get(
                    "loss",
                    float("inf")
                )
            )
        )

        # =================================================
        # RESTORE BEST EPOCH
        # =================================================

        self.best_epoch = int(
            checkpoint.get(
                "best_epoch",
                stored_epoch
            )
        )

        # =================================================
        # RESTORE BEST METRICS
        # =================================================

        self.best_metrics = checkpoint.get(
            "best_metrics",
            {
                "MAE": 0.0,
                "RMSE": 0.0,
                "PSNR": 0.0,
                "SNR": 0.0,
                "SSIM": 0.0
            }
        )

        # =================================================
        # RESTORE EARLY STOPPING
        # =================================================

        self.wait = int(
            checkpoint.get(
                "early_stopping_wait",
                0
            )
        )

        # =================================================
        # NEXT EPOCH
        # =================================================

        start_epoch = (
            stored_epoch + 1
        )

        # =================================================
        # DISPLAY RESUME INFORMATION
        # =================================================

        print()
        print("=" * 60)
        print(
            "CHECKPOINT LOADED SUCCESSFULLY"
        )
        print("=" * 60)

        print(
            f"Last Completed Epoch : "
            f"{stored_epoch + 1}"
        )

        print(
            f"Resuming from Epoch   : "
            f"{start_epoch + 1}"
        )

        print(
            f"Best Epoch            : "
            f"{self.best_epoch + 1}"
        )

        print(
            f"Best Validation Loss  : "
            f"{self.best_validation_loss:.6f}"
        )

        print(
            f"Early-Stopping Wait   : "
            f"{self.wait}"
        )

        print("=" * 60)

        return start_epoch

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        train_dataloader,
        validation_dataloader,
        epochs,
        resume=True
    ):
        """
        Complete training procedure.
        """

        # =================================================
        # VALIDATE EPOCH COUNT
        # =================================================

        if epochs <= 0:

            raise ValueError(
                "epochs must be greater than zero."
            )

        start_time = time.time()

        # =================================================
        # CHECKPOINT PATH
        # =================================================

        latest_checkpoint = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth"
        )

        # =================================================
        # RESUME
        # =================================================

        if (
            resume
            and
            os.path.exists(
                latest_checkpoint
            )
        ):

            start_epoch = self.load_checkpoint(
                latest_checkpoint
            )

        else:

            start_epoch = 0

        # =================================================
        # CHECK WHETHER TRAINING IS ALREADY COMPLETE
        # =================================================

        if start_epoch >= epochs:

            print()
            print("=" * 60)
            print(
                "Training already completed."
            )
            print(
                f"Checkpoint epoch: "
                f"{start_epoch}"
            )
            print(
                f"Requested epochs: "
                f"{epochs}"
            )
            print("=" * 60)

            self.writer.close()

            return

        # =================================================
        # EPOCH LOOP
        # =================================================

        for epoch in range(
            start_epoch,
            epochs
        ):

            self.current_epoch = epoch

            print()
            print("=" * 70)
            print(
                f"Epoch {epoch + 1}/{epochs}"
            )
            print("=" * 70)

            # =================================================
            # TRAIN
            # =================================================

            train_losses = self.train_epoch(
                train_dataloader
            )

            # =================================================
            # VALIDATION
            # =================================================

            validation_losses = (
                self.validate_epoch(
                    validation_dataloader
                )
            )

            # =================================================
            # LEARNING-RATE SCHEDULER
            # =================================================

            self.scheduler.step(
                validation_losses["total"]
            )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # =================================================
            # TRAINING HISTORY
            # =================================================

            self.history["total"].append(
                train_losses["total"]
            )

            self.history["mae"].append(
                train_losses["mae"]
            )

            self.history["physics"].append(
                train_losses["physics"]
            )

            self.history["uncertainty"].append(
                train_losses["uncertainty"]
            )

            self.history["ssim"].append(
                train_losses["ssim"]
            )

            self.history["learning_rate"].append(
                current_lr
            )

            self.history["gradient_norm"].append(
                train_losses["gradient_norm"]
            )

            self.history["maximum_gradient"].append(
                train_losses["maximum_gradient"]
            )

            # =================================================
            # VALIDATION HISTORY
            # =================================================

            self.validation_history["total"].append(
                validation_losses["total"]
            )

            self.validation_history["mae"].append(
                validation_losses["mae"]
            )

            self.validation_history["physics"].append(
                validation_losses["physics"]
            )

            self.validation_history["uncertainty"].append(
                validation_losses["uncertainty"]
            )

            self.validation_history["ssim"].append(
                validation_losses["ssim"]
            )

            self.validation_history["metric_mae"].append(
                validation_losses["metric_mae"]
            )

            self.validation_history["metric_rmse"].append(
                validation_losses["metric_rmse"]
            )

            self.validation_history["metric_psnr"].append(
                validation_losses["metric_psnr"]
            )

            self.validation_history["metric_snr"].append(
                validation_losses["metric_snr"]
            )

            self.validation_history["metric_ssim"].append(
                validation_losses["metric_ssim"]
            )

            # =================================================
            # TENSORBOARD
            # =================================================

            self.writer.add_scalar(
                "Loss/Train_Total",
                train_losses["total"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Validation_Total",
                validation_losses["total"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_MAE",
                train_losses["mae"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Validation_MAE",
                validation_losses["mae"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_Physics",
                train_losses["physics"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Validation_Physics",
                validation_losses["physics"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_Uncertainty",
                train_losses["uncertainty"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Validation_Uncertainty",
                validation_losses["uncertainty"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_SSIM",
                train_losses["ssim"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Validation_SSIM",
                validation_losses["ssim"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_MAE",
                validation_losses["metric_mae"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_RMSE",
                validation_losses["metric_rmse"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_PSNR",
                validation_losses["metric_psnr"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_SNR",
                validation_losses["metric_snr"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_SSIM",
                validation_losses["metric_ssim"],
                epoch
            )

            self.writer.add_scalar(
                "Gradient/Train_Norm",
                train_losses["gradient_norm"],
                epoch
            )

            self.writer.add_scalar(
                "Gradient/Train_Maximum",
                train_losses["maximum_gradient"],
                epoch
            )

            self.writer.add_scalar(
                "Learning_Rate",
                current_lr,
                epoch
            )

            # =================================================
            # CONSOLE REPORT
            # =================================================

            print()
            print(
                "TRAINING"
            )
            print(
                "-" * 40
            )

            print(
                f"Total Loss       : "
                f"{train_losses['total']:.6f}"
            )

            print(
                f"MAE Loss         : "
                f"{train_losses['mae']:.6f}"
            )

            print(
                f"Physics Loss     : "
                f"{train_losses['physics']:.6e}"
            )

            print(
                f"Uncertainty Loss : "
                f"{train_losses['uncertainty']:.6f}"
            )

            print(
                f"SSIM Loss        : "
                f"{train_losses['ssim']:.6f}"
            )

            print(
                f"Gradient Norm    : "
                f"{train_losses['gradient_norm']:.6e}"
            )

            print(
                f"Maximum Gradient : "
                f"{train_losses['maximum_gradient']:.6e}"
            )

            print()
            print(
                "VALIDATION"
            )
            print(
                "-" * 40
            )

            print(
                f"Total Loss       : "
                f"{validation_losses['total']:.6f}"
            )

            print(
                f"MAE Loss         : "
                f"{validation_losses['mae']:.6f}"
            )

            print(
                f"Physics Loss     : "
                f"{validation_losses['physics']:.6e}"
            )

            print(
                f"Uncertainty Loss : "
                f"{validation_losses['uncertainty']:.6f}"
            )

            print(
                f"SSIM Loss        : "
                f"{validation_losses['ssim']:.6f}"
            )

            print()
            print(
                "RECONSTRUCTION METRICS"
            )
            print(
                "-" * 40
            )

            print(
                f"MAE  : "
                f"{validation_losses['metric_mae']:.6f}"
            )

            print(
                f"RMSE : "
                f"{validation_losses['metric_rmse']:.6f}"
            )

            print(
                f"PSNR : "
                f"{validation_losses['metric_psnr']:.3f} dB"
            )

            print(
                f"SNR  : "
                f"{validation_losses['metric_snr']:.3f} dB"
            )

            print(
                f"SSIM : "
                f"{validation_losses['metric_ssim']:.6f}"
            )

            print(
                f"Learning Rate : "
                f"{current_lr:.8f}"
            )

            # =================================================
            # CSV LOG
            # =================================================

            with open(
                self.log_file,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    epoch + 1,

                    train_losses["total"],

                    validation_losses["total"],

                    train_losses["mae"],

                    validation_losses["mae"],

                    train_losses["physics"],

                    validation_losses["physics"],

                    train_losses["uncertainty"],

                    validation_losses["uncertainty"],

                    train_losses["ssim"],

                    validation_losses["ssim"],

                    validation_losses["metric_rmse"],

                    validation_losses["metric_psnr"],

                    validation_losses["metric_snr"],

                    validation_losses["metric_ssim"],

                    current_lr,

                    train_losses["gradient_norm"],

                    train_losses["maximum_gradient"]
                ])

        # =================================================
        # BEST MODEL
        # =================================================

            if (
                validation_losses["total"]
                <
                self.best_validation_loss
            ):

                # -------------------------------------------------
                # Update best validation loss.
                # -------------------------------------------------

                self.best_validation_loss = (
                    validation_losses["total"]
                )

                # -------------------------------------------------
                # Update best epoch.
                # -------------------------------------------------

                self.best_epoch = epoch

                # -------------------------------------------------
                # Update best metrics.
                # -------------------------------------------------

                self.best_metrics = {

                    "MAE":
                        validation_losses[
                            "metric_mae"
                        ],

                    "RMSE":
                        validation_losses[
                            "metric_rmse"
                        ],

                    "PSNR":
                        validation_losses[
                            "metric_psnr"
                        ],

                    "SNR":
                        validation_losses[
                            "metric_snr"
                        ],

                    "SSIM":
                        validation_losses[
                            "metric_ssim"
                        ]
                }

                # -------------------------------------------------
                # Reset early stopping.
                # -------------------------------------------------

                self.wait = 0

                # =================================================
                # SAVE BEST MODEL
                # =================================================

                best_checkpoint = {

                    "epoch":
                        int(epoch),

                    "model_state_dict":
                        self.model.state_dict(),

                    "optimizer_state_dict":
                        self.optimizer.state_dict(),

                    "scheduler_state_dict":
                        self.scheduler.state_dict(),

                    "best_validation_loss":
                        float(
                            self.best_validation_loss
                        ),

                    "best_epoch":
                        int(
                            self.best_epoch
                        ),

                    "best_metrics":
                        dict(
                            self.best_metrics
                        ),

                    "early_stopping_wait":
                        int(
                            self.wait
                        ),

                    "loss":
                        float(
                            validation_losses[
                                "total"
                            ]
                        )
                }

                torch.save(
                    best_checkpoint,
                    os.path.join(
                        self.checkpoint_directory,
                        "best_model.pth"
                    )
                )

                print()
                print(
                    "NEW BEST MODEL FOUND"
                )

                print(
                    f"Best Epoch : "
                    f"{self.best_epoch + 1}"
                )

                print(
                    f"Best Loss  : "
                    f"{self.best_validation_loss:.6f}"
                )

                print(
                    f"Best SSIM  : "
                    f"{self.best_metrics['SSIM']:.6f}"
                )

            else:

                self.wait += 1

                print(
                    f"Early Stopping Counter: "
                    f"{self.wait}/{self.patience}"
                )

            # =================================================
            # SAVE LATEST CHECKPOINT
            # =================================================

            self.save_checkpoint(
                epoch=epoch,
                loss=validation_losses["total"]
            )

            # =================================================
            # EARLY STOPPING
            # =================================================

            if self.wait >= self.patience:

                print()
                print("=" * 60)
                print(
                    "Early stopping triggered."
                )
                print("=" * 60)

                break

        # =================================================
        # TRAINING SUMMARY
        # =================================================

        training_time = (
            time.time()
            -
            start_time
        )

        self.save_training_summary(
            best_epoch=self.best_epoch,
            best_validation_loss=(
                self.best_validation_loss
            ),
            best_metrics=self.best_metrics,
            training_time=training_time
        )

        # =================================================
        # CLOSE TENSORBOARD
        # =================================================

        self.writer.close()

    # =====================================================
    # TRAINING SUMMARY
    # =====================================================

    def save_training_summary(
        self,
        best_epoch,
        best_validation_loss,
        best_metrics,
        training_time
    ):
        """
        Save final training summary.
        """

        report_directory = (
            self.experiment.reports
        )

        os.makedirs(
            report_directory,
            exist_ok=True
        )

        report_file = os.path.join(
            report_directory,
            "training_summary.txt"
        )

        with open(
            report_file,
            "w"
        ) as file:

            file.write(
                "Physics-Informed 3D Seismic "
                "Reconstruction Training Summary\n"
            )

            file.write(
                "=" * 60 + "\n\n"
            )

            file.write(
                f"Best Epoch            : "
                f"{best_epoch + 1}\n"
            )

            file.write(
                f"Best Validation Loss  : "
                f"{best_validation_loss:.6f}\n\n"
            )

            file.write(
                f"Best MAE              : "
                f"{best_metrics['MAE']:.6f}\n"
            )

            file.write(
                f"Best RMSE             : "
                f"{best_metrics['RMSE']:.6f}\n"
            )

            file.write(
                f"Best PSNR             : "
                f"{best_metrics['PSNR']:.3f} dB\n"
            )

            file.write(
                f"Best SNR              : "
                f"{best_metrics['SNR']:.3f} dB\n"
            )

            file.write(
                f"Best SSIM             : "
                f"{best_metrics['SSIM']:.6f}\n\n"
            )

            file.write(
                f"Training Time         : "
                f"{training_time:.2f} seconds\n"
            )

            file.write(
                f"Total Epochs Trained  : "
                f"{len(self.history['total'])}\n"
            )

            # -------------------------------------------------
            # Gradient diagnostics.
            # -------------------------------------------------

            if self.history["gradient_norm"]:

                file.write(
                    f"\nAverage Gradient Norm : "
                    f"{sum(self.history['gradient_norm']) / len(self.history['gradient_norm']):.6e}\n"
                )

            if self.history["maximum_gradient"]:

                file.write(
                    f"Maximum Gradient Seen : "
                    f"{max(self.history['maximum_gradient']):.6e}\n"
                )