"""
=========================================================
Trainer Integration Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

This test verifies the complete Trainer pipeline:

    Dataset
       |
       v
    Model
       |
       v
    Composite Total Loss
       |
       v
    Backpropagation
       |
       v
    Optimizer
       |
       v
    Validation
       |
       v
    Metrics
       |
       +---- Checkpoint
       +---- CSV logging
       +---- TensorBoard logging
       +---- Visualization
       +---- Resume training

Tensor convention:

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import os
import shutil
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer.trainer import Trainer
from losses.total_loss import TotalLoss
from utils.experiment_manager import ExperimentManager


# =========================================================
# TEST CONFIGURATION
# =========================================================

TEST_SEED = 42

BATCH_SIZE = 1

CHANNELS = 1

DEPTH = 8

HEIGHT = 8

WIDTH = 8

DX = 10.0

DY = 10.0

DZ = 10.0

LEARNING_RATE = 1.0e-4


# =========================================================
# REPRODUCIBILITY
# =========================================================

torch.manual_seed(TEST_SEED)


# =========================================================
# TEST MODEL
# =========================================================

class DummyPhysicsInformed3DNet(nn.Module):
    """
    Very small model used only for testing Trainer.

    The purpose is NOT to test the neural-network architecture.

    The purpose is to test whether Trainer correctly handles
    the required three outputs:

        reconstruction
        travel_time
        log_variance

    Output shapes:

        [B,C,D,H,W]
    """

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # Simple learnable 3D convolution
        # -------------------------------------------------

        self.reconstruction_layer = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

        # -------------------------------------------------
        # Travel-time prediction layer
        # -------------------------------------------------

        self.travel_time_layer = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

        # -------------------------------------------------
        # Log-variance prediction layer
        # -------------------------------------------------

        self.log_variance_layer = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input seismic volume.

        Returns
        -------
        tuple
            reconstruction,
            travel_time,
            log_variance
        """

        reconstruction = (
            self.reconstruction_layer(x)
        )

        travel_time = (
            self.travel_time_layer(x)
        )

        log_variance = (
            self.log_variance_layer(x)
        )

        return (
            reconstruction,
            travel_time,
            log_variance
        )


# =========================================================
# TEST EXPERIMENT MANAGER
# =========================================================

class TestExperimentManager:
    """
    Lightweight experiment manager for testing.

    This prevents the test from writing files into the
    real experiment directory.
    """

    def __init__(self, root_directory):

        self.root = root_directory

        self.tensorboard = os.path.join(
            root_directory,
            "tensorboard"
        )

        self.global_checkpoints = os.path.join(
            root_directory,
            "checkpoints"
        )

        self.logs = os.path.join(
            root_directory,
            "logs"
        )

        self.training_progress = os.path.join(
            root_directory,
            "training_progress"
        )

        self.reports = os.path.join(
            root_directory,
            "reports"
        )

        # -------------------------------------------------
        # Create directories
        # -------------------------------------------------

        for directory in [
            self.tensorboard,
            self.global_checkpoints,
            self.logs,
            self.training_progress,
            self.reports
        ]:

            os.makedirs(
                directory,
                exist_ok=True
            )


# =========================================================
# CREATE TEST DATA
# =========================================================

def create_test_dataloader():
    """
    Create a small synthetic seismic dataset.

    Returns
    -------
    train_loader
    validation_loader
    """

    # -----------------------------------------------------
    # Generate synthetic input
    # -----------------------------------------------------

    inputs = torch.randn(
        4,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH
    )

    # -----------------------------------------------------
    # Ground-truth seismic data
    # -----------------------------------------------------

    targets = (
        0.5 * inputs
    )

    # -----------------------------------------------------
    # Missing-data mask
    #
    # The current Trainer validates and transfers the mask,
    # but does not yet use it in the loss.
    # -----------------------------------------------------

    mask = torch.ones_like(
        targets
    )

    # -----------------------------------------------------
    # Velocity model
    #
    # Positive P-wave velocity is required by the
    # Eikonal physics loss.
    # -----------------------------------------------------

    velocity_model = torch.full(
        (
            4,
            CHANNELS,
            DEPTH,
            HEIGHT,
            WIDTH
        ),
        2000.0
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = TensorDataset(
        inputs,
        targets,
        mask,
        velocity_model
    )

    # -----------------------------------------------------
    # Training subset
    # -----------------------------------------------------

    train_dataset = torch.utils.data.Subset(
        dataset,
        [0, 1]
    )

    # -----------------------------------------------------
    # Validation subset
    # -----------------------------------------------------

    validation_dataset = torch.utils.data.Subset(
        dataset,
        [2, 3]
    )

    # -----------------------------------------------------
    # Training DataLoader
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------------------------------
    # Validation DataLoader
    # -----------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader
    )


# =========================================================
# MAIN TEST
# =========================================================

def test_trainer():
    """
    Complete Trainer integration test.
    """

    print("=" * 70)

    print(
        "TESTING TRAINER INTEGRATION"
    )

    print("=" * 70)

    # =====================================================
    # TEMPORARY TEST DIRECTORY
    # =====================================================

    temporary_directory = tempfile.mkdtemp(
        prefix="trainer_test_"
    )

    print()
    print(
        f"Temporary test directory:\n"
        f"{temporary_directory}"
    )

    try:

        # =================================================
        # DEVICE
        # =================================================

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print()
        print(
            f"Testing device: {device}"
        )

        # =================================================
        # MODEL
        # =================================================

        model = (
            DummyPhysicsInformed3DNet()
        )

        # =================================================
        # TOTAL LOSS
        # =================================================

        criterion = TotalLoss(
            dx=DX,
            dy=DY,
            dz=DZ
        )

        # =================================================
        # OPTIMIZER
        # =================================================

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        # =================================================
        # EXPERIMENT MANAGER
        # =================================================

        experiment = (
            TestExperimentManager(
                temporary_directory
            )
        )

        # =================================================
        # TRAINER
        # =================================================

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            experiment_manager=experiment
        )

        print()
        print(
            "Trainer initialized successfully."
        )

        # =================================================
        # DATA
        # =================================================

        (
            train_loader,
            validation_loader
        ) = create_test_dataloader()

        print()
        print(
            "Training batches:",
            len(train_loader)
        )

        print(
            "Validation batches:",
            len(validation_loader)
        )

        # =================================================
        # TRAINING EPOCH
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING TRAINING EPOCH"
        )
        print("=" * 70)

        train_losses = trainer.train_epoch(
            train_loader
        )

        print()
        print(
            "Training losses:"
        )

        for name, value in train_losses.items():

            print(
                f"{name:20s}: {value:.6e}"
            )

        # -------------------------------------------------
        # Verify training losses
        # -------------------------------------------------

        for name, value in train_losses.items():

            assert torch.isfinite(
                torch.tensor(value)
            ), (
                f"Training loss '{name}' "
                "is not finite."
            )

        # =================================================
        # VERIFY PARAMETERS UPDATED
        # =================================================

        print()
        print(
            "Checking optimizer update..."
        )

        parameter_found = False

        for parameter in model.parameters():

            if parameter.requires_grad:

                parameter_found = True

                assert parameter.grad is not None, (
                    "Model parameter has no gradient."
                )

                assert torch.isfinite(
                    parameter.grad
                ).all(), (
                    "Model parameter contains "
                    "non-finite gradients."
                )

        assert parameter_found, (
            "No trainable model parameters found."
        )

        print(
            "Optimizer/backward propagation test passed."
        )

        # =================================================
        # VALIDATION EPOCH
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING VALIDATION EPOCH"
        )
        print("=" * 70)

        trainer.current_epoch = 0

        validation_losses = (
            trainer.validate_epoch(
                validation_loader
            )
        )

        print()
        print(
            "Validation results:"
        )

        for name, value in validation_losses.items():

            print(
                f"{name:20s}: {value:.6e}"
            )

        # -------------------------------------------------
        # Verify validation results
        # -------------------------------------------------

        for name, value in validation_losses.items():

            assert torch.isfinite(
                torch.tensor(value)
            ), (
                f"Validation result '{name}' "
                "is not finite."
            )

        # =================================================
        # CHECK RECONSTRUCTION METRICS
        # =================================================

        required_metrics = [
            "metric_mae",
            "metric_rmse",
            "metric_psnr",
            "metric_snr",
            "metric_ssim"
        ]

        for metric_name in required_metrics:

            assert metric_name in validation_losses, (
                f"Missing validation metric: "
                f"{metric_name}"
            )

        print()
        print(
            "Validation metric test passed."
        )

        # =================================================
        # CHECK VISUALIZATION
        # =================================================

        visualization_file = os.path.join(
            experiment.training_progress,
            "epoch_000.png"
        )

        assert os.path.exists(
            visualization_file
        ), (
            "Validation visualization was not created."
        )

        print()
        print(
            "Validation visualization test passed."
        )

        # =================================================
        # CHECKPOINT TEST
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING CHECKPOINT SYSTEM"
        )
        print("=" * 70)

        trainer.best_validation_loss = (
            validation_losses["total"]
        )

        trainer.best_epoch = 0

        trainer.save_checkpoint(
            epoch=0,
            loss=validation_losses["total"]
        )

        # -------------------------------------------------
        # Latest checkpoint
        # -------------------------------------------------

        latest_checkpoint = os.path.join(
            experiment.global_checkpoints,
            "latest_checkpoint.pth"
        )

        assert os.path.exists(
            latest_checkpoint
        ), (
            "latest_checkpoint.pth was not created."
        )

        # -------------------------------------------------
        # Historical checkpoint
        # -------------------------------------------------

        historical_checkpoint = os.path.join(
            experiment.global_checkpoints,
            "epoch_sensitivity",
            "epoch_0001.pth"
        )

        assert os.path.exists(
            historical_checkpoint
        ), (
            "Historical epoch checkpoint was not created."
        )

        # -------------------------------------------------
        # JSON training state
        # -------------------------------------------------

        training_state = os.path.join(
            experiment.global_checkpoints,
            "training_state.json"
        )

        assert os.path.exists(
            training_state
        ), (
            "training_state.json was not created."
        )

        print()
        print(
            "Checkpoint creation test passed."
        )

        # =================================================
        # CSV TEST
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING CSV LOGGING"
        )
        print("=" * 70)

        # The CSV file is created during Trainer
        # initialization.

        assert os.path.exists(
            trainer.log_file
        ), (
            "training_history.csv was not created."
        )

        print(
            "CSV initialization test passed."
        )

        # =================================================
        # FIT TEST
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING COMPLETE FIT PROCEDURE"
        )
        print("=" * 70)

        # -------------------------------------------------
        # Use a fresh trainer so that FIT itself is tested
        # from the beginning.
        # -------------------------------------------------

        model_fit = (
            DummyPhysicsInformed3DNet()
        )

        criterion_fit = TotalLoss(
            dx=DX,
            dy=DY,
            dz=DZ
        )

        optimizer_fit = torch.optim.Adam(
            model_fit.parameters(),
            lr=LEARNING_RATE
        )

        fit_directory = os.path.join(
            temporary_directory,
            "fit_test"
        )

        fit_experiment = (
            TestExperimentManager(
                fit_directory
            )
        )

        trainer_fit = Trainer(
            model=model_fit,
            criterion=criterion_fit,
            optimizer=optimizer_fit,
            device=device,
            experiment_manager=fit_experiment
        )

        trainer_fit.fit(
            train_dataloader=train_loader,
            validation_dataloader=validation_loader,
            epochs=2,
            resume=False
        )

        print()
        print(
            "FIT procedure completed successfully."
        )

        # =================================================
        # FIT OUTPUTS
        # =================================================

        fit_latest_checkpoint = os.path.join(
            fit_experiment.global_checkpoints,
            "latest_checkpoint.pth"
        )

        fit_best_checkpoint = os.path.join(
            fit_experiment.global_checkpoints,
            "best_model.pth"
        )

        fit_csv = os.path.join(
            fit_experiment.logs,
            "training_history.csv"
        )

        fit_summary = os.path.join(
            fit_experiment.reports,
            "training_summary.txt"
        )

        assert os.path.exists(
            fit_latest_checkpoint
        ), (
            "FIT did not create latest checkpoint."
        )

        assert os.path.exists(
            fit_best_checkpoint
        ), (
            "FIT did not create best_model.pth."
        )

        assert os.path.exists(
            fit_csv
        ), (
            "FIT did not create training_history.csv."
        )

        assert os.path.exists(
            fit_summary
        ), (
            "FIT did not create training_summary.txt."
        )

        print()
        print(
            "FIT output-file test passed."
        )

        # =================================================
        # RESUME TEST
        # =================================================

        print()
        print("=" * 70)
        print(
            "TESTING CHECKPOINT RESUME"
        )
        print("=" * 70)

        model_resume = (
            DummyPhysicsInformed3DNet()
        )

        criterion_resume = TotalLoss(
            dx=DX,
            dy=DY,
            dz=DZ
        )

        optimizer_resume = torch.optim.Adam(
            model_resume.parameters(),
            lr=LEARNING_RATE
        )

        resume_trainer = Trainer(
            model=model_resume,
            criterion=criterion_resume,
            optimizer=optimizer_resume,
            device=device,
            experiment_manager=fit_experiment
        )

        next_epoch = (
            resume_trainer.load_checkpoint(
                fit_latest_checkpoint
            )
        )

        print()
        print(
            f"Next epoch returned by checkpoint: "
            f"{next_epoch}"
        )

        # Two epochs were completed.
        #
        # Therefore:
        #
        # stored epoch = 1
        #
        # next zero-based epoch = 2

        assert next_epoch == 2, (
            "Checkpoint resume returned the wrong "
            f"next epoch. Expected 2, received "
            f"{next_epoch}."
        )

        print()
        print(
            "Checkpoint resume test passed."
        )

        # =================================================
        # FINAL RESULT
        # =================================================

        print()
        print("=" * 70)
        print(
            "TRAINER INTEGRATION TEST PASSED."
        )
        print("=" * 70)

    finally:

        # =================================================
        # CLOSE TENSORBOARD WRITER
        # =================================================

        # The Trainer normally closes its writer after
        # fit(). Explicitly close it here as well for tests
        # that stop before fit().

        try:
            trainer.writer.close()
        except Exception:
            pass

        # =================================================
        # REMOVE TEMPORARY DIRECTORY
        # =================================================

        shutil.rmtree(
            temporary_directory,
            ignore_errors=True
        )


# =========================================================
# TEST ENTRY POINT
# =========================================================

if __name__ == "__main__":

    test_trainer()