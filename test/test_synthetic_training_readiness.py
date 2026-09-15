"""
======================================================================
FINAL SYNTHETIC TRAINING READINESS TEST
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
This test verifies that the complete synthetic training pipeline
operates correctly as one integrated system:

    Synthetic Dataset
            |
            v
       DataLoader
            |
            v
        Network3D
            |
      +-----+-----+----------------+
      |           |                |
      v           v                v
 Reconstruction  Travel-Time   Log Variance
      |           |                |
      +-----------+----------------+
                  |
                  v
              TotalLoss
                  |
                  v
             Backpropagation
                  |
                  v
             Gradient Check
                  |
                  v
             Optimizer Step
                  |
                  v
              Validation
                  |
                  v
          Checkpoint Save/Load

This is an INTEGRATION TEST.

It does not modify:
    - Network architecture
    - Loss definitions
    - Dataset implementation
    - Trainer implementation
    - Global configuration

The test uses the current project APIs.

Author: Ormin Joseph
======================================================================
"""


# ====================================================================
# STANDARD LIBRARY
# ====================================================================

import os
import shutil
import tempfile


# ====================================================================
# PYTORCH
# ====================================================================

import torch

from torch.utils.data import DataLoader


# ====================================================================
# PROJECT IMPORTS
# ====================================================================

from dataset.synthetic_dataset import SyntheticSeismicDataset

from models.network import Network3D

from losses.total_loss import TotalLoss

from utils.config import (
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,
    DX,
    DY,
    DZ,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    SEED,
)


# ====================================================================
# TEST CONFIGURATION
# ====================================================================

# ------------------------------------------------------------
# Number of synthetic samples used by this integration test.
# ------------------------------------------------------------

TEST_NUM_SAMPLES = 2


# ------------------------------------------------------------
# Batch size.
#
# Batch size 1 is deliberately used because the current
# 3D seismic architecture is memory intensive.
# ------------------------------------------------------------

TEST_BATCH_SIZE = 1


# ------------------------------------------------------------
# Number of DataLoader workers.
#
# Zero workers gives deterministic and predictable behaviour
# during the integration test.
# ------------------------------------------------------------

TEST_NUM_WORKERS = 0


# ------------------------------------------------------------
# Temporary checkpoint directory.
#
# The test must NEVER overwrite a real training experiment.
# ------------------------------------------------------------

TEST_CHECKPOINT_NAME = "synthetic_training_readiness_checkpoint.pth"


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def header(title):
    """
    Print a clear test section header.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subheader(title):
    """
    Print a subsection header.
    """

    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def require(condition, message):
    """
    Raise RuntimeError if a test condition fails.
    """

    if not condition:
        raise RuntimeError(message)


def check_finite(name, tensor):
    """
    Verify that a tensor contains only finite values.
    """

    require(
        isinstance(tensor, torch.Tensor),
        f"{name} is not a torch.Tensor."
    )

    require(
        torch.isfinite(tensor).all().item(),
        f"{name} contains NaN or Inf values."
    )


def check_shape(name, tensor, expected_shape):
    """
    Verify tensor shape.
    """

    actual_shape = tuple(tensor.shape)

    require(
        actual_shape == tuple(expected_shape),
        (
            f"{name} shape mismatch.\n"
            f"Expected: {tuple(expected_shape)}\n"
            f"Received: {actual_shape}"
        )
    )


def gradient_statistics(model):
    """
    Inspect gradients after backpropagation.

    Returns
    -------
    gradient_norm : float
        Global L2 gradient norm.

    nonzero_parameters : int
        Number of trainable parameters receiving gradients.

    finite_gradients : bool
        Whether all observed gradients are finite.
    """

    squared_norm = 0.0
    nonzero_parameters = 0
    finite_gradients = True

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            finite_gradients = False

        norm = gradient.norm(2).item()

        squared_norm += norm ** 2

        if norm > 0.0:
            nonzero_parameters += 1

    gradient_norm = squared_norm ** 0.5

    return (
        gradient_norm,
        nonzero_parameters,
        finite_gradients,
    )


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """
    Configure deterministic random seeds for the integration test.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================
# DATASET TEST
# ====================================================================

def test_dataset_and_dataloader(device):
    """
    Construct the synthetic dataset and DataLoader and validate
    one complete batch.
    """

    header("1. SYNTHETIC DATASET + DATALOADER")

    print(
        f"Dataset samples : {TEST_NUM_SAMPLES}"
    )

    print(
        f"Cube size       : {SYNTHETIC_PATCH_SIZE}"
    )

    print(
        f"Missing rate    : "
        f"{SYNTHETIC_MISSING_PROBABILITY}"
    )

    # ------------------------------------------------------------
    # Create synthetic dataset.
    # ------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=TEST_NUM_SAMPLES,
        cube_size=SYNTHETIC_PATCH_SIZE,
        missing_probability=SYNTHETIC_MISSING_PROBABILITY,
        seed=SEED,
    )

    require(
        len(dataset) == TEST_NUM_SAMPLES,
        "Synthetic dataset length is incorrect."
    )

    # ------------------------------------------------------------
    # Create DataLoader.
    # ------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=TEST_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    require(
        len(loader) > 0,
        "Synthetic DataLoader contains no batches."
    )

    # ------------------------------------------------------------
    # Obtain first batch.
    #
    # Current SyntheticSeismicDataset returns:
    #
    #     input
    #     target
    #     mask
    #     velocity
    #     mask_type
    #     geological_mode
    # ------------------------------------------------------------

    batch = next(iter(loader))

    require(
        len(batch) == 6,
        (
            "Unexpected synthetic dataset batch structure.\n"
            f"Expected 6 items, received {len(batch)}."
        )
    )

    (
        inputs,
        targets,
        mask,
        velocity_model,
        mask_type,
        geological_mode,
    ) = batch

    # ------------------------------------------------------------
    # Validate tensor dimensions.
    # ------------------------------------------------------------

    expected_cube_shape = (
        TEST_BATCH_SIZE,
        1,
        *SYNTHETIC_PATCH_SIZE,
    )

    check_shape(
        "Input",
        inputs,
        expected_cube_shape,
    )

    check_shape(
        "Target",
        targets,
        expected_cube_shape,
    )

    check_shape(
        "Mask",
        mask,
        expected_cube_shape,
    )

    check_shape(
        "Velocity model",
        velocity_model,
        expected_cube_shape,
    )

    # ------------------------------------------------------------
    # Validate finiteness.
    # ------------------------------------------------------------

    check_finite(
        "Input",
        inputs,
    )

    check_finite(
        "Target",
        targets,
    )

    check_finite(
        "Mask",
        mask,
    )

    check_finite(
        "Velocity model",
        velocity_model,
    )

    # ------------------------------------------------------------
    # Validate velocity positivity.
    # ------------------------------------------------------------

    require(
        torch.all(velocity_model > 0).item(),
        "Velocity model contains non-positive values."
    )

    # ------------------------------------------------------------
    # Validate binary mask.
    # ------------------------------------------------------------

    unique_mask_values = torch.unique(mask)

    require(
        torch.all(
            (unique_mask_values == 0)
            | (unique_mask_values == 1)
        ).item(),
        (
            "Mask is not binary.\n"
            f"Unique values: {unique_mask_values}"
        )
    )

    # ------------------------------------------------------------
    # Validate input-target consistency.
    #
    # The current synthetic dataset defines:
    #
    #     input = target * mask
    #
    # Therefore this must hold exactly.
    # ------------------------------------------------------------

    consistency_error = (
        torch.abs(
            inputs - targets * mask
        ).max().item()
    )

    require(
        consistency_error == 0.0,
        (
            "Input-target-mask consistency failed.\n"
            f"Maximum error: {consistency_error}"
        )
    )

    print(
        "Dataset tensor validation : PASS"
    )

    print(
        "DataLoader validation      : PASS"
    )

    print(
        f"Geological mode            : {geological_mode[0]}"
    )

    print(
        f"Mask mode                  : {mask_type[0]}"
    )

    return (
        loader,
        inputs.to(device),
        targets.to(device),
        mask.to(device),
        velocity_model.to(device),
    )


# ====================================================================
# MODEL TEST
# ====================================================================

def test_network(
        model,
        inputs,
        targets,
        velocity_model
):
    """
    Verify the complete Network3D forward pass.
    """

    header("2. NETWORK3D FORWARD PASS")

    model.train()

    (
        reconstruction,
        travel_time,
        log_variance,
    ) = model(inputs)

    # ------------------------------------------------------------
    # Expected output shape.
    # ------------------------------------------------------------

    expected_shape = targets.shape

    check_shape(
        "Reconstruction",
        reconstruction,
        expected_shape,
    )

    check_shape(
        "Travel-time field",
        travel_time,
        velocity_model.shape,
    )

    check_shape(
        "Log variance",
        log_variance,
        expected_shape,
    )

    # ------------------------------------------------------------
    # Finiteness.
    # ------------------------------------------------------------

    check_finite(
        "Reconstruction",
        reconstruction,
    )

    check_finite(
        "Travel-time field",
        travel_time,
    )

    check_finite(
        "Log variance",
        log_variance,
    )

    # ------------------------------------------------------------
    # Travel time must be non-negative because the current
    # Network3D uses Softplus for the travel-time head.
    # ------------------------------------------------------------

    require(
        torch.all(travel_time >= 0).item(),
        "Travel-time field contains negative values."
    )

    print(
        "Reconstruction head : PASS"
    )

    print(
        "Travel-time head     : PASS"
    )

    print(
        "Aleatoric uncertainty head : PASS"
    )

    print(
        f"Reconstruction shape : "
        f"{tuple(reconstruction.shape)}"
    )

    print(
        f"Travel-time shape    : "
        f"{tuple(travel_time.shape)}"
    )

    print(
        f"Log-variance shape   : "
        f"{tuple(log_variance.shape)}"
    )

    return (
        reconstruction,
        travel_time,
        log_variance,
    )


# ====================================================================
# TOTAL LOSS TEST
# ====================================================================

def test_total_loss(
        criterion,
        reconstruction,
        targets,
        travel_time,
        velocity_model,
        log_variance
):
    """
    Verify the complete composite TotalLoss.
    """

    header("3. TOTAL LOSS")

    losses = criterion(
        reconstruction,
        targets,
        travel_time,
        velocity_model,
        log_variance,
    )

    # ------------------------------------------------------------
    # Expected loss components used by the current Trainer.
    # ------------------------------------------------------------

    required_keys = {
        "total",
        "mae",
        "physics",
        "uncertainty",
        "ssim",
    }

    missing_keys = (
        required_keys
        - set(losses.keys())
    )

    require(
        not missing_keys,
        (
            "TotalLoss is missing required "
            f"loss components: {missing_keys}"
        )
    )

    # ------------------------------------------------------------
    # Verify every returned loss is finite.
    # ------------------------------------------------------------

    for name, value in losses.items():

        if not isinstance(value, torch.Tensor):
            continue

        check_finite(
            f"Loss component '{name}'",
            value,
        )

    # ------------------------------------------------------------
    # Total loss must be scalar.
    # ------------------------------------------------------------

    require(
        losses["total"].ndim == 0,
        "Total loss is not scalar."
    )

    print(
        f"Total loss       : "
        f"{losses['total'].detach().item():.6e}"
    )

    print(
        f"MAE loss         : "
        f"{losses['mae'].detach().item():.6e}"
    )

    print(
        f"Physics loss     : "
        f"{losses['physics'].detach().item():.6e}"
    )

    print(
        f"Uncertainty loss : "
        f"{losses['uncertainty'].detach().item():.6e}"
    )

    print(
        f"SSIM loss        : "
        f"{losses['ssim'].detach().item():.6e}"
    )

    print(
        "TotalLoss forward pass : PASS"
    )

    return losses


# ====================================================================
# BACKPROPAGATION TEST
# ====================================================================

def test_backpropagation(
        model,
        optimizer,
        loss
):
    """
    Verify that the total loss propagates gradients through
    the trainable network.
    """

    header("4. BACKPROPAGATION + GRADIENT CHECK")

    optimizer.zero_grad(
        set_to_none=True
    )

    # ------------------------------------------------------------
    # Backpropagate the complete composite loss.
    # ------------------------------------------------------------

    loss.backward()

    (
        gradient_norm,
        nonzero_parameters,
        finite_gradients,
    ) = gradient_statistics(model)

    print(
        f"Gradient norm          : "
        f"{gradient_norm:.6e}"
    )

    print(
        f"Parameters with grad   : "
        f"{nonzero_parameters}"
    )

    print(
        f"Gradients finite       : "
        f"{finite_gradients}"
    )

    # ------------------------------------------------------------
    # Scientific integrity checks.
    # ------------------------------------------------------------

    require(
        gradient_norm > 0.0,
        "No gradient reached the trainable network."
    )

    require(
        nonzero_parameters > 0,
        "No trainable parameter received a gradient."
    )

    require(
        finite_gradients,
        "NaN or Inf detected in network gradients."
    )

    print(
        "Backpropagation : PASS"
    )

    print(
        "Gradient propagation : PASS"
    )


# ====================================================================
# OPTIMIZER TEST
# ====================================================================

def test_optimizer_step(
        model,
        optimizer
):
    """
    Verify that the optimizer can update model parameters.
    """

    header("5. OPTIMIZER STEP")

    # ------------------------------------------------------------
    # Record one parameter before the update.
    # ------------------------------------------------------------

    tracked_parameter = None

    for parameter in model.parameters():

        if parameter.requires_grad:

            tracked_parameter = parameter

            break

    require(
        tracked_parameter is not None,
        "No trainable model parameter found."
    )

    parameter_before = (
        tracked_parameter.detach()
        .clone()
    )

    # ------------------------------------------------------------
    # Apply optimizer update.
    # ------------------------------------------------------------

    optimizer.step()

    # ------------------------------------------------------------
    # Compare parameter after update.
    # ------------------------------------------------------------

    parameter_after = (
        tracked_parameter.detach()
    )

    parameter_change = torch.abs(
        parameter_after
        - parameter_before
    ).sum().item()

    print(
        f"Parameter update magnitude : "
        f"{parameter_change:.6e}"
    )

    require(
        parameter_change > 0.0,
        "Optimizer step did not update model parameters."
    )

    print(
        "Optimizer update : PASS"
    )


# ====================================================================
# VALIDATION TEST
# ====================================================================

def test_validation(
        model,
        loader,
        criterion,
        device
):
    """
    Execute a no-gradient validation pass using the same
    dataset and loss interfaces used by training.
    """

    header("6. VALIDATION / NO-GRADIENT PASS")

    model.eval()

    validation_batches = 0

    with torch.no_grad():

        for batch in loader:

            (
                inputs,
                targets,
                mask,
                velocity_model,
                mask_type,
                geological_mode,
            ) = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            velocity_model = velocity_model.to(device)

            # ----------------------------------------------------
            # Forward pass.
            # ----------------------------------------------------

            (
                reconstruction,
                travel_time,
                log_variance,
            ) = model(inputs)

            # ----------------------------------------------------
            # Validate outputs.
            # ----------------------------------------------------

            check_shape(
                "Validation reconstruction",
                reconstruction,
                targets.shape,
            )

            check_shape(
                "Validation travel time",
                travel_time,
                velocity_model.shape,
            )

            check_shape(
                "Validation log variance",
                log_variance,
                targets.shape,
            )

            check_finite(
                "Validation reconstruction",
                reconstruction,
            )

            check_finite(
                "Validation travel time",
                travel_time,
            )

            check_finite(
                "Validation log variance",
                log_variance,
            )

            # ----------------------------------------------------
            # Validation TotalLoss.
            # ----------------------------------------------------

            losses = criterion(
                reconstruction,
                targets,
                travel_time,
                velocity_model,
                log_variance,
            )

            check_finite(
                "Validation total loss",
                losses["total"],
            )

            validation_batches += 1

            # ----------------------------------------------------
            # Only one validation batch is required for this
            # integration test.
            # ----------------------------------------------------

            break

    require(
        validation_batches > 0,
        "Validation produced zero batches."
    )

    print(
        f"Validation total loss : "
        f"{losses['total'].item():.6e}"
    )

    print(
        "Validation forward pass : PASS"
    )

    print(
        "No-gradient validation : PASS"
    )


# ====================================================================
# CHECKPOINT TEST
# ====================================================================

def test_checkpoint(
        model,
        optimizer,
        device
):
    """
    Verify checkpoint save and reload without touching the
    real experiment output directory.
    """

    header("7. CHECKPOINT SAVE / RELOAD")

    temporary_directory = tempfile.mkdtemp(
        prefix="synthetic_training_readiness_"
    )

    checkpoint_path = os.path.join(
        temporary_directory,
        TEST_CHECKPOINT_NAME,
    )

    try:

        # --------------------------------------------------------
        # Save checkpoint.
        # --------------------------------------------------------

        checkpoint = {
            "epoch": 0,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        require(
            os.path.isfile(checkpoint_path),
            "Checkpoint file was not created."
        )

        print(
            f"Checkpoint saved : "
            f"{checkpoint_path}"
        )

        # --------------------------------------------------------
        # Create a new model.
        #
        # This verifies that the saved state can actually be
        # restored into a fresh Network3D instance.
        # --------------------------------------------------------

        restored_model = Network3D(
            use_uncertainty=True,
            use_residual=True,
            use_attention=True,
        ).to(device)

        restored_optimizer = torch.optim.Adam(
            restored_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # --------------------------------------------------------
        # Load checkpoint.
        # --------------------------------------------------------

        loaded_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        restored_model.load_state_dict(
            loaded_checkpoint[
                "model_state_dict"
            ]
        )

        restored_optimizer.load_state_dict(
            loaded_checkpoint[
                "optimizer_state_dict"
            ]
        )

        # --------------------------------------------------------
        # Verify one parameter matches exactly.
        # --------------------------------------------------------

        original_state = model.state_dict()
        restored_state = restored_model.state_dict()

        require(
            original_state.keys()
            == restored_state.keys(),
            "Checkpoint state keys do not match."
        )

        for key in original_state:

            require(
                torch.equal(
                    original_state[key],
                    restored_state[key],
                ),
                (
                    "Checkpoint restoration mismatch "
                    f"for parameter: {key}"
                )
            )

        print(
            "Checkpoint reload : PASS"
        )

    finally:

        # --------------------------------------------------------
        # Remove only the temporary test directory.
        # --------------------------------------------------------

        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


# ====================================================================
# MAIN INTEGRATION TEST
# ====================================================================

def main():
    """
    Run the complete synthetic training readiness test.
    """

    header(
        "FINAL SYNTHETIC TRAINING READINESS TEST"
    )

    print(
        "Physics-Informed 3D Encoder-Decoder Framework"
    )

    print(
        "with Predictive Uncertainty"
    )

    print()
    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Seed   : {SEED}"
    )

    # ------------------------------------------------------------
    # Configure reproducibility.
    # ------------------------------------------------------------

    set_seed(SEED)

    # ------------------------------------------------------------
    # Select device.
    #
    # If configuration requests CUDA but CUDA is unavailable,
    # fall back to CPU for the test rather than failing during
    # device selection.
    # ------------------------------------------------------------

    if str(DEVICE).lower().startswith("cuda"):

        if torch.cuda.is_available():

            device = torch.device("cuda")

        else:

            print(
                "CUDA requested but unavailable."
            )

            print(
                "Falling back to CPU for integration test."
            )

            device = torch.device("cpu")

    else:

        device = torch.device("cpu")

    print(
        f"Resolved device : {device}"
    )

    # ============================================================
    # 1. DATASET + DATALOADER
    # ============================================================

    (
        loader,
        inputs,
        targets,
        mask,
        velocity_model,
    ) = test_dataset_and_dataloader(
        device
    )

    # ============================================================
    # 2. NETWORK
    # ============================================================

    model = Network3D(
        use_uncertainty=True,
        use_residual=True,
        use_attention=True,
    ).to(device)

    (
        reconstruction,
        travel_time,
        log_variance,
    ) = test_network(
        model,
        inputs,
        targets,
        velocity_model,
    )

    # ============================================================
    # 3. TOTAL LOSS
    # ============================================================

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(device)

    losses = test_total_loss(
        criterion,
        reconstruction,
        targets,
        travel_time,
        velocity_model,
        log_variance,
    )

    # ============================================================
    # 4. OPTIMIZER
    # ============================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print()
    print(
        "Optimizer initialized : PASS"
    )

    # ============================================================
    # 5. BACKPROPAGATION
    # ============================================================

    test_backpropagation(
        model,
        optimizer,
        losses["total"],
    )

    # ============================================================
    # 6. OPTIMIZER STEP
    # ============================================================

    test_optimizer_step(
        model,
        optimizer,
    )

    # ============================================================
    # 7. VALIDATION
    # ============================================================

    test_validation(
        model,
        loader,
        criterion,
        device,
    )

    # ============================================================
    # 8. CHECKPOINT
    # ============================================================

    test_checkpoint(
        model,
        optimizer,
        device,
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    header(
        "FINAL RESULT"
    )

    print(
        "PASS : Synthetic Dataset"
    )

    print(
        "PASS : DataLoader"
    )

    print(
        "PASS : Network3D"
    )

    print(
        "PASS : Reconstruction head"
    )

    print(
        "PASS : Travel-time head"
    )

    print(
        "PASS : Aleatoric uncertainty head"
    )

    print(
        "PASS : TotalLoss"
    )

    print(
        "PASS : Backpropagation"
    )

    print(
        "PASS : Gradient propagation"
    )

    print(
        "PASS : Optimizer update"
    )

    print(
        "PASS : Validation"
    )

    print(
        "PASS : Checkpoint save/reload"
    )

    print()
    print("=" * 78)
    print(
        "ALL SYNTHETIC TRAINING READINESS TESTS PASSED"
    )
    print("=" * 78)
    print()

    print(
        "The synthetic training pipeline is "
        "integration-ready."
    )


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()