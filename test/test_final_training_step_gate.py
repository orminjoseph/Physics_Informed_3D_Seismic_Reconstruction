"""
=========================================================
Final Training-Step Gate
=========================================================

Purpose:
    Verify that the complete training pipeline can perform
    a short optimizer update before the full training run.

Checks:
    1. Dataset creation
    2. Batch construction
    3. Model forward pass
    4. Total loss calculation
    5. Gradient calculation
    6. Optimizer update
    7. Second forward pass after the update

The current SyntheticSeismicDataset returns:

    input_cube,
    target_cube,
    mask,
    velocity_model,
    mask_type,
    geological_mode

The gate uses only the first four tensors.

=========================================================
"""
import sys
from pathlib import Path

# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# Project imports
# =========================================================

from dataset.synthetic_dataset import SyntheticSeismicDataset
from models.network import Network3D
from losses.total_loss import TotalLoss

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.config import DEVICE


# =========================================================
# Reproducibility
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Utility functions
# =========================================================

def check_finite(name, tensor):
    """
    Check whether a tensor contains only finite values.
    """

    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} is not a torch.Tensor.")

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or infinite values."
        )


def print_tensor_status(name, tensor):
    """
    Print the shape and finite-value status of a tensor.
    """

    check_finite(name, tensor)

    print(
        f"{name:<16}: "
        f"shape={tuple(tensor.shape)} "
        f"finite=PASS"
    )


def detach_loss_dictionary(losses):
    """
    Detach loss values from the computation graph.

    This is used only when printing losses after an
    optimizer update.
    """

    return {
        name: (
            value.detach()
            if torch.is_tensor(value)
            else value
        )
        for name, value in losses.items()
    }


# =========================================================
# Forward pass and loss calculation
# =========================================================

def forward_and_loss(model, criterion, batch):
    """
    Perform one forward pass and calculate the complete loss.

    The current TotalLoss implementation does not accept a
    seismic mask. It expects velocity_model instead of velocity.
    """

    # -------------------------------------------------
    # 1. Unpack the batch
    # -------------------------------------------------
    inputs, targets, mask, velocity_model = batch[:4]

    # -------------------------------------------------
    # 2. Forward pass
    # -------------------------------------------------
    reconstruction, travel_time, log_variance = model(inputs)

    # -------------------------------------------------
    # 3. Calculate the complete loss
    # -------------------------------------------------
    losses = criterion(
        prediction=reconstruction,
        target=targets,
        travel_time=travel_time,
        velocity_model=velocity_model,
        log_variance=log_variance,
    )

    # -------------------------------------------------
    # 4. Return outputs and loss components
    # -------------------------------------------------
    return (
        reconstruction,
        travel_time,
        log_variance,
        losses,
    )

def main():

    print("=" * 60)
    print("FINAL TRAINING-STEP GATE")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    # -----------------------------------------------------
    # 1. Create the synthetic dataset
    # -----------------------------------------------------

    print("\n[1/7] Creating synthetic dataset...")

    dataset = SyntheticSeismicDataset()

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError(
            "Synthetic dataset is empty."
        )

    # -----------------------------------------------------
    # 2. Read one dataset sample
    # -----------------------------------------------------

    print("\n[2/7] Reading one dataset sample...")

    sample = dataset[0]

    if not isinstance(sample, (tuple, list)):
        raise RuntimeError(
            "Dataset sample must be a tuple or list."
        )

    if len(sample) < 4:
        raise RuntimeError(
            "Dataset sample must contain at least "
            "(input, target, mask, velocity_model)."
        )

    # The current dataset returns six values.
    #
    # Only the first four are required by the training gate.

    input_cube = sample[0]
    target_cube = sample[1]
    mask = sample[2]
    velocity_model = sample[3]

    print(f"Sample length: {len(sample)}")

    print_tensor_status(
        "input_cube",
        input_cube,
    )

    print_tensor_status(
        "target_cube",
        target_cube,
    )

    print_tensor_status(
        "mask",
        mask,
    )

    print_tensor_status(
        "velocity_model",
        velocity_model,
    )

    if torch.any(velocity_model <= 0):
        raise RuntimeError(
            "Velocity model contains non-positive values."
        )

    print("Velocity values: positive=PASS")

    # -----------------------------------------------------
    # 3. Construct a DataLoader
    # -----------------------------------------------------

    print("\n[3/7] Constructing training batch...")

    tensor_dataset = TensorDataset(
        input_cube.unsqueeze(0),
        target_cube.unsqueeze(0),
        mask.unsqueeze(0),
        velocity_model.unsqueeze(0),
    )

    loader = DataLoader(
        tensor_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    batch = tuple(
        tensor.to(DEVICE)
        for tensor in batch
    )

    inputs, targets, batch_mask, velocity = batch

    print_tensor_status("inputs", inputs)
    print_tensor_status("targets", targets)
    print_tensor_status("mask", batch_mask)
    print_tensor_status("velocity", velocity)

    # -----------------------------------------------------
    # 4. Create the model and loss function
    # -----------------------------------------------------

    print("\n[4/7] Creating model and loss function...")

    model = Network3D()

    model = model.to(DEVICE)

    criterion = TotalLoss(
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    print("Model creation: PASS")
    print("Loss creation: PASS")
    print("Optimizer creation: PASS")

    # -----------------------------------------------------
    # 5. Initial forward pass and loss calculation
    # -----------------------------------------------------

    print("\n[5/7] Running initial forward pass...")

    model.train()

    optimizer.zero_grad(set_to_none=True)

    (
        reconstruction_before,
        travel_time_before,
        log_variance_before,
        losses_before,
    ) = forward_and_loss(
        model,
        criterion,
        batch,
    )

    total_loss_before = losses_before["total"]

    check_finite(
        "reconstruction_before",
        reconstruction_before,
    )

    check_finite(
        "travel_time_before",
        travel_time_before,
    )

    check_finite(
        "log_variance_before",
        log_variance_before,
    )

    check_finite(
        "total_loss_before",
        total_loss_before,
    )

    print_tensor_status(
        "reconstruction",
        reconstruction_before,
    )

    print_tensor_status(
        "travel_time",
        travel_time_before,
    )

    print_tensor_status(
        "log_variance",
        log_variance_before,
    )

    print(
        f"Initial total loss: "
        f"{total_loss_before.item():.6f}"
    )

    # -----------------------------------------------------
    # 6. Backward pass and optimizer update
    # -----------------------------------------------------

    print("\n[6/7] Running backward pass and update...")

    total_loss_before.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            check_finite(
                "parameter_gradient",
                parameter.grad,
            )

            gradient_count += 1

    if gradient_count == 0:
        raise RuntimeError(
            "No parameter gradients were produced."
        )

    print(
        f"Parameters with gradients: "
        f"{gradient_count}"
    )

    optimizer.step()

    print("Backward pass: PASS")
    print("Optimizer update: PASS")

    # -----------------------------------------------------
    # 7. Forward pass after the update
    # -----------------------------------------------------

    print("\n[7/7] Running post-update forward pass...")

    model.eval()

    # Do not use torch.no_grad() here.
    #
    # The physics-informed loss may require autograd to
    # calculate spatial derivatives of the travel-time field.

    with torch.enable_grad():

        (
            reconstruction_after,
            travel_time_after,
            log_variance_after,
            losses_after,
        ) = forward_and_loss(
            model,
            criterion,
            batch,
        )

    total_loss_after = losses_after["total"]

    check_finite(
        "reconstruction_after",
        reconstruction_after,
    )

    check_finite(
        "travel_time_after",
        travel_time_after,
    )

    check_finite(
        "log_variance_after",
        log_variance_after,
    )

    check_finite(
        "total_loss_after",
        total_loss_after,
    )

    print_tensor_status(
        "reconstruction_after",
        reconstruction_after,
    )

    print_tensor_status(
        "travel_time_after",
        travel_time_after,
    )

    print_tensor_status(
        "log_variance_after",
        log_variance_after,
    )

    print(
        f"Post-update total loss: "
        f"{total_loss_after.item():.6f}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TRAINING-STEP GATE: PASS")
    print("=" * 60)

    print(
        "The dataset, model, losses, gradients, and "
        "optimizer update completed successfully."
    )


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    main()