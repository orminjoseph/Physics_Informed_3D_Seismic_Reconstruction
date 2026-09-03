"""
=========================================================
PHYSICS-INFORMED 3D TRAINING CONVERGENCE AUDIT
=========================================================

Purpose
-------

This test evaluates whether the Physics-Informed 3D
Encoder-Decoder network can reduce the active physics
loss during optimization.

The current project uses:

    L_total =
        lambda_mae L_mae
        +
        lambda_physics L_physics
        +
        lambda_uncertainty L_uncertainty
        +
        lambda_ssim L_ssim

with:

    L_physics =
        lambda_eikonal L_eikonal
        +
        lambda_source L_source
        +
        lambda_travel_time L_travel_time

Current configuration:

    LOSS_WEIGHTS["physics"] = 0.10

    PHYSICS_LOSS_WEIGHTS["eikonal"] = 1.0
    PHYSICS_LOSS_WEIGHTS["source"] = 1.0
    PHYSICS_LOSS_WEIGHTS["travel_time"] = 1.0

However:

    USE_SOURCE_LOSS = False
    USE_TRAVEL_TIME_LOSS = False
    USE_EIKONAL_LOSS = True

Therefore the active physics constraint is:

    L_physics = L_eikonal

The test verifies:

1. Configuration consistency
2. Network initialization
3. Forward propagation
4. Three-output interface
5. Travel-time positivity
6. Eikonal physics loss
7. Composite loss construction
8. Gradient propagation
9. Parameter updates
10. Physics-loss evolution
11. Travel-time gradient evolution
12. Numerical stability
13. Training convergence behavior

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    USE_PHYSICS_LOSS,
    USE_EIKONAL_LOSS,
    USE_SOURCE_LOSS,
    USE_TRAVEL_TIME_LOSS,
    USE_UNCERTAINTY,
)


# =========================================================
# CONFIGURATION
# =========================================================

PHYSICS_WEIGHT = LOSS_WEIGHTS["physics"]

EIKONAL_WEIGHT = PHYSICS_LOSS_WEIGHTS["eikonal"]

SOURCE_WEIGHT = PHYSICS_LOSS_WEIGHTS["source"]

TRAVEL_TIME_WEIGHT = PHYSICS_LOSS_WEIGHTS["travel_time"]


# ---------------------------------------------------------
# Number of optimization steps.
#
# A small number is intentionally used because the current
# 3D model contains approximately 96 million parameters and
# the development environment is CPU-based.
# ---------------------------------------------------------

NUM_STEPS = 5


# ---------------------------------------------------------
# Synthetic test volume.
# ---------------------------------------------------------

BATCH_SIZE = 1

CHANNELS = 1

DEPTH = 64

HEIGHT = 128

WIDTH = 128


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def print_section(title):
    """
    Print a formatted test section heading.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def tensor_statistics(name, tensor):
    """
    Print basic numerical statistics for a tensor.
    """

    tensor_detached = tensor.detach()

    print(
        f"{name:<30}: "
        f"min={tensor_detached.min().item():.6e}, "
        f"max={tensor_detached.max().item():.6e}, "
        f"mean={tensor_detached.mean().item():.6e}, "
        f"std={tensor_detached.std().item():.6e}"
    )


def assert_finite(name, tensor):
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )


def parameter_norm(model):
    """
    Calculate the global L2 norm of trainable parameters.
    """

    total = torch.tensor(
        0.0,
        device=next(model.parameters()).device
    )

    for parameter in model.parameters():

        if parameter.requires_grad:

            total = total + torch.sum(
                parameter.detach() ** 2
            )

    return torch.sqrt(total).item()


def gradient_norm(model):
    """
    Calculate the global L2 norm of model gradients.
    """

    total = torch.tensor(
        0.0,
        device=next(model.parameters()).device
    )

    for parameter in model.parameters():

        if parameter.grad is not None:

            total = total + torch.sum(
                parameter.grad.detach() ** 2
            )

    return torch.sqrt(total).item()


def mean_gradient_magnitude(model):
    """
    Calculate mean absolute gradient across all
    parameters that contain gradients.
    """

    values = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            values.append(
                parameter.grad.detach().abs().mean()
            )

    if not values:

        return 0.0

    return torch.stack(values).mean().item()


def maximum_gradient(model):
    """
    Calculate the maximum absolute gradient.
    """

    maximum = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            current = parameter.grad.detach().abs().max().item()

            maximum = max(
                maximum,
                current
            )

    return maximum


# =========================================================
# EIKONAL GRADIENT CALCULATION
# =========================================================

def compute_eikonal_gradient(
    travel_time,
    dx,
    dy,
    dz
):
    """
    Compute central finite-difference approximations
    of the spatial travel-time gradient.

    Tensor convention:

        [B, C, D, H, W]

    Therefore:

        D -> z
        H -> y
        W -> x
    """

    # -----------------------------------------------------
    # dT/dz
    # -----------------------------------------------------

    dz_gradient = torch.zeros_like(
        travel_time
    )

    dz_gradient[:, :, 1:-1, :, :] = (
        travel_time[:, :, 2:, :, :]
        -
        travel_time[:, :, :-2, :, :]
    ) / (2.0 * dz)

    dz_gradient[:, :, 0, :, :] = (
        travel_time[:, :, 1, :, :]
        -
        travel_time[:, :, 0, :, :]
    ) / dz

    dz_gradient[:, :, -1, :, :] = (
        travel_time[:, :, -1, :, :]
        -
        travel_time[:, :, -2, :, :]
    ) / dz

    # -----------------------------------------------------
    # dT/dy
    # -----------------------------------------------------

    dy_gradient = torch.zeros_like(
        travel_time
    )

    dy_gradient[:, :, :, 1:-1, :] = (
        travel_time[:, :, :, 2:, :]
        -
        travel_time[:, :, :, :-2, :]
    ) / (2.0 * dy)

    dy_gradient[:, :, :, 0, :] = (
        travel_time[:, :, :, 1, :]
        -
        travel_time[:, :, :, 0, :]
    ) / dy

    dy_gradient[:, :, :, -1, :] = (
        travel_time[:, :, :, -1, :]
        -
        travel_time[:, :, :, -2, :]
    ) / dy

    # -----------------------------------------------------
    # dT/dx
    # -----------------------------------------------------

    dx_gradient = torch.zeros_like(
        travel_time
    )

    dx_gradient[:, :, :, :, 1:-1] = (
        travel_time[:, :, :, :, 2:]
        -
        travel_time[:, :, :, :, :-2]
    ) / (2.0 * dx)

    dx_gradient[:, :, :, :, 0] = (
        travel_time[:, :, :, :, 1]
        -
        travel_time[:, :, :, :, 0]
    ) / dx

    dx_gradient[:, :, :, :, -1] = (
        travel_time[:, :, :, :, -1]
        -
        travel_time[:, :, :, :, -2]
    ) / dx

    return (
        dx_gradient,
        dy_gradient,
        dz_gradient
    )


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print_section(
        "PHYSICS-INFORMED 3D TRAINING "
        "CONVERGENCE AUDIT"
    )

    # =====================================================
    # DEVICE
    # =====================================================

    print_section(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(
        DEVICE
    )

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # PHYSICAL CONFIGURATION
    # =====================================================

    print_section(
        "PHYSICAL CONFIGURATION"
    )

    print(
        f"DX                         : {DX}"
    )

    print(
        f"DY                         : {DY}"
    )

    print(
        f"DZ                         : {DZ}"
    )

    print(
        f"Travel-time scale          : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"Physics loss weight        : "
        f"{PHYSICS_WEIGHT}"
    )

    print(
        f"Eikonal loss weight        : "
        f"{EIKONAL_WEIGHT}"
    )

    print(
        f"Source loss weight         : "
        f"{SOURCE_WEIGHT}"
    )

    print(
        f"Travel-time loss weight    : "
        f"{TRAVEL_TIME_WEIGHT}"
    )

    print(
        f"USE_PHYSICS_LOSS           : "
        f"{USE_PHYSICS_LOSS}"
    )

    print(
        f"USE_EIKONAL_LOSS          : "
        f"{USE_EIKONAL_LOSS}"
    )

    print(
        f"USE_SOURCE_LOSS           : "
        f"{USE_SOURCE_LOSS}"
    )

    print(
        f"USE_TRAVEL_TIME_LOSS      : "
        f"{USE_TRAVEL_TIME_LOSS}"
    )

    # =====================================================
    # CONFIGURATION CONSISTENCY
    # =====================================================

    print_section(
        "CONFIGURATION CONSISTENCY AUDIT"
    )

    if not USE_PHYSICS_LOSS:

        raise RuntimeError(
            "USE_PHYSICS_LOSS must be True for "
            "this convergence audit."
        )

    if not USE_EIKONAL_LOSS:

        raise RuntimeError(
            "USE_EIKONAL_LOSS must be True for "
            "this convergence audit."
        )

    if USE_SOURCE_LOSS:

        raise RuntimeError(
            "This audit assumes source supervision "
            "is disabled."
        )

    if USE_TRAVEL_TIME_LOSS:

        raise RuntimeError(
            "This audit assumes supervised travel-time "
            "loss is disabled."
        )

    if PHYSICS_WEIGHT < 0:

        raise RuntimeError(
            "Physics loss weight cannot be negative."
        )

    if EIKONAL_WEIGHT < 0:

        raise RuntimeError(
            "Eikonal loss weight cannot be negative."
        )

    print(
        "Physics configuration is internally consistent."
    )

    print(
        "Configuration consistency test: PASSED"
    )

    # =====================================================
    # SYNTHETIC INPUT
    # =====================================================

    print_section(
        "CREATING SYNTHETIC INPUT"
    )

    torch.manual_seed(42)

    incomplete_seismic = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    target_seismic = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    # -----------------------------------------------------
    # Physical velocity.
    #
    # A constant 2000 m/s velocity field is used for this
    # controlled convergence experiment.
    # -----------------------------------------------------

    velocity = torch.full(
        (
            BATCH_SIZE,
            CHANNELS,
            DEPTH,
            HEIGHT,
            WIDTH
        ),
        2000.0,
        device=device
    )

    tensor_statistics(
        "Input seismic volume",
        incomplete_seismic
    )

    tensor_statistics(
        "Target seismic volume",
        target_seismic
    )

    tensor_statistics(
        "Velocity field",
        velocity
    )

    # =====================================================
    # NETWORK INITIALIZATION
    # =====================================================

    print_section(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=USE_UNCERTAINTY,
        use_residual=True,
        use_attention=True
    ).to(device)

    model.train()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Network successfully initialized."
    )

    print(
        f"Trainable parameters       : "
        f"{total_parameters:,}"
    )

    # =====================================================
    # PHYSICS LOSS INITIALIZATION
    # =====================================================

    print_section(
        "INITIALIZING PHYSICS LOSS"
    )

    physics_loss_function = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=EIKONAL_WEIGHT,
        source_weight=SOURCE_WEIGHT,
        travel_time_weight=TRAVEL_TIME_WEIGHT
    )

    print(
        "PhysicsLoss successfully initialized."
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print(
        f"Learning rate              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay               : "
        f"{WEIGHT_DECAY}"
    )

    # =====================================================
    # INITIAL FORWARD PASS
    # =====================================================

    print_section(
        "INITIAL FORWARD PROPAGATION"
    )

    with torch.enable_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(
            incomplete_seismic
        )

    print(
        "Network returned three outputs."
    )

    # =====================================================
    # OUTPUT VALIDATION
    # =====================================================

    print_section(
        "INITIAL NETWORK OUTPUT AUDIT"
    )

    assert_finite(
        "Reconstruction",
        reconstruction
    )

    assert_finite(
        "Travel-time",
        travel_time
    )

    assert_finite(
        "Log variance",
        log_variance
    )

    if travel_time.min().item() <= 0:

        raise RuntimeError(
            "Predicted travel time is not strictly "
            "positive."
        )

    tensor_statistics(
        "Initial reconstruction",
        reconstruction
    )

    tensor_statistics(
        "Initial travel-time",
        travel_time
    )

    tensor_statistics(
        "Initial log variance",
        log_variance
    )

    print(
        "Initial output validation: PASSED"
    )

    # =====================================================
    # INITIAL PHYSICS LOSS
    # =====================================================

    print_section(
        "INITIAL PHYSICS LOSS"
    )

    physics_components = physics_loss_function(
        travel_time=travel_time,
        velocity=velocity
    )

    initial_physics_loss = (
        physics_components["total"]
    )

    initial_eikonal_loss = (
        physics_components["eikonal"]
    )

    print(
        f"Initial physics loss      : "
        f"{initial_physics_loss.item():.12e}"
    )

    print(
        f"Initial Eikonal loss      : "
        f"{initial_eikonal_loss.item():.12e}"
    )

    # =====================================================
    # INITIAL EIKONAL GRADIENT SCALE
    # =====================================================

    print_section(
        "INITIAL TRAVEL-TIME GRADIENT SCALE"
    )

    (
        dTdx,
        dTdy,
        dTdz
    ) = compute_eikonal_gradient(
        travel_time.detach(),
        DX,
        DY,
        DZ
    )

    gradient_magnitude = torch.sqrt(
        dTdx ** 2
        +
        dTdy ** 2
        +
        dTdz ** 2
        +
        1e-12
    )

    tensor_statistics(
        "|grad T|",
        gradient_magnitude
    )

    print(
        "Expected gradient for V=2000 m/s : "
        f"{1.0 / 2000.0:.6e} s/m"
    )

    print(
        "Initial observed gradient mean     : "
        f"{gradient_magnitude.mean().item():.6e}"
    )

    # =====================================================
    # INITIAL PARAMETER NORM
    # =====================================================

    initial_parameter_norm = parameter_norm(
        model
    )

    print_section(
        "INITIAL PARAMETER STATE"
    )

    print(
        f"Parameter L2 norm         : "
        f"{initial_parameter_norm:.12e}"
    )

    # =====================================================
    # TRAINING HISTORY
    # =====================================================

    physics_history = []

    eikonal_history = []

    total_loss_history = []

    gradient_history = []

    parameter_norm_history = []

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    print_section(
        "PHYSICS TRAINING CONVERGENCE"
    )

    print(
        f"{'Step':<8}"
        f"{'Physics Loss':<20}"
        f"{'Eikonal Loss':<20}"
        f"{'Total Loss':<20}"
        f"{'Grad Mean':<18}"
        f"{'Param Norm':<18}"
    )

    print("-" * 105)

    for step in range(
        1,
        NUM_STEPS + 1
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # Forward propagation.
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(
            incomplete_seismic
        )

        # -------------------------------------------------
        # Physics loss.
        # -------------------------------------------------

        physics_components = physics_loss_function(
            travel_time=travel_time,
            velocity=velocity
        )

        physics_loss = (
            physics_components["total"]
        )

        eikonal_loss = (
            physics_components["eikonal"]
        )

        # -------------------------------------------------
        # Reconstruction loss.
        #
        # MAE is used for this convergence audit.
        # -------------------------------------------------

        mae_loss = torch.mean(
            torch.abs(
                reconstruction
                -
                target_seismic
            )
        )

        # -------------------------------------------------
        # Uncertainty probe.
        #
        # This is intentionally kept simple for the audit.
        # The purpose here is to ensure that the uncertainty
        # branch remains differentiable.
        # -------------------------------------------------

        uncertainty_loss = torch.mean(
            torch.abs(
                log_variance
            )
        )

        # -------------------------------------------------
        # Composite loss.
        # -------------------------------------------------

        total_loss = (
            LOSS_WEIGHTS["mae"]
            *
            mae_loss
            +
            PHYSICS_WEIGHT
            *
            physics_loss
            +
            LOSS_WEIGHTS["uncertainty"]
            *
            uncertainty_loss
        )

        # -------------------------------------------------
        # Backward propagation.
        # -------------------------------------------------

        total_loss.backward()

        # -------------------------------------------------
        # Gradient statistics.
        # -------------------------------------------------

        current_gradient_mean = (
            mean_gradient_magnitude(model)
        )

        # -------------------------------------------------
        # Optimizer update.
        # -------------------------------------------------

        optimizer.step()

        # -------------------------------------------------
        # Parameter norm after update.
        # -------------------------------------------------

        current_parameter_norm = (
            parameter_norm(model)
        )

        # -------------------------------------------------
        # Store history.
        # -------------------------------------------------

        physics_history.append(
            physics_loss.detach().item()
        )

        eikonal_history.append(
            eikonal_loss.detach().item()
        )

        total_loss_history.append(
            total_loss.detach().item()
        )

        gradient_history.append(
            current_gradient_mean
        )

        parameter_norm_history.append(
            current_parameter_norm
        )

        print(
            f"{step:<8}"
            f"{physics_loss.item():<20.8e}"
            f"{eikonal_loss.item():<20.8e}"
            f"{total_loss.item():<20.8e}"
            f"{current_gradient_mean:<18.8e}"
            f"{current_parameter_norm:<18.8e}"
        )

    # =====================================================
    # FINAL FORWARD PASS
    # =====================================================

    print_section(
        "FINAL FORWARD PROPAGATION"
    )

    with torch.enable_grad():

        (
            final_reconstruction,
            final_travel_time,
            final_log_variance
        ) = model(
            incomplete_seismic
        )

    assert_finite(
        "Final reconstruction",
        final_reconstruction
    )

    assert_finite(
        "Final travel-time",
        final_travel_time
    )

    assert_finite(
        "Final log variance",
        final_log_variance
    )

    print(
        "Final network outputs are finite."
    )

    # =====================================================
    # FINAL PHYSICS LOSS
    # =====================================================

    print_section(
        "FINAL PHYSICS LOSS"
    )

    final_components = physics_loss_function(
        travel_time=final_travel_time,
        velocity=velocity
    )

    final_physics_loss = (
        final_components["total"]
    )

    final_eikonal_loss = (
        final_components["eikonal"]
    )

    print(
        f"Final physics loss        : "
        f"{final_physics_loss.item():.12e}"
    )

    print(
        f"Final Eikonal loss        : "
        f"{final_eikonal_loss.item():.12e}"
    )

    # =====================================================
    # PHYSICS LOSS CHANGE
    # =====================================================

    print_section(
        "PHYSICS LOSS CONVERGENCE ANALYSIS"
    )

    initial_value = (
        initial_physics_loss.detach().item()
    )

    final_value = (
        final_physics_loss.detach().item()
    )

    absolute_change = (
        final_value
        -
        initial_value
    )

    reduction = (
        initial_value
        -
        final_value
    )

    if abs(initial_value) > 1e-12:

        reduction_percentage = (
            reduction
            /
            abs(initial_value)
            *
            100.0
        )

    else:

        reduction_percentage = 0.0

    print(
        f"Initial physics loss     : "
        f"{initial_value:.12e}"
    )

    print(
        f"Final physics loss       : "
        f"{final_value:.12e}"
    )

    print(
        f"Absolute change          : "
        f"{absolute_change:.12e}"
    )

    print(
        f"Physics loss reduction   : "
        f"{reduction_percentage:.6f}%"
    )

    # =====================================================
    # PHYSICS LOSS TREND
    # =====================================================

    print_section(
        "PHYSICS LOSS TREND"
    )

    for index, value in enumerate(
        physics_history,
        start=1
    ):

        print(
            f"Step {index:<3}: "
            f"{value:.12e}"
        )

    # =====================================================
    # PARAMETER UPDATE AUDIT
    # =====================================================

    print_section(
        "PARAMETER UPDATE AUDIT"
    )

    final_parameter_norm = (
        parameter_norm(model)
    )

    parameter_change = (
        final_parameter_norm
        -
        initial_parameter_norm
    )

    print(
        f"Initial parameter norm    : "
        f"{initial_parameter_norm:.12e}"
    )

    print(
        f"Final parameter norm      : "
        f"{final_parameter_norm:.12e}"
    )

    print(
        f"Parameter norm change     : "
        f"{parameter_change:.12e}"
    )

    if abs(parameter_change) > 0.0:

        print(
            "Model parameters changed during optimization."
        )

        print(
            "Parameter update test: PASSED"
        )

    else:

        raise RuntimeError(
            "Model parameters did not change during "
            "optimization."
        )

    # =====================================================
    # FINAL TRAVEL-TIME GRADIENT
    # =====================================================

    print_section(
        "FINAL TRAVEL-TIME GRADIENT SCALE"
    )

    (
        final_dTdx,
        final_dTdy,
        final_dTdz
    ) = compute_eikonal_gradient(
        final_travel_time.detach(),
        DX,
        DY,
        DZ
    )

    final_gradient_magnitude = torch.sqrt(
        final_dTdx ** 2
        +
        final_dTdy ** 2
        +
        final_dTdz ** 2
        +
        1e-12
    )

    tensor_statistics(
        "Final |grad T|",
        final_gradient_magnitude
    )

    print(
        "Expected |grad T| for "
        "V=2000 m/s        : "
        f"{1.0 / 2000.0:.6e} s/m"
    )

    print(
        "Final observed mean |grad T| : "
        f"{final_gradient_magnitude.mean().item():.6e}"
    )

    # =====================================================
    # FINAL TRAVEL-TIME VALIDATION
    # =====================================================

    print_section(
        "FINAL TRAVEL-TIME VALIDATION"
    )

    print(
        f"Minimum travel time       : "
        f"{final_travel_time.min().item():.12e}"
    )

    print(
        f"Maximum travel time       : "
        f"{final_travel_time.max().item():.12e}"
    )

    print(
        f"Mean travel time          : "
        f"{final_travel_time.mean().item():.12e}"
    )

    if final_travel_time.min().item() <= 0:

        raise RuntimeError(
            "Final travel-time field contains "
            "non-positive values."
        )

    print(
        "Final travel-time positivity: PASSED"
    )

    # =====================================================
    # NUMERICAL STABILITY
    # =====================================================

    print_section(
        "NUMERICAL STABILITY AUDIT"
    )

    assert_finite(
        "Physics history",
        torch.tensor(
            physics_history
        )
    )

    assert_finite(
        "Eikonal history",
        torch.tensor(
            eikonal_history
        )
    )

    assert_finite(
        "Total loss history",
        torch.tensor(
            total_loss_history
        )
    )

    assert_finite(
        "Gradient history",
        torch.tensor(
            gradient_history
        )
    )

    print(
        "All training-history values are finite."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # =====================================================
    # CONVERGENCE INTERPRETATION
    # =====================================================

    print_section(
        "CONVERGENCE INTERPRETATION"
    )

    if final_value < initial_value:

        print(
            "The physics loss decreased during optimization."
        )

        print(
            "Physics-loss descent: OBSERVED"
        )

    elif final_value == initial_value:

        print(
            "The physics loss remained unchanged."
        )

        print(
            "Physics-loss descent: NOT OBSERVED"
        )

    else:

        print(
            "The physics loss increased during optimization."
        )

        print(
            "Physics-loss descent: NOT OBSERVED"
        )

    # -----------------------------------------------------
    # Check whether at least one later step is lower than
    # the initial value.
    # -----------------------------------------------------

    if any(
        value < initial_value
        for value in physics_history
    ):

        print(
            "At least one optimization step reduced "
            "the physics loss relative to initialization."
        )

        descent_detected = True

    else:

        print(
            "No optimization step reduced the physics loss "
            "relative to initialization."
        )

        descent_detected = False

    # =====================================================
    # FINAL AUDIT DECISION
    # =====================================================

    print_section(
        "PHYSICS TRAINING CONVERGENCE AUDIT RESULT"
    )

    if (
        descent_detected
        and
        abs(parameter_change) > 0.0
    ):

        print(
            "Physics-loss descent was successfully observed."
        )

        print(
            "The network responded to optimization."
        )

        print(
            "PHYSICS TRAINING CONVERGENCE AUDIT PASSED."
        )

    else:

        print(
            "Physics-loss convergence was not demonstrated "
            "during this short optimization experiment."
        )

        print(
            "This does NOT necessarily indicate a model "
            "failure; additional training steps or "
            "hyperparameter tuning may be required."
        )

        print(
            "PHYSICS TRAINING CONVERGENCE AUDIT COMPLETED "
            "WITH NO CONVERGENCE DETECTED."
        )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()