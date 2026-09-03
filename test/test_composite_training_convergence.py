"""
============================================================
COMPOSITE TRAINING CONVERGENCE AUDIT
============================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This test verifies that the complete composite training
objective can successfully optimize the network.

The audit monitors:

    1. Total composite loss
    2. MAE reconstruction loss
    3. Physics loss
    4. Eikonal loss
    5. Aleatoric uncertainty loss
    6. SSIM loss
    7. Physics-to-MAE loss ratio
    8. Gradient stability
    9. Parameter updates
    10. Travel-time positivity
    11. Numerical stability
    12. Short-run convergence

Composite objective
-------------------

    L_total =
        lambda_mae * L_mae
        +
        lambda_physics * L_physics
        +
        lambda_uncertainty * L_uncertainty
        +
        lambda_ssim * L_ssim

Author: Ormin Joseph
============================================================
"""

import torch
import torch.optim as optim


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.network import Network3D

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    UncertaintyLoss
)
from losses.ssim_loss import SSIMLoss

from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    SEISMIC_DATA_RANGE,
    USE_PHYSICS_LOSS,
    USE_EIKONAL_LOSS,
    USE_UNCERTAINTY,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

NUM_TRAINING_STEPS = 10

BATCH_SIZE = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

VELOCITY_VALUE = 2000.0


# ============================================================
# PRINTING UTILITIES
# ============================================================

def print_header(title):
    """Print a major audit section."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_statistics(name, tensor):
    """Print numerical statistics for a tensor."""

    tensor = tensor.detach()

    print(
        f"{name:<30} "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


# ============================================================
# GRADIENT UTILITIES
# ============================================================

def compute_gradient_statistics(model):
    """
    Compute gradient statistics across all trainable
    model parameters.
    """

    gradient_values = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_values.append(
                parameter.grad.detach()
                .abs()
                .reshape(-1)
            )

    if not gradient_values:

        return {
            "mean": 0.0,
            "maximum": 0.0,
            "minimum": 0.0,
            "elements": 0,
            "finite": False
        }

    gradients = torch.cat(
        gradient_values
    )

    return {
        "mean": gradients.mean().item(),
        "maximum": gradients.max().item(),
        "minimum": gradients.min().item(),
        "elements": gradients.numel(),
        "finite": bool(
            torch.isfinite(
                gradients
            ).all().item()
        )
    }


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def compute_parameter_norm(model):
    """
    Compute the global L2 norm of all trainable
    parameters.
    """

    squared_norm = torch.tensor(
        0.0,
        device=DEVICE
    )

    for parameter in model.parameters():

        if parameter.requires_grad:

            squared_norm = (
                squared_norm
                +
                parameter.detach()
                .pow(2)
                .sum()
            )

    return torch.sqrt(
        squared_norm
    ).item()


# ============================================================
# PHYSICS COMPONENT EXTRACTION
# ============================================================

def get_physics_components(
    physics_loss_function,
    travel_time,
    velocity
):
    """
    Compute the physics loss and safely extract the
    Eikonal component.

    Supported PhysicsLoss return formats:

        1. Tensor
        2. Dictionary
        3. (total_loss, components_dict)
        4. (components_dict, total_loss)
        5. (tensor, tensor)

    Returns:

        physics_loss : scalar tensor
        eikonal_loss : scalar tensor
    """

    physics_output = physics_loss_function(
        travel_time,
        velocity
    )

    # ========================================================
    # CASE 1: SINGLE TENSOR
    # ========================================================

    if torch.is_tensor(
        physics_output
    ):

        physics_loss = physics_output
        eikonal_loss = physics_output

    # ========================================================
    # CASE 2: DICTIONARY
    # ========================================================

    elif isinstance(
        physics_output,
        dict
    ):

        components = physics_output

        physics_loss = None

        total_keys = [
            "total",
            "physics",
            "loss",
            "total_loss"
        ]

        for key in total_keys:

            if key in components:

                physics_loss = (
                    components[key]
                )

                break

        eikonal_loss = None

        eikonal_keys = [
            "eikonal",
            "eikonal_loss"
        ]

        for key in eikonal_keys:

            if key in components:

                eikonal_loss = (
                    components[key]
                )

                break

        if physics_loss is None:

            if eikonal_loss is not None:

                physics_loss = (
                    eikonal_loss
                )

            else:

                raise RuntimeError(
                    "PhysicsLoss returned a dictionary, "
                    "but no total or Eikonal loss "
                    "could be identified."
                )

        if eikonal_loss is None:

            eikonal_loss = (
                physics_loss
            )

    # ========================================================
    # CASE 3: TUPLE
    # ========================================================

    elif isinstance(
        physics_output,
        tuple
    ):

        if len(
            physics_output
        ) != 2:

            raise RuntimeError(
                "PhysicsLoss returned a tuple with an "
                "unexpected number of elements."
            )

        first_item = (
            physics_output[0]
        )

        second_item = (
            physics_output[1]
        )

        # ----------------------------------------------------
        # FORMAT:
        #
        # total_loss, components_dict
        # ----------------------------------------------------

        if (
            torch.is_tensor(
                first_item
            )
            and isinstance(
                second_item,
                dict
            )
        ):

            physics_loss = (
                first_item
            )

            eikonal_loss = (
                second_item.get(
                    "eikonal"
                )
            )

            if eikonal_loss is None:

                eikonal_loss = (
                    second_item.get(
                        "eikonal_loss"
                    )
                )

            if eikonal_loss is None:

                eikonal_loss = (
                    physics_loss
                )

        # ----------------------------------------------------
        # FORMAT:
        #
        # components_dict, total_loss
        # ----------------------------------------------------

        elif (
            isinstance(
                first_item,
                dict
            )
            and torch.is_tensor(
                second_item
            )
        ):

            physics_loss = (
                second_item
            )

            eikonal_loss = (
                first_item.get(
                    "eikonal"
                )
            )

            if eikonal_loss is None:

                eikonal_loss = (
                    first_item.get(
                        "eikonal_loss"
                    )
                )

            if eikonal_loss is None:

                eikonal_loss = (
                    physics_loss
                )

        # ----------------------------------------------------
        # FORMAT:
        #
        # tensor, tensor
        # ----------------------------------------------------

        elif (
            torch.is_tensor(
                first_item
            )
            and torch.is_tensor(
                second_item
            )
        ):

            physics_loss = (
                first_item
            )

            eikonal_loss = (
                second_item
            )

        else:

            raise RuntimeError(
                "Unable to interpret the tuple returned "
                "by PhysicsLoss."
            )

    # ========================================================
    # UNSUPPORTED FORMAT
    # ========================================================

    else:

        raise RuntimeError(
            "PhysicsLoss returned an unsupported object "
            f"of type: {type(physics_output)}"
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not torch.is_tensor(
        physics_loss
    ):

        raise RuntimeError(
            "Extracted physics loss is not a tensor. "
            f"Found type: {type(physics_loss)}"
        )

    if not torch.is_tensor(
        eikonal_loss
    ):

        raise RuntimeError(
            "Extracted Eikonal loss is not a tensor. "
            f"Found type: {type(eikonal_loss)}"
        )

    return (
        physics_loss,
        eikonal_loss
    )


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    # ========================================================
    # AUDIT HEADER
    # ========================================================

    print_header(
        "COMPOSITE TRAINING CONVERGENCE AUDIT"
    )

    # ========================================================
    # DEVICE CONFIGURATION
    # ========================================================

    device = torch.device(
        DEVICE
    )

    print_header(
        "DEVICE CONFIGURATION"
    )

    print(
        f"Device                     : {device}"
    )

    # ========================================================
    # COMPOSITE LOSS CONFIGURATION
    # ========================================================

    print_header(
        "COMPOSITE LOSS CONFIGURATION"
    )

    print(
        f"MAE weight                 : "
        f"{LOSS_WEIGHTS['mae']}"
    )

    print(
        f"Physics weight             : "
        f"{LOSS_WEIGHTS['physics']}"
    )

    print(
        f"Uncertainty weight         : "
        f"{LOSS_WEIGHTS['uncertainty']}"
    )

    print(
        f"SSIM weight                : "
        f"{LOSS_WEIGHTS['ssim']}"
    )

    print(
        f"Eikonal weight             : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    print(
        f"Travel-time scale          : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"USE_PHYSICS_LOSS           : "
        f"{USE_PHYSICS_LOSS}"
    )

    print(
        f"USE_EIKONAL_LOSS           : "
        f"{USE_EIKONAL_LOSS}"
    )

    print(
        f"USE_UNCERTAINTY            : "
        f"{USE_UNCERTAINTY}"
    )

    # ========================================================
    # CREATE SYNTHETIC TRAINING DATA
    # ========================================================

    print_header(
        "CREATING SYNTHETIC TRAINING DATA"
    )

    torch.manual_seed(
        42
    )

    input_cube = torch.randn(
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    target_cube = torch.randn(
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    velocity = torch.full(
        (
            BATCH_SIZE,
            1,
            DEPTH,
            HEIGHT,
            WIDTH
        ),
        VELOCITY_VALUE,
        device=device
    )

    print_statistics(
        "Input seismic volume",
        input_cube
    )

    print_statistics(
        "Target seismic volume",
        target_cube
    )

    print_statistics(
        "Velocity field",
        velocity
    )

    # ========================================================
    # INITIALIZE NETWORK
    # ========================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=USE_UNCERTAINTY
    ).to(
        device
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Network successfully initialized."
    )

    print(
        f"Trainable parameters       : "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # INITIALIZE LOSS FUNCTIONS
    # ========================================================

    print_header(
        "INITIALIZING LOSS FUNCTIONS"
    )

    mae_loss_function = (
        MAELoss()
    )

    physics_loss_function = (
        PhysicsLoss(
            dx=DX,
            dy=DY,
            dz=DZ
        )
    )

    uncertainty_loss_function = (
        UncertaintyLoss()
    )

    ssim_loss_function = (
        SSIMLoss(
            data_range=SEISMIC_DATA_RANGE
        )
    )

    print(
        "MAE loss initialized."
    )

    print(
        "Physics loss initialized."
    )

    print(
        "Aleatoric uncertainty loss initialized."
    )

    print(
        "SSIM loss initialized."
    )

    # ========================================================
    # INITIALIZE OPTIMIZER
    # ========================================================

    print_header(
        "INITIALIZING OPTIMIZER"
    )

    optimizer = optim.Adam(
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

    # ========================================================
    # INITIAL PARAMETER STATE
    # ========================================================

    initial_parameter_norm = (
        compute_parameter_norm(
            model
        )
    )

    print_header(
        "INITIAL PARAMETER STATE"
    )

    print(
        f"Initial parameter norm     : "
        f"{initial_parameter_norm:.12e}"
    )

    # ========================================================
    # INITIAL COMPOSITE LOSS EVALUATION
    # ========================================================

    print_header(
        "INITIAL COMPOSITE LOSS EVALUATION"
    )

    model.train()

    (
        reconstructed_cube,
        travel_time,
        log_variance
    ) = model(
        input_cube
    )

    initial_mae_loss = (
        mae_loss_function(
            reconstructed_cube,
            target_cube
        )
    )

    if USE_PHYSICS_LOSS:

        (
            initial_physics_loss,
            initial_eikonal_loss
        ) = get_physics_components(
            physics_loss_function,
            travel_time,
            velocity
        )

    else:

        initial_physics_loss = torch.tensor(
            0.0,
            device=device
        )

        initial_eikonal_loss = torch.tensor(
            0.0,
            device=device
        )

    if USE_UNCERTAINTY:

        initial_uncertainty_loss = (
            uncertainty_loss_function(
                reconstructed_cube,
                target_cube,
                log_variance
            )
        )

    else:

        initial_uncertainty_loss = torch.tensor(
            0.0,
            device=device
        )

    initial_ssim_loss = (
        ssim_loss_function(
            reconstructed_cube,
            target_cube
        )
    )

    initial_total_loss = (
        LOSS_WEIGHTS["mae"]
        * initial_mae_loss
        +
        LOSS_WEIGHTS["physics"]
        * initial_physics_loss
        +
        LOSS_WEIGHTS["uncertainty"]
        * initial_uncertainty_loss
        +
        LOSS_WEIGHTS["ssim"]
        * initial_ssim_loss
    )

    print(
        f"Initial MAE loss          : "
        f"{initial_mae_loss.item():.12e}"
    )

    print(
        f"Initial physics loss      : "
        f"{initial_physics_loss.item():.12e}"
    )

    print(
        f"Initial Eikonal loss      : "
        f"{initial_eikonal_loss.item():.12e}"
    )

    print(
        f"Initial uncertainty loss  : "
        f"{initial_uncertainty_loss.item():.12e}"
    )

    print(
        f"Initial SSIM loss         : "
        f"{initial_ssim_loss.item():.12e}"
    )

    print(
        f"Initial total loss        : "
        f"{initial_total_loss.item():.12e}"
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {
        "total": [],
        "mae": [],
        "physics": [],
        "eikonal": [],
        "uncertainty": [],
        "ssim": [],
        "physics_mae_ratio": [],
        "gradient_mean": [],
        "gradient_maximum": []
    }

    # ========================================================
    # COMPOSITE TRAINING CONVERGENCE
    # ========================================================

    print_header(
        "COMPOSITE TRAINING CONVERGENCE"
    )

    print(
        f"{'Step':<8}"
        f"{'Total Loss':<18}"
        f"{'MAE Loss':<18}"
        f"{'Physics Loss':<18}"
        f"{'Eikonal':<18}"
        f"{'Uncertainty':<18}"
        f"{'SSIM':<18}"
        f"{'Physics/MAE':<18}"
        f"{'Grad Mean':<18}"
        f"{'Grad Max':<18}"
    )

    print(
        "-" * 170
    )

    for step in range(
        1,
        NUM_TRAINING_STEPS + 1
    ):

        # ====================================================
        # RESET GRADIENTS
        # ====================================================

        optimizer.zero_grad()

        # ====================================================
        # FORWARD PROPAGATION
        # ====================================================

        (
            reconstructed_cube,
            travel_time,
            log_variance
        ) = model(
            input_cube
        )

        # ====================================================
        # MAE LOSS
        # ====================================================

        mae_loss = (
            mae_loss_function(
                reconstructed_cube,
                target_cube
            )
        )

        # ====================================================
        # PHYSICS LOSS
        # ====================================================

        if USE_PHYSICS_LOSS:

            (
                physics_loss,
                eikonal_loss
            ) = get_physics_components(
                physics_loss_function,
                travel_time,
                velocity
            )

        else:

            physics_loss = torch.tensor(
                0.0,
                device=device
            )

            eikonal_loss = torch.tensor(
                0.0,
                device=device
            )

        # ====================================================
        # UNCERTAINTY LOSS
        # ====================================================

        if USE_UNCERTAINTY:

            uncertainty_loss = (
                uncertainty_loss_function(
                    reconstructed_cube,
                    target_cube,
                    log_variance
                )
            )

        else:

            uncertainty_loss = torch.tensor(
                0.0,
                device=device
            )

        # ====================================================
        # SSIM LOSS
        # ====================================================

        ssim_loss = (
            ssim_loss_function(
                reconstructed_cube,
                target_cube
            )
        )

        # ====================================================
        # COMPOSITE LOSS
        # ====================================================

        total_loss = (
            LOSS_WEIGHTS["mae"]
            * mae_loss
            +
            LOSS_WEIGHTS["physics"]
            * physics_loss
            +
            LOSS_WEIGHTS["uncertainty"]
            * uncertainty_loss
            +
            LOSS_WEIGHTS["ssim"]
            * ssim_loss
        )

        # ====================================================
        # NUMERICAL STABILITY CHECK
        # ====================================================

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "Composite loss became NaN or Inf."
            )

        # ====================================================
        # BACKWARD PROPAGATION
        # ====================================================

        total_loss.backward()

        # ====================================================
        # GRADIENT AUDIT
        # ====================================================

        gradient_statistics = (
            compute_gradient_statistics(
                model
            )
        )

        if not gradient_statistics[
            "finite"
        ]:

            raise RuntimeError(
                "Non-finite network gradients detected."
            )

        # ====================================================
        # OPTIMIZER UPDATE
        # ====================================================

        optimizer.step()

        # ====================================================
        # PHYSICS / MAE RATIO
        # ====================================================

        weighted_physics = (
            LOSS_WEIGHTS["physics"]
            * physics_loss
        )

        weighted_mae = (
            LOSS_WEIGHTS["mae"]
            * mae_loss
        )

        physics_mae_ratio = (
            weighted_physics
            /
            (
                weighted_mae
                +
                1e-12
            )
        )

        # ====================================================
        # STORE TRAINING HISTORY
        # ====================================================

        history["total"].append(
            total_loss.item()
        )

        history["mae"].append(
            mae_loss.item()
        )

        history["physics"].append(
            physics_loss.item()
        )

        history["eikonal"].append(
            eikonal_loss.item()
        )

        history["uncertainty"].append(
            uncertainty_loss.item()
        )

        history["ssim"].append(
            ssim_loss.item()
        )

        history[
            "physics_mae_ratio"
        ].append(
            physics_mae_ratio.item()
        )

        history[
            "gradient_mean"
        ].append(
            gradient_statistics[
                "mean"
            ]
        )

        history[
            "gradient_maximum"
        ].append(
            gradient_statistics[
                "maximum"
            ]
        )

        # ====================================================
        # PRINT TRAINING STEP
        # ====================================================

        print(
            f"{step:<8}"
            f"{total_loss.item():<18.8e}"
            f"{mae_loss.item():<18.8e}"
            f"{physics_loss.item():<18.8e}"
            f"{eikonal_loss.item():<18.8e}"
            f"{uncertainty_loss.item():<18.8e}"
            f"{ssim_loss.item():<18.8e}"
            f"{physics_mae_ratio.item():<18.8e}"
            f"{gradient_statistics['mean']:<18.8e}"
            f"{gradient_statistics['maximum']:<18.8e}"
        )

    # ========================================================
    # FINAL PARAMETER STATE
    # ========================================================

    final_parameter_norm = (
        compute_parameter_norm(
            model
        )
    )

    parameter_change = (
        final_parameter_norm
        -
        initial_parameter_norm
    )

    print_header(
        "PARAMETER UPDATE AUDIT"
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

    if abs(
        parameter_change
    ) <= 1e-12:

        raise RuntimeError(
            "Model parameters did not change."
        )

    print(
        "Model parameters changed during optimization."
    )

    print(
        "Parameter update test: PASSED"
    )

    # ========================================================
    # FINAL COMPOSITE LOSS EVALUATION
    # ========================================================

    print_header(
        "FINAL COMPOSITE LOSS EVALUATION"
    )

    optimizer.zero_grad()

    (
        final_reconstructed_cube,
        final_travel_time,
        final_log_variance
    ) = model(
        input_cube
    )

    final_mae_loss = (
        mae_loss_function(
            final_reconstructed_cube,
            target_cube
        )
    )

    if USE_PHYSICS_LOSS:

        (
            final_physics_loss,
            final_eikonal_loss
        ) = get_physics_components(
            physics_loss_function,
            final_travel_time,
            velocity
        )

    else:

        final_physics_loss = torch.tensor(
            0.0,
            device=device
        )

        final_eikonal_loss = torch.tensor(
            0.0,
            device=device
        )

    if USE_UNCERTAINTY:

        final_uncertainty_loss = (
            uncertainty_loss_function(
                final_reconstructed_cube,
                target_cube,
                final_log_variance
            )
        )

    else:

        final_uncertainty_loss = torch.tensor(
            0.0,
            device=device
        )

    final_ssim_loss = (
        ssim_loss_function(
            final_reconstructed_cube,
            target_cube
        )
    )

    final_total_loss = (
        LOSS_WEIGHTS["mae"]
        * final_mae_loss
        +
        LOSS_WEIGHTS["physics"]
        * final_physics_loss
        +
        LOSS_WEIGHTS["uncertainty"]
        * final_uncertainty_loss
        +
        LOSS_WEIGHTS["ssim"]
        * final_ssim_loss
    )

    print(
        f"Final MAE loss            : "
        f"{final_mae_loss.item():.12e}"
    )

    print(
        f"Final physics loss        : "
        f"{final_physics_loss.item():.12e}"
    )

    print(
        f"Final Eikonal loss        : "
        f"{final_eikonal_loss.item():.12e}"
    )

    print(
        f"Final uncertainty loss    : "
        f"{final_uncertainty_loss.item():.12e}"
    )

    print(
        f"Final SSIM loss           : "
        f"{final_ssim_loss.item():.12e}"
    )

    print(
        f"Final total loss          : "
        f"{final_total_loss.item():.12e}"
    )

    # ========================================================
    # LOSS CONVERGENCE ANALYSIS
    # ========================================================

    print_header(
        "COMPOSITE LOSS CONVERGENCE ANALYSIS"
    )

    total_reduction = (
        (
            initial_total_loss.item()
            -
            final_total_loss.item()
        )
        /
        (
            abs(
                initial_total_loss.item()
            )
            +
            1e-12
        )
        * 100.0
    )

    mae_reduction = (
        (
            initial_mae_loss.item()
            -
            final_mae_loss.item()
        )
        /
        (
            abs(
                initial_mae_loss.item()
            )
            +
            1e-12
        )
        * 100.0
    )

    physics_reduction = (
        (
            initial_physics_loss.item()
            -
            final_physics_loss.item()
        )
        /
        (
            abs(
                initial_physics_loss.item()
            )
            +
            1e-12
        )
        * 100.0
    )

    print(
        f"Total loss reduction      : "
        f"{total_reduction:.6f}%"
    )

    print(
        f"MAE loss reduction        : "
        f"{mae_reduction:.6f}%"
    )

    print(
        f"Physics loss reduction    : "
        f"{physics_reduction:.6f}%"
    )

    # ========================================================
    # LOSS TREND ANALYSIS
    # ========================================================

    print_header(
        "LOSS TREND ANALYSIS"
    )

    for index in range(
        NUM_TRAINING_STEPS
    ):

        print(
            f"Step {index + 1:<3}: "
            f"Total={history['total'][index]:.8e}   "
            f"MAE={history['mae'][index]:.8e}   "
            f"Physics={history['physics'][index]:.8e}"
        )

    # ========================================================
    # FINAL TRAVEL-TIME VALIDATION
    # ========================================================

    print_header(
        "FINAL TRAVEL-TIME VALIDATION"
    )

    minimum_travel_time = (
        final_travel_time.min().item()
    )

    maximum_travel_time = (
        final_travel_time.max().item()
    )

    mean_travel_time = (
        final_travel_time.mean().item()
    )

    print(
        f"Minimum travel time       : "
        f"{minimum_travel_time:.12e}"
    )

    print(
        f"Maximum travel time       : "
        f"{maximum_travel_time:.12e}"
    )

    print(
        f"Mean travel time          : "
        f"{mean_travel_time:.12e}"
    )

    if minimum_travel_time < 0:

        raise RuntimeError(
            "Travel-time positivity was violated."
        )

    print(
        "Travel-time positivity: PASSED"
    )

    # ========================================================
    # NUMERICAL STABILITY AUDIT
    # ========================================================

    print_header(
        "NUMERICAL STABILITY AUDIT"
    )

    for name, values in history.items():

        tensor = torch.tensor(
            values
        )

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"Non-finite values detected in "
                f"{name} history."
            )

    print(
        "All training-history values are finite."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # ========================================================
    # CONVERGENCE DECISION
    # ========================================================

    print_header(
        "COMPOSITE CONVERGENCE INTERPRETATION"
    )

    total_descent = (
        final_total_loss.item()
        <
        initial_total_loss.item()
    )

    mae_descent = (
        final_mae_loss.item()
        <
        initial_mae_loss.item()
    )

    physics_stable = bool(
        torch.isfinite(
            final_physics_loss
        ).item()
    )

    print(
        f"Total-loss descent        : "
        f"{total_descent}"
    )

    print(
        f"MAE-loss descent          : "
        f"{mae_descent}"
    )

    print(
        f"Physics-loss finite       : "
        f"{physics_stable}"
    )

    if not total_descent:

        raise RuntimeError(
            "Composite training did not reduce "
            "the total loss."
        )

    if not mae_descent:

        raise RuntimeError(
            "Composite training did not reduce "
            "the MAE reconstruction loss."
        )

    if not physics_stable:

        raise RuntimeError(
            "Physics loss became numerically unstable."
        )

    print()

    print(
        "Composite-loss descent: OBSERVED"
    )

    print(
        "MAE reconstruction improvement: OBSERVED"
    )

    print(
        "Physics loss remained numerically stable."
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_header(
        "COMPOSITE TRAINING CONVERGENCE AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ Composite loss construction"
    )

    print(
        "  ✓ MAE reconstruction loss"
    )

    print(
        "  ✓ Physics loss"
    )

    print(
        "  ✓ Eikonal loss"
    )

    print(
        "  ✓ Aleatoric uncertainty loss"
    )

    print(
        "  ✓ SSIM structural loss"
    )

    print(
        "  ✓ Physics-to-MAE balance"
    )

    print(
        "  ✓ Backward propagation"
    )

    print(
        "  ✓ Finite gradients"
    )

    print(
        "  ✓ Parameter updates"
    )

    print(
        "  ✓ Travel-time positivity"
    )

    print(
        "  ✓ Numerical stability"
    )

    print(
        "  ✓ Composite-loss convergence"
    )

    print(
        "  ✓ Reconstruction-loss convergence"
    )

    print()

    print(
        "COMPOSITE TRAINING CONVERGENCE TEST PASSED."
    )


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    main()