"""
======================================================================
COMPLETE TOTAL LOSS INTEGRATION AUDIT
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

This audit validates the complete loss integration pathway:

    Incomplete Seismic Input
            |
            v
        Network3D
            |
            +------------------+
            |        |         |
            v        v         v
    Reconstruction  Travel-T  Log Variance
            |        |         |
            |        |         |
            v        v         v
          MAE    Physics   Uncertainty
            |        |         |
            +--------+---------+
                     |
                     v
                    SSIM
                     |
                     v
                TotalLoss
                     |
                     v
               Weighted Sum
                     |
                     v
                 Backward()

The audit verifies:

    1. Network forward propagation
    2. Output shape consistency
    3. Finite reconstruction output
    4. Finite travel-time output
    5. Finite log-variance output
    6. MAE loss
    7. Physics loss
    8. Heteroscedastic aleatoric uncertainty loss
    9. SSIM loss
   10. Composite total loss
   11. Weighted-loss decomposition
   12. Independent total-loss verification
   13. Backward propagation
   14. Network gradient propagation
   15. Numerical stability

Author: Ormin Joseph
======================================================================
"""

import torch

from models.network import Network3D

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss

from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    UncertaintyLoss
)

from losses.ssim_loss import SSIMLoss

from losses.total_loss import TotalLoss

from utils.config import (
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    DX,
    DY,
    DZ,
    SEED
)


# ======================================================================
# DEVICE
# ======================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# AUDIT TENSOR SIZE
# ======================================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32


# ======================================================================
# VELOCITY CONFIGURATION
# ======================================================================

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 4500.0


# ======================================================================
# PRINT UTILITIES
# ======================================================================

def print_header(title):
    """Print a formatted audit section header."""

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


def print_stats(name, tensor):
    """Print numerical statistics for a tensor."""

    tensor = tensor.detach()

    print(f"{name}:")

    print(
        f"    shape : "
        f"{tuple(tensor.shape)}"
    )

    print(
        f"    min   : "
        f"{tensor.min().item():.6e}"
    )

    print(
        f"    max   : "
        f"{tensor.max().item():.6e}"
    )

    print(
        f"    mean  : "
        f"{tensor.mean().item():.6e}"
    )

    print(
        f"    std   : "
        f"{tensor.std().item():.6e}"
    )

    print(
        f"    absmax: "
        f"{tensor.abs().max().item():.6e}"
    )


def assert_finite(name, tensor):
    """Confirm that a tensor contains only finite values."""

    if not torch.isfinite(
        tensor
    ).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )

    print(
        f"{name}: finite values confirmed."
    )


def calculate_gradient_statistics(model):
    """
    Calculate network parameter-gradient statistics.
    """

    gradient_norm_squared = 0.0

    maximum_gradient = 0.0

    parameters_with_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        parameters_with_gradients += 1

        gradient = parameter.grad.detach()

        if not torch.isfinite(
            gradient
        ).all():

            raise RuntimeError(
                "Network parameter gradient "
                "contains NaN or Inf."
            )

        gradient_norm = (
            gradient.norm().item()
        )

        gradient_norm_squared += (
            gradient_norm ** 2
        )

        gradient_maximum = (
            gradient.abs().max().item()
        )

        maximum_gradient = max(
            maximum_gradient,
            gradient_maximum
        )

    total_gradient_norm = (
        gradient_norm_squared ** 0.5
    )

    return (
        total_gradient_norm,
        maximum_gradient,
        parameters_with_gradients
    )


# ======================================================================
# SYNTHETIC INPUT
# ======================================================================

def create_synthetic_input():
    """
    Create a normalized synthetic incomplete seismic volume.
    """

    seismic_input = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE
    )

    seismic_input = torch.tanh(
        seismic_input
    )

    return seismic_input


# ======================================================================
# SYNTHETIC TARGET
# ======================================================================

def create_synthetic_target():
    """
    Create a synthetic complete seismic target.

    The target remains in approximately [-1, 1].
    """

    target = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE
    )

    target = torch.tanh(
        target
    )

    return target


# ======================================================================
# SYNTHETIC VELOCITY
# ======================================================================

def create_synthetic_velocity():
    """
    Create a positive heterogeneous physical velocity model.
    """

    z = torch.linspace(
        0.0,
        1.0,
        DEPTH,
        device=DEVICE
    ).view(
        1,
        1,
        DEPTH,
        1,
        1
    )

    y = torch.linspace(
        0.0,
        1.0,
        HEIGHT,
        device=DEVICE
    ).view(
        1,
        1,
        1,
        HEIGHT,
        1
    )

    x = torch.linspace(
        0.0,
        1.0,
        WIDTH,
        device=DEVICE
    ).view(
        1,
        1,
        1,
        1,
        WIDTH
    )

    normalized_velocity = (

        0.50

        + 0.25 * z

        + 0.15 * y

        + 0.10 * x
    )

    normalized_velocity = (

        normalized_velocity
        -
        normalized_velocity.min()

    ) / (

        normalized_velocity.max()
        -
        normalized_velocity.min()
    )

    velocity = (

        VELOCITY_MIN

        +

        (
            VELOCITY_MAX
            -
            VELOCITY_MIN
        )

        *
        normalized_velocity
    )

    velocity = velocity.expand(

        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH

    ).clone()

    return velocity


# ======================================================================
# MAIN AUDIT
# ======================================================================

def main():

    # ==================================================================
    # REPRODUCIBILITY
    # ==================================================================

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    # ==================================================================
    # AUDIT HEADER
    # ==================================================================

    print_header(
        "COMPLETE TOTAL LOSS INTEGRATION AUDIT"
    )

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Tensor shape        : "
        f"({BATCH_SIZE}, "
        f"{CHANNELS}, "
        f"{DEPTH}, "
        f"{HEIGHT}, "
        f"{WIDTH})"
    )

    print()

    print(
        "Composite loss weights:"
    )

    for name, value in LOSS_WEIGHTS.items():

        print(
            f"    {name:<15}: "
            f"{value}"
        )


    # ==================================================================
    # CREATE SYNTHETIC DATA
    # ==================================================================

    print_header(
        "CREATING SYNTHETIC DATA"
    )

    seismic_input = (
        create_synthetic_input()
    )

    target = (
        create_synthetic_target()
    )

    velocity = (
        create_synthetic_velocity()
    )


    print_stats(
        "Input",
        seismic_input
    )

    print_stats(
        "Target",
        target
    )

    print_stats(
        "Velocity",
        velocity
    )


    assert_finite(
        "Input",
        seismic_input
    )

    assert_finite(
        "Target",
        target
    )

    assert_finite(
        "Velocity",
        velocity
    )


    velocity_positive = torch.all(
        velocity > 0.0
    ).item()


    print(
        "Velocity positivity check: "
        f"{'PASS' if velocity_positive else 'FAIL'}"
    )


    if not velocity_positive:

        raise RuntimeError(
            "Velocity contains non-positive values."
        )


    # ==================================================================
    # INITIALIZE NETWORK
    # ==================================================================

    print_header(
        "INITIALIZING NETWORK3D"
    )


    network = Network3D(

        in_channels=CHANNELS,

        out_channels=CHANNELS,

        use_uncertainty=True,

        use_residual=True,

        use_attention=True

    ).to(
        DEVICE
    )


    network.train()


    print(
        "Network3D initialized successfully."
    )


    # ==================================================================
    # INITIALIZE INDIVIDUAL LOSSES
    # ==================================================================

    print_header(
        "INITIALIZING LOSS COMPONENTS"
    )


    mae_loss_function = (
        MAELoss()
    ).to(
        DEVICE
    )


    physics_loss_function = (
        PhysicsLoss(

            dx=DX,

            dy=DY,

            dz=DZ,

            eikonal_weight=
            PHYSICS_LOSS_WEIGHTS[
                "eikonal"
            ],

            source_weight=
            PHYSICS_LOSS_WEIGHTS[
                "source"
            ],

            travel_time_weight=
            PHYSICS_LOSS_WEIGHTS[
                "travel_time"
            ]

        ).to(
            DEVICE
        )
    )


    uncertainty_loss_function = (
        UncertaintyLoss()
    ).to(
        DEVICE
    )


    ssim_loss_function = (
        SSIMLoss()
    ).to(
        DEVICE
    )


    print(
        "MAE loss initialized successfully."
    )

    print(
        "Physics loss initialized successfully."
    )

    print(
        "Heteroscedastic aleatoric "
        "uncertainty loss initialized successfully."
    )

    print(
        "SSIM loss initialized successfully."
    )

    # =====================================================
    # INITIALIZING TOTAL LOSS
    # =====================================================

    print_header(
        "INITIALIZING TOTAL LOSS"
    )

    total_loss_function = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    ).to(DEVICE)

    print(
        "TotalLoss initialized successfully."
    )

    # ==================================================================
    # NETWORK FORWARD PASS
    # ==================================================================

    print_header(
        "NETWORK FORWARD PASS"
    )


    network.zero_grad(
        set_to_none=True
    )


    (
        reconstruction,
        travel_time,
        log_variance
    ) = network(
        seismic_input
    )


    print(
        "Forward propagation completed."
    )


    expected_shape = (
        seismic_input.shape
    )


    shapes_valid = (

        reconstruction.shape
        ==
        expected_shape

        and

        travel_time.shape
        ==
        expected_shape

        and

        log_variance.shape
        ==
        expected_shape
    )


    print(
        "Network output shapes: "
        f"{'PASS' if shapes_valid else 'FAIL'}"
    )


    if not shapes_valid:

        raise RuntimeError(
            "Network output shape validation failed."
        )


    # ==================================================================
    # NETWORK OUTPUT AUDIT
    # ==================================================================

    print_header(
        "NETWORK OUTPUT AUDIT"
    )


    print_stats(
        "Reconstruction",
        reconstruction
    )

    print_stats(
        "Travel Time",
        travel_time
    )

    print_stats(
        "Log Variance",
        log_variance
    )


    assert_finite(
        "Reconstruction",
        reconstruction
    )

    assert_finite(
        "Travel Time",
        travel_time
    )

    assert_finite(
        "Log Variance",
        log_variance
    )


    # ==================================================================
    # INDIVIDUAL MAE LOSS
    # ==================================================================

    print_header(
        "MAE LOSS AUDIT"
    )


    mae_value = (
        mae_loss_function(
            reconstruction,
            target
        )
    )


    print(
        f"MAE loss : "
        f"{mae_value.item():.6e}"
    )


    assert_finite(
        "MAE loss",
        mae_value
    )


    # ==================================================================
    # INDIVIDUAL PHYSICS LOSS
    # ==================================================================

    print_header(
        "PHYSICS LOSS AUDIT"
    )


    physics_components = (
        physics_loss_function(

            travel_time=travel_time,

            velocity=velocity
        )
    )


    physics_value = (
        physics_components[
            "total"
        ]
    )


    for key, value in physics_components.items():

        print(
            f"{key:<25}: "
            f"{value.item():.6e}"
        )


    assert_finite(
        "Physics loss",
        physics_value
    )


    # ==================================================================
    # ALEATORIC UNCERTAINTY LOSS
    # ==================================================================

    print_header(
        "HETEROSCEDASTIC ALEATORIC "
        "UNCERTAINTY LOSS AUDIT"
    )


    uncertainty_value = (
        uncertainty_loss_function(

            prediction=reconstruction,

            target=target,

            log_variance=log_variance
        )
    )


    print(
        f"Uncertainty loss : "
        f"{uncertainty_value.item():.6e}"
    )


    assert_finite(
        "Uncertainty loss",
        uncertainty_value
    )


    # ==================================================================
    # SSIM LOSS
    # ==================================================================

    print_header(
        "SSIM LOSS AUDIT"
    )


    ssim_value = (
        ssim_loss_function(

            reconstruction,

            target
        )
    )


    print(
        f"SSIM loss : "
        f"{ssim_value.item():.6e}"
    )


    assert_finite(
        "SSIM loss",
        ssim_value
    )


    # ==================================================================
    # COMPLETE TOTAL LOSS
    # ==================================================================

    print_header(
        "TOTAL LOSS FORWARD INTEGRATION"
    )


    total_components = (
        total_loss_function(

            prediction=reconstruction,

            target=target,

            travel_time=travel_time,

            log_variance=log_variance,

            velocity_model=velocity
        )
    )


    print(
        "TotalLoss forward propagation completed."
    )


    # ------------------------------------------------------------------
    # Print all returned components.
    # ------------------------------------------------------------------

    print()

    print(
        "TOTAL LOSS COMPONENTS"
    )

    print(
        "-" * 70
    )


    for key, value in total_components.items():

        if torch.is_tensor(
            value
        ):

            print(
                f"{key:<25}: "
                f"{value.item():.6e}"
            )


    # ==================================================================
    # FINITE VALUE AUDIT
    # ==================================================================

    print_header(
        "TOTAL LOSS FINITE VALUE AUDIT"
    )


    for key, value in total_components.items():

        if torch.is_tensor(
            value
        ):

            assert_finite(
                key,
                value
            )


    # ==================================================================
    # COMPONENT CONSISTENCY AUDIT
    # ==================================================================

    print_header(
        "INDIVIDUAL COMPONENT CONSISTENCY"
    )


    component_checks = {


        "mae":

            mae_value,


        "physics":

            physics_value,


        "uncertainty":

            uncertainty_value,


        "ssim":

            ssim_value
    }


    component_consistency_pass = True


    for name, expected_value in (
        component_checks.items()
    ):

        if name not in total_components:

            print(
                f"{name}: NOT FOUND "
                "in TotalLoss output."
            )

            component_consistency_pass = False

            continue


        actual_value = (
            total_components[name]
        )


        difference = (
            actual_value
            -
            expected_value
        ).abs().item()


        print(
            f"{name:<15} difference: "
            f"{difference:.6e}"
        )


        if difference >= 1.0e-6:

            component_consistency_pass = False


    print()

    print(
        "Component consistency check: "
        f"{'PASS' if component_consistency_pass else 'FAIL'}"
    )


    if not component_consistency_pass:

        raise RuntimeError(
            "Individual loss components do not match "
            "the TotalLoss components."
        )


    # ==================================================================
    # WEIGHTED LOSS DECOMPOSITION
    # ==================================================================

    print_header(
        "WEIGHTED TOTAL LOSS DECOMPOSITION"
    )


    expected_weighted_mae = (

        LOSS_WEIGHTS["mae"]

        *

        total_components[
            "mae"
        ]
    )


    expected_weighted_physics = (

        LOSS_WEIGHTS["physics"]

        *

        total_components[
            "physics"
        ]
    )


    expected_weighted_uncertainty = (

        LOSS_WEIGHTS["uncertainty"]

        *

        total_components[
            "uncertainty"
        ]
    )


    expected_weighted_ssim = (

        LOSS_WEIGHTS["ssim"]

        *

        total_components[
            "ssim"
        ]
    )


    independently_calculated_total = (

        expected_weighted_mae

        +

        expected_weighted_physics

        +

        expected_weighted_uncertainty

        +

        expected_weighted_ssim
    )


    print(
        f"Weighted MAE         : "
        f"{expected_weighted_mae.item():.6e}"
    )

    print(
        f"Weighted Physics     : "
        f"{expected_weighted_physics.item():.6e}"
    )

    print(
        f"Weighted Uncertainty : "
        f"{expected_weighted_uncertainty.item():.6e}"
    )

    print(
        f"Weighted SSIM        : "
        f"{expected_weighted_ssim.item():.6e}"
    )

    print()

    print(
        f"Independent total    : "
        f"{independently_calculated_total.item():.6e}"
    )


    # ==================================================================
    # TOTAL LOSS CONSISTENCY
    # ==================================================================

    if "total" not in total_components:

        raise RuntimeError(
            "TotalLoss output dictionary does not "
            "contain 'total'."
        )


    total_value = (
        total_components[
            "total"
        ]
    )


    total_difference = (

        total_value

        -

        independently_calculated_total

    ).abs().item()


    print(
        f"TotalLoss total      : "
        f"{total_value.item():.6e}"
    )

    print(
        f"Total difference     : "
        f"{total_difference:.6e}"
    )


    total_decomposition_pass = (

        total_difference
        <
        1.0e-6
    )


    print(
        "Total decomposition check: "
        f"{'PASS' if total_decomposition_pass else 'FAIL'}"
    )


    if not total_decomposition_pass:

        raise RuntimeError(
            "TotalLoss does not match the independently "
            "calculated weighted sum."
        )


    # ==================================================================
    # BACKWARD PROPAGATION AUDIT
    # ==================================================================

    print_header(
        "TOTAL LOSS BACKWARD AUDIT"
    )


    network.zero_grad(
        set_to_none=True
    )


    (
        reconstruction,
        travel_time,
        log_variance
    ) = network(
        seismic_input
    )


    backward_components = (
        total_loss_function(

            prediction=reconstruction,

            target=target,

            travel_time=travel_time,

            log_variance=log_variance,

            velocity_model=velocity
        )
    )


    backward_loss = (
        backward_components[
            "total"
        ]
    )


    print(
        f"Total loss before backward: "
        f"{backward_loss.item():.6e}"
    )


    backward_loss.backward()


    (
        network_gradient_norm,
        maximum_parameter_gradient,
        parameters_with_gradients
    ) = calculate_gradient_statistics(
        network
    )


    print(
        f"Network gradient norm       : "
        f"{network_gradient_norm:.6e}"
    )

    print(
        f"Maximum parameter gradient  : "
        f"{maximum_parameter_gradient:.6e}"
    )

    print(
        f"Parameters with gradients   : "
        f"{parameters_with_gradients}"
    )


    backward_pass = (

        parameters_with_gradients
        >
        0

        and

        torch.isfinite(
            torch.tensor(
                network_gradient_norm
            )
        )

        and

        torch.isfinite(
            torch.tensor(
                maximum_parameter_gradient
            )
        )
    )


    print(
        "Backward propagation check: "
        f"{'PASS' if backward_pass else 'FAIL'}"
    )


    if not backward_pass:

        raise RuntimeError(
            "Total loss backward propagation failed."
        )


    # ==================================================================
    # FINAL RESULT
    # ==================================================================

    print_header(
        "TOTAL LOSS INTEGRATION AUDIT RESULT"
    )


    print(
        "Network forward pass          : PASS"
    )

    print(
        "Output shape validation       : "
        f"{'PASS' if shapes_valid else 'FAIL'}"
    )

    print(
        "MAE loss                      : PASS"
    )

    print(
        "Physics loss                  : PASS"
    )

    print(
        "Aleatoric uncertainty loss    : PASS"
    )

    print(
        "SSIM loss                     : PASS"
    )

    print(
        "Loss component consistency    : "
        f"{'PASS' if component_consistency_pass else 'FAIL'}"
    )

    print(
        "Total loss decomposition      : "
        f"{'PASS' if total_decomposition_pass else 'FAIL'}"
    )

    print(
        "Backward propagation          : "
        f"{'PASS' if backward_pass else 'FAIL'}"
    )

    print(
        "Numerical stability           : PASS"
    )


    print()

    print(
        "=" * 70
    )

    print(
        "COMPLETE TOTAL LOSS INTEGRATION "
        "AUDIT COMPLETED."
    )

    print(
        "=" * 70
    )


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()