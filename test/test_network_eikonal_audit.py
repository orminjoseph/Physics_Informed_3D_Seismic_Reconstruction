"""
=========================================================
NETWORK EIKONAL PHYSICS AUDIT
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Phase B3
--------
Network-level audit of the stabilized Eikonal formulation.

Network:
    Network3D

Physics:
    PhysicsLoss

Stabilized Eikonal equation:

    |grad T| = 1 / V

    V |grad T| = 1

Residual:

    R_eikonal = V |grad T| - 1

Loss:

    L_eikonal = mean(R_eikonal^2)

This audit verifies the complete computational pathway:

    Input seismic volume
            |
            v
        Network3D
            |
            +------------------+
            |                  |
            v                  v
    reconstructed_cube     travel_time
                               |
                               v
                         PhysicsLoss
                               |
                               v
                       V |grad T| - 1
                               |
                               v
                         Eikonal loss
                               |
                               v
                           backward

IMPORTANT
---------
This audit does NOT use the old:

    V^2 |grad T|^2 - 1

formulation.

The stabilized first-order formulation is the only
Eikonal residual used by this test.

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

DX = 25.0
DY = 25.0
DZ = 10.0

# Synthetic physical velocity range [m/s]
VELOCITY_MIN = 1500.0
VELOCITY_MAX = 4500.0


# =========================================================
# PRINT UTILITIES
# =========================================================

def print_header(title):
    """Print a formatted audit section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_stats(name, tensor):
    """Print basic tensor statistics."""

    tensor_detached = tensor.detach()

    print(f"{name}:")
    print(
        f"    shape : "
        f"{tuple(tensor_detached.shape)}"
    )
    print(
        f"    min   : "
        f"{tensor_detached.min().item():.6e}"
    )
    print(
        f"    max   : "
        f"{tensor_detached.max().item():.6e}"
    )
    print(
        f"    mean  : "
        f"{tensor_detached.mean().item():.6e}"
    )
    print(
        f"    std   : "
        f"{tensor_detached.std().item():.6e}"
    )
    print(
        f"    absmax: "
        f"{tensor_detached.abs().max().item():.6e}"
    )


def check_finite(name, tensor):
    """Confirm that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )

    print(
        f"{name}: finite values confirmed."
    )


# =========================================================
# CREATE SYNTHETIC INPUT
# =========================================================

def create_synthetic_input():
    """
    Create a normalized synthetic incomplete seismic cube.
    """

    input_cube = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE
    )

    # Keep the synthetic input in a realistic normalized
    # amplitude range.
    input_cube = torch.tanh(input_cube)

    return input_cube


# =========================================================
# CREATE SYNTHETIC VELOCITY MODEL
# =========================================================

def create_synthetic_velocity():
    """
    Create a positive heterogeneous velocity field.

    Velocity varies smoothly between approximately
    1500 m/s and 4500 m/s.
    """

    z = torch.linspace(
        0.0,
        1.0,
        DEPTH,
        device=DEVICE
    ).view(
        1, 1, DEPTH, 1, 1
    )

    y = torch.linspace(
        0.0,
        1.0,
        HEIGHT,
        device=DEVICE
    ).view(
        1, 1, 1, HEIGHT, 1
    )

    x = torch.linspace(
        0.0,
        1.0,
        WIDTH,
        device=DEVICE
    ).view(
        1, 1, 1, 1, WIDTH
    )

    # Smooth heterogeneous normalized velocity.
    velocity_normalized = (
        0.50
        + 0.25 * z
        + 0.15 * y
        + 0.10 * x
    )

    # Normalize to [0, 1].
    velocity_normalized = (
        velocity_normalized
        - velocity_normalized.min()
    ) / (
        velocity_normalized.max()
        - velocity_normalized.min()
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
        velocity_normalized
    )

    velocity = velocity.expand(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH
    ).clone()

    return velocity


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print_header(
        "NETWORK EIKONAL PHYSICS AUDIT"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Tensor shape    : "
        f"({BATCH_SIZE}, {CHANNELS}, "
        f"{DEPTH}, {HEIGHT}, {WIDTH})"
    )

    print(
        f"dx              : {DX}"
    )

    print(
        f"dy              : {DY}"
    )

    print(
        f"dz              : {DZ}"
    )

    # =====================================================
    # CREATE INPUT
    # =====================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    input_cube = create_synthetic_input()

    print_stats(
        "Input",
        input_cube
    )

    check_finite(
        "Input",
        input_cube
    )

    # =====================================================
    # CREATE VELOCITY
    # =====================================================

    print_header(
        "CREATING SYNTHETIC VELOCITY MODEL"
    )

    velocity = create_synthetic_velocity()

    print_stats(
        "Velocity",
        velocity
    )

    check_finite(
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
            "Synthetic velocity contains "
            "non-positive values."
        )

    # =====================================================
    # INITIALIZE NETWORK
    # =====================================================

    print_header(
        "INITIALIZING NETWORK3D"
    )

    network = Network3D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(DEVICE)

    network.train()

    print(
        "Network3D initialized successfully."
    )

    # =====================================================
    # INITIALIZE PHYSICS LOSS
    # =====================================================

    print_header(
        "INITIALIZING STABILIZED EIKONAL PHYSICS LOSS"
    )

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=1.0,
        source_weight=1.0,
        travel_time_weight=1.0
    ).to(DEVICE)

    print(
        "PhysicsLoss initialized successfully."
    )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    print_header(
        "NETWORK3D FORWARD PASS"
    )

    network.zero_grad(
        set_to_none=True
    )

    (
        reconstruction,
        travel_time,
        log_variance
    ) = network(
        input_cube
    )

    print(
        "Forward propagation completed."
    )

    # -----------------------------------------------------
    # Output shape validation
    # -----------------------------------------------------

    expected_shape = input_cube.shape

    shapes_valid = (
        reconstruction.shape == expected_shape
        and
        travel_time.shape == expected_shape
        and
        log_variance.shape == expected_shape
    )

    print(
        "Network output shapes: "
        f"{'PASS' if shapes_valid else 'FAIL'}"
    )

    if not shapes_valid:
        raise RuntimeError(
            "Network output shape validation failed."
        )

    # =====================================================
    # COMPLETE NETWORK OUTPUT AUDIT
    # =====================================================

    print_header(
        "COMPLETE NETWORK OUTPUT AUDIT"
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

    check_finite(
        "Reconstruction",
        reconstruction
    )

    check_finite(
        "Travel Time",
        travel_time
    )

    check_finite(
        "Log Variance",
        log_variance
    )

    # =====================================================
    # TRAVEL-TIME OUTPUT AUDIT
    # =====================================================

    print_header(
        "TRAVEL-TIME OUTPUT AUDIT"
    )

    print_stats(
        "Network travel time",
        travel_time
    )

    check_finite(
        "Network travel time",
        travel_time
    )

    travel_time_positive = torch.all(
        travel_time >= 0.0
    ).item()

    print(
        "Travel-time positivity check: "
        f"{'PASS' if travel_time_positive else 'FAIL'}"
    )

    if not travel_time_positive:
        raise RuntimeError(
            "Network travel time contains "
            "negative values."
        )

    travel_time_nonzero = (
        travel_time.abs().max().item()
        >
        0.0
    )

    print(
        "Non-zero travel-time output: "
        f"{'PASS' if travel_time_nonzero else 'FAIL'}"
    )

    if not travel_time_nonzero:
        raise RuntimeError(
            "Network travel-time output is identically zero."
        )

    # =====================================================
    # TRAVEL-TIME GRADIENT AUDIT
    # =====================================================

    print_header(
        "NETWORK TRAVEL-TIME DERIVATIVE AUDIT"
    )

    (
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared,
        gradient_magnitude
    ) = physics_loss.travel_time_gradient(
        travel_time
    )

    print_stats(
        "dT/dz",
        dT_dz
    )

    print_stats(
        "dT/dy",
        dT_dy
    )

    print_stats(
        "dT/dx",
        dT_dx
    )

    check_finite(
        "dT/dz",
        dT_dz
    )

    check_finite(
        "dT/dy",
        dT_dy
    )

    check_finite(
        "dT/dx",
        dT_dx
    )

    # =====================================================
    # GRADIENT MAGNITUDE AUDIT
    # =====================================================

    print_header(
        "NETWORK GRADIENT MAGNITUDE AUDIT"
    )

    print_stats(
        "|grad T|^2",
        gradient_squared
    )

    print_stats(
        "|grad T|",
        gradient_magnitude
    )

    check_finite(
        "|grad T|^2",
        gradient_squared
    )

    check_finite(
        "|grad T|",
        gradient_magnitude
    )

    gradient_nonnegative = torch.all(
        gradient_squared >= 0.0
    ).item()

    print(
        "Gradient-squared non-negativity: "
        f"{'PASS' if gradient_nonnegative else 'FAIL'}"
    )

    # =====================================================
    # STABILIZED EIKONAL RESIDUAL AUDIT
    # =====================================================

    print_header(
        "STABILIZED NETWORK EIKONAL AUDIT"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # The residual is obtained directly from the current
    # PhysicsLoss implementation.
    #
    # Therefore this audit is guaranteed to test:
    #
    #     R = V |grad T| - 1
    #
    # and NOT:
    #
    #     V^2 |grad T|^2 - 1
    # -----------------------------------------------------

    residual = physics_loss.eikonal_residual(
        travel_time=travel_time,
        velocity=velocity
    )

    print_stats(
        "Eikonal residual R = V|grad T| - 1",
        residual
    )

    check_finite(
        "Eikonal residual",
        residual
    )

    # -----------------------------------------------------
    # Independently verify the residual equation.
    # -----------------------------------------------------

    expected_residual = (
        velocity
        *
        gradient_magnitude
        -
        1.0
    )

    residual_error = (
        residual
        -
        expected_residual
    ).abs().max().item()

    print(
        "Residual equation error : "
        f"{residual_error:.6e}"
    )

    residual_equation_pass = (
        residual_error < 1.0e-10
    )

    print(
        "Residual equation check: "
        f"{'PASS' if residual_equation_pass else 'FAIL'}"
    )

    if not residual_equation_pass:
        raise RuntimeError(
            "The Eikonal residual does not match "
            "V|grad T| - 1."
        )

    # =====================================================
    # STABILIZED EIKONAL LOSS AUDIT
    # =====================================================

    print_header(
        "STABILIZED EIKONAL LOSS"
    )

    eikonal_loss = physics_loss.eikonal_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    print(
        f"Eikonal loss : "
        f"{eikonal_loss.item():.6e}"
    )

    check_finite(
        "Eikonal loss",
        eikonal_loss
    )

    # -----------------------------------------------------
    # Verify loss directly from residual.
    # -----------------------------------------------------

    expected_eikonal_loss = (
        residual.pow(2).mean()
    )

    loss_error = abs(
        eikonal_loss.item()
        -
        expected_eikonal_loss.item()
    )

    print(
        "Loss equation error : "
        f"{loss_error:.6e}"
    )

    loss_equation_pass = (
        loss_error < 1.0e-10
    )

    print(
        "Loss equation check: "
        f"{'PASS' if loss_equation_pass else 'FAIL'}"
    )

    if not loss_equation_pass:
        raise RuntimeError(
            "Eikonal loss does not equal "
            "mean((V|grad T|-1)^2)."
        )

    # =====================================================
    # COMPLETE PHYSICS LOSS AUDIT
    # =====================================================

    print_header(
        "COMPLETE PHYSICS LOSS AUDIT"
    )

    physics_components = physics_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    for key, value in physics_components.items():

        print(
            f"{key:<25}: "
            f"{value.item():.6e}"
        )

    decomposition_error = abs(
        physics_components["total"].item()
        -
        (
            physics_components[
                "weighted_eikonal"
            ].item()
            +
            physics_components[
                "weighted_source"
            ].item()
            +
            physics_components[
                "weighted_travel_time"
            ].item()
        )
    )

    print(
        "Total decomposition error : "
        f"{decomposition_error:.6e}"
    )

    decomposition_pass = (
        decomposition_error < 1.0e-10
    )

    print(
        "Physics-loss decomposition check: "
        f"{'PASS' if decomposition_pass else 'FAIL'}"
    )

    # =====================================================
    # NETWORK BACKWARD PROPAGATION AUDIT
    # =====================================================

    print_header(
        "NETWORK EIKONAL BACKWARD AUDIT"
    )

    network.zero_grad(
        set_to_none=True
    )

    backward_loss = physics_loss.eikonal_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    print(
        "Eikonal loss before backward: "
        f"{backward_loss.item():.6e}"
    )

    backward_loss.backward()

    # -----------------------------------------------------
    # Collect network parameter gradients.
    # -----------------------------------------------------

    gradient_norm_squared = 0.0
    maximum_parameter_gradient = 0.0
    parameters_with_gradients = 0

    for parameter in network.parameters():

        if parameter.grad is None:
            continue

        parameters_with_gradients += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            raise RuntimeError(
                "Network parameter gradient "
                "contains NaN or Inf."
            )

        grad_norm = (
            parameter.grad.detach()
            .norm()
            .item()
        )

        gradient_norm_squared += (
            grad_norm ** 2
        )

        parameter_max = (
            parameter.grad.detach()
            .abs()
            .max()
            .item()
        )

        maximum_parameter_gradient = max(
            maximum_parameter_gradient,
            parameter_max
        )

    network_gradient_norm = (
        gradient_norm_squared ** 0.5
    )

    print(
        "Network gradient norm       : "
        f"{network_gradient_norm:.6e}"
    )

    print(
        "Maximum parameter gradient  : "
        f"{maximum_parameter_gradient:.6e}"
    )

    print(
        "Parameters with gradients   : "
        f"{parameters_with_gradients}"
    )

    backward_pass = (
        parameters_with_gradients > 0
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
        "Network Eikonal backward check: "
        f"{'PASS' if backward_pass else 'FAIL'}"
    )

    # =====================================================
    # TRAVEL-TIME SCALE SENSITIVITY
    # =====================================================

    print_header(
        "TRAVEL-TIME SCALE SENSITIVITY AUDIT"
    )

    print(
        f"{'Scale':>14}"
        f"{'Eikonal Loss':>22}"
        f"{'Max Gradient':>22}"
    )

    print(
        "-" * 60
    )

    scales = [
        1.0,
        0.1,
        0.01,
        0.001
    ]

    scale_results = []

    # Use evaluation mode so that this diagnostic is not
    # affected by training-mode normalization statistics.
    network.eval()

    for scale in scales:

        network.zero_grad(
            set_to_none=True
        )

        (
            _,
            scaled_travel_time,
            _
        ) = network(
            input_cube
        )

        scaled_travel_time = (
            scaled_travel_time
            *
            scale
        )

        scaled_loss = (
            physics_loss.eikonal_loss(
                travel_time=scaled_travel_time,
                velocity=velocity
            )
        )

        scaled_loss.backward()

        max_gradient = 0.0

        for parameter in network.parameters():

            if parameter.grad is None:
                continue

            parameter_gradient = (
                parameter.grad.detach()
                .abs()
                .max()
                .item()
            )

            max_gradient = max(
                max_gradient,
                parameter_gradient
            )

        print(
            f"{scale:14.4e}"
            f"{scaled_loss.item():22.6e}"
            f"{max_gradient:22.6e}"
        )

        scale_results.append(
            (
                scale,
                scaled_loss.item(),
                max_gradient
            )
        )

    print()
    print(
        "Scale sensitivity audit completed."
    )

    # =====================================================
    # DIAGNOSTIC INTERPRETATION
    # =====================================================

    print_header(
        "DIAGNOSTIC INTERPRETATION"
    )

    mean_velocity = (
        velocity.mean().item()
    )

    mean_travel_time = (
        travel_time.mean().item()
    )

    rms_gradient = torch.sqrt(
        gradient_squared.mean()
    ).item()

    mean_normalized_eikonal = (
        (
            velocity
            *
            gradient_magnitude
        )
        .mean()
        .item()
    )

    rms_residual = torch.sqrt(
        residual.pow(2).mean()
    ).item()

    print(
        f"Mean velocity magnitude : "
        f"{mean_velocity:.6e} m/s"
    )

    print(
        f"Mean travel time        : "
        f"{mean_travel_time:.6e} s"
    )

    print(
        f"Maximum |T|             : "
        f"{travel_time.abs().max().item():.6e} s"
    )

    print(
        f"RMS |grad T|            : "
        f"{rms_gradient:.6e} s/m"
    )

    print(
        f"Mean V|grad T|          : "
        f"{mean_normalized_eikonal:.6e}"
    )

    print(
        f"RMS Eikonal residual    : "
        f"{rms_residual:.6e}"
    )

    print(
        f"Eikonal loss            : "
        f"{eikonal_loss.item():.6e}"
    )

    print(
        f"Network gradient norm   : "
        f"{network_gradient_norm:.6e}"
    )

    print(
        f"Maximum network gradient: "
        f"{maximum_parameter_gradient:.6e}"
    )

    # =====================================================
    # DIAGNOSTIC CLASSIFICATION
    # =====================================================

    print()
    print(
        "DIAGNOSTIC CLASSIFICATION"
    )

    print(
        "-" * 70
    )

    if mean_normalized_eikonal < 1.0e-6:

        print(
            "WARNING: V|grad T| is extremely small."
        )

    elif mean_normalized_eikonal < 1.0:

        print(
            "V|grad T| is below the Eikonal target "
            "of 1.0."
        )

    else:

        print(
            "V|grad T| is at or above the Eikonal "
            "target scale."
        )

    if network_gradient_norm == 0.0:

        print(
            "WARNING: Network Eikonal gradient is zero."
        )

    elif torch.isfinite(
        torch.tensor(network_gradient_norm)
    ):

        print(
            "Network Eikonal parameter gradients "
            "are finite."
        )

    else:

        print(
            "WARNING: Network Eikonal gradient "
            "is non-finite."
        )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print_header(
        "NETWORK EIKONAL AUDIT RESULT"
    )

    print(
        "Network3D forward pass       : "
        "PASS"
    )

    print(
        "Output shape validation      : "
        "PASS"
    )

    print(
        "Travel-time output           : "
        "PASS"
    )

    print(
        "Spatial derivatives          : "
        "PASS"
    )

    print(
        "Gradient magnitude           : "
        "PASS"
    )

    print(
        "V|grad T| - 1 residual       : "
        f"{'PASS' if residual_equation_pass else 'FAIL'}"
    )

    print(
        "Stabilized Eikonal loss      : "
        f"{'PASS' if loss_equation_pass else 'FAIL'}"
    )

    print(
        "Physics-loss decomposition   : "
        f"{'PASS' if decomposition_pass else 'FAIL'}"
    )

    print(
        "Backward propagation         : "
        f"{'PASS' if backward_pass else 'FAIL'}"
    )

    print(
        "Numerical stability          : "
        "PASS"
    )

    # =====================================================
    # IMPORTANT CONCLUSION
    # =====================================================

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This B3 audit validates the complete "
        "Network3D-to-Eikonal computational pathway."
    )

    print()
    print(
        "The active Eikonal formulation is:"
    )

    print(
        "    R_eikonal = V |grad T| - 1"
    )

    print()
    print(
        "The active Eikonal loss is:"
    )

    print(
        "    L_eikonal = mean((V|grad T| - 1)^2)"
    )

    print()
    print(
        "The old V^2|grad T|^2 formulation is "
        "NOT used by this audit."
    )

    print()
    print(
        "The physical velocity field is supplied "
        "externally and is not predicted by Network3D."
    )

    print()
    print(
        "NETWORK EIKONAL PHYSICS AUDIT COMPLETED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()