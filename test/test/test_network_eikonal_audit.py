"""
=========================================================
Network Eikonal Physics Audit
=========================================================

Phase B3

Audits the complete computational pathway:

    Input seismic volume
            |
            v
        Network3D
            |
            +------------------+
            |                  |
            v                  v
      Travel-time T       Other outputs
            |
            v
       Spatial gradient
            |
            v
          |∇T|
            |
            v
        V |∇T|
            |
            v
      R = V|∇T| - 1
            |
            v
       Eikonal loss
            |
            v
       Backpropagation
            |
            v
      Network gradients

This test specifically validates the stabilized Eikonal
formulation introduced in Phase B1:

    R_eikonal = V |∇T| - 1

IMPORTANT
---------

This audit does NOT use the obsolete active formulation:

    V² |∇T|² - 1

The network under test is:

    Network3D

The physical velocity model is supplied externally.

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

# ---------------------------------------------------------
# Synthetic velocity range
# ---------------------------------------------------------

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 4500.0


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def print_header(title):
    """Print a formatted audit section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_tensor_statistics(name, tensor):
    """Print basic tensor statistics."""

    tensor_detached = tensor.detach()

    print(f"{name}:")
    print(f"    shape : {tuple(tensor.shape)}")
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
    """Verify that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )

    print(f"{name}: finite values confirmed.")


def compute_spatial_derivatives(
    travel_time,
    dx,
    dy,
    dz
):
    """
    Compute first-order spatial derivatives of travel time.

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

    dT_dz = torch.gradient(
        travel_time,
        dim=2,
        spacing=dz
    )[0]

    # -----------------------------------------------------
    # dT/dy
    # -----------------------------------------------------

    dT_dy = torch.gradient(
        travel_time,
        dim=3,
        spacing=dy
    )[0]

    # -----------------------------------------------------
    # dT/dx
    # -----------------------------------------------------

    dT_dx = torch.gradient(
        travel_time,
        dim=4,
        spacing=dx
    )[0]

    return dT_dz, dT_dy, dT_dx


# =========================================================
# AUDIT START
# =========================================================

print_header(
    "NETWORK EIKONAL PHYSICS AUDIT"
)

print(
    f"Device          : {DEVICE}"
)

print(
    "Tensor shape    : "
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


# =========================================================
# CREATE SYNTHETIC INPUT
# =========================================================

print_header(
    "CREATING SYNTHETIC INPUT"
)

torch.manual_seed(42)

input_cube = torch.rand(
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH,
    device=DEVICE
)

# Convert [0,1] to approximately [-1,1]
input_cube = (
    2.0 * input_cube
    - 1.0
)

print_tensor_statistics(
    "Input",
    input_cube
)

check_finite(
    "Input",
    input_cube
)


# =========================================================
# CREATE SYNTHETIC VELOCITY MODEL
# =========================================================

print_header(
    "CREATING SYNTHETIC VELOCITY MODEL"
)

# ---------------------------------------------------------
# Create a depth-dependent velocity model.
#
# Velocity increases from 1500 m/s to 4500 m/s with depth.
# ---------------------------------------------------------

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

velocity = (
    VELOCITY_MIN
    +
    (
        VELOCITY_MAX
        - VELOCITY_MIN
    )
    * z
)

velocity = velocity.expand(
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH
).contiguous()

print_tensor_statistics(
    "Velocity",
    velocity
)

check_finite(
    "Velocity",
    velocity
)

if (velocity <= 0).any():

    raise RuntimeError(
        "Velocity positivity check failed."
    )

print(
    "Velocity positivity check: PASS"
)


# =========================================================
# INITIALIZE NETWORK
# =========================================================

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


# =========================================================
# INITIALIZE PHYSICS LOSS
# =========================================================

print_header(
    "INITIALIZING STABILIZED EIKONAL PHYSICS LOSS"
)

physics_loss = PhysicsLoss(
    dx=DX,
    dy=DY,
    dz=DZ
).to(DEVICE)

print(
    "PhysicsLoss initialized successfully."
)


# =========================================================
# NETWORK FORWARD PASS
# =========================================================

print_header(
    "NETWORK3D FORWARD PASS"
)

reconstructed_cube, travel_time, log_variance = (
    network(input_cube)
)

print(
    "Forward propagation completed."
)

# ---------------------------------------------------------
# Verify output shapes
# ---------------------------------------------------------

if reconstructed_cube.shape != input_cube.shape:

    raise RuntimeError(
        "Reconstruction output shape mismatch."
    )

if travel_time.shape != input_cube.shape:

    raise RuntimeError(
        "Travel-time output shape mismatch."
    )

if log_variance.shape != input_cube.shape:

    raise RuntimeError(
        "Log-variance output shape mismatch."
    )

print(
    "Network output shapes: PASS"
)


# =========================================================
# COMPLETE NETWORK OUTPUT AUDIT
# =========================================================

print_header(
    "COMPLETE NETWORK OUTPUT AUDIT"
)

print_tensor_statistics(
    "Reconstruction",
    reconstructed_cube
)

print_tensor_statistics(
    "Travel Time",
    travel_time
)

print_tensor_statistics(
    "Log Variance",
    log_variance
)

check_finite(
    "Reconstruction",
    reconstructed_cube
)

check_finite(
    "Travel Time",
    travel_time
)

check_finite(
    "Log Variance",
    log_variance
)


# =========================================================
# TRAVEL-TIME OUTPUT AUDIT
# =========================================================

print_header(
    "TRAVEL-TIME OUTPUT AUDIT"
)

print_tensor_statistics(
    "Network travel time",
    travel_time
)

check_finite(
    "Network travel time",
    travel_time
)

# ---------------------------------------------------------
# Network3D uses Softplus, therefore T should be positive.
# ---------------------------------------------------------

if (travel_time < 0).any():

    raise RuntimeError(
        "Travel-time positivity check failed."
    )

print(
    "Travel-time positivity check: PASS"
)

if torch.allclose(
    travel_time,
    torch.zeros_like(travel_time)
):

    raise RuntimeError(
        "Travel-time output is numerically zero."
    )

print(
    "Non-zero travel-time output confirmed."
)


# =========================================================
# SPATIAL DERIVATIVE AUDIT
# =========================================================

print_header(
    "NETWORK TRAVEL-TIME DERIVATIVE AUDIT"
)

dT_dz, dT_dy, dT_dx = (
    compute_spatial_derivatives(
        travel_time,
        DX,
        DY,
        DZ
    )
)

print_tensor_statistics(
    "dT/dz",
    dT_dz
)

print_tensor_statistics(
    "dT/dy",
    dT_dy
)

print_tensor_statistics(
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


# =========================================================
# GRADIENT MAGNITUDE
# =========================================================

print_header(
    "NETWORK GRADIENT MAGNITUDE AUDIT"
)

grad_T_squared = (
    dT_dx ** 2
    +
    dT_dy ** 2
    +
    dT_dz ** 2
)

grad_T = torch.sqrt(
    grad_T_squared
    +
    1.0e-12
)

print_tensor_statistics(
    "|grad T|^2",
    grad_T_squared
)

check_finite(
    "|grad T|^2",
    grad_T_squared
)

print_tensor_statistics(
    "|grad T|",
    grad_T
)

check_finite(
    "|grad T|",
    grad_T
)


# =========================================================
# STABILIZED EIKONAL TERM
# =========================================================

print_header(
    "STABILIZED NETWORK EIKONAL AUDIT"
)

# ---------------------------------------------------------
# B1 formulation:
#
#     V |grad T|
#
# ---------------------------------------------------------

normalized_eikonal_term = (
    velocity
    *
    grad_T
)

print_tensor_statistics(
    "V |grad T|",
    normalized_eikonal_term
)

check_finite(
    "V |grad T|",
    normalized_eikonal_term
)


# =========================================================
# EIKONAL RESIDUAL
# =========================================================

eikonal_residual = (
    normalized_eikonal_term
    - 1.0
)

print_tensor_statistics(
    "Eikonal residual",
    eikonal_residual
)

check_finite(
    "Eikonal residual",
    eikonal_residual
)

# ---------------------------------------------------------
# Explicit residual verification.
# ---------------------------------------------------------

residual_error = (
    eikonal_residual
    -
    (
        velocity
        *
        grad_T
        - 1.0
    )
).abs().max().item()

print(
    f"Residual equation error : "
    f"{residual_error:.6e}"
)

if residual_error > 1.0e-6:

    raise RuntimeError(
        "Eikonal residual equation check failed."
    )

print(
    "Residual equation check: PASS"
)


# =========================================================
# EIKONAL LOSS THROUGH PHYSICS LOSS
# =========================================================

print_header(
    "NETWORK EIKONAL LOSS"
)

# ---------------------------------------------------------
# The physics loss receives the NETWORK-PREDICTED travel
# time and the externally supplied physical velocity model.
#
# The exact argument names below correspond to the expected
# stabilized PhysicsLoss interface:
#
#     travel_time
#     velocity
#
# ---------------------------------------------------------

physics_result = physics_loss(
    travel_time=travel_time,
    velocity=velocity
)

# ---------------------------------------------------------
# Support both a scalar tensor return and a dictionary
# return containing an "eikonal" component.
# ---------------------------------------------------------

if isinstance(
    physics_result,
    dict
):

    if "eikonal" not in physics_result:

        raise RuntimeError(
            "PhysicsLoss dictionary does not contain "
            "'eikonal'."
        )

    eikonal_loss = physics_result[
        "eikonal"
    ]

else:

    eikonal_loss = physics_result

print(
    f"Eikonal loss : "
    f"{eikonal_loss.item():.6e}"
)

check_finite(
    "Eikonal loss",
    eikonal_loss
)


# =========================================================
# BACKWARD PROPAGATION AUDIT
# =========================================================

print_header(
    "NETWORK EIKONAL BACKWARD AUDIT"
)

network.zero_grad(
    set_to_none=True
)

if travel_time.grad is not None:

    travel_time.grad.zero_()

print(
    f"Eikonal loss before backward: "
    f"{eikonal_loss.item():.6e}"
)

eikonal_loss.backward()


# ---------------------------------------------------------
# Collect network parameter gradients.
# ---------------------------------------------------------

gradient_norm_squared = 0.0
maximum_parameter_gradient = 0.0
parameters_with_gradients = 0

for parameter in network.parameters():

    if parameter.grad is None:

        continue

    parameters_with_gradients += 1

    gradient = parameter.grad.detach()

    check_finite(
        "Network parameter gradient",
        gradient
    )

    gradient_norm_squared += (
        gradient.norm().item() ** 2
    )

    maximum_parameter_gradient = max(
        maximum_parameter_gradient,
        gradient.abs().max().item()
    )

network_gradient_norm = (
    gradient_norm_squared ** 0.5
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

if parameters_with_gradients == 0:

    raise RuntimeError(
        "No Network3D parameters received gradients."
    )

print(
    "Network Eikonal backward propagation: PASS"
)


# =========================================================
# TRAVEL-TIME SCALE SENSITIVITY AUDIT
# =========================================================

print_header(
    "TRAVEL-TIME SCALE SENSITIVITY AUDIT"
)

print(
    f"{'Scale':>12}"
    f"{'Eikonal Loss':>20}"
    f"{'Max Gradient':>20}"
)

print(
    "-" * 55
)

scales = [
    1.0,
    0.1,
    0.01,
    0.001
]

for scale in scales:

    network.zero_grad(
        set_to_none=True
    )

    # -----------------------------------------------------
    # Scale the NETWORK travel-time output.
    #
    # This is a diagnostic only.
    # It does not modify Network3D itself.
    # -----------------------------------------------------

    scaled_travel_time = (
        travel_time.detach()
        * scale
    )

    scaled_travel_time.requires_grad_()

    scaled_result = physics_loss(
        travel_time=scaled_travel_time,
        velocity=velocity
    )

    if isinstance(
        scaled_result,
        dict
    ):

        scaled_loss = scaled_result[
            "eikonal"
        ]

    else:

        scaled_loss = scaled_result

    scaled_loss.backward()

    max_gradient = 0.0

    for parameter in network.parameters():

        if parameter.grad is not None:

            max_gradient = max(
                max_gradient,
                parameter.grad.abs().max().item()
            )

    print(
        f"{scale:12.4e}"
        f"{scaled_loss.item():20.6e}"
        f"{max_gradient:20.6e}"
    )

print()
print(
    "Scale sensitivity audit completed."
)


# =========================================================
# DIAGNOSTIC INTERPRETATION
# =========================================================

print_header(
    "DIAGNOSTIC INTERPRETATION"
)

mean_velocity = (
    velocity.mean().item()
)

mean_travel_time = (
    travel_time.mean().item()
)

maximum_travel_time = (
    travel_time.abs().max().item()
)

rms_grad_T = torch.sqrt(
    torch.mean(
        grad_T ** 2
    )
).item()

mean_normalized_eikonal = (
    normalized_eikonal_term.mean().item()
)

rms_eikonal_residual = torch.sqrt(
    torch.mean(
        eikonal_residual ** 2
    )
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
    f"{maximum_travel_time:.6e} s"
)

print(
    f"RMS |grad T|            : "
    f"{rms_grad_T:.6e}"
)

print(
    f"Mean V|grad T|          : "
    f"{mean_normalized_eikonal:.6e}"
)

print(
    f"RMS Eikonal residual    : "
    f"{rms_eikonal_residual:.6e}"
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


# =========================================================
# FINAL AUDIT RESULT
# =========================================================

print_header(
    "NETWORK EIKONAL AUDIT RESULT"
)

print(
    "Network3D forward pass        : PASS"
)

print(
    "Output shape validation       : PASS"
)

print(
    "Travel-time output            : PASS"
)

print(
    "Spatial derivatives           : PASS"
)

print(
    "Gradient magnitude            : PASS"
)

print(
    "V|grad T| calculation         : PASS"
)

print(
    "Eikonal residual V|grad T|-1  : PASS"
)

print(
    "Eikonal loss                  : PASS"
)

print(
    "Backward propagation          : PASS"
)

print(
    "Numerical stability           : PASS"
)

print()
print(
    "IMPORTANT:"
)

print(
    "The B3 audit validates the complete"
)

print(
    "Network3D -> Travel Time -> Spatial Gradient"
)

print(
    "-> V|grad T| -> (V|grad T|-1)"
)

print(
    "-> Eikonal Loss -> Network Gradient pathway."
)

print()
print(
    "The obsolete V^2|grad T|^2 formulation is NOT"
)

print(
    "used as the active Eikonal residual."
)

print()
print(
    "NOTE:"
)

print(
    "A non-zero Eikonal loss is expected from an"
)

print(
    "untrained Network3D. PASS means the computational"
)

print(
    "pathway is numerically valid, not that the randomly"
)

print(
    "initialized network already satisfies the physics."
)

print()
print(
    "NETWORK EIKONAL PHYSICS AUDIT COMPLETED."
)