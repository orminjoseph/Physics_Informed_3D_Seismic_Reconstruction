"""
=========================================================
PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION
PHYSICS-LOSS SPATIAL RESIDUAL AUDIT
=========================================================

Purpose
-------

This test performs a spatial audit of the Eikonal physics
residual produced by the PhysicsInformed3DUNet.

The audit verifies:

1. Travel-time field validity.
2. Physical velocity validity.
3. Spatial derivative validity.
4. Eikonal residual statistics.
5. Residual percentiles.
6. Boundary versus interior residual behaviour.
7. Fraction of voxels satisfying residual tolerances.
8. Eikonal loss consistency.
9. Gradient propagation through the physics loss.
10. Numerical stability.

Governing equation
------------------

The Eikonal equation is:

    |∇T|² = 1 / V²

where:

    T = travel time [s]
    V = P-wave velocity [m/s]

Residual:

    R = V² |∇T|² - 1

A physically consistent travel-time field should produce:

    R ≈ 0

Tensor convention
-----------------

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import sys

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
)


# =========================================================
# PRINT UTILITIES
# =========================================================

def print_header(title):
    """Print a formatted audit section header."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_stat(name, tensor):
    """Print basic tensor statistics."""

    tensor = tensor.detach()

    print(
        f"{name:<30}: "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


# =========================================================
# SPATIAL DERIVATIVE FUNCTION
# =========================================================

def compute_spatial_derivatives(
    travel_time,
    dx,
    dy,
    dz
):
    """
    Compute first-order spatial derivatives of the
    travel-time field.

    Tensor convention:

        [B, C, D, H, W]

    Therefore:

        D -> z direction
        H -> y direction
        W -> x direction
    """

    if travel_time.ndim != 5:

        raise ValueError(
            "travel_time must have shape "
            "[B, C, D, H, W]."
        )

    # -----------------------------------------------------
    # Central differences in the interior.
    # Forward/backward differences are used at boundaries.
    # -----------------------------------------------------

    dT_dz = torch.zeros_like(travel_time)
    dT_dy = torch.zeros_like(travel_time)
    dT_dx = torch.zeros_like(travel_time)

    # -----------------------------------------------------
    # Z derivative
    # -----------------------------------------------------

    dT_dz[:, :, 1:-1, :, :] = (
        travel_time[:, :, 2:, :, :]
        -
        travel_time[:, :, :-2, :, :]
    ) / (2.0 * dz)

    dT_dz[:, :, 0, :, :] = (
        travel_time[:, :, 1, :, :]
        -
        travel_time[:, :, 0, :, :]
    ) / dz

    dT_dz[:, :, -1, :, :] = (
        travel_time[:, :, -1, :, :]
        -
        travel_time[:, :, -2, :, :]
    ) / dz

    # -----------------------------------------------------
    # Y derivative
    # -----------------------------------------------------

    dT_dy[:, :, :, 1:-1, :] = (
        travel_time[:, :, :, 2:, :]
        -
        travel_time[:, :, :, :-2, :]
    ) / (2.0 * dy)

    dT_dy[:, :, :, 0, :] = (
        travel_time[:, :, :, 1, :]
        -
        travel_time[:, :, :, 0, :]
    ) / dy

    dT_dy[:, :, :, -1, :] = (
        travel_time[:, :, :, -1, :]
        -
        travel_time[:, :, :, -2, :]
    ) / dy

    # -----------------------------------------------------
    # X derivative
    # -----------------------------------------------------

    dT_dx[:, :, :, :, 1:-1] = (
        travel_time[:, :, :, :, 2:]
        -
        travel_time[:, :, :, :, :-2]
    ) / (2.0 * dx)

    dT_dx[:, :, :, :, 0] = (
        travel_time[:, :, :, :, 1]
        -
        travel_time[:, :, :, :, 0]
    ) / dx

    dT_dx[:, :, :, :, -1] = (
        travel_time[:, :, :, :, -1]
        -
        travel_time[:, :, :, :, -2]
    ) / dx

    return dT_dz, dT_dy, dT_dx


# =========================================================
# EIKONAL RESIDUAL
# =========================================================

def compute_eikonal_residual(
    travel_time,
    velocity,
    dx,
    dy,
    dz
):
    """
    Compute the normalized Eikonal residual:

        R = V² |∇T|² - 1
    """

    dT_dz, dT_dy, dT_dx = (
        compute_spatial_derivatives(
            travel_time,
            dx,
            dy,
            dz
        )
    )

    gradient_squared = (
        dT_dz.pow(2)
        +
        dT_dy.pow(2)
        +
        dT_dx.pow(2)
    )

    residual = (
        velocity.pow(2)
        *
        gradient_squared
        -
        1.0
    )

    return (
        residual,
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared
    )


# =========================================================
# PERCENTILE UTILITY
# =========================================================

def percentile(tensor, q):
    """
    Compute percentile using torch.quantile.
    """

    tensor = tensor.detach().flatten()

    return torch.quantile(
        tensor,
        q
    ).item()


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print_header(
        "PHYSICS LOSS SPATIAL RESIDUAL AUDIT"
    )

    # =====================================================
    # DEVICE CONFIGURATION
    # =====================================================

    print_header(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(DEVICE)

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # PHYSICAL CONFIGURATION
    # =====================================================

    print_header(
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
        f"{LOSS_WEIGHTS['physics']}"
    )

    print(
        f"Eikonal weight             : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    # =====================================================
    # CREATE SYNTHETIC INPUT
    # =====================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    torch.manual_seed(42)

    input_cube = torch.randn(
        1,
        1,
        64,
        128,
        128,
        device=device
    )

    print(
        f"Input shape                : "
        f"{tuple(input_cube.shape)}"
    )

    print_stat(
        "Input seismic volume",
        input_cube
    )

    # =====================================================
    # CREATE PHYSICAL VELOCITY
    # =====================================================

    print_header(
        "CREATING PHYSICAL VELOCITY FIELD"
    )

    velocity = torch.full(
        input_cube.shape,
        2000.0,
        device=device
    )

    print_stat(
        "Velocity field",
        velocity
    )

    # =====================================================
    # INITIALIZE NETWORK
    # =====================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(device)

    model.eval()

    print(
        "Network successfully initialized."
    )

    # =====================================================
    # PARAMETER COUNT
    # =====================================================

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters       : "
        f"{parameter_count:,}"
    )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    print_header(
        "FORWARD PROPAGATION"
    )

    with torch.no_grad():

        reconstruction, travel_time, log_variance = (
            model(input_cube)
        )

    print(
        "Network returned three outputs."
    )

    # =====================================================
    # OUTPUT VALIDATION
    # =====================================================

    print_header(
        "NETWORK OUTPUT VALIDATION"
    )

    if not torch.isfinite(
        travel_time
    ).all():

        raise RuntimeError(
            "Travel-time contains NaN or Inf values."
        )

    if not torch.isfinite(
        velocity
    ).all():

        raise RuntimeError(
            "Velocity contains NaN or Inf values."
        )

    if not torch.isfinite(
        reconstruction
    ).all():

        raise RuntimeError(
            "Reconstruction contains NaN or Inf values."
        )

    print(
        "Travel-time field is finite."
    )

    print(
        "Velocity field is finite."
    )

    print(
        "Reconstruction field is finite."
    )

    print_stat(
        "Travel-time field",
        travel_time
    )

    # =====================================================
    # TRAVEL-TIME POSITIVITY
    # =====================================================

    print_header(
        "TRAVEL-TIME POSITIVITY AUDIT"
    )

    minimum_travel_time = (
        travel_time.min().item()
    )

    print(
        f"Minimum travel time        : "
        f"{minimum_travel_time:.6e}"
    )

    if minimum_travel_time < 0.0:

        raise RuntimeError(
            "Travel-time contains negative values."
        )

    print(
        "Travel-time positivity: PASSED"
    )

    # =====================================================
    # EIKONAL RESIDUAL
    # =====================================================

    print_header(
        "COMPUTING EIKONAL RESIDUAL"
    )

    (
        residual,
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared
    ) = compute_eikonal_residual(
        travel_time,
        velocity,
        DX,
        DY,
        DZ
    )

    print(
        "Eikonal residual successfully computed."
    )

    # =====================================================
    # DERIVATIVE STATISTICS
    # =====================================================

    print_header(
        "SPATIAL DERIVATIVE STATISTICS"
    )

    print_stat(
        "dT/dz",
        dT_dz
    )

    print_stat(
        "dT/dy",
        dT_dy
    )

    print_stat(
        "dT/dx",
        dT_dx
    )

    # =====================================================
    # GRADIENT MAGNITUDE
    # =====================================================

    print_header(
        "GRADIENT MAGNITUDE"
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared.clamp_min(0.0)
    )

    print_stat(
        "|grad T|",
        gradient_magnitude
    )

    # =====================================================
    # EXPECTED GRADIENT
    # =====================================================

    expected_gradient = (
        1.0 / 2000.0
    )

    print(
        f"Expected gradient magnitude : "
        f"{expected_gradient:.6e} s/m"
    )

    observed_gradient = (
        gradient_magnitude.mean().item()
    )

    gradient_ratio = (
        observed_gradient
        /
        expected_gradient
    )

    print(
        f"Observed gradient magnitude : "
        f"{observed_gradient:.6e} s/m"
    )

    print(
        f"Observed / expected ratio    : "
        f"{gradient_ratio:.6e}"
    )

    # =====================================================
    # RESIDUAL STATISTICS
    # =====================================================

    print_header(
        "EIKONAL RESIDUAL STATISTICS"
    )

    absolute_residual = (
        residual.abs()
    )

    print_stat(
        "Eikonal residual",
        residual
    )

    print_stat(
        "|Eikonal residual|",
        absolute_residual
    )

    print(
        f"Maximum absolute residual   : "
        f"{absolute_residual.max().item():.6e}"
    )

    print(
        f"Mean absolute residual      : "
        f"{absolute_residual.mean().item():.6e}"
    )

    # =====================================================
    # RESIDUAL PERCENTILES
    # =====================================================

    print_header(
        "EIKONAL RESIDUAL PERCENTILES"
    )

    for q in (
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.999
    ):

        value = percentile(
            absolute_residual,
            q
        )

        print(
            f"{q * 100:6.1f}th percentile       : "
            f"{value:.6e}"
        )

    # =====================================================
    # RESIDUAL TOLERANCE ANALYSIS
    # =====================================================

    print_header(
        "RESIDUAL TOLERANCE ANALYSIS"
    )

    total_voxels = (
        absolute_residual.numel()
    )

    for tolerance in (
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00
    ):

        count = (
            absolute_residual <= tolerance
        ).sum().item()

        percentage = (
            100.0
            *
            count
            /
            total_voxels
        )

        print(
            f"|R| <= {tolerance:<5.2f}              : "
            f"{percentage:8.4f}%"
        )

    # =====================================================
    # BOUNDARY MASK
    # =====================================================

    print_header(
        "BOUNDARY VERSUS INTERIOR RESIDUAL AUDIT"
    )

    depth = residual.shape[2]
    height = residual.shape[3]
    width = residual.shape[4]

    if depth < 3 or height < 3 or width < 3:

        raise RuntimeError(
            "Tensor is too small for interior/boundary "
            "analysis."
        )

    boundary_mask = torch.zeros_like(
        residual,
        dtype=torch.bool
    )

    boundary_mask[:, :, 0, :, :] = True
    boundary_mask[:, :, -1, :, :] = True

    boundary_mask[:, :, :, 0, :] = True
    boundary_mask[:, :, :, -1, :] = True

    boundary_mask[:, :, :, :, 0] = True
    boundary_mask[:, :, :, :, -1] = True

    interior_mask = ~boundary_mask

    boundary_residual = (
        absolute_residual[
            boundary_mask
        ]
    )

    interior_residual = (
        absolute_residual[
            interior_mask
        ]
    )

    print(
        f"Boundary voxel count       : "
        f"{boundary_residual.numel():,}"
    )

    print(
        f"Interior voxel count       : "
        f"{interior_residual.numel():,}"
    )

    print(
        f"Boundary mean |R|          : "
        f"{boundary_residual.mean().item():.6e}"
    )

    print(
        f"Interior mean |R|          : "
        f"{interior_residual.mean().item():.6e}"
    )

    print(
        f"Boundary maximum |R|       : "
        f"{boundary_residual.max().item():.6e}"
    )

    print(
        f"Interior maximum |R|       : "
        f"{interior_residual.max().item():.6e}"
    )

    boundary_ratio = (
        boundary_residual.mean().item()
        /
        interior_residual.mean().item()
        if interior_residual.mean().item() > 0
        else float("inf")
    )

    print(
        f"Boundary / interior ratio  : "
        f"{boundary_ratio:.6e}"
    )

    # =====================================================
    # PHYSICS LOSS
    # =====================================================

    print_header(
        "PHYSICS LOSS FORWARD PASS"
    )

    physics_loss_function = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=PHYSICS_LOSS_WEIGHTS[
            "eikonal"
        ],
        source_weight=PHYSICS_LOSS_WEIGHTS[
            "source"
        ],
        travel_time_weight=PHYSICS_LOSS_WEIGHTS[
            "travel_time"
        ]
    )

    physics_loss_function.eval()

    # Use a fresh graph-enabled travel-time tensor
    # for differentiability verification.

    _, travel_time_grad, _ = model(
        input_cube.requires_grad_(True)
    )

    physics_result = physics_loss_function(
        travel_time=travel_time_grad,
        velocity=velocity
    )

    physics_eikonal = physics_result[
        "eikonal"
    ]

    physics_total = physics_result[
        "total"
    ]

    print(
        f"Physics total loss         : "
        f"{physics_total.item():.6e}"
    )

    print(
        f"Physics Eikonal loss       : "
        f"{physics_eikonal.item():.6e}"
    )

    # =====================================================
    # DIRECT VERSUS PHYSICSLOSS
    # =====================================================

    print_header(
        "DIRECT EIKONAL CONSISTENCY"
    )

    direct_loss = (
        residual.pow(2).mean()
    )

    print(
        f"Direct Eikonal loss        : "
        f"{direct_loss.item():.6e}"
    )

    print(
        f"PhysicsLoss Eikonal        : "
        f"{physics_eikonal.item():.6e}"
    )

    difference = abs(
        direct_loss.item()
        -
        physics_eikonal.item()
    )

    print(
        f"Absolute difference        : "
        f"{difference:.6e}"
    )

    if difference > 1e-4:

        raise RuntimeError(
            "Direct Eikonal calculation does not "
            "match PhysicsLoss implementation."
        )

    print(
        "Eikonal implementation consistency: PASSED"
    )

    # =====================================================
    # DIFFERENTIABILITY
    # =====================================================

    print_header(
        "EIKONAL DIFFERENTIABILITY AUDIT"
    )

    print(
        f"Travel-time requires_grad  : "
        f"{travel_time_grad.requires_grad}"
    )

    print(
        f"Eikonal requires_grad      : "
        f"{physics_eikonal.requires_grad}"
    )

    if not physics_eikonal.requires_grad:

        raise RuntimeError(
            "Physics Eikonal loss does not require "
            "gradients."
        )

    physics_eikonal.backward()

    gradients_found = False
    finite_gradients = True

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients_found = True

            if not torch.isfinite(
                parameter.grad
            ).all():

                finite_gradients = False

                break

    if not gradients_found:

        raise RuntimeError(
            "No network gradients were generated."
        )

    if not finite_gradients:

        raise RuntimeError(
            "Non-finite network gradients detected."
        )

    print(
        "Eikonal gradient propagation: PASSED"
    )

    # =====================================================
    # RESIDUAL EXTREME CHECK
    # =====================================================

    print_header(
        "NUMERICAL STABILITY CHECK"
    )

    if not torch.isfinite(
        residual
    ).all():

        raise RuntimeError(
            "Eikonal residual contains NaN or Inf."
        )

    if not torch.isfinite(
        gradient_magnitude
    ).all():

        raise RuntimeError(
            "Gradient magnitude contains NaN or Inf."
        )

    print(
        "Residual contains only finite values."
    )

    print(
        "Gradient magnitude contains only finite values."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # =====================================================
    # FINAL INTERPRETATION
    # =====================================================

    print_header(
        "SPATIAL RESIDUAL AUDIT INTERPRETATION"
    )

    print(
        f"Mean |R|                  : "
        f"{absolute_residual.mean().item():.6e}"
    )

    print(
        f"Median |R|                : "
        f"{percentile(absolute_residual, 0.50):.6e}"
    )

    print(
        f"95th percentile |R|       : "
        f"{percentile(absolute_residual, 0.95):.6e}"
    )

    print(
        f"99th percentile |R|       : "
        f"{percentile(absolute_residual, 0.99):.6e}"
    )

    print(
        f"Boundary/interior ratio    : "
        f"{boundary_ratio:.6e}"
    )

    if boundary_ratio > 2.0:

        print(
            "NOTE:"
        )

        print(
            "Boundary residuals are substantially larger "
            "than interior residuals. This may indicate "
            "finite-difference boundary effects."
        )

    elif boundary_ratio < 0.5:

        print(
            "NOTE:"
        )

        print(
            "Interior residuals are substantially larger "
            "than boundary residuals."
        )

    else:

        print(
            "Boundary and interior residual magnitudes "
            "are of comparable scale."
        )

    # =====================================================
    # FINAL SUCCESS
    # =====================================================

    print_header(
        "PHYSICS LOSS SPATIAL RESIDUAL AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ Travel-time field validity"
    )

    print(
        "  ✓ Physical velocity validity"
    )

    print(
        "  ✓ Spatial derivative computation"
    )

    print(
        "  ✓ Eikonal residual computation"
    )

    print(
        "  ✓ Residual statistical distribution"
    )

    print(
        "  ✓ Residual percentile analysis"
    )

    print(
        "  ✓ Residual tolerance analysis"
    )

    print(
        "  ✓ Boundary/interior comparison"
    )

    print(
        "  ✓ Direct Eikonal consistency"
    )

    print(
        "  ✓ PhysicsLoss consistency"
    )

    print(
        "  ✓ Eikonal differentiability"
    )

    print(
        "  ✓ Network gradient propagation"
    )

    print(
        "  ✓ Numerical stability"
    )

    print()
    print(
        "PHYSICS LOSS SPATIAL RESIDUAL AUDIT COMPLETED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("=" * 60)
        print(
            "PHYSICS LOSS SPATIAL RESIDUAL AUDIT FAILED"
        )
        print("=" * 60)

        print(
            f"Error: {error}"
        )

        sys.exit(1)