"""
======================================================================
PHYSICS LOSS IMPLEMENTATION AUDIT
======================================================================

Purpose
-------
Determine exactly which Eikonal formulation is implemented by the
PhysicsLoss class currently imported by Python.

This test does NOT modify:
    - PhysicsLoss
    - Network3D
    - TotalLoss
    - config.py
    - LOSS_WEIGHTS

It compares the actual PhysicsLoss output against candidate
formulations.

Candidate formulations:

A. Stabilized:
       R = V |grad T| - 1

B. Normalized squared:
       R = V^2 |grad T|^2 - 1

C. Dimensional:
       R = |grad T|^2 - 1/V^2

The test also checks the exact derivative implementation.

Author: Ormin Joseph
======================================================================
"""

import torch

from losses.physics_loss import PhysicsLoss


# ======================================================================
# CONFIGURATION
# ======================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

BATCH_SIZE = 1
CHANNELS = 1
DEPTH = 16
HEIGHT = 32
WIDTH = 32

DX = 1.0
DY = 1.0
DZ = 1.0

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 5000.0

EPSILON = 1.0e-12


# ======================================================================
# UTILITIES
# ======================================================================

def header(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def finite(tensor, name):

    ok = bool(
        torch.isfinite(tensor).all()
    )

    print(
        f"{name:<35}: "
        f"{'PASS' if ok else 'FAIL'}"
    )

    if not ok:
        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )


def statistics(name, tensor):

    tensor = tensor.detach()

    print()
    print(name)

    print(
        f"    min   : {tensor.min().item():.8e}"
    )

    print(
        f"    max   : {tensor.max().item():.8e}"
    )

    print(
        f"    mean  : {tensor.mean().item():.8e}"
    )

    print(
        f"    std   : {tensor.std().item():.8e}"
    )


# ======================================================================
# CREATE TRAVEL-TIME FIELD
# ======================================================================

def create_travel_time():

    torch.manual_seed(SEED)

    travel_time = torch.rand(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE,
    )

    # Smooth the field slightly by repeated averaging.
    #
    # This is only for diagnostic purposes.
    for _ in range(3):

        travel_time = (
            travel_time
            + torch.roll(
                travel_time,
                shifts=1,
                dims=2,
            )
            + torch.roll(
                travel_time,
                shifts=-1,
                dims=2,
            )
        ) / 3.0

    travel_time.requires_grad_(True)

    return travel_time


# ======================================================================
# CREATE VELOCITY
# ======================================================================

def create_velocity():

    velocity_axis = torch.linspace(
        VELOCITY_MIN,
        VELOCITY_MAX,
        DEPTH,
        device=DEVICE,
    )

    velocity = velocity_axis.view(
        1,
        1,
        DEPTH,
        1,
        1,
    ).expand(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
    ).contiguous()

    return velocity


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "PHYSICS LOSS IMPLEMENTATION AUDIT"
    )

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Shape  : "
        f"({BATCH_SIZE}, {CHANNELS}, "
        f"{DEPTH}, {HEIGHT}, {WIDTH})"
    )

    print(
        f"dx     : {DX}"
    )

    print(
        f"dy     : {DY}"
    )

    print(
        f"dz     : {DZ}"
    )

    # --------------------------------------------------------------
    # Create data
    # --------------------------------------------------------------

    header(
        "1. CREATE DIAGNOSTIC FIELDS"
    )

    travel_time = create_travel_time()

    velocity = create_velocity()

    finite(
        travel_time,
        "Travel time",
    )

    finite(
        velocity,
        "Velocity",
    )

    statistics(
        "Travel time",
        travel_time,
    )

    statistics(
        "Velocity",
        velocity,
    )

    # --------------------------------------------------------------
    # PhysicsLoss
    # --------------------------------------------------------------

    header(
        "2. INITIALIZE PHYSICSLOSS"
    )

    physics = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(DEVICE)

    print(
        f"PhysicsLoss class : "
        f"{physics.__class__.__name__}"
    )

    print(
        f"PhysicsLoss module: "
        f"{physics.__class__.__module__}"
    )

    if hasattr(
        physics,
        "eps"
    ):

        print(
            f"PhysicsLoss eps    : "
            f"{physics.eps:.8e}"
        )

    else:

        print(
            "PhysicsLoss eps    : NOT EXPOSED"
        )

    # --------------------------------------------------------------
    # Derivatives
    # --------------------------------------------------------------

    header(
        "3. EXACT PHYSICSLOSS DERIVATIVES"
    )

    dT_dz = physics._derivative(
        travel_time,
        spacing=DZ,
        dimension=2,
    )

    dT_dy = physics._derivative(
        travel_time,
        spacing=DY,
        dimension=3,
    )

    dT_dx = physics._derivative(
        travel_time,
        spacing=DX,
        dimension=4,
    )

    finite(
        dT_dx,
        "dT/dx",
    )

    finite(
        dT_dy,
        "dT/dy",
    )

    finite(
        dT_dz,
        "dT/dz",
    )

    statistics(
        "dT/dx",
        dT_dx,
    )

    statistics(
        "dT/dy",
        dT_dy,
    )

    statistics(
        "dT/dz",
        dT_dz,
    )

    # --------------------------------------------------------------
    # Gradient squared
    # --------------------------------------------------------------

    header(
        "4. GRADIENT MAGNITUDE"
    )

    gradient_squared = (
        dT_dx.pow(2)
        + dT_dy.pow(2)
        + dT_dz.pow(2)
    )

    finite(
        gradient_squared,
        "|grad T|²",
    )

    eps = (
        physics.eps
        if hasattr(physics, "eps")
        else EPSILON
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared.clamp_min(eps)
    )

    finite(
        gradient_magnitude,
        "|grad T|",
    )

    statistics(
        "|grad T|",
        gradient_magnitude,
    )

    # --------------------------------------------------------------
    # Candidate A
    # --------------------------------------------------------------

    header(
        "5. CANDIDATE A — STABILIZED FORM"
    )

    residual_stabilized = (
        velocity
        * gradient_magnitude
        - 1.0
    )

    loss_stabilized = (
        residual_stabilized.pow(2)
        .mean()
    )

    statistics(
        "Stabilized residual",
        residual_stabilized,
    )

    print(
        f"Stabilized loss : "
        f"{loss_stabilized.item():.10e}"
    )

    # --------------------------------------------------------------
    # Candidate B
    # --------------------------------------------------------------

    header(
        "6. CANDIDATE B — NORMALIZED SQUARED FORM"
    )

    residual_squared = (
        velocity.pow(2)
        * gradient_squared
        - 1.0
    )

    loss_squared = (
        residual_squared.pow(2)
        .mean()
    )

    statistics(
        "Normalized squared residual",
        residual_squared,
    )

    print(
        f"Normalized squared loss : "
        f"{loss_squared.item():.10e}"
    )

    # --------------------------------------------------------------
    # Candidate C
    # --------------------------------------------------------------

    header(
        "7. CANDIDATE C — DIMENSIONAL FORM"
    )

    safe_velocity = torch.clamp(
        velocity,
        min=eps,
    )

    residual_dimensional = (
        gradient_squared
        - 1.0 / safe_velocity.pow(2)
    )

    loss_dimensional = (
        residual_dimensional.pow(2)
        .mean()
    )

    statistics(
        "Dimensional residual",
        residual_dimensional,
    )

    print(
        f"Dimensional loss : "
        f"{loss_dimensional.item():.10e}"
    )

    # --------------------------------------------------------------
    # Actual PhysicsLoss residual
    # --------------------------------------------------------------

    header(
        "8. ACTUAL PHYSICSLOSS RESIDUAL"
    )

    if hasattr(
        physics,
        "eikonal_residual"
    ):

        actual_residual = physics.eikonal_residual(
            travel_time,
            velocity,
        )

        finite(
            actual_residual,
            "Actual PhysicsLoss residual",
        )

        statistics(
            "Actual PhysicsLoss residual",
            actual_residual,
        )

    else:

        actual_residual = None

        print(
            "PhysicsLoss.eikonal_residual(): "
            "NOT AVAILABLE"
        )

    # --------------------------------------------------------------
    # Actual Eikonal loss
    # --------------------------------------------------------------

    header(
        "9. ACTUAL PHYSICSLOSS EIKONAL LOSS"
    )

    if hasattr(
        physics,
        "eikonal_loss"
    ):

        actual_loss = physics.eikonal_loss(
            travel_time,
            velocity,
        )

    else:

        result = physics(
            travel_time,
            velocity,
        )

        if isinstance(
            result,
            dict,
        ):

            actual_loss = result[
                "eikonal"
            ]

        else:

            actual_loss = result

    finite(
        actual_loss,
        "Actual PhysicsLoss loss",
    )

    print(
        f"Actual PhysicsLoss loss : "
        f"{actual_loss.item():.10e}"
    )

    # --------------------------------------------------------------
    # Compare actual loss with candidates
    # --------------------------------------------------------------

    header(
        "10. FORMULATION IDENTIFICATION"
    )

    error_stabilized = abs(
        actual_loss.item()
        - loss_stabilized.item()
    )

    error_squared = abs(
        actual_loss.item()
        - loss_squared.item()
    )

    error_dimensional = abs(
        actual_loss.item()
        - loss_dimensional.item()
    )

    print(
        f"Error vs stabilized form       : "
        f"{error_stabilized:.10e}"
    )

    print(
        f"Error vs normalized squared     : "
        f"{error_squared:.10e}"
    )

    print(
        f"Error vs dimensional form       : "
        f"{error_dimensional:.10e}"
    )

    errors = {
        "stabilized": error_stabilized,
        "normalized_squared": error_squared,
        "dimensional": error_dimensional,
    }

    identified = min(
        errors,
        key=errors.get,
    )

    print()
    print(
        f"Closest formulation: {identified}"
    )

    # --------------------------------------------------------------
    # Residual comparison
    # --------------------------------------------------------------

    if actual_residual is not None:

        print()
        print(
            "Residual maximum absolute errors:"
        )

        residual_errors = {
            "stabilized": (
                actual_residual
                - residual_stabilized
            ).abs().max().item(),

            "normalized_squared": (
                actual_residual
                - residual_squared
            ).abs().max().item(),

            "dimensional": (
                actual_residual
                - residual_dimensional
            ).abs().max().item(),
        }

        for name, value in residual_errors.items():

            print(
                f"    {name:<22}: "
                f"{value:.10e}"
            )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    header(
        "11. FINAL RESULT"
    )

    tolerance = 1.0e-6

    if (
        error_stabilized
        <= tolerance
    ):

        print(
            "PhysicsLoss implementation: "
            "STABILIZED V|grad T|-1"
        )

        print(
            "Implementation identification: PASS"
        )

    elif (
        error_squared
        <= tolerance
    ):

        print(
            "PhysicsLoss implementation: "
            "NORMALIZED V²|grad T|²-1"
        )

        print(
            "Implementation identification: PASS"
        )

    elif (
        error_dimensional
        <= tolerance
    ):

        print(
            "PhysicsLoss implementation: "
            "DIMENSIONAL |grad T|²-1/V²"
        )

        print(
            "Implementation identification: PASS"
        )

    else:

        print(
            "PhysicsLoss implementation does NOT "
            "match any tested formulation."
        )

        print(
            "Implementation identification: FAIL"
        )

    print()
    print(
        "NO MODEL OR CONFIGURATION WAS MODIFIED."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()