"""
=========================================================
Analytical Eikonal Physics Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Verify that PhysicsLoss correctly evaluates a known
analytical travel-time field satisfying the Eikonal
equation.

Analytical field:

    T(x) = x / V

Therefore:

    dT/dx = 1 / V

and:

    |grad T| = 1 / V

so:

    V |grad T| = 1

and:

    R_eikonal = 0

Author: Ormin Joseph
=========================================================
"""

import torch

from losses.physics_loss import PhysicsLoss
from utils.config import DX, DY, DZ


# =======================================================
# TEST CONFIGURATION
# =======================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

VELOCITY_VALUE = 3000.0


# =======================================================
# CREATE PHYSICS LOSS
# =======================================================

physics_loss = PhysicsLoss(
    dx=DX,
    dy=DY,
    dz=DZ,
    eikonal_weight=1.0,
    source_weight=1.0,
    travel_time_weight=1.0,
)


# =======================================================
# CREATE ANALYTICAL TRAVEL-TIME FIELD
# =======================================================

# Coordinate along the inline direction.
#
# Since x corresponds to dimension 4:
#
# [B, C, D, H, W]

x = torch.arange(
    WIDTH,
    dtype=torch.float32
)


# Convert sample index to physical distance.

x = x * DX


# Analytical travel time:
#
# T(x) = x / V

travel_time = (
    x
    /
    VELOCITY_VALUE
)


# Reshape to:
#
# [1, 1, 1, 1, W]

travel_time = travel_time.reshape(
    1,
    1,
    1,
    1,
    WIDTH
)


# Expand across depth and crossline.

travel_time = travel_time.expand(
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH
).clone()


# =======================================================
# CREATE CONSTANT VELOCITY FIELD
# =======================================================

velocity = torch.full_like(
    travel_time,
    VELOCITY_VALUE
)


# =======================================================
# CALCULATE GRADIENT
# =======================================================

(
    dT_dz,
    dT_dy,
    dT_dx,
    gradient_squared,
    gradient_magnitude,
) = physics_loss.travel_time_gradient(
    travel_time
)


# =======================================================
# CALCULATE EIKONAL RESIDUAL
# =======================================================

residual = physics_loss.eikonal_residual(
    travel_time,
    velocity
)


# =======================================================
# CALCULATE LOSS
# =======================================================

loss = physics_loss.eikonal_loss(
    travel_time,
    velocity
)


# =======================================================
# EXPECTED VALUES
# =======================================================

expected_gradient = (
    1.0
    /
    VELOCITY_VALUE
)


# =======================================================
# PRINT RESULTS
# =======================================================

print()
print("=" * 70)
print("ANALYTICAL EIKONAL PHYSICS TEST")
print("=" * 70)

print()
print("TEST FIELD")
print("-" * 70)

print(
    "T(x) = x / V"
)

print(
    f"Velocity:            {VELOCITY_VALUE:.8e} m/s"
)

print(
    f"dx:                  {DX:.8e}"
)


print()
print("EXPECTED")
print("-" * 70)

print(
    f"Expected dT/dx:      "
    f"{expected_gradient:.8e}"
)

print(
    "Expected dT/dy:      0"
)

print(
    "Expected dT/dz:      0"
)

print(
    "Expected Eikonal residual: 0"
)


print()
print("MEASURED")
print("-" * 70)

print(
    f"Mean dT/dx:          "
    f"{dT_dx.mean().item():.8e}"
)

print(
    f"Mean |dT/dx|:        "
    f"{dT_dx.abs().mean().item():.8e}"
)

print(
    f"Mean |dT/dy|:        "
    f"{dT_dy.abs().mean().item():.8e}"
)

print(
    f"Mean |dT/dz|:        "
    f"{dT_dz.abs().mean().item():.8e}"
)

print(
    f"Mean |grad T|:       "
    f"{gradient_magnitude.mean().item():.8e}"
)


print()
print("EIKONAL RESIDUAL")
print("-" * 70)

print(
    f"Minimum:             "
    f"{residual.min().item():.8e}"
)

print(
    f"Maximum:             "
    f"{residual.max().item():.8e}"
)

print(
    f"Mean:                "
    f"{residual.mean().item():.8e}"
)

print(
    f"Mean absolute:       "
    f"{residual.abs().mean().item():.8e}"
)


print()
print("EIKONAL LOSS")
print("-" * 70)

print(
    f"Loss:                "
    f"{loss.item():.8e}"
)


# =======================================================
# PASS / FAIL
# =======================================================

gradient_error = abs(
    dT_dx.abs().mean().item()
    -
    expected_gradient
)

residual_error = residual.abs().mean().item()


print()
print("=" * 70)


if gradient_error < 1e-7:

    print(
        "GRADIENT TEST: PASS"
    )

else:

    print(
        "GRADIENT TEST: FAIL"
    )


if residual_error < 1e-4:

    print(
        "EIKONAL RESIDUAL TEST: PASS"
    )

else:

    print(
        "EIKONAL RESIDUAL TEST: FAIL"
    )


if loss.item() < 1e-6:

    print(
        "EIKONAL LOSS TEST: PASS"
    )

else:

    print(
        "EIKONAL LOSS TEST: FAIL"
    )


if (
    gradient_error < 1e-7
    and residual_error < 1e-4
    and loss.item() < 1e-6
):

    print()
    print(
        "ANALYTICAL EIKONAL PHYSICS TEST PASSED"
    )

else:

    print()
    print(
        "ANALYTICAL EIKONAL PHYSICS TEST FAILED"
    )


print("=" * 70)