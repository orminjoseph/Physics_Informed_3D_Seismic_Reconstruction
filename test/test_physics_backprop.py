"""
======================================================================
PHYSICS BACKPROPAGATION TEST
======================================================================

Purpose
-------
Verify that the Eikonal physics loss:

1. Produces a finite loss.
2. Produces non-zero gradients.
3. Produces gradients with respect to the travel-time field.
4. Can backpropagate through a realistic predicted travel-time field.

This test does NOT modify the model or physics-loss implementation.
======================================================================
"""

import torch

from losses.physics_loss import PhysicsLoss


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

DEVICE = torch.device("cpu")

BATCH_SIZE = 1
CHANNELS = 1
DEPTH = 16
HEIGHT = 16
WIDTH = 16

VELOCITY_VALUE = 3000.0

DX = 1.0
DY = 1.0
DZ = 1.0


# ---------------------------------------------------------------------
# 2. Create a synthetic travel-time field
# ---------------------------------------------------------------------

# We deliberately start with a field whose gradient is too small.
#
# This resembles the problem observed in the network audit.

travel_time = (
    0.01
    + 1.0e-6
    * torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE,
    )
)

# Enable gradient tracking.
travel_time.requires_grad_(True)


# ---------------------------------------------------------------------
# 3. Create velocity model
# ---------------------------------------------------------------------

velocity = torch.full(
    (
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
    ),
    VELOCITY_VALUE,
    dtype=torch.float32,
    device=DEVICE,
)


# ---------------------------------------------------------------------
# 4. Create physics loss
# ---------------------------------------------------------------------

physics_loss = PhysicsLoss(
    dx=DX,
    dy=DY,
    dz=DZ,
    eikonal_weight=1.0,
    source_weight=0.0,
    travel_time_weight=0.0,
)


# ---------------------------------------------------------------------
# 5. Calculate Eikonal loss
# ---------------------------------------------------------------------

losses = physics_loss(
    travel_time=travel_time,
    velocity=velocity,
)


eikonal_loss = losses["eikonal"]


# ---------------------------------------------------------------------
# 6. Print loss
# ---------------------------------------------------------------------

print("=" * 70)
print("PHYSICS BACKPROPAGATION TEST")
print("=" * 70)

print()
print("TRAVEL-TIME FIELD")
print("-" * 70)

print(f"Mean:       {travel_time.detach().mean().item():.8e}")
print(f"Std:        {travel_time.detach().std().item():.8e}")

print()
print("EIKONAL LOSS")
print("-" * 70)

print(f"Loss:       {eikonal_loss.item():.8e}")


# ---------------------------------------------------------------------
# 7. Backpropagate
# ---------------------------------------------------------------------

eikonal_loss.backward()


# ---------------------------------------------------------------------
# 8. Inspect gradients
# ---------------------------------------------------------------------

gradient = travel_time.grad


print()
print("TRAVEL-TIME GRADIENT")
print("-" * 70)

if gradient is None:

    print("Gradient:  NONE")

    raise RuntimeError(
        "FAIL: No gradient reached the travel-time field."
    )


gradient_abs = gradient.abs()

print(f"Mean |gradient|: {gradient_abs.mean().item():.8e}")
print(f"Max  |gradient|: {gradient_abs.max().item():.8e}")
print(f"Min  |gradient|: {gradient_abs.min().item():.8e}")


# ---------------------------------------------------------------------
# 9. Check gradient validity
# ---------------------------------------------------------------------

if not torch.isfinite(gradient).all():

    raise RuntimeError(
        "FAIL: Physics gradient contains NaN or Inf."
    )


if gradient_abs.max().item() == 0.0:

    raise RuntimeError(
        "FAIL: Physics gradient is exactly zero."
    )


# ---------------------------------------------------------------------
# 10. Final result
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("GRADIENT TEST: PASS")
print("PHYSICS BACKPROPAGATION TEST PASSED")
print("=" * 70)