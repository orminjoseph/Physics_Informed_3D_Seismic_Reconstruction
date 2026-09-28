"""
=========================================================
Physics Loss Scaling Audit
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Diagnose the numerical scale of the Eikonal physics loss.

This test does NOT modify the model or physics loss.

It measures:

    1. velocity statistics
    2. travel-time statistics
    3. spatial gradient statistics
    4. Eikonal residual statistics
    5. Eikonal loss

The purpose is to determine whether the current
travel-time/velocity/spatial-sampling scales are physically
and numerically compatible.

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss
from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    VELOCITY_MIN,
    VELOCITY_MAX,
    TRAVEL_TIME_SCALE,
)


# =======================================================
# CONFIGURATION
# =======================================================

DEVICE = torch.device(DEVICE)

BATCH_SIZE = 1
CHANNELS = 1
DEPTH = 64
HEIGHT = 128
WIDTH = 128


# =======================================================
# CREATE TEST INPUT
# =======================================================

torch.manual_seed(42)

input_tensor = torch.randn(
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH,
    device=DEVICE,
)

input_tensor = torch.clamp(
    input_tensor,
    min=-1.0,
    max=1.0,
)


# =======================================================
# CREATE PHYSICS MODEL
# =======================================================

model = Network3D(
    in_channels=1,
    out_channels=1,
    use_attention=True,
    use_residual=True,
    use_uncertainty=True,
).to(DEVICE)


model.eval()


# =======================================================
# FORWARD PASS
# =======================================================

with torch.no_grad():

    reconstruction, travel_time, log_variance = model(
        input_tensor
    )


# =======================================================
# CREATE PHYSICAL VELOCITY FIELD
# =======================================================

velocity = torch.full_like(
    travel_time,
    fill_value=3000.0,
)


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
# CALCULATE TRAVEL-TIME GRADIENT
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
    velocity,
)


# =======================================================
# CALCULATE EIKONAL LOSS
# =======================================================

eikonal = physics_loss.eikonal_loss(
    travel_time,
    velocity,
)


# =======================================================
# PRINT HEADER
# =======================================================

print()
print("=" * 70)
print("PHYSICS LOSS SCALING AUDIT")
print("=" * 70)

print()
print(f"Device:             {DEVICE}")
print(
    f"Input shape:        {tuple(input_tensor.shape)}"
)

print()
print("SPATIAL SAMPLING")
print("-" * 70)

print(f"dx:                 {DX}")
print(f"dy:                 {DY}")
print(f"dz:                 {DZ}")

print()
print("TRAVEL-TIME SCALE")
print("-" * 70)

print(
    f"Configured scale:   {TRAVEL_TIME_SCALE}"
)


# =======================================================
# VELOCITY STATISTICS
# =======================================================

print()
print("VELOCITY")
print("-" * 70)

print(
    f"Minimum:            {velocity.min().item():.8e}"
)

print(
    f"Maximum:            {velocity.max().item():.8e}"
)

print(
    f"Mean:               {velocity.mean().item():.8e}"
)

print(
    f"Expected config min:{VELOCITY_MIN:.8e}"
)

print(
    f"Expected config max:{VELOCITY_MAX:.8e}"
)


# =======================================================
# TRAVEL-TIME STATISTICS
# =======================================================

print()
print("TRAVEL TIME")
print("-" * 70)

print(
    f"Minimum:            {travel_time.min().item():.8e}"
)

print(
    f"Maximum:            {travel_time.max().item():.8e}"
)

print(
    f"Mean:               {travel_time.mean().item():.8e}"
)

print(
    f"Std:                {travel_time.std().item():.8e}"
)


# =======================================================
# GRADIENT STATISTICS
# =======================================================

print()
print("TRAVEL-TIME GRADIENT")
print("-" * 70)

print(
    f"|dT/dz| mean:       "
    f"{dT_dz.abs().mean().item():.8e}"
)

print(
    f"|dT/dy| mean:       "
    f"{dT_dy.abs().mean().item():.8e}"
)

print(
    f"|dT/dx| mean:       "
    f"{dT_dx.abs().mean().item():.8e}"
)

print(
    f"|grad T| mean:      "
    f"{gradient_magnitude.mean().item():.8e}"
)

print(
    f"|grad T| max:       "
    f"{gradient_magnitude.max().item():.8e}"
)


# =======================================================
# PHYSICAL TARGET GRADIENT
# =======================================================

expected_gradient = 1.0 / 3000.0

print()
print("EXPECTED EIKONAL GRADIENT")
print("-" * 70)

print(
    f"1 / V:              "
    f"{expected_gradient:.8e}"
)

print(
    f"Actual / expected:  "
    f"{gradient_magnitude.mean().item() / expected_gradient:.8e}"
)


# =======================================================
# EIKONAL RESIDUAL
# =======================================================

print()
print("EIKONAL RESIDUAL")
print("-" * 70)

print(
    f"Residual min:       "
    f"{residual.min().item():.8e}"
)

print(
    f"Residual max:       "
    f"{residual.max().item():.8e}"
)

print(
    f"Residual mean:      "
    f"{residual.mean().item():.8e}"
)

print(
    f"|Residual| mean:    "
    f"{residual.abs().mean().item():.8e}"
)

print(
    f"Residual RMS:       "
    f"{torch.sqrt(residual.pow(2).mean()).item():.8e}"
)


# =======================================================
# EIKONAL LOSS
# =======================================================

print()
print("EIKONAL LOSS")
print("-" * 70)

print(
    f"Eikonal loss:       "
    f"{eikonal.item():.8e}"
)


# =======================================================
# INTERPRETATION HELPER
# =======================================================

print()
print("DIAGNOSTIC")
print("-" * 70)

actual_gradient = gradient_magnitude.mean().item()

ratio = (
    actual_gradient
    /
    expected_gradient
)

if ratio > 100.0:

    print(
        "WARNING: Predicted travel-time gradient is "
        "more than 100x the expected physical gradient."
    )

elif ratio < 0.01:

    print(
        "WARNING: Predicted travel-time gradient is "
        "less than 1% of the expected physical gradient."
    )

else:

    print(
        "Travel-time gradient is within a broad "
        "diagnostic range of the expected physical scale."
    )


print()
print("=" * 70)
print("PHYSICS SCALING AUDIT COMPLETE")
print("=" * 70)