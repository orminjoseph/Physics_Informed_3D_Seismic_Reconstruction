"""
======================================================================
NETWORK PHYSICS GRADIENT FLOW TEST
======================================================================

Purpose
-------
Verify that the Eikonal physics loss can backpropagate through:

    PhysicsLoss
        ↓
    Network travel-time output
        ↓
    Travel-time head parameters
        ↓
    Decoder parameters

This test does not modify the model.

======================================================================
"""

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss


# =====================================================================
# 1. Configuration
# =====================================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

VELOCITY_VALUE = 3000.0

DX = 1.0
DY = 1.0
DZ = 1.0


# =====================================================================
# 2. Create network
# =====================================================================

print("=" * 70)
print("NETWORK PHYSICS GRADIENT FLOW TEST")
print("=" * 70)

print()
print("CREATING NETWORK")
print("-" * 70)

model = Network3D(
    in_channels=CHANNELS,
    out_channels=1,
    use_attention=True,
    use_residual=True,
    use_uncertainty=True,
).to(DEVICE)

model.train()


# =====================================================================
# 3. Create synthetic input
# =====================================================================

inputs = torch.randn(
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH,
    device=DEVICE,
)


# =====================================================================
# 4. Create velocity model
# =====================================================================

velocity = torch.full(
    (
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH,
    ),
    VELOCITY_VALUE,
    dtype=torch.float32,
    device=DEVICE,
)


# =====================================================================
# 5. Forward pass
# =====================================================================

print()
print("NETWORK FORWARD PASS")
print("-" * 70)

reconstruction, travel_time, log_variance = model(inputs)


print(
    f"Reconstruction shape: "
    f"{tuple(reconstruction.shape)}"
)

print(
    f"Travel-time shape:    "
    f"{tuple(travel_time.shape)}"
)

print(
    f"Log-variance shape:   "
    f"{tuple(log_variance.shape)}"
)


# =====================================================================
# 6. Inspect travel-time output
# =====================================================================

print()
print("TRAVEL-TIME OUTPUT")
print("-" * 70)

print(
    f"Minimum: {travel_time.detach().min().item():.8e}"
)

print(
    f"Maximum: {travel_time.detach().max().item():.8e}"
)

print(
    f"Mean:    {travel_time.detach().mean().item():.8e}"
)

print(
    f"Std:     {travel_time.detach().std().item():.8e}"
)


# =====================================================================
# 7. Compute spatial gradients
# =====================================================================

physics_loss = PhysicsLoss(
    dx=DX,
    dy=DY,
    dz=DZ,
    eikonal_weight=1.0,
    source_weight=0.0,
    travel_time_weight=0.0,
)

dT_dz, dT_dy, dT_dx, gradient_squared, gradient_magnitude = (
    physics_loss.travel_time_gradient(
        travel_time
    )
)

print()
print("TRAVEL-TIME GRADIENT")
print("-" * 70)

print(
    f"Mean |dT/dz|: "
    f"{dT_dz.abs().mean().item():.8e}"
)

print(
    f"Mean |dT/dy|: "
    f"{dT_dy.abs().mean().item():.8e}"
)

print(
    f"Mean |dT/dx|: "
    f"{dT_dx.abs().mean().item():.8e}"
)

print(
    f"Mean |grad T|: "
    f"{gradient_magnitude.mean().item():.8e}"
)

print(
    f"Max  |grad T|: "
    f"{gradient_magnitude.max().item():.8e}"
)


# =====================================================================
# 8. Calculate Eikonal loss
# =====================================================================

losses = physics_loss(
    travel_time=travel_time,
    velocity=velocity,
)

eikonal_loss = losses["eikonal"]

print()
print("EIKONAL LOSS")
print("-" * 70)

print(
    f"Eikonal loss: "
    f"{eikonal_loss.item():.8e}"
)


# =====================================================================
# 9. Clear existing gradients
# =====================================================================

model.zero_grad(set_to_none=True)


# =====================================================================
# 10. Backpropagate
# =====================================================================

print()
print("BACKPROPAGATION")
print("-" * 70)

eikonal_loss.backward()


# =====================================================================
# 11. Inspect travel-time head gradients
# =====================================================================

travel_weight_gradient = (
    model.travel_time_head.weight.grad
)

travel_bias_gradient = (
    model.travel_time_head.bias.grad
)


print()
print("TRAVEL-TIME HEAD GRADIENTS")
print("-" * 70)

if travel_weight_gradient is None:

    print("Travel-time weight gradient: NONE")

else:

    print(
        "Travel-time weight gradient mean: "
        f"{travel_weight_gradient.abs().mean().item():.8e}"
    )

    print(
        "Travel-time weight gradient max:  "
        f"{travel_weight_gradient.abs().max().item():.8e}"
    )


if travel_bias_gradient is None:

    print("Travel-time bias gradient: NONE")

else:

    print(
        "Travel-time bias gradient mean: "
        f"{travel_bias_gradient.abs().mean().item():.8e}"
    )

    print(
        "Travel-time bias gradient max:  "
        f"{travel_bias_gradient.abs().max().item():.8e}"
    )


# =====================================================================
# 12. Inspect decoder gradient
# =====================================================================

decoder_gradient_values = []

for name, parameter in model.decoder.named_parameters():

    if parameter.grad is not None:

        decoder_gradient_values.append(
            parameter.grad.detach().abs().max().item()
        )


print()
print("DECODER GRADIENTS")
print("-" * 70)

if len(decoder_gradient_values) == 0:

    print("No decoder gradients found.")

else:

    print(
        f"Decoder parameters with gradients: "
        f"{len(decoder_gradient_values)}"
    )

    print(
        f"Maximum decoder gradient: "
        f"{max(decoder_gradient_values):.8e}"
    )


# =====================================================================
# 13. Validate gradients
# =====================================================================

if travel_weight_gradient is None:

    raise RuntimeError(
        "FAIL: No gradient reached travel-time head weights."
    )

if travel_bias_gradient is None:

    raise RuntimeError(
        "FAIL: No gradient reached travel-time head bias."
    )

if not torch.isfinite(travel_weight_gradient).all():

    raise RuntimeError(
        "FAIL: Travel-time weight gradients contain NaN or Inf."
    )

if not torch.isfinite(travel_bias_gradient).all():

    raise RuntimeError(
        "FAIL: Travel-time bias gradients contain NaN or Inf."
    )

if travel_weight_gradient.abs().max().item() == 0.0:

    raise RuntimeError(
        "FAIL: Travel-time weight gradients are exactly zero."
    )


# =====================================================================
# 14. Final result
# =====================================================================

print()
print("=" * 70)
print("NETWORK PHYSICS GRADIENT FLOW TEST: PASS")
print("=" * 70)