"""
=========================================================
Travel-Time Head Audit
=========================================================

Purpose
-------
Audit the travel-time output of the production
PhysicsInformed3DUNet without modifying production code.

The test examines:

1. Travel-time output magnitude
2. Spatial variation
3. Spatial gradients
4. Eikonal quantity V|grad T|
5. Eikonal residual
6. Gradient flow into the travel-time head
7. Gradient flow into the rest of the network

This is a diagnostic test only.
=========================================================
"""

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss
from utils.config import (
    DX,
    DY,
    DZ,
    PHYSICS_LOSS_WEIGHTS,
)


def describe(name, tensor):
    """Print basic statistics for a tensor."""

    print(f"\n{name}")
    print("-" * 60)

    print(f"Shape       : {tuple(tensor.shape)}")
    print(f"Min         : {tensor.min().item():.10e}")
    print(f"Max         : {tensor.max().item():.10e}")
    print(f"Mean        : {tensor.mean().item():.10e}")
    print(f"Std         : {tensor.std().item():.10e}")
    print(f"Abs mean    : {tensor.abs().mean().item():.10e}")
    print(f"Abs max     : {tensor.abs().max().item():.10e}")


def main():

    print("=" * 78)
    print("TRAVEL-TIME HEAD AUDIT")
    print("=" * 78)

    print("\nThis test does NOT modify production source code.")

    torch.manual_seed(42)

    device = torch.device("cpu")

    # -----------------------------------------------------
    # 1. Create production network
    # -----------------------------------------------------

    print("\n[1] Creating production network")
    print("-" * 78)

    model = Network3D()

    model = model.to(device)

    model.eval()

    print("PhysicsInformed3DUNet created successfully.")

    # -----------------------------------------------------
    # 2. Create controlled input
    # -----------------------------------------------------

    print("\n[2] Creating controlled input")
    print("-" * 78)

    input_shape = (1, 1, 64, 128, 128)

    inputs = torch.randn(
        input_shape,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    velocity_model = torch.full(
        input_shape,
        2650.0,
        dtype=torch.float32,
        device=device,
    )

    describe("Input", inputs)
    describe("Velocity model", velocity_model)

    # -----------------------------------------------------
    # 3. Forward pass
    # -----------------------------------------------------

    print("\n[3] Network forward pass")
    print("-" * 78)

    reconstruction, travel_time, log_variance = model(inputs)

    describe("Reconstruction", reconstruction)
    describe("Travel time", travel_time)
    describe("Log variance", log_variance)

    # -----------------------------------------------------
    # 4. Travel-time spatial variation
    # -----------------------------------------------------

    print("\n[4] Travel-time spatial variation")
    print("-" * 78)

    tt_range = (
        travel_time.max() - travel_time.min()
    ).item()

    tt_std = travel_time.std().item()

    tt_mean_abs = travel_time.abs().mean().item()

    print(f"Travel-time range     : {tt_range:.10e}")
    print(f"Travel-time std       : {tt_std:.10e}")
    print(f"Travel-time abs mean  : {tt_mean_abs:.10e}")

    if tt_std < 1e-5:
        print(
            "\nWARNING: Travel-time output is "
            "extremely spatially uniform."
        )
    else:
        print(
            "\nTravel-time output has "
            "measurable spatial variation."
        )

    # -----------------------------------------------------
    # 5. Physics gradient calculation
    # -----------------------------------------------------

    print("\n[5] Travel-time spatial gradients")
    print("-" * 78)

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=PHYSICS_LOSS_WEIGHTS["eikonal"],
        source_weight=PHYSICS_LOSS_WEIGHTS["source"],
        travel_time_weight=PHYSICS_LOSS_WEIGHTS["travel_time"],
    )

    (
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared,
        gradient_magnitude,
    ) = physics_loss.travel_time_gradient(travel_time)

    describe("dT/dx", dT_dx)
    describe("dT/dy", dT_dy)
    describe("dT/dz", dT_dz)
    describe("|grad T|", gradient_magnitude)

    # -----------------------------------------------------
    # 6. Eikonal quantity
    # -----------------------------------------------------

    print("\n[6] Eikonal quantity")
    print("-" * 78)

    eikonal_quantity = (
        velocity_model * gradient_magnitude
    )

    residual = eikonal_quantity - 1.0

    describe(
        "V |grad T|",
        eikonal_quantity,
    )

    describe(
        "Eikonal residual V|grad T|-1",
        residual,
    )

    eikonal_loss = torch.mean(
        residual ** 2
    )

    print(
        f"\nEikonal loss : "
        f"{eikonal_loss.item():.10e}"
    )

    # -----------------------------------------------------
    # 7. Isolate travel-time physics gradient
    # -----------------------------------------------------

    print("\n[7] Backpropagation through travel-time physics")
    print("-" * 78)

    model.zero_grad(set_to_none=True)

    if inputs.grad is not None:
        inputs.grad.zero_()

    eikonal_loss.backward()

    print(
        f"Input gradient norm : "
        f"{inputs.grad.norm().item():.10e}"
    )

    # -----------------------------------------------------
    # 8. Identify travel-time-related parameters
    # -----------------------------------------------------

    print("\n[8] Parameter gradient audit")
    print("-" * 78)

    travel_time_parameters = []
    other_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if "travel" in name.lower():
            travel_time_parameters.append(
                (name, parameter)
            )
        else:
            other_parameters.append(
                (name, parameter)
            )

    print(
        f"Travel-time-related parameter tensors : "
        f"{len(travel_time_parameters)}"
    )

    print(
        f"Other trainable parameter tensors      : "
        f"{len(other_parameters)}"
    )

    # -----------------------------------------------------
    # 9. Travel-time parameter gradients
    # -----------------------------------------------------

    print("\nTravel-time-related gradients")
    print("-" * 60)

    travel_missing = 0
    travel_nonfinite = 0

    for name, parameter in travel_time_parameters:

        if parameter.grad is None:

            travel_missing += 1

            print(
                f"{name}: MISSING GRADIENT"
            )

            continue

        grad = parameter.grad

        finite = torch.isfinite(grad).all().item()

        if not finite:
            travel_nonfinite += 1

        print(
            f"{name}"
            f"\n    shape       : {tuple(parameter.shape)}"
            f"\n    grad norm   : {grad.norm().item():.10e}"
            f"\n    grad abs max: {grad.abs().max().item():.10e}"
            f"\n    finite      : {finite}"
        )

    # -----------------------------------------------------
    # 10. Other parameter gradients
    # -----------------------------------------------------

    print("\nOther network gradients")
    print("-" * 60)

    other_missing = 0
    other_nonfinite = 0
    other_nonzero = 0

    for name, parameter in other_parameters:

        if parameter.grad is None:

            other_missing += 1
            continue

        grad = parameter.grad

        finite = torch.isfinite(grad).all().item()

        if not finite:
            other_nonfinite += 1

        if grad.abs().max().item() > 0.0:
            other_nonzero += 1

    print(
        f"Other parameters with missing gradients : "
        f"{other_missing}"
    )

    print(
        f"Other parameters with nonfinite gradients : "
        f"{other_nonfinite}"
    )

    print(
        f"Other parameters with nonzero gradients : "
        f"{other_nonzero}"
    )

    print("\nOther parameters with missing gradients")
    print("-" * 60)

    for name, parameter in other_parameters:

        if parameter.grad is None:
            print(
                f"{name}"
                f"\n    shape : {tuple(parameter.shape)}"
            )

    # -----------------------------------------------------
    # 11. Final diagnostic interpretation
    # -----------------------------------------------------

    print("\n[11] Diagnostic interpretation")
    print("-" * 78)

    if travel_missing > 0:
        print(
            "RESULT: Travel-time head has missing gradients."
        )

    elif travel_nonfinite > 0:
        print(
            "RESULT: Travel-time head has nonfinite gradients."
        )

    else:
        print(
            "RESULT: Travel-time head receives finite gradients."
        )

    if tt_std < 1e-5:
        print(
            "RESULT: Travel-time prediction is "
            "near-constant at initialization."
        )

    else:
        print(
            "RESULT: Travel-time prediction has "
            "non-negligible spatial variation."
        )

    print("\n" + "=" * 78)
    print("TRAVEL-TIME HEAD AUDIT: COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()