"""
=========================================================
Test: Eikonal Physics Loss
=========================================================
"""

import torch

from losses.physics_loss import (
    PhysicsLoss
)


def main():

    print("=" * 60)
    print("TESTING EIKONAL PHYSICS LOSS")
    print("=" * 60)

    # =================================================
    # ANALYTICAL EIKONAL TEST
    # =================================================

    test_analytical_eikonal_solution()

    print()
    print("EIKONAL PHYSICS LOSS TEST PASSED.")
    print("=" * 60)
    # =====================================================
    # CREATE TEST TENSORS
    # =====================================================

    batch_size = 1
    channels = 1

    depth = 16
    crossline = 32
    inline = 32

    # Predicted travel-time field.
    #
    # requires_grad=True allows us to test whether
    # gradients can flow through the physics loss.

    travel_time = torch.rand(
        batch_size,
        channels,
        depth,
        crossline,
        inline,
        requires_grad=True
    )

    # Physical P-wave velocity model [m/s].
    #
    # Typical values:
    #
    # 1500 m/s -> shallow sediments
    # 5000 m/s -> high-velocity rocks

    velocity = (
        1500.0
        +
        3500.0
        *
        torch.rand(
            batch_size,
            channels,
            depth,
            crossline,
            inline
        )
    )

    # =====================================================
    # INITIALIZE LOSS
    # =====================================================

    physics_loss = PhysicsLoss(

        dx=1.0,

        dy=1.0,

        dz=1.0,

        eikonal_weight=1.0,

        source_weight=1.0,

        travel_time_weight=1.0
    )

    # =====================================================
    # COMPUTE PHYSICS LOSS
    # =====================================================

    losses = physics_loss(

        travel_time=travel_time,

        velocity=velocity,

        source_indices=None,

        travel_time_target=None
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\nPhysics Loss Components:\n")

    for name, value in losses.items():

        print(
            f"{name:25s}: "
            f"{value.item():.10e}"
        )

    # =====================================================
    # BACKWARD PASS TEST
    # =====================================================

    print("\nTesting backward propagation...")

    total_loss = losses["total"]

    total_loss.backward()

    # =====================================================
    # CHECK GRADIENT
    # =====================================================

    if travel_time.grad is None:

        raise RuntimeError(
            "Gradient was not produced for travel_time."
        )

    if not torch.isfinite(
        travel_time.grad
    ).all():

        raise RuntimeError(
            "Gradient contains NaN or Inf."
        )

    print("Gradient successfully computed.")

    print(
        "Gradient mean:",
        travel_time.grad.mean().item()
    )

    print(
        "Gradient maximum:",
        travel_time.grad.abs().max().item()
    )

    print("\nEIKONAL PHYSICS LOSS TEST PASSED.")

    print("=" * 60)

def test_analytical_eikonal_solution():

    print()
    print("=" * 60)
    print("TESTING ANALYTICAL EIKONAL SOLUTION")
    print("=" * 60)

    # -------------------------------------------------
    # Physical parameters
    # -------------------------------------------------

    batch_size = 1
    channels = 1

    depth = 16
    crossline = 32
    inline = 32

    dx = 1.0
    dy = 1.0
    dz = 1.0

    # Constant P-wave velocity [m/s]

    velocity_value = 2000.0

    # -------------------------------------------------
    # Create Eikonal physics loss
    # -------------------------------------------------

    physics_loss = PhysicsLoss(
        dx=dx,
        dy=dy,
        dz=dz
    )

    # -------------------------------------------------
    # Create coordinate field
    # -------------------------------------------------

    x = torch.arange(
        inline,
        dtype=torch.float32
    ) * dx

    # -------------------------------------------------
    # Analytical travel-time solution
    #
    # T(x) = x / V
    #
    # Therefore:
    #
    # dT/dx = 1 / V
    # -------------------------------------------------

    travel_time = (
        x
        /
        velocity_value
    )

    # Reshape:
    #
    # [W]
    #
    # ->
    #
    # [B, C, D, H, W]

    travel_time = travel_time.view(
        1,
        1,
        1,
        1,
        inline
    )

    travel_time = travel_time.expand(
        batch_size,
        channels,
        depth,
        crossline,
        inline
    ).clone()

    # -------------------------------------------------
    # Constant velocity field
    # -------------------------------------------------

    velocity = torch.full_like(
        travel_time,
        velocity_value
    )

    # -------------------------------------------------
    # Compute residual
    # -------------------------------------------------

    residual = physics_loss.eikonal_residual(
        travel_time,
        velocity
    )

    # -------------------------------------------------
    # Compute loss
    # -------------------------------------------------

    loss = residual.pow(2).mean()

    print()
    print(
        "Analytical Eikonal loss:",
        loss.item()
    )

    print(
        "Maximum residual:",
        residual.abs().max().item()
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    assert torch.isfinite(loss)

    assert loss.item() < 1e-8

    print()
    print(
        "ANALYTICAL EIKONAL TEST PASSED."
    )

    print("=" * 60)

if __name__ == "__main__":

    main()