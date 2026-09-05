"""
=========================================================
Physics Loss Discrepancy Diagnostic Test
=========================================================

Purpose:
    Determine why the standalone TotalLoss test produced
    a very large physics loss (~1.53e6), while the actual
    Trainer one-batch integration produced a physics loss
    of approximately 0.3175.

This test DOES NOT modify production source code.

It compares:

    1. Real synthetic dataset + network travel time
    2. Direct PhysicsLoss calculation
    3. TotalLoss physics calculation
    4. Random travel-time field used by the old standalone test
    5. Eikonal gradient and residual statistics

Author: Ormin Joseph
=========================================================
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.synthetic_dataset import SyntheticSeismicDataset
from models.network import Network3D
from losses.physics_loss import PhysicsLoss
from losses.total_loss import TotalLoss

from utils.config import (
    DX,
    DY,
    DZ,
    DEVICE,
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
)


# =========================================================
# Utility functions
# =========================================================

def print_stats(name, tensor):
    """
    Print basic numerical statistics for a tensor.
    """

    tensor = tensor.detach()

    print(f"\n{name}")
    print("-" * 60)

    print(f"Shape       : {tuple(tensor.shape)}")
    print(f"Min         : {tensor.min().item():.10e}")
    print(f"Max         : {tensor.max().item():.10e}")
    print(f"Mean        : {tensor.mean().item():.10e}")
    print(f"Std         : {tensor.std().item():.10e}")
    print(f"Abs mean    : {tensor.abs().mean().item():.10e}")
    print(f"Abs max     : {tensor.abs().max().item():.10e}")

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )


def calculate_gradient_components(
    physics_loss,
    travel_time
):
    """
    Calculate spatial travel-time gradient components
    using the actual production PhysicsLoss API.
    """

    (
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared,
        gradient_magnitude
    ) = physics_loss.travel_time_gradient(
        travel_time
    )

    return (
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared,
        gradient_magnitude
    )


# =========================================================
# Main diagnostic
# =========================================================

def main():

    print("=" * 78)
    print("PHYSICS LOSS DISCREPANCY DIAGNOSTIC")
    print("=" * 78)

    print("\nThis test does NOT modify production source code.")

    # =====================================================
    # Reproducibility
    # =====================================================

    torch.manual_seed(42)

    print("\n[1] Configuration")
    print("-" * 78)

    print(f"Device              : {DEVICE}")
    print(f"DX                  : {DX}")
    print(f"DY                  : {DY}")
    print(f"DZ                  : {DZ}")

    print("\nLoss weights:")

    for name, value in LOSS_WEIGHTS.items():
        print(f"    {name:<15}: {value}")

    print("\nPhysics loss weights:")

    for name, value in PHYSICS_LOSS_WEIGHTS.items():
        print(f"    {name:<15}: {value}")

    # =====================================================
    # Create actual synthetic dataset
    # =====================================================

    print("\n[2] Creating real synthetic dataset")
    print("-" * 78)

    dataset = SyntheticSeismicDataset(
        num_samples=1
    )

    sample = dataset[0]

    if len(sample) < 4:
        raise RuntimeError(
            "SyntheticSeismicDataset does not return "
            "(input, target, mask, velocity_model)."
        )

    inputs = sample[0]
    target = sample[1]
    mask = sample[2]
    velocity = sample[3]

    # Convert [C,D,H,W] to [B,C,D,H,W]
    inputs = inputs.unsqueeze(0)
    target = target.unsqueeze(0)
    mask = mask.unsqueeze(0)
    velocity = velocity.unsqueeze(0)

    inputs = inputs.to(DEVICE)
    target = target.to(DEVICE)
    mask = mask.to(DEVICE)
    velocity = velocity.to(DEVICE)

    print_stats(
        "Input",
        inputs
    )

    print_stats(
        "Target",
        target
    )

    print_stats(
        "Velocity model",
        velocity
    )

    # =====================================================
    # Initialize production network
    # =====================================================

    print("\n[3] Creating production network")
    print("-" * 78)

    model = Network3D().to(DEVICE)

    model.eval()

    print(
        "PhysicsInformed3DUNet created successfully."
    )

    # =====================================================
    # Network forward pass
    # =====================================================

    print("\n[4] Network forward pass")
    print("-" * 78)

    with torch.no_grad():

        reconstruction, travel_time, log_variance = (
            model(inputs)
        )

    print_stats(
        "Network reconstruction",
        reconstruction
    )

    print_stats(
        "Network travel time",
        travel_time
    )

    print_stats(
        "Network log variance",
        log_variance
    )

    # =====================================================
    # PhysicsLoss
    # =====================================================

    print("\n[5] Direct PhysicsLoss diagnostic")
    print("-" * 78)

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=PHYSICS_LOSS_WEIGHTS["eikonal"],
        source_weight=PHYSICS_LOSS_WEIGHTS["source"],
        travel_time_weight=PHYSICS_LOSS_WEIGHTS["travel_time"],
    ).to(DEVICE)

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Re-enable gradients for travel time because we want
    # to inspect the physical gradient field.
    # -----------------------------------------------------

    travel_time = travel_time.detach().requires_grad_(True)

    # =====================================================
    # Calculate spatial derivatives
    # =====================================================

    print("\nCalculating spatial travel-time derivatives...")

    dT_dx, dT_dy, dT_dz, gradient_squared, gradient_magnitude = (physics_loss.travel_time_gradient(travel_time
        )
    )

    print_stats(
        "dT/dx",
        dT_dx
    )

    print_stats(
        "dT/dy",
        dT_dy
    )

    print_stats(
        "dT/dz",
        dT_dz
    )

    # =====================================================
    # Gradient magnitude
    # =====================================================

    gradient_squared = (
        dT_dx.pow(2)
        +
        dT_dy.pow(2)
        +
        dT_dz.pow(2)
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared
        +
        physics_loss.eps
    )

    print_stats(
        "|grad T|",
        gradient_magnitude
    )

    # =====================================================
    # V |grad T|
    # =====================================================

    velocity_gradient = (
        velocity
        *
        gradient_magnitude
    )

    print_stats(
        "V |grad T|",
        velocity_gradient
    )

    # =====================================================
    # Eikonal residual
    # =====================================================

    eikonal_residual = (
        velocity_gradient
        -
        1.0
    )

    print_stats(
        "Eikonal residual V|grad T|-1",
        eikonal_residual
    )

    # =====================================================
    # Direct eikonal loss
    # =====================================================

    direct_eikonal = (
        eikonal_residual.pow(2).mean()
    )

    print(
        "\nDirect manually calculated Eikonal loss:"
    )

    print(
        f"{direct_eikonal.item():.10e}"
    )

    # =====================================================
    # PhysicsLoss residual
    # =====================================================

    physics_residual = (
        physics_loss.eikonal_residual(
            travel_time,
            velocity
        )
    )

    residual_difference = (
        eikonal_residual
        -
        physics_residual
    ).abs().max()

    print(
        "\nPhysicsLoss residual comparison:"
    )

    print(
        "Maximum absolute difference:"
    )

    print(
        f"{residual_difference.item():.10e}"
    )

    if residual_difference.item() > 1e-6:

        raise RuntimeError(
            "Manual Eikonal residual does not match "
            "PhysicsLoss implementation."
        )

    print(
        "Residual implementation check: PASS"
    )

    # =====================================================
    # Full PhysicsLoss
    # =====================================================

    physics_components = physics_loss(
        travel_time,
        velocity
    )

    print("\nPhysicsLoss output")
    print("-" * 60)

    for name, value in physics_components.items():

        print(
            f"{name:<20}: "
            f"{value.item():.10e}"
        )

    # =====================================================
    # Compare Eikonal component
    # =====================================================

    physics_eikonal = (
        physics_components["eikonal"]
    )

    eikonal_difference = (
        direct_eikonal
        -
        physics_eikonal
    ).abs()

    print(
        "\nManual vs PhysicsLoss Eikonal difference:"
    )

    print(
        f"{eikonal_difference.item():.10e}"
    )

    if eikonal_difference.item() > 1e-5:

        raise RuntimeError(
            "Manual Eikonal loss does not match "
            "PhysicsLoss Eikonal component."
        )

    print(
        "Eikonal loss implementation check: PASS"
    )

    # =====================================================
    # TotalLoss comparison
    # =====================================================

    print("\n[6] TotalLoss comparison")
    print("-" * 78)

    total_loss = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    ).to(DEVICE)

    # Reconstruction and log variance do not need gradients
    # for this comparison.
    reconstruction_for_loss = (
        reconstruction.detach()
    )

    log_variance_for_loss = (
        log_variance.detach()
    )

    total_losses = total_loss(
        prediction=reconstruction_for_loss,
        target=target,
        travel_time=travel_time,
        velocity_model=velocity,
        log_variance=log_variance_for_loss
    )

    print("\nTotalLoss physics component:")

    print(
        f"{total_losses['physics'].item():.10e}"
    )

    physics_difference = (
        total_losses["physics"]
        -
        physics_components["total"]
    ).abs()

    print(
        "\nDirect PhysicsLoss vs TotalLoss difference:"
    )

    print(
        f"{physics_difference.item():.10e}"
    )

    if physics_difference.item() > 1e-5:

        raise RuntimeError(
            "TotalLoss physics component does not "
            "match direct PhysicsLoss."
        )

    print(
        "TotalLoss physics consistency: PASS"
    )

    # =====================================================
    # OLD STANDALONE TEST TRAVEL-TIME FIELD
    # =====================================================

    print("\n[7] Reproducing old standalone test")
    print("-" * 78)

    old_random_travel_time = torch.rand(
        travel_time.shape,
        device=DEVICE,
        requires_grad=True
    )

    print_stats(
        "Old standalone random travel time",
        old_random_travel_time
    )

    old_physics_components = physics_loss(
        old_random_travel_time,
        velocity
    )

    print(
        "\nOld standalone random travel-time"
        " PhysicsLoss:"
    )

    for name, value in old_physics_components.items():

        print(
            f"{name:<20}: "
            f"{value.item():.10e}"
        )

    # =====================================================
    # Comparison summary
    # =====================================================

    print("\n[8] FINAL COMPARISON")
    print("=" * 78)

    print(
        "\nNetwork-generated travel-time physics:"
    )

    print(
        f"{physics_components['total'].item():.10e}"
    )

    print(
        "\nOld random travel-time physics:"
    )

    print(
        f"{old_physics_components['total'].item():.10e}"
    )

    ratio = (
        old_physics_components["total"]
        /
        (
            physics_components["total"]
            +
            1e-12
        )
    )

    print(
        "\nRandom/network physics-loss ratio:"
    )

    print(
        f"{ratio.item():.10e}"
    )

    print("\n" + "=" * 78)

    print(
        "PHYSICS LOSS DISCREPANCY DIAGNOSTIC: PASSED"
    )

    print("=" * 78)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    main()