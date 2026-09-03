"""
=========================================================
Version 3A - Forward Integration Test
=========================================================

Tests the complete forward computational path:

    DataLoader
        ↓
    Network3D
        ↓
    TotalLoss

No optimizer or backward pass is performed.

Tensor convention:

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D

from losses.total_loss import TotalLoss

from utils.config import (
    BATCH_SIZE,
    DX,
    DY,
    DZ,
    USE_UNCERTAINTY,
    USE_RESIDUAL,
    USE_ATTENTION,
)


def test_version3_forward_integration():

    print()
    print("=" * 70)
    print("VERSION 3A - FORWARD INTEGRATION TEST")
    print("=" * 70)

    # =====================================================
    # 1. BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print()
    print("Dataset:")
    print(type(dataset))
    print("Dataset size:", len(dataset))

    # =====================================================
    # 2. SPLIT DATASET
    # =====================================================

    train_dataset, validation_dataset = (
        split_dataset(dataset)
    )

    print()
    print("Training dataset size:")
    print(len(train_dataset))

    print()
    print("Validation dataset size:")
    print(len(validation_dataset))

    # =====================================================
    # 3. CREATE TRAINING DATALOADER
    # =====================================================

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =====================================================
    # 4. GET ONE BATCH
    # =====================================================

    batch = next(iter(train_loader))

    input_cube, target, mask, velocity = batch

    print()
    print("Input batch:")
    print(input_cube.shape)

    print()
    print("Target batch:")
    print(target.shape)

    print()
    print("Mask batch:")
    print(mask.shape)

    print()
    print("Velocity batch:")
    print(velocity.shape)

    # =====================================================
    # 5. VALIDATE INPUT BATCH
    # =====================================================

    assert isinstance(
        input_cube,
        torch.Tensor
    )

    assert isinstance(
        target,
        torch.Tensor
    )

    assert isinstance(
        mask,
        torch.Tensor
    )

    assert isinstance(
        velocity,
        torch.Tensor
    )

    assert input_cube.ndim == 5
    assert target.ndim == 5
    assert mask.ndim == 5
    assert velocity.ndim == 5

    assert input_cube.shape == target.shape
    assert input_cube.shape == mask.shape
    assert input_cube.shape == velocity.shape

    assert torch.isfinite(
        input_cube
    ).all()

    assert torch.isfinite(
        target
    ).all()

    assert torch.isfinite(
        mask
    ).all()

    assert torch.isfinite(
        velocity
    ).all()

    # =====================================================
    # 6. CREATE NETWORK
    # =====================================================

    print()
    print("Creating Network3D...")

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=USE_UNCERTAINTY,
        use_residual=USE_RESIDUAL,
        use_attention=USE_ATTENTION
    )

    model.eval()

    print()
    print("Network created successfully.")

    # =====================================================
    # 7. FORWARD PASS
    # =====================================================

    print()
    print("Running network forward pass...")

    with torch.no_grad():

        reconstruction, travel_time, log_variance = (
            model(input_cube)
        )

    # =====================================================
    # 8. DISPLAY NETWORK OUTPUTS
    # =====================================================

    print()
    print("Network outputs:")
    print(
        "Reconstruction:",
        reconstruction.shape
    )

    print(
        "Travel time:",
        travel_time.shape
    )

    print(
        "Log variance:",
        log_variance.shape
    )

    # =====================================================
    # 9. VALIDATE NETWORK OUTPUTS
    # =====================================================

    assert reconstruction.shape == input_cube.shape

    assert travel_time.shape == input_cube.shape

    assert log_variance.shape == input_cube.shape

    assert torch.isfinite(
        reconstruction
    ).all()

    assert torch.isfinite(
        travel_time
    ).all()

    assert torch.isfinite(
        log_variance
    ).all()

    # Travel time must be non-negative
    # because Network3D uses Softplus.

    assert travel_time.min() >= 0

    # =====================================================
    # 10. CREATE TOTAL LOSS
    # =====================================================

    print()
    print("Creating TotalLoss...")

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    )

    print()
    print("TotalLoss created successfully.")

    # =====================================================
    # 11. CALCULATE COMPOSITE LOSS
    # =====================================================

    print()
    print("Calculating composite loss...")

    loss_components = criterion(

        prediction=reconstruction,

        target=target,

        travel_time=travel_time,

        velocity_model=velocity,

        log_variance=log_variance,

        source_indices=None,

        travel_time_target=None
    )

    # =====================================================
    # 12. DISPLAY LOSS COMPONENTS
    # =====================================================

    print()
    print("Loss components:")
    print(
        "MAE:",
        loss_components["mae"].item()
    )

    print(
        "Physics:",
        loss_components["physics"].item()
    )

    print(
        "  Eikonal:",
        loss_components["eikonal"].item()
    )

    print(
        "  Source:",
        loss_components["source"].item()
    )

    print(
        "  Travel time:",
        loss_components["travel_time"].item()
    )

    print(
        "Uncertainty:",
        loss_components["uncertainty"].item()
    )

    print(
        "SSIM:",
        loss_components["ssim"].item()
    )

    print()
    print(
        "Weighted MAE:",
        loss_components["weighted_mae"].item()
    )

    print(
        "Weighted Physics:",
        loss_components["weighted_physics"].item()
    )

    print(
        "Weighted Uncertainty:",
        loss_components[
            "weighted_uncertainty"
        ].item()
    )

    print(
        "Weighted SSIM:",
        loss_components["weighted_ssim"].item()
    )

    print()
    print(
        "TOTAL LOSS:",
        loss_components["total"].item()
    )

    # =====================================================
    # 13. VALIDATE LOSS OUTPUTS
    # =====================================================

    expected_keys = [

        "mae",

        "physics",

        "uncertainty",

        "ssim",

        "eikonal",

        "source",

        "travel_time",

        "weighted_mae",

        "weighted_physics",

        "weighted_uncertainty",

        "weighted_ssim",

        "total"
    ]

    for key in expected_keys:

        assert key in loss_components

        assert isinstance(
            loss_components[key],
            torch.Tensor
        )

        assert torch.isfinite(
            loss_components[key]
        ).all()

    # =====================================================
    # 14. VALIDATE TOTAL LOSS
    # =====================================================

    calculated_total = (

        loss_components["weighted_mae"]

        +

        loss_components["weighted_physics"]

        +

        loss_components[
            "weighted_uncertainty"
        ]

        +

        loss_components["weighted_ssim"]
    )

    assert torch.allclose(
        loss_components["total"],
        calculated_total,
        rtol=1e-5,
        atol=1e-6
    )

    # =====================================================
    # 15. SUCCESS
    # =====================================================

    print()
    print("=" * 70)
    print("VERSION 3A FORWARD INTEGRATION TEST PASSED.")
    print("=" * 70)