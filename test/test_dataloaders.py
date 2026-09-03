"""
=========================================================
DataLoader Pipeline Test
=========================================================

Tests the complete dataset -> split -> DataLoader pipeline.

Author: Ormin Joseph
=========================================================
"""

import torch

from dataset.build_dataloaders import (
    build_dataloaders
)


def test_dataloaders():

    print()
    print("=" * 60)
    print("DATALOADERS TEST")
    print("=" * 60)

    # =====================================================
    # Build DataLoaders
    # =====================================================

    (
        train_loader,
        validation_loader
    ) = build_dataloaders()

    # =====================================================
    # Validate DataLoader objects
    # =====================================================

    assert train_loader is not None

    assert validation_loader is not None

    # =====================================================
    # Display dataset sizes
    # =====================================================

    print()
    print(
        "Training Dataset Size:",
        len(train_loader.dataset)
    )

    print(
        "Validation Dataset Size:",
        len(validation_loader.dataset)
    )

    # =====================================================
    # Validate non-empty datasets
    # =====================================================

    assert len(train_loader.dataset) > 0

    assert len(validation_loader.dataset) > 0

    # =====================================================
    # Obtain training batch
    # =====================================================

    train_batch = next(
        iter(train_loader)
    )

    (
        train_input,
        train_target,
        train_mask,
        train_velocity
    ) = train_batch

    # =====================================================
    # Display training batch shapes
    # =====================================================

    print()
    print("Training Batch:")
    print(
        "Input:",
        train_input.shape
    )

    print(
        "Target:",
        train_target.shape
    )

    print(
        "Mask:",
        train_mask.shape
    )

    print(
        "Velocity:",
        train_velocity.shape
    )

    # =====================================================
    # Validate training tensors
    # =====================================================

    assert isinstance(
        train_input,
        torch.Tensor
    )

    assert isinstance(
        train_target,
        torch.Tensor
    )

    assert isinstance(
        train_mask,
        torch.Tensor
    )

    assert isinstance(
        train_velocity,
        torch.Tensor
    )

    # =====================================================
    # Validate training dimensions
    # =====================================================

    assert train_input.ndim == 5

    assert train_target.ndim == 5

    assert train_mask.ndim == 5

    assert train_velocity.ndim == 5

    # =====================================================
    # Validate training shape consistency
    # =====================================================

    assert (
        train_input.shape
        ==
        train_target.shape
    )

    assert (
        train_input.shape
        ==
        train_mask.shape
    )

    assert (
        train_input.shape
        ==
        train_velocity.shape
    )

    # =====================================================
    # Validate batch size
    # =====================================================

    assert (
        train_input.shape[0]
        <=
        train_loader.batch_size
    )

    # =====================================================
    # Validate finite values
    # =====================================================

    assert torch.isfinite(
        train_input
    ).all()

    assert torch.isfinite(
        train_target
    ).all()

    assert torch.isfinite(
        train_mask
    ).all()

    assert torch.isfinite(
        train_velocity
    ).all()

    # =====================================================
    # Validate mask range
    # =====================================================

    assert train_mask.min() >= 0

    assert train_mask.max() <= 1

    # =====================================================
    # Obtain validation batch
    # =====================================================

    validation_batch = next(
        iter(validation_loader)
    )

    (
        validation_input,
        validation_target,
        validation_mask,
        validation_velocity
    ) = validation_batch

    # =====================================================
    # Display validation batch
    # =====================================================

    print()
    print("Validation Batch:")

    print(
        "Input:",
        validation_input.shape
    )

    print(
        "Target:",
        validation_target.shape
    )

    print(
        "Mask:",
        validation_mask.shape
    )

    print(
        "Velocity:",
        validation_velocity.shape
    )

    # =====================================================
    # Validate validation tensors
    # =====================================================

    assert isinstance(
        validation_input,
        torch.Tensor
    )

    assert isinstance(
        validation_target,
        torch.Tensor
    )

    assert isinstance(
        validation_mask,
        torch.Tensor
    )

    assert isinstance(
        validation_velocity,
        torch.Tensor
    )

    # =====================================================
    # Validate validation dimensions
    # =====================================================

    assert validation_input.ndim == 5

    assert validation_target.ndim == 5

    assert validation_mask.ndim == 5

    assert validation_velocity.ndim == 5

    # =====================================================
    # Validate validation shape consistency
    # =====================================================

    assert (
        validation_input.shape
        ==
        validation_target.shape
    )

    assert (
        validation_input.shape
        ==
        validation_mask.shape
    )

    assert (
        validation_input.shape
        ==
        validation_velocity.shape
    )

    # =====================================================
    # Final message
    # =====================================================

    print()
    print("=" * 60)
    print("DATALOADERS TEST PASSED")
    print("=" * 60)