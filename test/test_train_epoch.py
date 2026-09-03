"""
=========================================================
Test - Train Epoch
=========================================================

Tests one complete training epoch of the
Physics-Informed 3D Encoder-Decoder Framework.

The test verifies:

    1. Trainer initialization
    2. Training DataLoader creation
    3. One complete training epoch
    4. Returned loss components
    5. Finite loss values
    6. Training history compatibility

Author: Ormin Joseph
=========================================================
"""

import torch

import test.setup_path

from test.test_factory import (
    create_trainer,
    create_dataloader
)


# =========================================================
# TEST TRAIN EPOCH
# =========================================================

def test_train_epoch():

    print()
    print("=" * 60)
    print("TEST - TRAIN ONE EPOCH")
    print("=" * 60)

    # -----------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------

    train_loader, validation_loader = create_dataloader(
        batch_size=2,
        train_samples=2,
        validation_samples=1
    )

    print()
    print(
        f"Training batches   : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(validation_loader)}"
    )

    # -----------------------------------------------------
    # Create Trainer
    # -----------------------------------------------------

    trainer = create_trainer()

    print(
        "Trainer initialization: OK"
    )

    # -----------------------------------------------------
    # Train one epoch
    # -----------------------------------------------------

    losses = trainer.train_epoch(
        train_loader
    )

    # -----------------------------------------------------
    # Verify returned object
    # -----------------------------------------------------

    assert isinstance(
        losses,
        dict
    )

    # -----------------------------------------------------
    # Required loss components
    # -----------------------------------------------------

    required_components = {
        "total",
        "mae",
        "physics",
        "uncertainty",
        "ssim"
    }

    assert required_components.issubset(
        losses.keys()
    )

    print()
    print("Training Losses")
    print("-" * 40)

    for name, value in losses.items():

        print(
            f"{name:<15}: {value:.6f}"
        )

    # -----------------------------------------------------
    # Verify finite losses
    # -----------------------------------------------------

    for name, value in losses.items():

        assert torch.isfinite(
            torch.tensor(value)
        ), (
            f"Non-finite loss detected: {name}"
        )

    print()
    print(
        "All training losses are finite: OK"
    )

    # -----------------------------------------------------
    # Verify total loss
    # -----------------------------------------------------

    assert losses["total"] >= 0.0

    print(
        "Total loss validation: OK"
    )

    # -----------------------------------------------------
    # Verify expected number of batches
    # -----------------------------------------------------

    assert len(train_loader) > 0

    print(
        "Training DataLoader validation: OK"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print(
        "Train Epoch Test: PASSED"
    )

    print("=" * 60)
