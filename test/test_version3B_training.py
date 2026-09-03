import os

import torch
from torch.optim import Adam

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer

from utils.config import (
    DX,
    DY,
    DZ
)


def test_version3B_training_integration():

    print()
    print("=" * 70)
    print("VERSION 3B - TRAINING INTEGRATION TEST")
    print("=" * 70)

    # =====================================================
    # 1. BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print()
    print("Dataset:")
    print(type(dataset))
    print("Dataset size:", len(dataset))

    assert len(dataset) > 1

    # =====================================================
    # 2. SPLIT DATASET
    # =====================================================

    train_dataset, validation_dataset = split_dataset(
        dataset
    )

    print()
    print("Training dataset size:")
    print(len(train_dataset))

    print()
    print("Validation dataset size:")
    print(len(validation_dataset))

    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0

    # =====================================================
    # 3. CREATE DATALOADERS
    # =====================================================

    train_loader = create_dataloader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    assert len(train_loader) > 0
    assert len(validation_loader) > 0

    print()
    print("Training batches:")
    print(len(train_loader))

    print()
    print("Validation batches:")
    print(len(validation_loader))

    # =====================================================
    # 4. DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:")
    print(device)

    # =====================================================
    # 5. CREATE NETWORK
    # =====================================================

    print()
    print("Creating Network3D...")

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    )

    # Move model to selected device.

    model = model.to(device)

    print("Network created successfully.")

    # =====================================================
    # 6. CREATE TOTAL LOSS
    # =====================================================

    print()
    print("Creating TotalLoss...")

    # -----------------------------------------------------
    # Synthetic physical sampling
    # -----------------------------------------------------
    #
    # The synthetic dataset defines one voxel as:
    #
    #     1 m × 1 m × 1 m
    #
    # Therefore:
    #
    #     DX = 1.0 m
    #     DY = 1.0 m
    #     DZ = 1.0 m
    #
    # These values are obtained directly from the
    # centralized project configuration.
    # -----------------------------------------------------

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    )

    print("TotalLoss created successfully.")

    print()
    print("Synthetic physical sampling:")
    print("DX:", DX)
    print("DY:", DY)
    print("DZ:", DZ)

    # =====================================================
    # 7. CREATE OPTIMIZER
    # =====================================================

    print()
    print("Creating Adam optimizer...")

    optimizer = Adam(
        model.parameters(),
        lr=1e-4
    )

    print("Adam optimizer created successfully.")

    # =====================================================
    # 8. CREATE TRAINER
    # =====================================================

    print()
    print("Creating Trainer...")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    print("Trainer created successfully.")

    # =====================================================
    # 9. TRAIN FOR ONE EPOCH
    # =====================================================

    print()
    print("=" * 70)
    print("STARTING ONE-EPOCH TRAINING SMOKE TEST")
    print("=" * 70)

    trainer.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=1,
        resume=False
    )

    # =====================================================
    # 10. CHECK TRAINING OUTPUTS
    # =====================================================

    print()
    print("=" * 70)
    print("CHECKING TRAINING OUTPUTS")
    print("=" * 70)

    checkpoint_directory = (
        trainer.checkpoint_directory
    )

    latest_checkpoint = os.path.join(
        checkpoint_directory,
        "latest_checkpoint.pth"
    )

    best_checkpoint = os.path.join(
        checkpoint_directory,
        "best_model.pth"
    )

    training_state = os.path.join(
        checkpoint_directory,
        "training_state.json"
    )

    training_history = os.path.join(
        trainer.experiment.logs,
        "training_history.csv"
    )

    training_summary = os.path.join(
        trainer.experiment.reports,
        "training_summary.txt"
    )

    # =====================================================
    # 11. VALIDATE OUTPUT FILES
    # =====================================================

    print()
    print("Latest checkpoint:")
    print(latest_checkpoint)

    assert os.path.exists(
        latest_checkpoint
    )

    print()
    print("Best model:")
    print(best_checkpoint)

    assert os.path.exists(
        best_checkpoint
    )

    print()
    print("Training state:")
    print(training_state)

    assert os.path.exists(
        training_state
    )

    print()
    print("Training history:")
    print(training_history)

    assert os.path.exists(
        training_history
    )

    print()
    print("Training summary:")
    print(training_summary)

    assert os.path.exists(
        training_summary
    )

    # =====================================================
    # 12. CHECK TRAINING STATE
    # =====================================================

    assert trainer.current_epoch == 0

    assert trainer.best_epoch >= 0

    assert torch.isfinite(
        torch.tensor(
            trainer.best_validation_loss
        )
    )

    # =====================================================
    # 13. SUCCESS
    # =====================================================

    print()
    print("=" * 70)
    print("VERSION 3B TRAINING INTEGRATION TEST PASSED.")
    print("=" * 70)
