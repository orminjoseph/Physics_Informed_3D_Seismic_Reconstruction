"""
=========================================================
Checkpoint and Resume Training Test
=========================================================

Verifies that:

1. Training can complete one epoch.
2. A latest checkpoint is created.
3. The checkpoint stores the completed epoch correctly.
4. A new Trainer can load the checkpoint.
5. Resume training starts from the NEXT epoch.
6. Training can continue to a second epoch.
7. The latest checkpoint is updated.
8. Epoch-specific checkpoints are created.

Author: Ormin Joseph
=========================================================
"""

import os
import torch
from torch.optim import Adam

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer


def test_resume_training():

    print()
    print("=" * 70)
    print("CHECKPOINT AND RESUME TRAINING TEST")
    print("=" * 70)

    # =====================================================
    # 1. BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print()
    print("Dataset size:")
    print(len(dataset))

    assert len(dataset) > 1

    # =====================================================
    # 2. SPLIT DATASET
    # =====================================================

    train_dataset, validation_dataset = split_dataset(
        dataset
    )

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
    # 5. CREATE FIRST MODEL
    # =====================================================

    print()
    print("Creating first Network3D...")

    model_1 = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    )

    model_1 = model_1.to(device)

    # =====================================================
    # 6. CREATE FIRST LOSS
    # =====================================================

    criterion_1 = TotalLoss(
        dx=1.0,
        dy=1.0,
        dz=1.0
    )

    # =====================================================
    # 7. CREATE FIRST OPTIMIZER
    # =====================================================

    optimizer_1 = Adam(
        model_1.parameters(),
        lr=1e-4
    )

    # =====================================================
    # 8. CREATE FIRST TRAINER
    # =====================================================

    trainer_1 = Trainer(
        model=model_1,
        criterion=criterion_1,
        optimizer=optimizer_1,
        device=device
    )

    # =====================================================
    # 9. FIRST TRAINING RUN
    # =====================================================
    #
    # Train for ONE epoch.
    #
    # Internally:
    #
    #     epoch = 0
    #
    # means:
    #
    #     Epoch 1 has been completed.
    #
    # The checkpoint should therefore store:
    #
    #     epoch = 0
    #
    # =====================================================

    print()
    print("=" * 70)
    print("FIRST RUN: TRAINING FOR ONE EPOCH")
    print("=" * 70)

    trainer_1.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=1,
        resume=False
    )

    # =====================================================
    # 10. CHECK LATEST CHECKPOINT
    # =====================================================

    checkpoint_directory = (
        trainer_1.checkpoint_directory
    )

    latest_checkpoint = os.path.join(
        checkpoint_directory,
        "latest_checkpoint.pth"
    )

    assert os.path.exists(
        latest_checkpoint
    )

    print()
    print("Latest checkpoint:")
    print(latest_checkpoint)

    # =====================================================
    # 11. READ CHECKPOINT
    # =====================================================

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device
    )

    stored_epoch = int(
        checkpoint["epoch"]
    )

    print()
    print("Stored checkpoint epoch:")
    print(stored_epoch)

    # Epoch 1 corresponds to internal epoch 0.
    assert stored_epoch == 0

    # =====================================================
    # 12. CREATE SECOND MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("CREATING SECOND TRAINER FOR RESUME TEST")
    print("=" * 70)

    model_2 = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    )

    model_2 = model_2.to(device)

    criterion_2 = TotalLoss(
        dx=1.0,
        dy=1.0,
        dz=1.0
    )

    optimizer_2 = Adam(
        model_2.parameters(),
        lr=1e-4
    )

    trainer_2 = Trainer(
        model=model_2,
        criterion=criterion_2,
        optimizer=optimizer_2,
        device=device
    )

    # =====================================================
    # 13. LOAD CHECKPOINT DIRECTLY
    # =====================================================

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    start_epoch = trainer_2.load_checkpoint(
        latest_checkpoint
    )

    print()
    print("Returned start epoch:")
    print(start_epoch)

    # =====================================================
    # 14. VERIFY RESUME POSITION
    # =====================================================
    #
    # Completed:
    #
    #     Epoch 1
    #
    # Internal checkpoint:
    #
    #     epoch = 0
    #
    # Therefore:
    #
    #     start_epoch = 1
    #
    # This means training resumes at:
    #
    #     Epoch 2
    #
    # =====================================================

    assert start_epoch == 1

    print()
    print("Resume position verified.")
    print("Training should continue from Epoch 2.")

    # =====================================================
    # 15. RESUME TRAINING TO TWO EPOCHS
    # =====================================================

    print()
    print("=" * 70)
    print("SECOND RUN: RESUMING TRAINING TO TWO EPOCHS")
    print("=" * 70)

    trainer_2.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=2,
        resume=True
    )

    # =====================================================
    # 16. CHECK UPDATED CHECKPOINT
    # =====================================================

    updated_checkpoint = torch.load(
        latest_checkpoint,
        map_location=device
    )

    updated_epoch = int(
        updated_checkpoint["epoch"]
    )

    print()
    print("Updated checkpoint epoch:")
    print(updated_epoch)

    # Epoch 2 corresponds to internal epoch 1.
    assert updated_epoch == 1

    # =====================================================
    # 17. VERIFY TRAINER STATE
    # =====================================================

    print()
    print("Trainer current epoch:")
    print(trainer_2.current_epoch)

    assert trainer_2.current_epoch == 1

    # =====================================================
    # 18. CHECK EPOCH CHECKPOINT
    # =====================================================

    epoch_directory = os.path.join(
        checkpoint_directory,
        "epoch_sensitivity"
    )

    epoch_0_checkpoint = os.path.join(
        epoch_directory,
        "epoch_0000.pth"
    )

    epoch_1_checkpoint = os.path.join(
        epoch_directory,
        "epoch_0001.pth"
    )

    print()
    print("Epoch 1 checkpoint:")
    print(epoch_0_checkpoint)

    print()
    print("Epoch 2 checkpoint:")
    print(epoch_1_checkpoint)

    assert os.path.exists(
        epoch_0_checkpoint
    )

    assert os.path.exists(
        epoch_1_checkpoint
    )

    # =====================================================
    # 19. CHECK TRAINING STATE JSON
    # =====================================================

    training_state = os.path.join(
        checkpoint_directory,
        "training_state.json"
    )

    assert os.path.exists(
        training_state
    )

    # =====================================================
    # 20. CHECK BEST MODEL
    # =====================================================

    best_model = os.path.join(
        checkpoint_directory,
        "best_model.pth"
    )

    assert os.path.exists(
        best_model
    )

    # =====================================================
    # 21. SUCCESS
    # =====================================================

    print()
    print("=" * 70)
    print("CHECKPOINT AND RESUME TEST PASSED.")
    print("=" * 70)