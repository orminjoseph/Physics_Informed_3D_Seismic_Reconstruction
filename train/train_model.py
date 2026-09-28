"""
=========================================================
Training Entry Point
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This script is the main training entry point.

Pipeline:

    Configuration
          |
          v
    Dataset Construction
          |
          v
    Train / Validation Split
          |
          v
    DataLoaders
          |
          v
    3D Encoder–Decoder Network
          |
          v
    Composite Total Loss
          |
          v
    Adam Optimizer
          |
          v
    Trainer
          |
          v
    Training + Validation
          |
          v
    Checkpoints + Logs + Reports

The actual training logic is implemented in:

    trainer/trainer.py

This file is responsible for assembling the complete
training pipeline and starting the training process.

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# IMPORT PYTORCH
# =========================================================

import torch


# =========================================================
# DATASET PIPELINE
# =========================================================

from dataset.build_dataset import build_dataset

from dataset.split_dataset import split_dataset

from dataset.dataloader import create_dataloader


# =========================================================
# MODEL
# =========================================================

from models.network import Network3D


# =========================================================
# COMPOSITE LOSS
# =========================================================

from losses.total_loss import TotalLoss


# =========================================================
# TRAINER
# =========================================================

from trainer.trainer import Trainer


# =========================================================
# CONFIGURATION
# =========================================================

from utils.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DX,
    DY,
    DZ,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    DATASET_MODE,
    EXPERIMENT_NAME
)


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def main():
    """
    Construct and execute the complete training pipeline.

    The function performs the following operations:

        1. Select the computational device.
        2. Build the configured dataset.
        3. Split the dataset into training and validation sets.
        4. Create training and validation DataLoaders.
        5. Construct the 3D reconstruction network.
        6. Construct the composite TotalLoss.
        7. Construct the Adam optimizer.
        8. Construct the Trainer.
        9. Start training.
    """

    # =====================================================
    # DISPLAY EXPERIMENT INFORMATION
    # =====================================================

    print("=" * 70)

    print(
        "PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION"
    )

    print(
        "TRAINING ENTRY POINT"
    )

    print("=" * 70)

    print(
        f"Dataset Mode    : {DATASET_MODE}"
    )

    print(
        f"Experiment Name : {EXPERIMENT_NAME}"
    )

    print(
        f"Configured Device: {DEVICE}"
    )

    print(
        f"Batch Size      : {BATCH_SIZE}"
    )

    print(
        f"Epochs          : {NUM_EPOCHS}"
    )

    print(
        f"Learning Rate   : {LEARNING_RATE}"
    )

    print(
        f"Weight Decay    : {WEIGHT_DECAY}"
    )

    print(
        f"Attention       : {USE_ATTENTION}"
    )

    print(
        f"Residual Blocks : {USE_RESIDUAL}"
    )

    print(
        f"Uncertainty     : {USE_UNCERTAINTY}"
    )

    print("=" * 70)


    # =====================================================
    # SELECT COMPUTATIONAL DEVICE
    # =====================================================
    #
    # The configuration specifies the desired device.
    #
    # Example:
    #
    #     DEVICE = "cpu"
    #
    # or:
    #
    #     DEVICE = "cuda"
    #
    # If CUDA is requested but unavailable, we stop with
    # a clear error rather than silently changing the
    # experiment configuration.
    # =====================================================

    configured_device = str(
        DEVICE
    ).lower()


    if configured_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested in utils.config.py, "
                "but CUDA is not available."
            )

        device = torch.device(
            "cuda"
        )

    elif configured_device == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        raise ValueError(
            "Unsupported DEVICE configuration.\n"
            f"Received: {DEVICE}\n"
            "Expected 'cpu' or 'cuda'."
        )


    # =====================================================
    # DISPLAY ACTUAL DEVICE
    # =====================================================

    print()

    print(
        f"Using device: {device}"
    )

    # =====================================================
    # CUDA INFORMATION
    # =====================================================

    if device.type == "cuda":

        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # =====================================================
    # BUILD DATASET
    # =====================================================
    #
    # build_dataset() reads DATASET_MODE and the associated
    # configuration from utils.config.
    #
    # Therefore this script does not hard-code:
    #
    #     Synthetic dataset
    #
    # or:
    #
    #     F3 dataset
    #
    # This keeps the training entry point reusable.
    # =====================================================

    print()

    print("=" * 70)

    print(
        "BUILDING DATASET"
    )

    print("=" * 70)

    dataset = build_dataset()


    # =====================================================
    # VALIDATE DATASET
    # =====================================================

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "The constructed dataset contains zero samples."
        )


    print(
        f"Total dataset samples: {len(dataset)}"
    )


    # =====================================================
    # SPLIT DATASET
    # =====================================================
    #
    # The split function uses the configured validation
    # split and seed.
    #
    # It returns:
    #
    #     train_dataset
    #     val_dataset
    # =====================================================

    print()

    print("=" * 70)

    print(
        "CREATING TRAIN / VALIDATION SPLIT"
    )

    print("=" * 70)

    train_dataset, val_dataset = split_dataset(
        dataset
    )


    # =====================================================
    # VALIDATE SPLIT
    # =====================================================

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset contains zero samples."
        )

    if len(val_dataset) == 0:

        raise RuntimeError(
            "Validation dataset contains zero samples."
        )


    print(
        f"Training samples   : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )


    # =====================================================
    # CREATE TRAINING DATALOADER
    # =====================================================

    print()

    print("=" * 70)

    print(
        "CREATING TRAINING DATALOADER"
    )

    print("=" * 70)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # =====================================================
    # CREATE VALIDATION DATALOADER
    # =====================================================

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # =====================================================
    # VALIDATE DATALOADERS
    # =====================================================

    if len(train_loader) == 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    if len(val_loader) == 0:

        raise RuntimeError(
            "Validation DataLoader contains zero batches."
        )


    print(
        f"Training batches   : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )


    # =====================================================
    # CONSTRUCT THE 3D NETWORK
    # =====================================================
    #
    # The architecture switches are taken directly from
    # utils.config.py.
    #
    # Current configuration:
    #
    #     USE_ATTENTION   = True
    #     USE_RESIDUAL    = True
    #     USE_UNCERTAINTY = True
    #
    # This produces the full proposed architecture.
    # =====================================================

    print()

    print("=" * 70)

    print(
        "BUILDING 3D NETWORK"
    )

    print("=" * 70)

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY
    )


    # =====================================================
    # MOVE MODEL TO DEVICE
    # =====================================================
    #
    # Trainer also moves the model to the selected device.
    #
    # Keeping the explicit operation here makes the model
    # device state clear before Trainer construction.
    # =====================================================

    model = model.to(
        device
    )


    # =====================================================
    # DISPLAY NUMBER OF TRAINABLE PARAMETERS
    # =====================================================

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )


    # =====================================================
    # CONSTRUCT COMPOSITE TOTAL LOSS
    # =====================================================
    #
    # TotalLoss combines:
    #
    #     MAE
    #     Physics / Eikonal
    #     Heteroscedastic Aleatoric Uncertainty
    #     SSIM
    #
    # Spatial derivatives use:
    #
    #     DX
    #     DY
    #     DZ
    #
    # from the configuration.
    # =====================================================

    print()

    print("=" * 70)

    print(
        "BUILDING TOTAL LOSS"
    )

    print("=" * 70)

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        use_uncertainty=USE_UNCERTAINTY
    )


    # =====================================================
    # CONSTRUCT ADAM OPTIMIZER
    # =====================================================

    print()

    print("=" * 70)

    print(
        "BUILDING OPTIMIZER"
    )

    print("=" * 70)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # =====================================================
    # CONSTRUCT TRAINER
    # =====================================================
    #
    # Trainer is responsible for:
    #
    #     training epochs
    #     validation
    #     gradient inspection
    #     gradient clipping
    #     scheduler
    #     checkpointing
    #     CSV logging
    #     TensorBoard logging
    #     training summary
    # =====================================================

    print()

    print("=" * 70)

    print(
        "INITIALIZING TRAINER"
    )

    print("=" * 70)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )


    # =====================================================
    # START TRAINING
    # =====================================================
    #
    # resume=True means that if a valid latest checkpoint
    # exists for this experiment, Trainer will attempt to
    # continue from it.
    #
    # Checkpoint locations are controlled by the
    # ExperimentManager used by Trainer.
    # =====================================================

    print()

    print("=" * 70)

    print(
        "STARTING TRAINING"
    )

    print("=" * 70)

    trainer.fit(
        train_loader,
        val_loader,
        NUM_EPOCHS,
        resume=True
    )

    # =====================================================
    # TRAINING COMPLETED
    # =====================================================

    print()

    print("=" * 70)

    print(
        "TRAINING ENTRY POINT COMPLETED"
    )

    print("=" * 70)


# =========================================================
# PYTHON ENTRY POINT
# =========================================================
#
# This ensures that main() executes only when this file
# is run directly as a module/script.
# =========================================================

if __name__ == "__main__":

    main()