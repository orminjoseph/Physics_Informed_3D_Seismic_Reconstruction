"""
=========================================================
Test Factory
=========================================================

Creates reusable objects for testing the
Physics-Informed 3D Encoder-Decoder Framework.

Author: Ormin Joseph
=========================================================
"""

import torch
from torch.utils.data import DataLoader

from models.network import Network3D
from dataset.synthetic_dataset import SyntheticSeismicDataset
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer
from utils.config import DEVICE


# =========================================================
# MODEL
# =========================================================

def create_model():
    """
    Create the Physics-Informed 3D Encoder-Decoder model.
    """

    return Network3D()


# =========================================================
# OPTIMIZER
# =========================================================

def create_optimizer(model):
    """
    Create the Adam optimizer.
    """

    return torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )


# =========================================================
# LOSS
# =========================================================

def create_loss():
    """
    Create the hybrid loss function.

    The synthetic test dataset uses unit spatial
    sampling intervals for the physics-loss test.
    """

    dx = 1.0   # Inline spacing [m]
    dy = 1.0   # Crossline spacing [m]
    dz = 1.0   # Depth spacing [m]

    return TotalLoss(
        dx=dx,
        dy=dy,
        dz=dz
    )

# =========================================================
# TRAINER
# =========================================================

def create_trainer():
    """
    Create a complete Trainer object.

    Trainer expects:

        model
        criterion
        optimizer
        device
    """

    model = create_model()

    optimizer = create_optimizer(
        model
    )

    loss_function = create_loss()

    return Trainer(
        model=model,
        criterion=loss_function,
        optimizer=optimizer,
        device=DEVICE
    )


# =========================================================
# DATASET
# =========================================================

def create_dataset(
    num_samples=8,
    cube_size=(64, 128, 128),
    missing_probability=0.30
):
    """
    Create a synthetic seismic dataset.
    """

    return SyntheticSeismicDataset(
        num_samples=num_samples,
        cube_size=cube_size,
        missing_probability=missing_probability
    )


# =========================================================
# DATALOADERS
# =========================================================

def create_dataloader(
    batch_size=2,
    train_samples=8,
    validation_samples=4
):
    """
    Create training and validation DataLoaders.

    Returns
    -------
    train_loader
        Training DataLoader.

    validation_loader
        Validation DataLoader.
    """

    train_dataset = create_dataset(
        num_samples=train_samples
    )

    validation_dataset = create_dataset(
        num_samples=validation_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader
    )
