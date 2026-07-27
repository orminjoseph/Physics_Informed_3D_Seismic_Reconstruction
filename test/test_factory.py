"""
=========================================================
Test Factory
=========================================================

Creates reusable objects for testing.

Author: Ormin Joseph
=========================================================
"""

import torch
from torch.utils.data import DataLoader

from models.network import PhysicsInformed3DUNet
from dataset.synthetic_dataset import SyntheticSeismicDataset
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer
from utils.config import DEVICE


def create_model():
    """
    Create a Physics-Informed 3D U-Net.
    """

    return PhysicsInformed3DUNet()


def create_optimizer(model):
    """
    Create the Adam optimizer.
    """

    return torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )


def create_loss():
    """
    Create the hybrid loss function.
    """

    return TotalLoss()


def create_trainer():
    """
    Create a complete trainer.
    """

    model = create_model()

    optimizer = create_optimizer(model)

    loss_function = create_loss()

    return Trainer(
        model,
        optimizer,
        loss_function,
        DEVICE
    )


def create_dataset(
    num_samples=8,
    cube_size=(64, 128, 128),
    missing_probability=0.30
):
    """
    Create a synthetic dataset.
    """

    return SyntheticSeismicDataset(
        num_samples=num_samples,
        cube_size=cube_size,
        missing_probability=missing_probability
    )


def create_dataloader(
        batch_size=2,
        train_samples=8,
        validation_samples=4
):
    """
    Create training and validation DataLoaders.
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

    return train_loader, validation_loader


