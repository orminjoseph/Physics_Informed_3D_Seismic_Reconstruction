"""
=========================================================
Physics-Informed 3D Seismic Reconstruction
Training Launcher
=========================================================

Main training entry point for the Physics-Informed
3D Encoder–Decoder Framework.

This script

• Creates datasets
• Creates dataloaders
• Builds the model
• Creates optimizer
• Creates trainer
• Resumes previous checkpoint automatically
• Starts training

Author: Ormin Joseph
=========================================================
"""

import torch

from torch.optim import Adam

from dataset.synthetic_dataset import SyntheticSeismicDataset

from dataset.dataloader import create_dataloader

from models.network import Network3D

from losses.total_loss import TotalLoss

from trainer.trainer import Trainer

from utils.config import *


def main():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("=" * 60)
    print("Training Device")
    print("=" * 60)
    print(DEVICE)

    dataset = SyntheticSeismicDataset(
        num_samples=10,
        cube_size=(32, 64, 64)
    )

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, validation_size]
    )

    train_loader = create_dataloader(
        train_dataset,
        shuffle=True
    )

    validation_loader = create_dataloader(
        validation_dataset,
        shuffle=False
    )

    model = Network3D()

    criterion = TotalLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE
    )

    #trainer.load_checkpoint()

    trainer.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=NUM_EPOCHS,
        resume=False
    )
   

if __name__ == "__main__":
    main()
