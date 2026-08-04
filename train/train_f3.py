"""
=========================================================
F3 Netherlands Training Script
=========================================================
"""

import torch

from torch.optim import Adam

from dataset.f3_dataset import F3Dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D

from losses.total_loss import TotalLoss

from trainer.trainer import Trainer

from utils.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RESUME_TRAINING
)

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


def main():

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("=" * 60)
    print("TRAINING DEVICE")
    print("=" * 60)
    print(DEVICE)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    from torch.utils.data import Subset

    dataset = Subset(
        dataset,
        range(100)
    )
    print()
    print("=" * 60)
    print("DEVELOPMENT DATASET")
    print("=" * 60)
    print("Number of Patches :", len(dataset))

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = (
        torch.utils.data.random_split(
            dataset,
            [train_size, validation_size]
        )
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

    trainer.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=NUM_EPOCHS,
        resume=RESUME_TRAINING
    )

if __name__ == "__main__":
    main()