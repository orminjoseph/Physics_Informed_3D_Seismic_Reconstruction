"""
=========================================================
Test Trainer
=========================================================
"""

import torch

from dataset.generated_dataset import SeismicDataset
from torch.utils.data import DataLoader

from models.network import Network3D
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer


def test_trainer():

    print()

    print("=" * 60)
    print("Testing Trainer")
    print("=" * 60)

    dataset = SeismicDataset(
        dataset_directory="datasets"
    )

    train_dataloader = DataLoader(

        dataset,

        batch_size=2,

        shuffle=True

    )
    validation_dataloader = DataLoader(

        dataset,

        batch_size=2,

        shuffle=False

    )


    model = Network3D()

    criterion = TotalLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=1e-3

    )

    trainer = Trainer(

        model,

        criterion,

        optimizer,

        device="cpu"

    )

    trainer.fit(

        train_dataloader,

        validation_dataloader,

        epochs=2

    )


    print()

    print("Trainer Test: PASSED")


def main():

    test_trainer()


if __name__ == "__main__":

    main()