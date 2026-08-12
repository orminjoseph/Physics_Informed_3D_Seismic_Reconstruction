from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from utils.training_config import (
    BATCH_SIZE
)


def build_dataloaders():

    dataset = build_dataset()

    train_dataset, validation_dataset = (
        split_dataset(dataset)
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader
    )