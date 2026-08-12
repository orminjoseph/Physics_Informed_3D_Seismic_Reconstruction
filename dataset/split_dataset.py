from torch.utils.data import random_split

from utils.training_config import VALIDATION_SPLIT


def split_dataset(dataset):

    total_size = len(dataset)

    validation_size = int(
        total_size * VALIDATION_SPLIT
    )

    train_size = (
        total_size - validation_size
    )

    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [train_size, validation_size]
        )
    )

    return (
        train_dataset,
        validation_dataset
    )