"""
=========================================================
Dataset Splitter
=========================================================

Splits a seismic dataset into training and validation
subsets.

The validation size is controlled by VALIDATION_SPLIT
from the global configuration.

Author: Ormin Joseph
=========================================================
"""

from torch.utils.data import random_split

from utils.config import (
    VALIDATION_SPLIT,
    SEED
)


def split_dataset(dataset):
    """
    Split a dataset into training and validation subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Complete seismic dataset.

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        Training subset.

    validation_dataset : torch.utils.data.Subset
        Validation subset.
    """

    # =====================================================
    # Validate dataset
    # =====================================================

    total_size = len(dataset)

    if total_size < 2:

        raise ValueError(
            "Dataset must contain at least 2 samples "
            "to create training and validation subsets."
        )

    # =====================================================
    # Calculate validation size
    # =====================================================

    validation_size = max(
        1,
        int(total_size * VALIDATION_SPLIT)
    )

    # =====================================================
    # Prevent validation set from consuming the
    # entire dataset
    # =====================================================

    validation_size = min(
        validation_size,
        total_size - 1
    )

    # =====================================================
    # Calculate training size
    # =====================================================

    train_size = (
        total_size
        -
        validation_size
    )

    # =====================================================
    # Reproducible random generator
    # =====================================================

    generator = None

    if SEED is not None:

        import torch

        generator = torch.Generator()

        generator.manual_seed(SEED)

    # =====================================================
    # Perform split
    # =====================================================

    train_dataset, validation_dataset = random_split(

        dataset,

        [
            train_size,
            validation_size
        ],

        generator=generator
    )

    # =====================================================
    # Return datasets
    # =====================================================

    return (
        train_dataset,
        validation_dataset
    )