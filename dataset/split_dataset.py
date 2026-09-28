"""
======================================================================
DATASET SPLITTER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Splits the complete dataset into:

    1. Training subset
    2. Validation subset

The validation fraction is controlled by:

    VALIDATION_SPLIT

from the global configuration file.

Reproducibility
---------------
The split is reproducible through the global:

    SEED

configuration parameter.

Important
---------
This module creates only the training and validation subsets.

An independent test dataset should remain separate from this
procedure and should be used only for final model evaluation.

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import torch

from torch.utils.data import random_split

from utils.config import (
    VALIDATION_SPLIT,
    SEED
)


# =====================================================================
# DATASET SPLITTING FUNCTION
# =====================================================================

def split_dataset(dataset):
    """
    Split a dataset into training and validation subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Complete dataset to be divided.

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        Training subset.

    validation_dataset : torch.utils.data.Subset
        Validation subset.

    Notes
    -----
    The validation size is determined by VALIDATION_SPLIT.

    The split is reproducible when SEED is defined.
    """

    # -----------------------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------------------

    if dataset is None:

        raise ValueError(
            "Dataset cannot be None."
        )

    total_size = len(dataset)

    if total_size < 2:

        raise ValueError(
            "Dataset must contain at least 2 samples "
            "to create training and validation subsets."
        )

    # -----------------------------------------------------------------
    # Validate validation split
    # -----------------------------------------------------------------

    if not 0.0 < VALIDATION_SPLIT < 1.0:

        raise ValueError(
            "VALIDATION_SPLIT must be greater than 0 "
            "and less than 1."
        )

    # -----------------------------------------------------------------
    # Calculate validation size
    # -----------------------------------------------------------------
    #
    # At least one sample is reserved for validation.
    #
    # However, the validation set must never consume the entire
    # dataset because at least one sample is required for training.

    validation_size = max(
        1,
        int(
            round(
                total_size
                *
                VALIDATION_SPLIT
            )
        )
    )

    validation_size = min(
        validation_size,
        total_size - 1
    )

    # -----------------------------------------------------------------
    # Calculate training size
    # -----------------------------------------------------------------

    train_size = (
        total_size
        -
        validation_size
    )

    # -----------------------------------------------------------------
    # Validate resulting split
    # -----------------------------------------------------------------

    if train_size < 1:

        raise RuntimeError(
            "Training subset contains no samples."
        )

    if validation_size < 1:

        raise RuntimeError(
            "Validation subset contains no samples."
        )

    # -----------------------------------------------------------------
    # Create reproducible random generator
    # -----------------------------------------------------------------

    generator = torch.Generator()

    if SEED is not None:

        generator.manual_seed(
            int(SEED)
        )

    # -----------------------------------------------------------------
    # Perform random split
    # -----------------------------------------------------------------

    train_dataset, validation_dataset = random_split(

        dataset,

        [
            train_size,
            validation_size
        ],

        generator=generator
    )

    # -----------------------------------------------------------------
    # Verify split sizes
    # -----------------------------------------------------------------

    if (
        len(train_dataset)
        +
        len(validation_dataset)
        !=
        total_size
    ):

        raise RuntimeError(
            "Training and validation subsets do not "
            "account for the complete dataset."
        )

    # -----------------------------------------------------------------
    # Display split information
    # -----------------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(
        f"Total samples      : {total_size}"
    )

    print(
        f"Training samples    : {len(train_dataset)}"
    )

    print(
        f"Validation samples  : {len(validation_dataset)}"
    )

    print(
        f"Validation fraction : "
        f"{len(validation_dataset) / total_size:.4f}"
    )

    print(
        f"Random seed         : {SEED}"
    )

    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # Return datasets
    # -----------------------------------------------------------------

    return (
        train_dataset,
        validation_dataset
    )


# =====================================================================
# END OF MODULE
# =====================================================================
