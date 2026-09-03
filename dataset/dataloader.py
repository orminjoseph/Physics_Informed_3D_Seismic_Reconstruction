"""
=========================================================
PyTorch DataLoader
=========================================================

Creates PyTorch DataLoaders for:

    Training
    Validation
    Testing

This module converts Dataset objects into mini-batches
for neural-network training and evaluation.

Tensor convention:

    Dataset sample:
        [C, D, H, W]

    DataLoader batch:
        [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

from torch.utils.data import DataLoader

from utils.config import (
    PIN_MEMORY,
    PERSISTENT_WORKERS
)


def create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
):
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset or Dataset subset.

    batch_size : int
        Number of samples per mini-batch.

    shuffle : bool
        Whether samples should be shuffled.

    num_workers : int
        Number of worker processes used for loading data.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.
    """

    # =====================================================
    # Validate dataset
    # =====================================================

    if dataset is None:

        raise ValueError(
            "dataset cannot be None."
        )

    # =====================================================
    # Validate batch size
    # =====================================================

    if batch_size < 1:

        raise ValueError(
            "batch_size must be at least 1."
        )

    # =====================================================
    # Configure persistent workers
    # =====================================================

    # PyTorch requires num_workers > 0 when
    # persistent_workers is enabled.

    persistent_workers = (
        PERSISTENT_WORKERS
        and num_workers > 0
    )

    # =====================================================
    # Create DataLoader
    # =====================================================

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        # -------------------------------------------------
        # Pin memory is primarily useful when transferring
        # batches from CPU memory to a CUDA GPU.
        #
        # The value is controlled centrally in config.py.
        # -------------------------------------------------

        pin_memory=PIN_MEMORY,

        # -------------------------------------------------
        # Persistent workers are only valid when
        # num_workers > 0.
        # -------------------------------------------------

        persistent_workers=persistent_workers
    )

    return loader