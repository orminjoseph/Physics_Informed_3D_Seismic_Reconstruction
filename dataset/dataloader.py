"""
======================================================================
PYTORCH DATALOADER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Creates PyTorch DataLoaders for:

    1. Training
    2. Validation
    3. Testing

The DataLoader converts individual dataset samples into mini-batches
for neural-network training and evaluation.

Tensor convention
-----------------

Individual dataset sample:

    [C, D, H, W]

DataLoader batch:

    [B, C, D, H, W]

where:

    B = batch size
    C = channel
    D = depth
    H = crossline
    W = inline

The actual dataset construction is handled by:

    dataset/build_dataset.py

The training/validation split is handled by:

    dataset/split_dataset.py

This module is responsible only for DataLoader construction.

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

from torch.utils.data import DataLoader

from utils.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS
)


# =====================================================================
# DATALOADER FACTORY
# =====================================================================

def create_dataloader(
    dataset,
    batch_size=None,
    shuffle=True,
    num_workers=None
):
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset or Dataset subset to be loaded.

    batch_size : int, optional
        Number of samples per mini-batch.

        If None, BATCH_SIZE from utils.config is used.

    shuffle : bool, default=True
        Whether samples should be shuffled.

        Recommended:
            Training   -> True
            Validation -> False
            Testing    -> False

    num_workers : int, optional
        Number of worker processes used for loading data.

        If None, NUM_WORKERS from utils.config is used.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured PyTorch DataLoader.

    Notes
    -----
    The DataLoader does not modify the tensor dimensions explicitly.

    If the dataset returns:

        [C, D, H, W]

    PyTorch batching produces:

        [B, C, D, H, W]
    """

    # =================================================================
    # 1. VALIDATE DATASET
    # =================================================================

    if dataset is None:

        raise ValueError(
            "dataset cannot be None."
        )

    # =================================================================
    # 2. USE CONFIGURATION DEFAULTS
    # =================================================================

    if batch_size is None:

        batch_size = BATCH_SIZE

    if num_workers is None:

        num_workers = NUM_WORKERS

    # =================================================================
    # 3. VALIDATE BATCH SIZE
    # =================================================================

    if not isinstance(
        batch_size,
        int
    ):

        raise TypeError(
            "batch_size must be an integer."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be at least 1."
        )

    # =================================================================
    # 4. VALIDATE NUMBER OF WORKERS
    # =================================================================

    if not isinstance(
        num_workers,
        int
    ):

        raise TypeError(
            "num_workers must be an integer."
        )

    if num_workers < 0:

        raise ValueError(
            "num_workers cannot be negative."
        )

    # =================================================================
    # 5. CONFIGURE PERSISTENT WORKERS
    # =================================================================
    #
    # PyTorch requires:
    #
    #     persistent_workers=True
    #
    # only when:
    #
    #     num_workers > 0
    #
    # Therefore, even if PERSISTENT_WORKERS is True in config.py,
    # it will automatically be disabled when num_workers == 0.

    persistent_workers = (
        bool(PERSISTENT_WORKERS)
        and
        num_workers > 0
    )

    # =================================================================
    # 6. CREATE DATALOADER
    # =================================================================

    loader = DataLoader(

        dataset=dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=bool(
            PIN_MEMORY
        ),

        persistent_workers=persistent_workers
    )

    # =================================================================
    # 7. DISPLAY CONFIGURATION
    # =================================================================

    print()
    print("-" * 60)
    print("DATALOADER CREATED")
    print("-" * 60)

    print(
        f"Dataset size       : {len(dataset)}"
    )

    print(
        f"Batch size         : {batch_size}"
    )

    print(
        f"Shuffle            : {shuffle}"
    )

    print(
        f"Workers            : {num_workers}"
    )

    print(
        f"Pin memory         : {PIN_MEMORY}"
    )

    print(
        f"Persistent workers : {persistent_workers}"
    )

    print("-" * 60)
    print()

    # =================================================================
    # 8. RETURN DATALOADER
    # =================================================================

    return loader


# =====================================================================
# END OF MODULE
# =====================================================================
