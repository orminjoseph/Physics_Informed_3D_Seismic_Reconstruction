"""
=========================================================
Training and Validation DataLoaders
=========================================================

Builds the complete data-loading pipeline:

    Dataset
       ↓
    Train/Validation Split
       ↓
    Training DataLoader
    Validation DataLoader

Author: Ormin Joseph
=========================================================
"""

from dataset.build_dataset import (
    build_dataset
)

from dataset.split_dataset import (
    split_dataset
)

from dataset.dataloader import (
    create_dataloader
)

from utils.config import (
    BATCH_SIZE
)


def build_dataloaders():
    """
    Build training and validation DataLoaders.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training DataLoader.

    validation_loader : torch.utils.data.DataLoader
        Validation DataLoader.
    """

    # =====================================================
    # 1. Build complete dataset
    # =====================================================

    dataset = build_dataset()

    # =====================================================
    # 2. Split dataset
    # =====================================================

    (
        train_dataset,
        validation_dataset
    ) = split_dataset(
        dataset
    )

    # =====================================================
    # 3. Create training DataLoader
    # =====================================================

    train_loader = create_dataloader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    # =====================================================
    # 4. Create validation DataLoader
    # =====================================================

    validation_loader = create_dataloader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    # =====================================================
    # 5. Return both loaders
    # =====================================================

    return (
        train_loader,
        validation_loader
    )