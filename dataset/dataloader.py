"""
=========================================================
PyTorch DataLoader
=========================================================

Creates PyTorch DataLoaders for

• Training
• Validation
• Testing

This module wraps the SeismicDataset and provides
mini-batches for efficient deep learning training.

Author: Ormin Joseph
=========================================================
"""

from torch.utils.data import DataLoader


def create_dataloader(

        dataset,

        batch_size=2,

        shuffle=True,

        num_workers=0

):
    """
    Create a PyTorch DataLoader
    from an already-created Dataset.
    """

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=True

    )
