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

from dataset.seismic_dataset import SeismicDataset
def create_dataloader(
        dataset_directory="datasets",
        batch_size=2,
        shuffle=True,
        num_workers=0
):
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset_directory : str
        Folder containing the dataset.

    batch_size : int
        Number of samples per mini-batch.

    shuffle : bool
        Shuffle dataset before each epoch.

    num_workers : int
        Number of worker processes.

    Returns
    -------
    DataLoader
    """
    # ------------------------------------------
    # Create dataset
    # ------------------------------------------

    dataset = SeismicDataset(

        dataset_directory=dataset_directory

    )
    # ------------------------------------------
    # Create DataLoader
    # ------------------------------------------

    dataloader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=True

    )
    # ------------------------------------------
    # Return DataLoader
    # ------------------------------------------

    return dataloader