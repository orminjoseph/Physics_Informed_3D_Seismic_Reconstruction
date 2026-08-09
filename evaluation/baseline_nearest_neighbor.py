"""
=========================================================
Nearest Neighbor Reconstruction Baseline
=========================================================

Simple seismic reconstruction baseline.

Author: Ormin Joseph
=========================================================
"""

import torch


def nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
):
    """
    Replace missing voxels with a simple global estimate.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Shape (1, D, H, W)

    mask : torch.Tensor
        Shape (1, D, H, W)

    Returns
    -------
    reconstructed_cube
    """

    reconstructed = corrupted_cube.clone()

    known_values = reconstructed[mask == 1]

    if len(known_values) == 0:
        return reconstructed

    fill_value = known_values.mean()

    reconstructed[mask == 0] = fill_value

    return reconstructed