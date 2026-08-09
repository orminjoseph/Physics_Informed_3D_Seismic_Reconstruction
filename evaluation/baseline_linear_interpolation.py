"""
=========================================================
Linear Interpolation Baseline
=========================================================

Simple seismic interpolation baseline.

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn.functional as F


def linear_interpolation_reconstruction(
        corrupted_cube,
        mask
):
    """
    Simple 3D smoothing interpolation.

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

    # add batch dimension
    x = reconstructed.unsqueeze(0)

    # average neighboring voxels
    smoothed = F.avg_pool3d(
        x,
        kernel_size=3,
        stride=1,
        padding=1
    )

    smoothed = smoothed.squeeze(0)

    reconstructed[mask == 0] = smoothed[mask == 0]

    return reconstructed