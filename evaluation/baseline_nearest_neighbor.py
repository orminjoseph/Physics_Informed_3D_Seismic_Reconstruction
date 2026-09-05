"""
=========================================================
Nearest Neighbor Reconstruction Baseline
=========================================================

Efficient spatial nearest-neighbor seismic reconstruction.

For each missing voxel, the value of the nearest observed
voxel is used for reconstruction.

Expected input convention:

    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

where:

    mask == 1 -> observed voxel
    mask == 0 -> missing voxel

The reconstruction preserves all observed voxels and
replaces only missing voxels.

Author: Ormin Joseph
=========================================================
"""

import torch
import numpy as np

from scipy.ndimage import distance_transform_edt


def nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
):
    """
    Reconstruct missing seismic voxels using efficient
    spatial nearest-neighbor interpolation.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Corrupted seismic cube with shape:

            (C, D, H, W)

    mask : torch.Tensor
        Observation mask with the same shape.

            1 -> observed
            0 -> missing

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with the same shape.

    Notes
    -----
    The method uses a Euclidean distance transform to
    identify the nearest observed voxel for each missing
    voxel.

    Observed voxels are left unchanged.
    """

    # -----------------------------------------------------
    # Validate input types
    # -----------------------------------------------------

    if not isinstance(corrupted_cube, torch.Tensor):
        corrupted_cube = torch.as_tensor(
            corrupted_cube,
            dtype=torch.float32
        )

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(
            mask,
            dtype=torch.float32
        )

    # -----------------------------------------------------
    # Preserve the original device
    # -----------------------------------------------------

    original_device = corrupted_cube.device

    corrupted_cube = corrupted_cube.float()
    mask = mask.float()

    # -----------------------------------------------------
    # Validate dimensions
    # -----------------------------------------------------

    if corrupted_cube.ndim != 4:
        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received shape: {tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received shape: {tuple(mask.shape)}"
        )

    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have identical "
            "shapes. "
            f"Received {tuple(corrupted_cube.shape)} and "
            f"{tuple(mask.shape)}."
        )

    # -----------------------------------------------------
    # Validate numerical values
    # -----------------------------------------------------

    if not torch.isfinite(corrupted_cube).all():
        raise ValueError(
            "corrupted_cube contains non-finite values."
        )

    if not torch.isfinite(mask).all():
        raise ValueError(
            "mask contains non-finite values."
        )

    # -----------------------------------------------------
    # Create reconstruction
    # -----------------------------------------------------

    reconstructed = corrupted_cube.clone()

    channels, depth, height, width = corrupted_cube.shape

    # -----------------------------------------------------
    # Process each channel independently
    # -----------------------------------------------------

    for channel in range(channels):

        cube = corrupted_cube[channel]
        channel_mask = mask[channel]

        # -------------------------------------------------
        # Convert the observation mask to NumPy.
        #
        # The distance-transform operation is performed
        # on the CPU. The final reconstruction is returned
        # to the original device.
        # -------------------------------------------------

        mask_numpy = channel_mask.detach().cpu().numpy()

        observed = mask_numpy == 1
        missing = mask_numpy == 0

        # -------------------------------------------------
        # If there are no missing voxels, nothing to do.
        # -------------------------------------------------

        if not missing.any():
            continue

        # -------------------------------------------------
        # If there are no observed voxels, reconstruction
        # is impossible.
        # -------------------------------------------------

        if not observed.any():
            raise ValueError(
                f"Channel {channel} contains no observed "
                "voxels. Nearest-neighbor reconstruction "
                "cannot be performed."
            )

        # -------------------------------------------------
        # Compute the nearest observed voxel.
        #
        # distance_transform_edt identifies, for every
        # missing voxel, the nearest zero-valued voxel.
        #
        # Therefore:
        #
        #     observed == True  -> 0
        #     missing == True   -> 1
        #
        # The returned indices point to the nearest
        # observed voxel.
        # -------------------------------------------------

        _, nearest_indices = distance_transform_edt(
            missing,
            return_distances=True,
            return_indices=True
        )

        # -------------------------------------------------
        # Convert nearest-neighbor indices to tensors.
        # -------------------------------------------------

        nearest_positions = torch.from_numpy(
            nearest_indices
        ).long()

        # -------------------------------------------------
        # Copy the nearest observed values.
        #
        # The indices have shape:
        #
        #     (3, D, H, W)
        #
        # corresponding to:
        #
        #     depth, height, width
        # -------------------------------------------------

        nearest_values = cube[
            nearest_positions[0],
            nearest_positions[1],
            nearest_positions[2]
        ]

        # -------------------------------------------------
        # Replace only missing voxels.
        # -------------------------------------------------

        reconstructed[channel][missing] = nearest_values[missing]

    # -----------------------------------------------------
    # Return reconstruction on the original device.
    # -----------------------------------------------------

    return reconstructed.to(original_device)