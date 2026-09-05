"""
=========================================================
Nearest Neighbor Reconstruction Baseline
=========================================================

Simple spatial nearest-neighbor seismic reconstruction
baseline.

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


def nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
):
    """
    Reconstruct missing seismic voxels using spatial
    nearest-neighbor interpolation.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Corrupted seismic cube with shape:

            (C, D, H, W)

        where C is the channel dimension.

    mask : torch.Tensor
        Observation mask with the same shape as
        corrupted_cube.

            1 -> observed
            0 -> missing

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with the same shape
        as corrupted_cube.

    Notes
    -----
    The method searches for the nearest observed voxel
    using Euclidean distance in the 3D spatial domain
    (D, H, W).

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

    # -----------------------------------------------------
    # Process each channel independently
    # -----------------------------------------------------

    channels, depth, height, width = corrupted_cube.shape

    for channel in range(channels):

        cube = corrupted_cube[channel]
        channel_mask = mask[channel]

        # -------------------------------------------------
        # Locate observed and missing voxels
        # -------------------------------------------------

        observed_positions = torch.nonzero(
            channel_mask == 1,
            as_tuple=False
        )

        missing_positions = torch.nonzero(
            channel_mask == 0,
            as_tuple=False
        )

        # -------------------------------------------------
        # If there are no missing voxels, nothing to do.
        # -------------------------------------------------

        if missing_positions.numel() == 0:
            continue

        # -------------------------------------------------
        # If there are no observed voxels, reconstruction
        # is impossible.
        # -------------------------------------------------

        if observed_positions.numel() == 0:
            raise ValueError(
                f"Channel {channel} contains no observed "
                "voxels. Nearest-neighbor reconstruction "
                "cannot be performed."
            )

        # -------------------------------------------------
        # Compute nearest observed voxel for each missing
        # voxel.
        #
        # torch.cdist computes Euclidean distances between:
        #
        #     missing voxel coordinates
        #
        # and
        #
        #     observed voxel coordinates.
        # -------------------------------------------------

        distances = torch.cdist(
            missing_positions.float(),
            observed_positions.float(),
            p=2
        )

        nearest_indices = torch.argmin(
            distances,
            dim=1
        )

        nearest_positions = observed_positions[
            nearest_indices
        ]

        # -------------------------------------------------
        # Copy values from nearest observed voxels.
        # -------------------------------------------------

        reconstructed[channel][
            missing_positions[:, 0],
            missing_positions[:, 1],
            missing_positions[:, 2]
        ] = cube[
            nearest_positions[:, 0],
            nearest_positions[:, 1],
            nearest_positions[:, 2]
        ]

    return reconstructed