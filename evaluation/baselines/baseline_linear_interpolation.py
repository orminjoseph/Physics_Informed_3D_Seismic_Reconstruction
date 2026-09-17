"""
=========================================================
Linear Interpolation Baseline
=========================================================

Simple 3D linear interpolation baseline for seismic
reconstruction.

The method estimates missing voxels from observed seismic
values using separable linear interpolation along the three
spatial dimensions.

Expected input convention:

    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

where:

    mask == 1 -> observed voxel
    mask == 0 -> missing voxel

Observed voxels are preserved and only missing voxels are
reconstructed.

Author: Ormin Joseph
=========================================================
"""

import torch


def _interpolate_along_dimension(
        cube,
        observed_mask,
        dimension
):
    """
    Perform one-dimensional linear interpolation along
    a specified spatial dimension.

    Parameters
    ----------
    cube : torch.Tensor
        Seismic cube with shape (D, H, W).

    observed_mask : torch.Tensor
        Boolean observation mask with shape (D, H, W).

    dimension : int
        Spatial dimension along which interpolation is
        performed:

            0 -> depth
            1 -> height
            2 -> width

    Returns
    -------
    torch.Tensor
        Interpolated cube.
    """

    result = cube.clone()

    # Move interpolation dimension to the last axis.
    values = result.movedim(dimension, -1)
    mask = observed_mask.movedim(dimension, -1)

    original_shape = values.shape

    # Flatten all dimensions except the interpolation axis.
    values = values.reshape(-1, values.shape[-1])
    mask = mask.reshape(-1, mask.shape[-1])

    axis = torch.arange(
        values.shape[-1],
        device=values.device,
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Process every 1-D seismic profile independently.
    # ---------------------------------------------------------

    for row in range(values.shape[0]):

        observed = mask[row]

        observed_indices = torch.nonzero(
            observed,
            as_tuple=False
        ).flatten()

        # -----------------------------------------------------
        # CASE 1:
        # No observed samples exist on this profile.
        #
        # Linear interpolation is impossible because there is
        # no value from which to interpolate.
        #
        # IMPORTANT:
        # Do not index observed_positions in this case.
        # Leave this profile unchanged for this dimension.
        # Subsequent spatial dimensions may still provide
        # information for reconstruction.
        # -----------------------------------------------------

        if observed_indices.numel() == 0:
            continue

        # -----------------------------------------------------
        # CASE 2:
        # All samples are already observed.
        # Nothing needs to be interpolated.
        # -----------------------------------------------------

        if observed_indices.numel() == values.shape[1]:
            continue

        # -----------------------------------------------------
        # CASE 3:
        # Only one observed sample exists.
        #
        # True linear interpolation cannot be calculated from
        # a single point. Therefore, use that observed value
        # to fill the missing locations on this profile.
        # -----------------------------------------------------

        if observed_indices.numel() == 1:

            values[row, ~observed] = values[
                row,
                observed_indices[0]
            ]

            continue

        # -----------------------------------------------------
        # CASE 4:
        # Two or more observed samples exist.
        #
        # Normal piecewise linear interpolation can now be
        # performed.
        # -----------------------------------------------------

        observed_positions = observed_indices.to(
            dtype=torch.float32
        )

        observed_values = values[
            row,
            observed_indices
        ]

        missing = ~observed

        missing_positions = torch.nonzero(
            missing,
            as_tuple=False
        ).flatten()

        if missing_positions.numel() == 0:
            continue

        missing_positions_float = missing_positions.to(
            dtype=torch.float32
        )

        # -----------------------------------------------------
        # Find the nearest observed points on either side
        # of each missing location.
        # -----------------------------------------------------

        right_index = torch.searchsorted(
            observed_positions,
            missing_positions_float
        )

        # Clamp to valid interpolation intervals.
        #
        # Minimum = 1 ensures that left_index = right_index - 1
        # remains valid.
        #
        # Maximum = N-1 ensures that right_index is valid.
        #
        # Values outside the observed range are handled below
        # using nearest observed values.
        # -----------------------------------------------------

        right_index = torch.clamp(
            right_index,
            min=1,
            max=observed_positions.numel() - 1
        )

        left_index = right_index - 1

        x0 = observed_positions[left_index]
        x1 = observed_positions[right_index]

        y0 = observed_values[left_index]
        y1 = observed_values[right_index]

        # -----------------------------------------------------
        # Linear interpolation:
        #
        # y = y0 + (x-x0)/(x1-x0) * (y1-y0)
        # -----------------------------------------------------

        denominator = x1 - x0

        interpolated = (
            y0
            + (
                (missing_positions_float - x0)
                / denominator
            )
            * (y1 - y0)
        )

        # -----------------------------------------------------
        # Handle missing values outside the observed range.
        #
        # Linear interpolation cannot be performed beyond
        # the first/last observed point, so nearest observed
        # values are used there.
        # -----------------------------------------------------

        before_first = (
            missing_positions_float
            < observed_positions[0]
        )

        after_last = (
            missing_positions_float
            > observed_positions[-1]
        )

        interpolated[before_first] = observed_values[0]
        interpolated[after_last] = observed_values[-1]

        # -----------------------------------------------------
        # Write interpolated values only into missing locations.
        # -----------------------------------------------------

        values[
            row,
            missing_positions
        ] = interpolated

    # Restore original shape and dimension order.
    values = values.reshape(original_shape)

    return values.movedim(-1, dimension)


def linear_interpolation_reconstruction(
        corrupted_cube,
        mask
):
    """
    Reconstruct missing seismic voxels using linear
    interpolation along the three spatial dimensions.

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
    The method:

        1. Preserves observed voxels.
        2. Interpolates along depth.
        3. Interpolates along height.
        4. Interpolates along width.

    Missing locations that cannot be bracketed by observed
    samples are assigned the nearest available observed
    value when at least two observed samples exist.

    If an individual 1-D profile contains no observed samples,
    that profile is left unchanged for that interpolation
    dimension because interpolation is mathematically
    impossible without an observed reference value.
    """

    # ---------------------------------------------------------
    # Convert inputs to tensors when necessary.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Validate dimensions.
    # ---------------------------------------------------------

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:

        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(mask.shape)}"
        )

    if corrupted_cube.shape != mask.shape:

        raise ValueError(
            "corrupted_cube and mask must have identical "
            "shapes. "
            f"Received {tuple(corrupted_cube.shape)} and "
            f"{tuple(mask.shape)}."
        )

    # ---------------------------------------------------------
    # Validate numerical values.
    # ---------------------------------------------------------

    if not torch.isfinite(corrupted_cube).all():

        raise ValueError(
            "corrupted_cube contains non-finite values."
        )

    if not torch.isfinite(mask).all():

        raise ValueError(
            "mask contains non-finite values."
        )

    # ---------------------------------------------------------
    # Convert mask to Boolean observation mask.
    # ---------------------------------------------------------

    observed_mask = mask == 1

    # ---------------------------------------------------------
    # Check that at least one observed voxel exists somewhere
    # in the entire cube.
    # ---------------------------------------------------------

    if not observed_mask.any():

        raise ValueError(
            "The mask contains no observed voxels. "
            "Linear interpolation cannot be performed."
        )

    # ---------------------------------------------------------
    # Start with the original corrupted cube.
    # ---------------------------------------------------------

    reconstructed = corrupted_cube.clone()

    # ---------------------------------------------------------
    # Process every channel independently.
    # ---------------------------------------------------------

    for channel in range(corrupted_cube.shape[0]):

        cube = reconstructed[channel]

        channel_mask = observed_mask[channel]

        # -----------------------------------------------------
        # If the channel is already complete, leave it unchanged.
        # -----------------------------------------------------

        if channel_mask.all():
            continue

        # -----------------------------------------------------
        # Interpolate successively along:
        #
        #   depth  -> dimension 0
        #   height -> dimension 1
        #   width  -> dimension 2
        #
        # The same observation mask is deliberately retained.
        # It represents the ORIGINAL measured-data locations.
        # -----------------------------------------------------

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=0
        )

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=1
        )

        cube = _interpolate_along_dimension(
            cube,
            channel_mask,
            dimension=2
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Restore the original observed values exactly.
        #
        # Interpolation must never modify measured seismic data.
        # -----------------------------------------------------

        cube[channel_mask] = corrupted_cube[
            channel
        ][channel_mask]

        reconstructed[channel] = cube

    # ---------------------------------------------------------
    # Final numerical validation.
    # ---------------------------------------------------------

    if not torch.isfinite(reconstructed).all():

        raise RuntimeError(
            "Linear interpolation produced non-finite "
            "values."
        )

    return reconstructed