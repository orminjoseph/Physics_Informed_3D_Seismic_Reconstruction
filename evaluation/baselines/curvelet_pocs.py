"""
======================================================================
CURVELET POCS RECONSTRUCTION BASELINE
======================================================================

Curvelet-domain Projection Onto Convex Sets (POCS) baseline for
3-D seismic data reconstruction.

This implementation uses the Uniform Discrete Curvelet Transform
(UDCT) provided by the Python curvelets package.

Transform:
    UDCT.forward()
    UDCT.backward()

Reconstruction strategy:
    1. Start from the incomplete seismic volume.
    2. Transform the current estimate into the curvelet domain.
    3. Apply coefficient thresholding.
    4. Transform back to the seismic domain.
    5. Enforce exact observed-data consistency.
    6. Repeat until the maximum number of iterations is reached
       or convergence is achieved.

Important:
    Observed seismic samples are NEVER modified.

Input convention:
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask convention:
    1 = observed sample
    0 = missing sample

Output:
    reconstructed_cube : (C, D, H, W)

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import numpy as np
import torch

from curvelets.numpy import UDCT


# =====================================================================
# INTERNAL VALIDATION
# =====================================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Validate the corrupted seismic cube and observation mask.
    """

    # ---------------------------------------------------------------
    # Validate tensor types.
    # ---------------------------------------------------------------

    if not isinstance(corrupted_cube, torch.Tensor):
        raise TypeError(
            "corrupted_cube must be a torch.Tensor."
        )

    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    # ---------------------------------------------------------------
    # Validate dimensionality.
    #
    # Expected:
    #     (C, D, H, W)
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Validate identical shapes.
    # ---------------------------------------------------------------

    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have identical shapes. "
            f"Received {tuple(corrupted_cube.shape)} and "
            f"{tuple(mask.shape)}."
        )

    # ---------------------------------------------------------------
    # Validate finite input values.
    # ---------------------------------------------------------------

    if not torch.isfinite(corrupted_cube).all():
        raise ValueError(
            "corrupted_cube contains NaN or infinite values."
        )

    # ---------------------------------------------------------------
    # Validate finite mask values.
    # ---------------------------------------------------------------

    if not torch.isfinite(mask).all():
        raise ValueError(
            "mask contains NaN or infinite values."
        )

    # ---------------------------------------------------------------
    # Validate binary mask.
    # ---------------------------------------------------------------

    unique_mask_values = torch.unique(mask)

    if not torch.all(
        (unique_mask_values == 0)
        | (unique_mask_values == 1)
    ):
        raise ValueError(
            "mask must contain only 0 and 1 values. "
            f"Received: {unique_mask_values.tolist()}"
        )


# =====================================================================
# CURVELET POCS FOR ONE CHANNEL
# =====================================================================

def _curvelet_pocs_channel(
    corrupted_channel: np.ndarray,
    mask_channel: np.ndarray,
    transform: UDCT,
    iterations: int,
    threshold: float,
    threshold_decay: float,
    tolerance: float,
):
    """
    Perform Curvelet-domain POCS reconstruction for one channel.

    Parameters
    ----------
    corrupted_channel : np.ndarray
        Incomplete seismic volume with shape (D, H, W).

    mask_channel : np.ndarray
        Binary observation mask with shape (D, H, W).

    transform : UDCT
        Initialized 3-D UDCT transform.

    iterations : int
        Maximum number of POCS iterations.

    threshold : float
        Initial curvelet coefficient soft-threshold.

    threshold_decay : float
        Multiplicative threshold decay after each iteration.

    tolerance : float
        Relative convergence tolerance.

    Returns
    -------
    np.ndarray
        Reconstructed seismic volume.
    """

    # ---------------------------------------------------------------
    # Convert inputs to float32.
    # ---------------------------------------------------------------

    current = np.asarray(
        corrupted_channel,
        dtype=np.float32,
    ).copy()

    mask_channel = np.asarray(
        mask_channel,
        dtype=np.float32,
    )

    # ---------------------------------------------------------------
    # Preserve the original observed seismic samples.
    # ---------------------------------------------------------------

    observed_data = current.copy()

    # ---------------------------------------------------------------
    # Initial threshold.
    # ---------------------------------------------------------------

    current_threshold = float(threshold)

    # ---------------------------------------------------------------
    # POCS iterations.
    # ---------------------------------------------------------------

    for _ in range(iterations):

        # -----------------------------------------------------------
        # Forward UDCT transform.
        # -----------------------------------------------------------

        coefficients = transform.forward(
            current
        )

        # -----------------------------------------------------------
        # Apply coefficient soft-thresholding.
        #
        # UDCT returns a structured coefficient object.
        # The package exposes its vector representation through
        # the vect() method.
        #
        # The coefficient structure itself is retained so that
        # reconstruction can be performed through backward().
        # -----------------------------------------------------------

        coefficient_vector = transform.vect(coefficients)

        coefficient_vector = np.asarray(
            coefficient_vector
        )

        # -----------------------------------------------------------
        # Soft thresholding.
        #
        # Works for real or complex coefficients.
        # -----------------------------------------------------------

        magnitude = np.abs(
            coefficient_vector
        )

        shrink_factor = np.maximum(
            1.0
            - current_threshold / (
                magnitude + 1.0e-12
            ),
            0.0,
        )

        thresholded_vector = (
            coefficient_vector
            * shrink_factor
        )

        # -----------------------------------------------------------
        # Reconstruct the coefficient structure.
        # -----------------------------------------------------------

        coefficients_thresholded = (
            transform.struct(
                thresholded_vector
            )
        )

        # -----------------------------------------------------------
        # Backward UDCT transform.
        # -----------------------------------------------------------

        reconstructed = transform.backward(
            coefficients_thresholded
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=np.float32,
        )

        # -----------------------------------------------------------
        # Validate reconstruction.
        # -----------------------------------------------------------

        if not np.isfinite(
            reconstructed
        ).all():
            raise FloatingPointError(
                "Curvelet POCS produced NaN or infinite values."
            )

        # -----------------------------------------------------------
        # Projection onto the observed-data constraint set.
        #
        # Observed samples are restored EXACTLY.
        # -----------------------------------------------------------

        current = (
            mask_channel * observed_data
            + (1.0 - mask_channel)
            * reconstructed
        )

        # -----------------------------------------------------------
        # Calculate relative change.
        # -----------------------------------------------------------

        denominator = (
            np.linalg.norm(current.ravel())
            + 1.0e-12
        )

        relative_change = (
            np.linalg.norm(
                (
                    current
                    - reconstructed
                ).ravel()
            )
            / denominator
        )

        # -----------------------------------------------------------
        # Reduce threshold for the next iteration.
        # -----------------------------------------------------------

        current_threshold *= (
            threshold_decay
        )

        # -----------------------------------------------------------
        # Convergence check.
        # -----------------------------------------------------------

        if relative_change < tolerance:
            break

    # ---------------------------------------------------------------
    # Final exact observed-data projection.
    # ---------------------------------------------------------------

    current = (
        mask_channel * observed_data
        + (1.0 - mask_channel) * current
    )

    return current.astype(
        np.float32,
        copy=False,
    )


# =====================================================================
# PUBLIC CURVELET POCS FUNCTION
# =====================================================================

def curvelet_pocs_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    num_scales: int = 3,
    wedges_per_direction: int = 3,
    iterations: int = 12,
    threshold: float = 0.05,
    threshold_decay: float = 0.90,
    tolerance: float = 1.0e-5,
):
    """
    Reconstruct a 3-D seismic cube using Curvelet POCS.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Incomplete seismic cube with shape (C, D, H, W).

    mask : torch.Tensor
        Binary observation mask with shape (C, D, H, W).

    num_scales : int
        Number of UDCT scales including the lowpass scale.

    wedges_per_direction : int
        Number of angular wedges per direction at the coarsest scale.

    iterations : int
        Maximum number of POCS iterations.

    threshold : float
        Initial curvelet coefficient threshold.

    threshold_decay : float
        Threshold decay factor.

    tolerance : float
        Relative convergence tolerance.

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with shape (C, D, H, W).
    """

    # =================================================================
    # VALIDATE INPUTS
    # =================================================================

    _validate_inputs(
        corrupted_cube,
        mask,
    )

    # =================================================================
    # VALIDATE PARAMETERS
    # =================================================================

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must be >= 0."
        )

    if not (
        0.0 < threshold_decay <= 1.0
    ):
        raise ValueError(
            "threshold_decay must satisfy "
            "0 < threshold_decay <= 1."
        )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be > 0."
        )

    if num_scales < 2:
        raise ValueError(
            "num_scales must be >= 2."
        )

    if wedges_per_direction < 3:
        raise ValueError(
            "wedges_per_direction must be >= 3."
        )

    # =================================================================
    # SAVE ORIGINAL DEVICE AND DTYPE
    # =================================================================

    original_device = corrupted_cube.device
    original_dtype = corrupted_cube.dtype

    # =================================================================
    # CONVERT TO CPU NUMPY
    # =================================================================

    corrupted_np = (
        corrupted_cube
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    mask_np = (
        mask
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    # =================================================================
    # SPATIAL VOLUME SHAPE
    # =================================================================

    volume_shape = tuple(
        corrupted_np.shape[1:]
    )

    # =================================================================
    # CREATE ONE 3-D UDCT TRANSFORM
    #
    # The transform is shared across channels because all channels
    # have identical spatial dimensions.
    # =================================================================

    transform = UDCT(
        shape=volume_shape,
        num_scales=num_scales,
        wedges_per_direction=wedges_per_direction,
        transform_kind="real",
    )

    # =================================================================
    # RECONSTRUCT CHANNELS
    # =================================================================

    reconstructed_np = np.empty_like(
        corrupted_np,
        dtype=np.float32,
    )

    for channel_index in range(
        corrupted_np.shape[0]
    ):

        reconstructed_np[
            channel_index
        ] = _curvelet_pocs_channel(
            corrupted_channel=(
                corrupted_np[channel_index]
            ),
            mask_channel=(
                mask_np[channel_index]
            ),
            transform=transform,
            iterations=iterations,
            threshold=threshold,
            threshold_decay=threshold_decay,
            tolerance=tolerance,
        )

    # =================================================================
    # FINAL OBSERVED-DATA PRESERVATION
    # =================================================================

    reconstructed_np = (
        mask_np * corrupted_np
        + (1.0 - mask_np)
        * reconstructed_np
    )

    # =================================================================
    # CONVERT BACK TO TORCH
    # =================================================================

    reconstructed_cube = torch.from_numpy(
        reconstructed_np
    )

    # =================================================================
    # RESTORE ORIGINAL DEVICE
    # =================================================================

    reconstructed_cube = (
        reconstructed_cube
        .to(
            device=original_device,
            dtype=original_dtype,
        )
    )

    # =================================================================
    # FINAL VALIDATION
    # =================================================================

    if not torch.isfinite(
        reconstructed_cube
    ).all():
        raise FloatingPointError(
            "Final Curvelet POCS reconstruction "
            "contains NaN or infinite values."
        )

    # =================================================================
    # FINAL OBSERVED-DATA CHECK
    # =================================================================

    observed_difference = (
        (
            reconstructed_cube
            - corrupted_cube
        )
        * mask
    ).abs().max()

    if observed_difference > 1.0e-6:
        raise RuntimeError(
            "Observed seismic samples were not "
            "preserved exactly. "
            f"Maximum difference: "
            f"{observed_difference.item():.6e}"
        )

    return reconstructed_cube