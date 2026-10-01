"""
=============================================================
f-x Prediction Seismic Interpolation Baseline
=============================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Classical seismic interpolation baseline based on
prediction-error filtering in the frequency-space (f-x)
domain.

Method
-------
The implementation performs:

    1. Fourier transformation along the temporal/depth axis.
    2. Identification of sufficiently observed spatial traces.
    3. Frequency-by-frequency spatial prediction.
    4. Forward and reverse spatial prediction.
    5. Combination of the spatial predictions.
    6. Inverse Fourier transformation.
    7. Exact observed-data consistency.
    8. Iterative refinement.

Tensor convention
-----------------
Input seismic cube:

    [C, D, H, W]

where:

    C = seismic channel
    D = temporal/depth dimension
    H = spatial dimension 1
    W = spatial dimension 2

Mask convention
---------------
    1 = observed
    0 = missing

Important methodological constraint
-----------------------------------
This baseline is independent of the proposed neural network.

The function uses ONLY:

    corrupted_cube
    mask

Ground-truth/target data are NEVER passed to the
reconstruction algorithm.

Important note
--------------
Classical f-x prediction is naturally designed for
missing-trace interpolation.

For arbitrary missing-voxel mechanisms, a spatial trace is
considered a reliable predictor only when its temporal/depth
samples satisfy the configured trace-observation criterion.

All originally observed samples are restored exactly.

Author: Ormin Joseph
=============================================================
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

# These values are module defaults.
#
# For the controlled experimental matrix, the corresponding
# experimental settings should be centralized in:
#
#     utils/config.py
#
# and supplied by the controlled-matrix evaluation script.

DEFAULT_PREDICTION_ORDER = 4

DEFAULT_ITERATIONS = 2

DEFAULT_MIN_TRACE_OBSERVED_FRACTION = 0.80

DEFAULT_CONVERGENCE_TOLERANCE = 1.0e-5


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and standardize the input seismic cube and mask.

    Parameters
    ----------
    corrupted_cube:
        Seismic cube with shape (C, D, H, W).

    mask:
        Binary observation mask with shape (C, D, H, W).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Validated floating-point tensors.
    """

    # --------------------------------------------------------
    # Convert NumPy arrays or array-like objects to tensors.
    # --------------------------------------------------------

    if not isinstance(corrupted_cube, torch.Tensor):

        corrupted_cube = torch.as_tensor(
            corrupted_cube,
            dtype=torch.float32,
        )

    if not isinstance(mask, torch.Tensor):

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32,
        )

    # --------------------------------------------------------
    # Preserve the original device of the seismic cube.
    #
    # Conversion to float32 is deliberate because the
    # classical NumPy implementation works in floating point.
    # --------------------------------------------------------

    corrupted_cube = corrupted_cube.float()

    mask = mask.to(
        device=corrupted_cube.device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Check dimensionality.
    #
    # Expected:
    #
    #     [C, D, H, W]
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Check shape agreement.
    # --------------------------------------------------------

    if corrupted_cube.shape != mask.shape:

        raise ValueError(
            "corrupted_cube and mask must have identical shapes. "
            f"Cube: {tuple(corrupted_cube.shape)}, "
            f"Mask: {tuple(mask.shape)}"
        )

    # --------------------------------------------------------
    # Check finite values.
    # --------------------------------------------------------

    if not torch.isfinite(corrupted_cube).all():

        raise ValueError(
            "corrupted_cube contains NaN or Inf values."
        )

    if not torch.isfinite(mask).all():

        raise ValueError(
            "mask contains NaN or Inf values."
        )

    # --------------------------------------------------------
    # Check binary mask.
    # --------------------------------------------------------

    if not torch.all(
        (mask == 0.0)
        | (mask == 1.0)
    ):

        unique_values = torch.unique(mask)

        raise ValueError(
            "mask must contain only 0.0 and 1.0. "
            f"Found values: {unique_values.tolist()}"
        )

    # --------------------------------------------------------
    # At least one observed sample must exist.
    # --------------------------------------------------------

    if not (mask == 1.0).any():

        raise ValueError(
            "The mask contains no observed seismic samples."
        )

    return corrupted_cube, mask


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def _validate_parameters(
    prediction_order: int,
    iterations: int,
    min_trace_observed_fraction: float,
    convergence_tolerance: float,
) -> None:
    """
    Validate f-x prediction parameters.
    """

    # --------------------------------------------------------
    # Prediction order.
    # --------------------------------------------------------

    if not isinstance(
        prediction_order,
        int,
    ):

        raise TypeError(
            "prediction_order must be an integer."
        )

    if prediction_order < 1:

        raise ValueError(
            "prediction_order must be >= 1."
        )

    # --------------------------------------------------------
    # Number of iterations.
    # --------------------------------------------------------

    if not isinstance(
        iterations,
        int,
    ):

        raise TypeError(
            "iterations must be an integer."
        )

    if iterations < 1:

        raise ValueError(
            "iterations must be >= 1."
        )

    # --------------------------------------------------------
    # Minimum trace observation fraction.
    # --------------------------------------------------------

    if not (
        0.0
        < min_trace_observed_fraction
        <= 1.0
    ):

        raise ValueError(
            "min_trace_observed_fraction must satisfy "
            "0 < value <= 1."
        )

    # --------------------------------------------------------
    # Convergence tolerance.
    # --------------------------------------------------------

    if convergence_tolerance <= 0.0:

        raise ValueError(
            "convergence_tolerance must be > 0."
        )


# ============================================================
# TRACE OBSERVATION ANALYSIS
# ============================================================

def _trace_observation_fraction(
    observation_mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate the observed fraction of every spatial trace.

    Parameters
    ----------
    observation_mask:
        Boolean mask with shape (D, H, W).

    Returns
    -------
    ndarray
        Trace observation fractions with shape (H, W).

    Notes
    -----
    The depth/time axis is D.

    Therefore:

        fraction[h, w]

    represents the proportion of temporal/depth samples
    observed along the spatial trace at (h, w).
    """

    return np.mean(
        observation_mask,
        axis=0,
    )


# ============================================================
# PREDICTION COEFFICIENT ESTIMATION
# ============================================================

def _estimate_prediction_coefficients(
    trace: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Estimate complex-valued linear prediction coefficients.

    The model is:

        x[n] ≈ a1*x[n-1] + ... + ap*x[n-p]

    Parameters
    ----------
    trace:
        Complex-valued spatial-frequency trace.

    order:
        Prediction filter order.

    Returns
    -------
    ndarray
        Complex-valued prediction coefficients.
    """

    # --------------------------------------------------------
    # Convert to complex representation.
    # --------------------------------------------------------

    trace = np.asarray(
        trace,
        dtype=np.complex128,
    )

    # --------------------------------------------------------
    # Remove non-finite values.
    # --------------------------------------------------------

    if not np.isfinite(trace).all():

        return np.zeros(
            order,
            dtype=np.complex128,
        )

    # --------------------------------------------------------
    # There must be enough samples for the predictor.
    # --------------------------------------------------------

    if trace.size <= order:

        return np.zeros(
            order,
            dtype=np.complex128,
        )

    # --------------------------------------------------------
    # Construct the least-squares system.
    # --------------------------------------------------------

    number_of_rows = trace.size - order

    A = np.empty(
        (
            number_of_rows,
            order,
        ),
        dtype=np.complex128,
    )

    b = np.empty(
        number_of_rows,
        dtype=np.complex128,
    )

    for row, index in enumerate(
        range(
            order,
            trace.size,
        )
    ):

        A[row, :] = trace[
            index - order:index
        ][::-1]

        b[row] = trace[index]

    # --------------------------------------------------------
    # Solve the complex least-squares problem.
    # --------------------------------------------------------

    try:

        coefficients, _, _, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )

    except np.linalg.LinAlgError:

        return np.zeros(
            order,
            dtype=np.complex128,
        )

    # --------------------------------------------------------
    # Numerical safety.
    # --------------------------------------------------------

    if not np.isfinite(coefficients).all():

        return np.zeros(
            order,
            dtype=np.complex128,
        )

    return coefficients


# ============================================================
# ONE-DIMENSIONAL F-X PREDICTION
# ============================================================

def _fx_predict_1d(
    spatial_trace: np.ndarray,
    observed: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Perform forward and reverse linear prediction on a
    one-dimensional spatial-frequency trace.

    Parameters
    ----------
    spatial_trace:
        Complex-valued spatial-frequency trace.

    observed:
        Boolean spatial observation mask.

    order:
        Prediction filter order.

    Returns
    -------
    ndarray
        Reconstructed complex-valued spatial trace.
    """

    # --------------------------------------------------------
    # Make independent copies.
    # --------------------------------------------------------

    values = np.asarray(
        spatial_trace,
        dtype=np.complex128,
    ).copy()

    observed = np.asarray(
        observed,
        dtype=bool,
    )

    # --------------------------------------------------------
    # No missing samples.
    # --------------------------------------------------------

    if observed.all():

        return values

    # --------------------------------------------------------
    # No observed samples.
    #
    # Prediction cannot be estimated from this trace.
    # --------------------------------------------------------

    if not observed.any():

        return values

    # --------------------------------------------------------
    # The coefficient estimation must use actual observed
    # spatial samples.
    # --------------------------------------------------------

    observed_values = values[observed]

    if observed_values.size <= order:

        return values

    # --------------------------------------------------------
    # Estimate prediction coefficients.
    # --------------------------------------------------------

    coefficients = (
        _estimate_prediction_coefficients(
            observed_values,
            order,
        )
    )

    # --------------------------------------------------------
    # Forward prediction.
    #
    # Missing positions are filled using the current
    # reconstructed spatial sequence.
    # --------------------------------------------------------

    forward = values.copy()

    for index in range(
        order,
        forward.size,
    ):

        if observed[index]:

            continue

        previous = forward[
            index - order:index
        ][::-1]

        prediction = np.dot(
            coefficients,
            previous,
        )

        if np.isfinite(prediction):

            forward[index] = prediction

    # --------------------------------------------------------
    # Reverse prediction.
    # --------------------------------------------------------

    backward = values[::-1].copy()

    reversed_observed = observed[::-1]

    for index in range(
        order,
        backward.size,
    ):

        if reversed_observed[index]:

            continue

        previous = backward[
            index - order:index
        ][::-1]

        prediction = np.dot(
            coefficients,
            previous,
        )

        if np.isfinite(prediction):

            backward[index] = prediction

    backward = backward[::-1]

    # --------------------------------------------------------
    # Combine forward and backward predictions.
    # --------------------------------------------------------

    missing = ~observed

    forward_values = forward[missing]

    backward_values = backward[missing]

    valid_forward = np.isfinite(
        forward_values
    )

    valid_backward = np.isfinite(
        backward_values
    )

    # --------------------------------------------------------
    # Both predictions valid.
    # --------------------------------------------------------

    both_valid = (
        valid_forward
        & valid_backward
    )

    reconstructed = values.copy()

    reconstructed_missing = reconstructed[missing]

    reconstructed_missing[both_valid] = (
        0.5 * forward_values[both_valid]
        +
        0.5 * backward_values[both_valid]
    )

    # --------------------------------------------------------
    # Forward prediction only.
    # --------------------------------------------------------

    forward_only = (
        valid_forward
        & ~valid_backward
    )

    reconstructed_missing[forward_only] = (
        forward_values[forward_only]
    )

    # --------------------------------------------------------
    # Backward prediction only.
    # --------------------------------------------------------

    backward_only = (
        ~valid_forward
        & valid_backward
    )

    reconstructed_missing[backward_only] = (
        backward_values[backward_only]
    )

    reconstructed[missing] = (
        reconstructed_missing
    )

    # --------------------------------------------------------
    # Restore observed values.
    # --------------------------------------------------------

    reconstructed[observed] = (
        values[observed]
    )

    return reconstructed


# ============================================================
# FREQUENCY-DOMAIN SPATIAL PREDICTION
# ============================================================

def _predict_frequency_plane(
    frequency_plane: np.ndarray,
    trace_observed_h: np.ndarray,
    trace_observed_w: np.ndarray,
    prediction_order: int,
) -> np.ndarray:
    """
    Apply f-x prediction to one frequency plane.

    Parameters
    ----------
    frequency_plane:
        Complex-valued spatial plane with shape (H, W).

    trace_observed_h:
        Boolean trace mask for prediction along H.

    trace_observed_w:
        Boolean trace mask for prediction along W.

    prediction_order:
        Prediction filter order.

    Returns
    -------
    ndarray
        Reconstructed complex-valued frequency plane.
    """

    plane = np.asarray(
        frequency_plane,
        dtype=np.complex128,
    ).copy()

    original_plane = plane.copy()

    # --------------------------------------------------------
    # Prediction along H.
    #
    # For each W location, predict missing H traces.
    # --------------------------------------------------------

    for width_index in range(
        plane.shape[1]
    ):

        trace = plane[
            :,
            width_index
        ]

        observed = trace_observed_h[
            :,
            width_index
        ]

        plane[
            :,
            width_index
        ] = _fx_predict_1d(
            trace,
            observed,
            prediction_order,
        )

    # --------------------------------------------------------
    # Prediction along W.
    #
    # For each H location, predict missing W traces.
    # --------------------------------------------------------

    for height_index in range(
        plane.shape[0]
    ):

        trace = plane[
            height_index,
            :
        ]

        observed = trace_observed_w[
            height_index,
            :
        ]

        plane[
            height_index,
            :
        ] = _fx_predict_1d(
            trace,
            observed,
            prediction_order,
        )

    # --------------------------------------------------------
    # Numerical safety.
    # --------------------------------------------------------

    invalid = ~np.isfinite(plane)

    if invalid.any():

        plane[invalid] = (
            original_plane[invalid]
        )

    return plane


# ============================================================
# PUBLIC F-X RECONSTRUCTION
# ============================================================

def fx_prediction_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    prediction_order: int = DEFAULT_PREDICTION_ORDER,
    iterations: int = DEFAULT_ITERATIONS,
    min_trace_observed_fraction: float = (
        DEFAULT_MIN_TRACE_OBSERVED_FRACTION
    ),
    convergence_tolerance: float = (
        DEFAULT_CONVERGENCE_TOLERANCE
    ),
) -> torch.Tensor:
    """
    Reconstruct a 3-D seismic cube using f-x prediction.

    Parameters
    ----------
    corrupted_cube:
        Incomplete seismic cube:

            (C, D, H, W)

    mask:
        Binary observation mask:

            1 = observed
            0 = missing

    prediction_order:
        Order of the spatial linear prediction filter.

    iterations:
        Number of alternating f-x refinement iterations.

    min_trace_observed_fraction:
        Minimum fraction of temporal/depth samples that must
        be observed for a spatial trace to be used as a
        reliable predictor.

    convergence_tolerance:
        Relative convergence threshold.

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with the same shape,
        device and dtype as the input.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    corrupted_cube, mask = _validate_inputs(
        corrupted_cube,
        mask,
    )

    _validate_parameters(
        prediction_order=prediction_order,
        iterations=iterations,
        min_trace_observed_fraction=(
            min_trace_observed_fraction
        ),
        convergence_tolerance=(
            convergence_tolerance
        ),
    )

    # ========================================================
    # PRESERVE ORIGINAL PROPERTIES
    # ========================================================

    original_device = (
        corrupted_cube.device
    )

    original_dtype = (
        corrupted_cube.dtype
    )

    # ========================================================
    # MOVE TO CPU / NUMPY
    # ========================================================

    cube = (
        corrupted_cube.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )

    observation_mask = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(
            bool,
            copy=False,
        )
    )

    # ========================================================
    # INITIAL RECONSTRUCTION
    # ========================================================

    reconstructed = cube.copy()

    # ========================================================
    # PROCESS EACH CHANNEL
    # ========================================================

    for channel in range(
        cube.shape[0]
    ):

        channel_mask = (
            observation_mask[channel]
        )

        channel_reconstruction = (
            reconstructed[channel]
        )

        # ----------------------------------------------------
        # Determine the observation fraction of each spatial
        # trace.
        #
        # Shape:
        #
        #     (H, W)
        # ----------------------------------------------------

        trace_fraction = (
            _trace_observation_fraction(
                channel_mask
            )
        )

        # ----------------------------------------------------
        # Trace along H is considered available when its
        # temporal/depth observation fraction reaches the
        # configured threshold.
        # ----------------------------------------------------

        trace_observed_h = (
            trace_fraction
            >= min_trace_observed_fraction
        )

        # ----------------------------------------------------
        # The same trace-availability map is used when
        # predicting in the W direction.
        # ----------------------------------------------------

        trace_observed_w = (
            trace_fraction
            >= min_trace_observed_fraction
        )

        # ====================================================
        # ITERATIVE F-X RECONSTRUCTION
        # ====================================================

        for iteration in range(
            iterations
        ):

            previous_iteration = (
                channel_reconstruction.copy()
            )

            # ------------------------------------------------
            # Fourier transform along temporal/depth axis.
            #
            # Real-valued input:
            #
            #     (D, H, W)
            #
            # becomes:
            #
            #     (F, H, W)
            # ------------------------------------------------

            frequency_cube = np.fft.rfft(
                channel_reconstruction,
                axis=0,
            )

            # ------------------------------------------------
            # Process every positive frequency.
            # ------------------------------------------------

            for frequency in range(
                frequency_cube.shape[0]
            ):

                frequency_plane = (
                    frequency_cube[frequency]
                )

                frequency_cube[frequency] = (
                    _predict_frequency_plane(
                        frequency_plane=frequency_plane,
                        trace_observed_h=(
                            trace_observed_h
                        ),
                        trace_observed_w=(
                            trace_observed_w
                        ),
                        prediction_order=(
                            prediction_order
                        ),
                    )
                )

            # ------------------------------------------------
            # Inverse Fourier transform.
            # ------------------------------------------------

            updated = np.fft.irfft(
                frequency_cube,
                n=channel_reconstruction.shape[0],
                axis=0,
            )

            updated = updated.astype(
                np.float64,
                copy=False,
            )

            # ------------------------------------------------
            # Numerical safety.
            # ------------------------------------------------

            if not np.isfinite(updated).all():

                raise FloatingPointError(
                    "f-x prediction produced NaN or Inf "
                    f"values in channel {channel}."
                )

            # ------------------------------------------------
            # CRITICAL DATA CONSISTENCY STEP
            #
            # Every originally observed sample is restored
            # exactly after each iteration.
            # ------------------------------------------------

            updated[channel_mask] = (
                cube[channel][channel_mask]
            )

            # ------------------------------------------------
            # Calculate relative change.
            # ------------------------------------------------

            difference = np.linalg.norm(
                updated
                - previous_iteration
            )

            reference = max(
                np.linalg.norm(
                    previous_iteration
                ),
                1.0e-12,
            )

            relative_change = (
                difference / reference
            )

            # ------------------------------------------------
            # Accept updated reconstruction.
            # ------------------------------------------------

            channel_reconstruction = (
                updated
            )

            # ------------------------------------------------
            # Stop if converged.
            # ------------------------------------------------

            if (
                relative_change
                < convergence_tolerance
            ):

                break

        # ----------------------------------------------------
        # Final exact observed-data restoration for channel.
        # ----------------------------------------------------

        channel_reconstruction[channel_mask] = (
            cube[channel][channel_mask]
        )

        reconstructed[channel] = (
            channel_reconstruction
        )

    # ========================================================
    # GLOBAL NUMERICAL VALIDATION
    # ========================================================

    if not np.isfinite(
        reconstructed
    ).all():

        raise FloatingPointError(
            "Final f-x reconstruction contains "
            "NaN or Inf values."
        )

    # ========================================================
    # FINAL GLOBAL DATA CONSISTENCY
    # ========================================================

    reconstructed[
        observation_mask
    ] = cube[
        observation_mask
    ]

    # ========================================================
    # CONVERT BACK TO TORCH
    # ========================================================

    result = torch.from_numpy(
        reconstructed
    ).to(
        device=original_device,
        dtype=original_dtype,
    )

    # ========================================================
    # SHAPE VALIDATION
    # ========================================================

    if result.shape != corrupted_cube.shape:

        raise RuntimeError(
            "f-x prediction changed the input shape. "
            f"Input: {tuple(corrupted_cube.shape)}, "
            f"Output: {tuple(result.shape)}"
        )

    # ========================================================
    # FINITE-VALUE VALIDATION
    # ========================================================

    if not torch.isfinite(result).all():

        raise RuntimeError(
            "f-x prediction produced NaN or Inf values."
        )

    # ========================================================
    # EXACT OBSERVED-DATA PRESERVATION
    # ========================================================

    observed_values = (
        mask == 1.0
    )

    if observed_values.any():

        observed_difference = torch.max(
            torch.abs(
                result[observed_values]
                -
                corrupted_cube[observed_values]
            )
        )

        if (
            observed_difference.item()
            > 1.0e-6
        ):

            raise RuntimeError(
                "f-x prediction modified observed "
                "seismic samples. "
                f"Maximum difference: "
                f"{observed_difference.item():.6e}"
            )

    return result


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "f-x Prediction baseline module loaded."
    )

    print(
        "Default prediction order:",
        DEFAULT_PREDICTION_ORDER,
    )

    print(
        "Default iterations:",
        DEFAULT_ITERATIONS,
    )

    print(
        "Default minimum trace observation fraction:",
        DEFAULT_MIN_TRACE_OBSERVED_FRACTION,
    )

    print(
        "Default convergence tolerance:",
        DEFAULT_CONVERGENCE_TOLERANCE,
    )