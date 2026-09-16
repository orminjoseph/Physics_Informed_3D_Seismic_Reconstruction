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

The method performs:

    1. Fourier transform along the temporal/depth axis.
    2. Frequency-by-frequency prediction in the spatial axis.
    3. Complex-valued linear prediction.
    4. Reconstruction through inverse Fourier transform.
    5. Preservation of all originally observed samples.

Tensor convention
-----------------
Input seismic cube:

    [C, D, H, W]

where:

    C = seismic channel
    D = time/depth samples
    H = spatial dimension 1
    W = spatial dimension 2

Mask convention
---------------
    1 = observed
    0 = missing

Important
---------
This implementation is a classical prediction-based baseline.
It is intentionally independent of the proposed neural network.

Author: Ormin Joseph
=============================================================
"""

import torch
import numpy as np


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

def _validate_inputs(
    corrupted_cube,
    mask
):
    """
    Validate input seismic cube and observation mask.
    """

    # --------------------------------------------------------
    # Convert NumPy arrays to PyTorch tensors.
    # --------------------------------------------------------

    if not isinstance(
        corrupted_cube,
        torch.Tensor
    ):
        corrupted_cube = torch.as_tensor(
            corrupted_cube,
            dtype=torch.float32
        )

    if not isinstance(
        mask,
        torch.Tensor
    ):
        mask = torch.as_tensor(
            mask,
            dtype=torch.float32
        )

    # --------------------------------------------------------
    # Force floating-point representation.
    # --------------------------------------------------------

    corrupted_cube = corrupted_cube.float()
    mask = mask.float()

    # --------------------------------------------------------
    # Check dimensionality.
    #
    # Expected:
    #
    # [C, D, H, W]
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
    # Check shape consistency.
    # --------------------------------------------------------

    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have "
            "identical shapes. "
            f"Received {tuple(corrupted_cube.shape)} "
            f"and {tuple(mask.shape)}."
        )

    # --------------------------------------------------------
    # Check numerical validity.
    # --------------------------------------------------------

    if not torch.isfinite(
        corrupted_cube
    ).all():
        raise ValueError(
            "corrupted_cube contains NaN or Inf."
        )

    if not torch.isfinite(
        mask
    ).all():
        raise ValueError(
            "mask contains NaN or Inf."
        )

    # --------------------------------------------------------
    # Check that observed samples exist.
    # --------------------------------------------------------

    if not (mask == 1).any():
        raise ValueError(
            "The mask contains no observed samples."
        )

    return corrupted_cube, mask


# ============================================================
# 1D LINEAR PREDICTION COEFFICIENTS
# ============================================================

def _estimate_prediction_coefficients(
    trace,
    order
):
    """
    Estimate a complex-valued autoregressive prediction
    filter from a spatial-frequency trace.

    Parameters
    ----------
    trace : numpy.ndarray
        Complex-valued spatial-frequency trace.

    order : int
        Prediction filter order.

    Returns
    -------
    numpy.ndarray
        Complex prediction coefficients.
    """

    trace = np.asarray(
        trace,
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Require enough samples for the prediction filter.
    # --------------------------------------------------------

    if trace.size <= order:
        return np.zeros(
            order,
            dtype=np.complex128
        )

    # --------------------------------------------------------
    # Construct the least-squares prediction system.
    #
    # x[n] ≈ a1*x[n-1] + ... + ap*x[n-p]
    # --------------------------------------------------------

    rows = []
    targets = []

    for index in range(
        order,
        trace.size
    ):
        rows.append(
            trace[
                index - order:index
            ][::-1]
        )

        targets.append(
            trace[index]
        )

    A = np.asarray(
        rows,
        dtype=np.complex128
    )

    b = np.asarray(
        targets,
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Solve complex least-squares problem.
    # --------------------------------------------------------

    coefficients, _, _, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None
    )

    return coefficients


# ============================================================
# 1D F-X PREDICTION
# ============================================================

def _fx_predict_1d(
    spatial_trace,
    observed,
    order
):
    """
    Perform prediction-based interpolation along one
    spatial dimension.

    Parameters
    ----------
    spatial_trace : numpy.ndarray
        Complex-valued spatial trace.

    observed : numpy.ndarray
        Boolean observation mask.

    order : int
        Prediction filter order.

    Returns
    -------
    numpy.ndarray
        Reconstructed complex-valued trace.
    """

    values = np.asarray(
        spatial_trace,
        dtype=np.complex128
    ).copy()

    observed = np.asarray(
        observed,
        dtype=bool
    )

    # --------------------------------------------------------
    # If all values are observed, nothing needs to be done.
    # --------------------------------------------------------

    if observed.all():
        return values

    # --------------------------------------------------------
    # If no samples are observed, prediction is impossible.
    # --------------------------------------------------------

    if not observed.any():
        return values

    # --------------------------------------------------------
    # Estimate prediction coefficients using observed values.
    #
    # Missing samples are temporarily excluded from the
    # coefficient-estimation sequence.
    # --------------------------------------------------------

    observed_values = values[observed]

    if observed_values.size <= order:
        return values

    coefficients = (
        _estimate_prediction_coefficients(
            observed_values,
            order
        )
    )

    # --------------------------------------------------------
    # Forward prediction.
    # --------------------------------------------------------

    reconstructed = values.copy()

    for index in range(
        order,
        values.size
    ):

        # ----------------------------------------------------
        # Only reconstruct missing samples.
        # ----------------------------------------------------

        if observed[index]:
            continue

        previous = reconstructed[
            index - order:index
        ][::-1]

        reconstructed[index] = np.dot(
            coefficients,
            previous
        )

    # --------------------------------------------------------
    # Backward prediction.
    #
    # This improves reconstruction of gaps where information
    # is available predominantly from the opposite direction.
    # --------------------------------------------------------

    reversed_values = reconstructed[::-1].copy()
    reversed_observed = observed[::-1]

    for index in range(
        order,
        reversed_values.size
    ):

        if reversed_observed[index]:
            continue

        previous = reversed_values[
            index - order:index
        ][::-1]

        reversed_values[index] = np.dot(
            coefficients,
            previous
        )

    # --------------------------------------------------------
    # Average forward and backward predictions.
    # --------------------------------------------------------

    missing = ~observed

    reconstructed[missing] = (
        0.5 * reconstructed[missing]
        +
        0.5 * reversed_values[::-1][missing]
    )

    # --------------------------------------------------------
    # Restore all observed samples exactly.
    # --------------------------------------------------------

    reconstructed[observed] = (
        values[observed]
    )

    return reconstructed


# ============================================================
# F-X PREDICTION RECONSTRUCTION
# ============================================================

def fx_prediction_reconstruction(
    corrupted_cube,
    mask,
    prediction_order=4,
    iterations=2
):
    """
    Reconstruct missing seismic samples using an f-x
    prediction-based interpolation strategy.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Input seismic cube:

            [C, D, H, W]

    mask : torch.Tensor
        Observation mask:

            1 = observed
            0 = missing

    prediction_order : int
        Order of the linear prediction filter.

    iterations : int
        Number of alternating prediction/refinement passes.

    Returns
    -------
    torch.Tensor
        Reconstructed cube with the same shape as input.

    Notes
    -----
    The Fourier transform is applied along the depth/time
    dimension.

    Prediction is then performed independently at each
    frequency.

    Observed samples are restored after every iteration.
    """

    # --------------------------------------------------------
    # Validate input.
    # --------------------------------------------------------

    corrupted_cube, mask = _validate_inputs(
        corrupted_cube,
        mask
    )

    # --------------------------------------------------------
    # Validate parameters.
    # --------------------------------------------------------

    if prediction_order < 1:
        raise ValueError(
            "prediction_order must be >= 1."
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1."
        )

    # --------------------------------------------------------
    # Preserve device and dtype.
    # --------------------------------------------------------

    device = corrupted_cube.device

    original_dtype = (
        corrupted_cube.dtype
    )

    # --------------------------------------------------------
    # Work in NumPy because the baseline uses complex-valued
    # least-squares prediction.
    # --------------------------------------------------------

    cube = (
        corrupted_cube.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    observation_mask = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    # --------------------------------------------------------
    # Make a working copy.
    # --------------------------------------------------------

    reconstructed = cube.copy()

    # ========================================================
    # ITERATIVE F-X RECONSTRUCTION
    # ========================================================

    for _ in range(iterations):

        # ----------------------------------------------------
        # Transform along depth/time.
        #
        # Result:
        #
        # [C, F, H, W]
        #
        # where F is the frequency dimension.
        # ----------------------------------------------------

        frequency_cube = np.fft.fft(
            reconstructed,
            axis=1
        )

        # ----------------------------------------------------
        # Process each channel.
        # ----------------------------------------------------

        for channel in range(
            frequency_cube.shape[0]
        ):

            # ------------------------------------------------
            # Process each frequency.
            # ------------------------------------------------

            for frequency in range(
                frequency_cube.shape[1]
            ):

                # --------------------------------------------
                # Extract spatial plane.
                # --------------------------------------------

                spatial_plane = (
                    frequency_cube[
                        channel,
                        frequency
                    ]
                )

                # --------------------------------------------
                # Reconstruct along H dimension.
                # --------------------------------------------

                for width_index in range(
                    spatial_plane.shape[1]
                ):

                    trace = spatial_plane[
                        :,
                        width_index
                    ]

                    # Observation pattern is taken from
                    # one representative depth-independent
                    # spatial line.
                    #
                    # This is appropriate for trace-style
                    # missing-data masks.
                    # ----------------------------------------

                    observed = observation_mask[
                        channel,
                        0,
                        :,
                        width_index
                    ]

                    if observed.all():
                        continue

                    spatial_plane[
                        :,
                        width_index
                    ] = _fx_predict_1d(
                        trace,
                        observed,
                        prediction_order
                    )

                # --------------------------------------------
                # Reconstruct along W dimension.
                # --------------------------------------------

                for height_index in range(
                    spatial_plane.shape[0]
                ):

                    trace = spatial_plane[
                        height_index,
                        :
                    ]

                    observed = observation_mask[
                        channel,
                        0,
                        height_index,
                        :
                    ]

                    if observed.all():
                        continue

                    spatial_plane[
                        height_index,
                        :
                    ] = _fx_predict_1d(
                        trace,
                        observed,
                        prediction_order
                    )

                frequency_cube[
                    channel,
                    frequency
                ] = spatial_plane

        # ----------------------------------------------------
        # Inverse Fourier transform.
        # ----------------------------------------------------

        reconstructed = np.fft.ifft(
            frequency_cube,
            axis=1
        ).real

        # ----------------------------------------------------
        # CRITICAL:
        #
        # Restore all observed samples exactly.
        #
        # A baseline must never modify measurements that
        # were actually available.
        # ----------------------------------------------------

        reconstructed[
            observation_mask
        ] = cube[
            observation_mask
        ]

    # --------------------------------------------------------
    # Convert back to PyTorch.
    # --------------------------------------------------------

    result = torch.from_numpy(
        reconstructed
    ).to(
        device=device,
        dtype=original_dtype
    )

    # --------------------------------------------------------
    # Final numerical validation.
    # --------------------------------------------------------

    if not torch.isfinite(result).all():
        raise RuntimeError(
            "f-x prediction produced NaN or Inf values."
        )

    # --------------------------------------------------------
    # Final shape validation.
    # --------------------------------------------------------

    if result.shape != corrupted_cube.shape:
        raise RuntimeError(
            "f-x prediction changed the input shape. "
            f"Input: {tuple(corrupted_cube.shape)}, "
            f"Output: {tuple(result.shape)}"
        )

    # --------------------------------------------------------
    # Final observed-data preservation check.
    # --------------------------------------------------------

    observed_difference = (
        result[mask == 1]
        -
        corrupted_cube[mask == 1]
    ).abs().max()

    if observed_difference > 1e-6:
        raise RuntimeError(
            "f-x prediction modified observed samples. "
            f"Maximum difference: "
            f"{observed_difference.item():.6e}"
        )

    return result