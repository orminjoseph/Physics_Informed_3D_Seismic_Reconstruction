"""
=========================================================
Compressive Sensing Baseline
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Baseline method:
    Compressive Sensing (CS)

Methodological basis:
    Sparse seismic reconstruction in a 3D wavelet domain
    using iterative soft-thresholding and data consistency.

Input convention:
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask convention:
    1 -> observed
    0 -> missing

The method performs:

    1. 3D wavelet decomposition
    2. Soft-thresholding of wavelet coefficients
    3. Inverse wavelet reconstruction
    4. Exact observed-data consistency
    5. Iteration until the prescribed number of iterations
       is completed or convergence is reached

Observed seismic samples are NEVER modified.

This implementation is intended as a classical sparse-
recovery baseline for comparison with:

    - Nearest Neighbor
    - Linear Interpolation
    - f-x Prediction
    - Curvelet POCS
    - Dictionary Learning
    - Proposed Physics-Informed 3D Network

Author: Ormin Joseph
=========================================================
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "PyWavelets is required for the Compressive Sensing "
        "baseline. Install it with:\n\n"
        "pip install PyWavelets"
    ) from exc


# =========================================================
# DEFAULT PARAMETERS
# =========================================================

DEFAULT_WAVELET = "db4"

DEFAULT_LEVEL = 3

DEFAULT_ITERATIONS = 12

DEFAULT_THRESHOLD = 0.05

DEFAULT_THRESHOLD_DECAY = 0.90

DEFAULT_TOLERANCE = 1.0e-5


# =========================================================
# INPUT VALIDATION
# =========================================================

def _validate_inputs(
        corrupted_cube: torch.Tensor,
        mask: torch.Tensor,
):
    """
    Validate the input seismic cube and observation mask.
    """

    # -----------------------------------------------------
    # Check tensor types
    # -----------------------------------------------------

    if not isinstance(
        corrupted_cube,
        torch.Tensor
    ):
        raise TypeError(
            "corrupted_cube must be a torch.Tensor."
        )

    if not isinstance(
        mask,
        torch.Tensor
    ):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    # -----------------------------------------------------
    # Check dimensions
    # -----------------------------------------------------

    if corrupted_cube.ndim != 4:
        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received {tuple(corrupted_cube.shape)}."
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received {tuple(mask.shape)}."
        )

    # -----------------------------------------------------
    # Check shape agreement
    # -----------------------------------------------------

    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have "
            "identical shapes.\n"
            f"Cube: {tuple(corrupted_cube.shape)}\n"
            f"Mask: {tuple(mask.shape)}"
        )

    # -----------------------------------------------------
    # Check finite values
    # -----------------------------------------------------

    if not torch.isfinite(
        corrupted_cube
    ).all():
        raise ValueError(
            "corrupted_cube contains NaN or Inf values."
        )

    if not torch.isfinite(
        mask
    ).all():
        raise ValueError(
            "mask contains NaN or Inf values."
        )

    # -----------------------------------------------------
    # Check binary mask
    # -----------------------------------------------------

    unique_mask_values = torch.unique(mask)

    for value in unique_mask_values:
        if not (
            torch.isclose(
                value,
                torch.tensor(
                    0.0,
                    device=value.device,
                    dtype=value.dtype
                )
            )
            or
            torch.isclose(
                value,
                torch.tensor(
                    1.0,
                    device=value.device,
                    dtype=value.dtype
                )
            )
        ):
            raise ValueError(
                "mask must contain only 0 and 1."
            )


# =========================================================
# PARAMETER VALIDATION
# =========================================================

def _validate_parameters(
        wavelet: str,
        level: int,
        iterations: int,
        threshold: float,
        threshold_decay: float,
        tolerance: float,
):
    """
    Validate CS algorithm parameters.
    """

    if not isinstance(
        wavelet,
        str
    ):
        raise TypeError(
            "wavelet must be a string."
        )

    if level < 1:
        raise ValueError(
            "level must be >= 1."
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be >= 1."
        )

    if threshold <= 0:
        raise ValueError(
            "threshold must be > 0."
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

    # -----------------------------------------------------
    # Verify wavelet exists
    # -----------------------------------------------------

    try:
        pywt.Wavelet(wavelet)
    except Exception as exc:
        raise ValueError(
            f"Unknown wavelet: {wavelet}"
        ) from exc


# =========================================================
# SOFT THRESHOLDING
# =========================================================

def _soft_threshold(
        coefficients,
        threshold,
):
    """
    Apply element-wise soft thresholding.

    S_lambda(x)
        = sign(x) * max(|x| - lambda, 0)
    """

    return np.sign(coefficients) * np.maximum(
        np.abs(coefficients) - threshold,
        0.0
    )


# =========================================================
# WAVELET THRESHOLDING
# =========================================================

def _threshold_wavelet_coefficients(
        coefficients,
        threshold,
):
    """
    Apply soft thresholding to all wavelet coefficient
    arrays.

    The approximation coefficients are preserved while
    detail coefficients are sparsity-promoted.
    """

    thresholded = [
        coefficients[0].copy()
    ]

    # -----------------------------------------------------
    # Threshold detail coefficients
    # -----------------------------------------------------

    for detail_level in coefficients[1:]:

        thresholded_details = {}

        for key, values in detail_level.items():

            thresholded_details[key] = (
                _soft_threshold(
                    values,
                    threshold
                )
            )

        thresholded.append(
            thresholded_details
        )

    return thresholded


# =========================================================
# SINGLE-CHANNEL WAVELET RECONSTRUCTION
# =========================================================

def _wavelet_reconstruct(
        volume: np.ndarray,
        wavelet: str,
        level: int,
        threshold: float,
):
    """
    Perform one sparse wavelet reconstruction step.

    Parameters
    ----------
    volume : ndarray
        3D seismic volume (D, H, W).

    wavelet : str
        Wavelet family.

    level : int
        Wavelet decomposition level.

    threshold : float
        Soft-threshold value.

    Returns
    -------
    ndarray
        Wavelet-domain sparse reconstruction.
    """

    # -----------------------------------------------------
    # Wavelet decomposition
    # -----------------------------------------------------

    coefficients = pywt.wavedecn(
        volume,
        wavelet=wavelet,
        level=level,
        mode="periodization"
    )

    # -----------------------------------------------------
    # Sparse coefficient thresholding
    # -----------------------------------------------------

    thresholded = (
        _threshold_wavelet_coefficients(
            coefficients,
            threshold
        )
    )

    # -----------------------------------------------------
    # Inverse wavelet transform
    # -----------------------------------------------------

    reconstructed = pywt.waverecn(
        thresholded,
        wavelet=wavelet,
        mode="periodization"
    )

    # -----------------------------------------------------
    # Crop possible padding differences
    # -----------------------------------------------------

    reconstructed = reconstructed[
        :volume.shape[0],
        :volume.shape[1],
        :volume.shape[2]
    ]

    return reconstructed.astype(
        np.float32,
        copy=False
    )


# =========================================================
# SINGLE-CHANNEL CS RECONSTRUCTION
# =========================================================

def _compressive_sensing_channel(
        corrupted,
        observed_mask,
        wavelet,
        level,
        iterations,
        threshold,
        threshold_decay,
        tolerance,
):
    """
    Reconstruct one seismic channel using iterative
    wavelet-domain sparse recovery.

    The observed data are enforced after every iteration.
    """

    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------

    corrupted = np.asarray(
        corrupted,
        dtype=np.float32
    )

    observed_mask = np.asarray(
        observed_mask,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Initial estimate
    #
    # Start from the observed incomplete volume.
    # Missing values remain zero initially.
    # -----------------------------------------------------

    reconstruction = (
        corrupted.copy()
    )

    previous = (
        reconstruction.copy()
    )

    current_threshold = (
        float(threshold)
    )

    # =====================================================
    # ITERATIVE SPARSE RECOVERY
    # =====================================================

    for iteration in range(
        iterations
    ):

        # -------------------------------------------------
        # Sparse reconstruction in wavelet domain
        # -------------------------------------------------

        sparse_estimate = (
            _wavelet_reconstruct(
                reconstruction,
                wavelet=wavelet,
                level=level,
                threshold=current_threshold
            )
        )

        # -------------------------------------------------
        # DATA CONSISTENCY
        #
        # Observed samples must remain exactly equal to
        # the original observed input.
        # -------------------------------------------------

        reconstruction = (
            observed_mask * corrupted
            +
            (1.0 - observed_mask)
            * sparse_estimate
        )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        if not np.isfinite(
            reconstruction
        ).all():
            raise FloatingPointError(
                "CS reconstruction produced "
                "NaN or Inf values."
            )

        # -------------------------------------------------
        # Convergence measurement
        # -------------------------------------------------

        difference = np.linalg.norm(
            reconstruction - previous
        )

        reference = max(
            np.linalg.norm(previous),
            1.0e-12
        )

        relative_change = (
            difference / reference
        )

        # -------------------------------------------------
        # Update threshold
        #
        # Threshold decreases gradually, allowing the
        # reconstruction to recover progressively finer
        # seismic structures.
        # -------------------------------------------------

        current_threshold *= (
            threshold_decay
        )

        # -------------------------------------------------
        # Store current estimate
        # -------------------------------------------------

        previous = (
            reconstruction.copy()
        )

        # -------------------------------------------------
        # Convergence criterion
        # -------------------------------------------------

        if relative_change < tolerance:
            break

    return reconstruction


# =========================================================
# PUBLIC CS RECONSTRUCTION FUNCTION
# =========================================================

def compressive_sensing_reconstruction(
        corrupted_cube: torch.Tensor,
        mask: torch.Tensor,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_LEVEL,
        iterations: int = DEFAULT_ITERATIONS,
        threshold: float = DEFAULT_THRESHOLD,
        threshold_decay: float = DEFAULT_THRESHOLD_DECAY,
        tolerance: float = DEFAULT_TOLERANCE,
):
    """
    Reconstruct a 3D seismic cube using compressive sensing.

    Parameters
    ----------
    corrupted_cube : torch.Tensor
        Incomplete seismic cube.

        Shape:
            (C, D, H, W)

    mask : torch.Tensor
        Binary observation mask.

        1 -> observed
        0 -> missing

    wavelet : str
        Sparsifying wavelet basis.

        Default:
            db4

    level : int
        3D wavelet decomposition level.

    iterations : int
        Maximum number of sparse-recovery iterations.

    threshold : float
        Initial soft-threshold.

    threshold_decay : float
        Multiplicative threshold decay.

    tolerance : float
        Relative convergence tolerance.

    Returns
    -------
    torch.Tensor
        Reconstructed cube with the same shape as
        corrupted_cube.
    """

    # =====================================================
    # VALIDATION
    # =====================================================

    _validate_inputs(
        corrupted_cube,
        mask
    )

    _validate_parameters(
        wavelet,
        level,
        iterations,
        threshold,
        threshold_decay,
        tolerance
    )

    # =====================================================
    # WORK IN CPU FLOAT32 FOR PYWT
    # =====================================================

    original_device = (
        corrupted_cube.device
    )

    original_dtype = (
        corrupted_cube.dtype
    )

    corrupted_cpu = (
        corrupted_cube.detach()
        .cpu()
        .float()
    )

    mask_cpu = (
        mask.detach()
        .cpu()
        .float()
    )

    # =====================================================
    # CONVERT TO NUMPY
    # =====================================================

    corrupted_numpy = (
        corrupted_cpu.numpy()
    )

    mask_numpy = (
        mask_cpu.numpy()
    )

    # =====================================================
    # OUTPUT ARRAY
    # =====================================================

    reconstructed_numpy = (
        corrupted_numpy.copy()
    )

    # =====================================================
    # PROCESS EACH CHANNEL
    # =====================================================

    number_of_channels = (
        corrupted_numpy.shape[0]
    )

    for channel in range(
        number_of_channels
    ):

        reconstructed_numpy[channel] = (
            _compressive_sensing_channel(
                corrupted_numpy[channel],
                mask_numpy[channel],
                wavelet=wavelet,
                level=level,
                iterations=iterations,
                threshold=threshold,
                threshold_decay=threshold_decay,
                tolerance=tolerance
            )
        )

    # =====================================================
    # FINAL OBSERVED-DATA PRESERVATION
    # =====================================================

    reconstructed_numpy = (
        mask_numpy * corrupted_numpy
        +
        (1.0 - mask_numpy)
        * reconstructed_numpy
    )

    # =====================================================
    # FINAL NUMERICAL VALIDATION
    # =====================================================

    if not np.isfinite(
        reconstructed_numpy
    ).all():
        raise FloatingPointError(
            "Final CS reconstruction contains "
            "NaN or Inf values."
        )

    # =====================================================
    # CONVERT BACK TO TORCH
    # =====================================================

    reconstructed = torch.from_numpy(
        reconstructed_numpy
    )

    # Restore original dtype.
    if original_dtype != torch.float32:
        reconstructed = reconstructed.to(
            dtype=original_dtype
        )

    # Restore original device.
    reconstructed = reconstructed.to(
        device=original_device
    )

    # =====================================================
    # SHAPE VALIDATION
    # =====================================================

    if reconstructed.shape != (
        corrupted_cube.shape
    ):
        raise RuntimeError(
            "CS reconstruction changed the "
            "input shape.\n"
            f"Input: "
            f"{tuple(corrupted_cube.shape)}\n"
            f"Output: "
            f"{tuple(reconstructed.shape)}"
        )

    # =====================================================
    # EXACT OBSERVED-DATA VALIDATION
    # =====================================================

    observed_difference = (
        torch.abs(
            reconstructed[mask == 1]
            -
            corrupted_cube[mask == 1]
        )
    )

    if observed_difference.numel() > 0:

        maximum_difference = (
            observed_difference.max()
        )

        if maximum_difference > 1.0e-6:

            raise RuntimeError(
                "CS reconstruction failed to "
                "preserve observed seismic samples.\n"
                f"Maximum difference: "
                f"{maximum_difference.item():.6e}"
            )

    return reconstructed


# =========================================================
# MODULE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "Compressive Sensing baseline module loaded."
    )

    print(
        "Default wavelet:",
        DEFAULT_WAVELET
    )

    print(
        "Default level:",
        DEFAULT_LEVEL
    )

    print(
        "Default iterations:",
        DEFAULT_ITERATIONS
    )

    print(
        "Default threshold:",
        DEFAULT_THRESHOLD
    )