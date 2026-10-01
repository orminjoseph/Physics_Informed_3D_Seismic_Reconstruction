"""
======================================================================
CURVELET POCS CONTROLLED-MATRIX BASELINE
======================================================================

Curvelet-domain Projection Onto Convex Sets (POCS) baseline for
3-D seismic data reconstruction.

This implementation uses the Uniform Discrete Curvelet Transform
(UDCT) provided by the Python curvelets package.

Controlled experimental design
-------------------------------
Geological modes:
    1. horizontal
    2. dipping
    3. faulted
    4. folded
    5. complex
    6. highly_complex

Missing-data mechanisms:
    1. random_voxels
    2. missing_traces
    3. missing_inlines
    4. missing_crosslines
    5. missing_blocks

Missing rates:
    10%, 20%, 30%, 40%, 50%

Seeds:
    42, 43, 44, 45, 46

Total controlled cases:
    6 × 5 × 5 × 5 = 750

Reconstruction strategy
-----------------------
    1. Start from the incomplete seismic volume.
    2. Transform the current estimate into the curvelet domain.
    3. Apply coefficient soft-thresholding.
    4. Transform back to the seismic domain.
    5. Project onto the observed-data constraint set.
    6. Repeat until convergence or the iteration limit is reached.

Important
---------
Observed seismic samples are NEVER modified.

The reconstruction algorithm never accesses the ground-truth
target. Ground truth is used only after reconstruction for
evaluation metrics.

Input convention
----------------
corrupted_cube : (C, D, H, W)
mask           : (C, D, H, W)

Mask convention
---------------
1 = observed sample
0 = missing sample

Output
------
Reconstructed seismic cube:
    (C, D, H, W)

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import random
import time
from pathlib import Path

import numpy as np
import torch

from curvelets.numpy import UDCT
from skimage.metrics import structural_similarity

from dataset.synthetic_dataset import SyntheticSeismicDataset

from utils.config import (
    BASELINE_CUBE_SIZE,
    BASELINE_NUM_SAMPLES,
    CONTROLLED_MATRIX_CASE_LIMIT,
    CURVELET_ITERATIONS,
    CURVELET_NUM_SCALES,
    CURVELET_THRESHOLD,
    CURVELET_THRESHOLD_DECAY,
    CURVELET_TOLERANCE,
    OBSERVED_PRESERVATION_TOLERANCE,
    REPORT_DIR,
    SEISMIC_DATA_RANGE,
    CURVELET_WEDGES_PER_DIRECTION,
)


# =====================================================================
# CONTROLLED EXPERIMENT FACTORS
# =====================================================================

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
]


MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible controlled experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# GENERAL VALIDATION
# =====================================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Validate corrupted seismic cube and observation mask.
    """

    if not isinstance(
        corrupted_cube,
        torch.Tensor,
    ):
        raise TypeError(
            "corrupted_cube must be a torch.Tensor."
        )

    if not isinstance(
        mask,
        torch.Tensor,
    ):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

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
            "corrupted_cube and mask must have identical shapes. "
            f"Received {tuple(corrupted_cube.shape)} and "
            f"{tuple(mask.shape)}."
        )

    if not corrupted_cube.is_floating_point():
        raise TypeError(
            "corrupted_cube must use a floating-point dtype."
        )

    if not torch.isfinite(
        corrupted_cube
    ).all():
        raise ValueError(
            "corrupted_cube contains NaN or infinite values."
        )

    if not torch.isfinite(
        mask
    ).all():
        raise ValueError(
            "mask contains NaN or infinite values."
        )

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
) -> np.ndarray:
    """
    Perform Curvelet-domain POCS reconstruction for one seismic
    channel.

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
        Initial curvelet coefficient threshold.

    threshold_decay : float
        Multiplicative threshold decay.

    tolerance : float
        Relative convergence tolerance.

    Returns
    -------
    np.ndarray
        Reconstructed seismic volume with shape (D, H, W).
    """

    # -----------------------------------------------------------------
    # Convert to float32.
    # -----------------------------------------------------------------

    corrupted_channel = np.asarray(
        corrupted_channel,
        dtype=np.float32,
    )

    mask_channel = np.asarray(
        mask_channel,
        dtype=np.float32,
    )

    # -----------------------------------------------------------------
    # Initial estimate.
    #
    # Missing samples are initialized to zero.
    # Observed samples retain their measured values.
    # -----------------------------------------------------------------

    current = (
        mask_channel * corrupted_channel
    ).astype(
        np.float32,
        copy=False,
    )

    # -----------------------------------------------------------------
    # Save observed data separately.
    #
    # This array is NEVER modified.
    # -----------------------------------------------------------------

    observed_data = (
        corrupted_channel.copy()
    )

    # -----------------------------------------------------------------
    # Keep track of the current threshold.
    # -----------------------------------------------------------------

    current_threshold = float(
        threshold
    )

    # -----------------------------------------------------------------
    # POCS iterations.
    # -----------------------------------------------------------------

    for iteration_index in range(
        iterations
    ):

        # =============================================================
        # Save previous iterate for convergence analysis.
        # =============================================================

        previous = current.copy()

        # =============================================================
        # Forward curvelet transform.
        # =============================================================

        coefficients = transform.forward(
            current
        )

        # =============================================================
        # Convert structured coefficients into vector form.
        # =============================================================

        coefficient_vector = transform.vect(
            coefficients
        )

        coefficient_vector = np.asarray(
            coefficient_vector
        )

        # =============================================================
        # Validate curvelet coefficients.
        # =============================================================

        if not np.isfinite(
            coefficient_vector.real
        ).all():
            raise FloatingPointError(
                "Curvelet coefficient real part "
                "contains NaN or infinite values."
            )

        if np.iscomplexobj(
            coefficient_vector
        ):
            if not np.isfinite(
                coefficient_vector.imag
            ).all():
                raise FloatingPointError(
                    "Curvelet coefficient imaginary part "
                    "contains NaN or infinite values."
                )

        # =============================================================
        # Soft thresholding.
        #
        # For each coefficient:
        #
        #   c_new =
        #       c * max(1 - threshold/|c|, 0)
        #
        # This preserves the phase of complex coefficients.
        # =============================================================

        magnitude = np.abs(
            coefficient_vector
        )

        shrink_factor = np.maximum(
            1.0
            - current_threshold
            / (
                magnitude
                + 1.0e-12
            ),
            0.0,
        )

        thresholded_vector = (
            coefficient_vector
            * shrink_factor
        )

        # =============================================================
        # Reconstruct the structured curvelet coefficient object.
        # =============================================================

        thresholded_coefficients = (
            transform.struct(
                thresholded_vector
            )
        )

        # =============================================================
        # Inverse curvelet transform.
        # =============================================================

        reconstructed = transform.backward(
            thresholded_coefficients
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=np.float32,
        )

        # =============================================================
        # Validate reconstructed volume.
        # =============================================================

        if not np.isfinite(
            reconstructed
        ).all():
            raise FloatingPointError(
                "Curvelet POCS produced NaN or "
                "infinite values."
            )

        # =============================================================
        # Projection onto observed-data constraint set.
        #
        # Observed samples are restored from the original corrupted
        # input, NOT from the reconstructed volume.
        # =============================================================

        current = (
            mask_channel * observed_data
            + (
                1.0 - mask_channel
            ) * reconstructed
        ).astype(
            np.float32,
            copy=False,
        )

        # =============================================================
        # Calculate relative change between successive projected
        # iterates.
        # =============================================================

        difference_norm = np.linalg.norm(
            (
                current
                - previous
            ).ravel()
        )

        previous_norm = (
            np.linalg.norm(
                previous.ravel()
            )
            + 1.0e-12
        )

        relative_change = (
            difference_norm
            / previous_norm
        )

        # =============================================================
        # Reduce threshold for the next iteration.
        # =============================================================

        current_threshold *= (
            threshold_decay
        )

        # =============================================================
        # Convergence test.
        # =============================================================

        if relative_change < tolerance:
            break

    # -----------------------------------------------------------------
    # Final exact observed-data projection.
    # -----------------------------------------------------------------

    current = (
        mask_channel * observed_data
        + (
            1.0 - mask_channel
        ) * current
    ).astype(
        np.float32,
        copy=False,
    )

    return current


# =====================================================================
# PUBLIC CURVELET POCS FUNCTION
# =====================================================================

def curvelet_pocs_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    num_scales: int | None = None,
    wedges_per_direction: int | None = None,
    iterations: int | None = None,
    threshold: float | None = None,
    threshold_decay: float | None = None,
    tolerance: float | None = None,
) -> torch.Tensor:
    """
    Reconstruct a 3-D seismic cube using Curvelet-domain POCS.

    Parameters default to the centralized values in utils.config.

    Input
    -----
    corrupted_cube:
        Tensor with shape (C, D, H, W).

    mask:
        Binary tensor with shape (C, D, H, W).

    Returns
    -------
    torch.Tensor
        Reconstructed cube with shape (C, D, H, W).
    """

    # =================================================================
    # VALIDATE INPUTS
    # =================================================================

    _validate_inputs(
        corrupted_cube,
        mask,
    )

    # =================================================================
    # RESOLVE CONFIGURATION VALUES
    # =================================================================

    if num_scales is None:
        num_scales = (
            CURVELET_NUM_SCALES
        )

    if wedges_per_direction is None:
        wedges_per_direction = (
            CURVELET_WEDGES_PER_DIRECTION
        )

    if iterations is None:
        iterations = (
            CURVELET_ITERATIONS
        )

    if threshold is None:
        threshold = (
            CURVELET_THRESHOLD
        )

    if threshold_decay is None:
        threshold_decay = (
            CURVELET_THRESHOLD_DECAY
        )

    if tolerance is None:
        tolerance = (
            CURVELET_TOLERANCE
        )

    # =================================================================
    # VALIDATE PARAMETERS
    # =================================================================

    if num_scales < 2:
        raise ValueError(
            "num_scales must be >= 2."
        )

    if wedges_per_direction < 3:
        raise ValueError(
            "wedges_per_direction must be >= 3."
        )

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

    # =================================================================
    # SAVE ORIGINAL DEVICE AND DTYPE
    # =================================================================

    original_device = (
        corrupted_cube.device
    )

    original_dtype = (
        corrupted_cube.dtype
    )

    # =================================================================
    # MOVE DATA TO CPU / NUMPY
    #
    # The curvelets.numpy implementation operates on NumPy arrays.
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
    # EXTRACT SPATIAL VOLUME SIZE
    # =================================================================

    volume_shape = tuple(
        corrupted_np.shape[1:]
    )

    # =================================================================
    # INITIALIZE ONE UDCT TRANSFORM
    #
    # The same transform is reused for all channels because the
    # spatial dimensions are identical.
    # =================================================================

    transform = UDCT(
        shape=volume_shape,
        num_scales=num_scales,
        wedges_per_direction=(
            wedges_per_direction
        ),
        transform_kind="real",
    )

    # =================================================================
    # OUTPUT ARRAY
    # =================================================================

    reconstructed_np = np.empty_like(
        corrupted_np,
        dtype=np.float32,
    )

    # =================================================================
    # RECONSTRUCT EACH CHANNEL
    # =================================================================

    for channel_index in range(
        corrupted_np.shape[0]
    ):

        reconstructed_np[
            channel_index
        ] = _curvelet_pocs_channel(
            corrupted_channel=(
                corrupted_np[
                    channel_index
                ]
            ),
            mask_channel=(
                mask_np[
                    channel_index
                ]
            ),
            transform=transform,
            iterations=iterations,
            threshold=threshold,
            threshold_decay=threshold_decay,
            tolerance=tolerance,
        )

    # =================================================================
    # FINAL DATA-CONSISTENCY PROJECTION
    # =================================================================

    reconstructed_np = (
        mask_np * corrupted_np
        + (
            1.0 - mask_np
        ) * reconstructed_np
    ).astype(
        np.float32,
        copy=False,
    )

    # =================================================================
    # CONVERT NUMPY -> TORCH
    # =================================================================

    reconstructed_cube = (
        torch.from_numpy(
            reconstructed_np
        )
    )

    # =================================================================
    # RESTORE ORIGINAL DEVICE AND DTYPE
    # =================================================================

    reconstructed_cube = (
        reconstructed_cube.to(
            device=original_device,
            dtype=original_dtype,
        )
    )

    # =================================================================
    # FINAL FINITE-VALUE CHECK
    # =================================================================

    if not torch.isfinite(
        reconstructed_cube
    ).all():
        raise FloatingPointError(
            "Final Curvelet POCS reconstruction "
            "contains NaN or infinite values."
        )

    # =================================================================
    # EXACT OBSERVED-DATA CHECK
    # =================================================================

    observed_difference = (
        (
            reconstructed_cube
            - corrupted_cube
        )
        * mask
    ).abs().max()

    if (
        observed_difference.item()
        > OBSERVED_PRESERVATION_TOLERANCE
    ):
        raise RuntimeError(
            "Observed seismic samples were not "
            "preserved within the configured tolerance. "
            f"Maximum difference: "
            f"{observed_difference.item():.6e}; "
            f"Tolerance: "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.6e}"
        )

    return reconstructed_cube


# =====================================================================
# METRIC FUNCTIONS
# =====================================================================

def _mae(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Mean Absolute Error.
    """

    return float(
        np.mean(
            np.abs(
                prediction
                - target
            )
        )
    )


def _rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Root Mean Squared Error.
    """

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - target
                ) ** 2
            )
        )
    )


def _psnr(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Peak Signal-to-Noise Ratio.
    """

    mse = np.mean(
        (
            prediction
            - target
        ) ** 2
    )

    if mse <= 0:
        return float("inf")

    return float(
        10.0
        * np.log10(
            (
                SEISMIC_DATA_RANGE ** 2
            )
            / mse
        )
    )


def _snr(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Signal-to-Noise Ratio.
    """

    signal_power = np.sum(
        target ** 2
    )

    noise_power = np.sum(
        (
            target
            - prediction
        ) ** 2
    )

    if noise_power <= 0:
        return float("inf")

    if signal_power <= 0:
        return float("-inf")

    return float(
        10.0
        * np.log10(
            signal_power
            / noise_power
        )
    )


def _ssim(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    3-D Structural Similarity Index.

    The calculation is performed directly on the seismic volume.
    """

    minimum_dimension = min(
        prediction.shape
    )

    if minimum_dimension < 3:
        return float("nan")

    # ---------------------------------------------------------------
    # Use a window size compatible with the seismic volume.
    # ---------------------------------------------------------------

    window_size = min(
        7,
        minimum_dimension,
    )

    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        return float("nan")

    return float(
        structural_similarity(
            target,
            prediction,
            data_range=(
                SEISMIC_DATA_RANGE
            ),
            win_size=window_size,
        )
    )


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Calculate global and missing-region reconstruction metrics.
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    mask = np.asarray(
        mask
    ).astype(bool)

    missing = ~mask

    result = {
        "MAE": _mae(
            prediction,
            target,
        ),
        "RMSE": _rmse(
            prediction,
            target,
        ),
        "PSNR": _psnr(
            prediction,
            target,
        ),
        "SNR": _snr(
            prediction,
            target,
        ),
        "SSIM": _ssim(
            prediction,
            target,
        ),
    }

    # -----------------------------------------------------------------
    # Missing-region metrics.
    # -----------------------------------------------------------------

    if np.any(missing):

        missing_prediction = (
            prediction[missing]
        )

        missing_target = (
            target[missing]
        )

        result[
            "Missing_MAE"
        ] = _mae(
            missing_prediction,
            missing_target,
        )

        result[
            "Missing_RMSE"
        ] = _rmse(
            missing_prediction,
            missing_target,
        )

    else:

        result[
            "Missing_MAE"
        ] = 0.0

        result[
            "Missing_RMSE"
        ] = 0.0

    return result


# =====================================================================
# CONTROLLED CASE GENERATION
# =====================================================================

def build_controlled_cases() -> list[dict]:
    """
    Build the complete deterministic 750-case experimental matrix.

    6 geological modes
    × 5 missing mechanisms
    × 5 missing rates
    × 5 seeds
    = 750 cases.
    """

    cases = []

    case_id = 1

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    cases.append(
                        {
                            "Case_ID": case_id,
                            "geological_mode": (
                                geological_mode
                            ),
                            "mask_mode": (
                                mask_mode
                            ),
                            "missing_rate": (
                                missing_rate
                            ),
                            "seed": seed,
                        }
                    )

                    case_id += 1

    return cases


# =====================================================================
# CONTROLLED CASE SELECTION
# =====================================================================

def select_cases_for_execution(
    cases: list[dict],
) -> list[dict]:
    """
    Select either the complete 750-case matrix or a deterministic
    smoke-test subset.

    CONTROLLED_MATRIX_CASE_LIMIT = None
        -> execute all 750 cases.

    CONTROLLED_MATRIX_CASE_LIMIT = 10
        -> execute the first 10 deterministic cases.
    """

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):
        return cases

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        < 1
    ):
        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must "
            "be >= 1 or None."
        )

    return cases[
        :CONTROLLED_MATRIX_CASE_LIMIT
    ]


# =====================================================================
# RUN ONE CONTROLLED EXPERIMENT
# =====================================================================

def run_single_experiment(
    case: dict,
) -> dict:
    """
    Run one Curvelet POCS controlled experiment.

    Ground truth is never supplied to the reconstruction function.
    """

    # =================================================================
    # EXTRACT CASE PARAMETERS
    # =================================================================

    case_id = case[
        "Case_ID"
    ]

    geological_mode = case[
        "geological_mode"
    ]

    mask_mode = case[
        "mask_mode"
    ]

    missing_rate = case[
        "missing_rate"
    ]

    seed = case[
        "seed"
    ]

    # =================================================================
    # SET SEED
    # =================================================================

    set_seed(seed)

    # =================================================================
    # CREATE SYNTHETIC DATASET FOR THIS CONTROLLED CASE
    # =================================================================

    dataset = SyntheticSeismicDataset(
        num_samples=(
            BASELINE_NUM_SAMPLES
        ),
        cube_size=(
            BASELINE_CUBE_SIZE
        ),
        missing_probability=(
            missing_rate
        ),
        geological_mode=(
            geological_mode
        ),
        mask_mode=(
            mask_mode
        ),
        seed=seed,
    )

    # =================================================================
    # GET THE CONTROLLED SAMPLE
    # =================================================================

    sample = dataset[0]

    (
        corrupted_cube,
        target_cube,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = sample

    # =================================================================
    # VALIDATE SAMPLE TYPES
    # =================================================================

    if not isinstance(
        corrupted_cube,
        torch.Tensor,
    ):
        raise TypeError(
            "Dataset corrupted_cube must be a torch.Tensor."
        )

    if not isinstance(
        target_cube,
        torch.Tensor,
    ):
        raise TypeError(
            "Dataset target_cube must be a torch.Tensor."
        )

    if not isinstance(
        mask,
        torch.Tensor,
    ):
        raise TypeError(
            "Dataset mask must be a torch.Tensor."
        )

    # =================================================================
    # VALIDATE TARGET SHAPE
    # =================================================================

    if (
        target_cube.shape
        != corrupted_cube.shape
    ):
        raise ValueError(
            "Target and corrupted cube shapes differ. "
            f"Target: {tuple(target_cube.shape)}, "
            f"Corrupted: {tuple(corrupted_cube.shape)}"
        )

    # =================================================================
    # VALIDATE CONTROLLED SHAPE
    # =================================================================

    expected_shape = (
        1,
        *BASELINE_CUBE_SIZE,
    )

    if (
        tuple(corrupted_cube.shape)
        != expected_shape
    ):
        raise ValueError(
            "Unexpected controlled cube shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(corrupted_cube.shape)}."
        )

    # =================================================================
    # VALIDATE FINITE DATA
    # =================================================================

    if not torch.isfinite(
        corrupted_cube
    ).all():
        raise ValueError(
            "Corrupted cube contains NaN or infinite values."
        )

    if not torch.isfinite(
        target_cube
    ).all():
        raise ValueError(
            "Target cube contains NaN or infinite values."
        )

    # =================================================================
    # VALIDATE ACTUAL DATASET METADATA
    # =================================================================

    if (
        actual_mask_mode
        != mask_mode
    ):
        raise RuntimeError(
            "Dataset returned a different mask mode. "
            f"Requested: {mask_mode}; "
            f"Received: {actual_mask_mode}"
        )

    if (
        actual_geological_mode
        != geological_mode
    ):
        raise RuntimeError(
            "Dataset returned a different geological mode. "
            f"Requested: {geological_mode}; "
            f"Received: {actual_geological_mode}"
        )

    # =================================================================
    # CALCULATE ACTUAL MISSING RATE
    # =================================================================

    mask_np = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    actual_missing_rate = float(
        np.mean(
            mask_np == 0
        )
    )

    # =================================================================
    # RUN CURVELET POCS
    # =================================================================

    start_time = time.perf_counter()

    reconstruction = (
        curvelet_pocs_reconstruction(
            corrupted_cube=(
                corrupted_cube
            ),
            mask=mask,
        )
    )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    # =================================================================
    # FINAL OBSERVED-DATA PRESERVATION CHECK
    # =================================================================

    preservation_error = float(
        (
            (
                reconstruction
                - corrupted_cube
            )
            * mask
        ).abs().max().item()
    )

    preservation_pass = (
        preservation_error
        <= OBSERVED_PRESERVATION_TOLERANCE
    )

    if not preservation_pass:
        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum difference: "
            f"{preservation_error:.6e}"
        )

    # =================================================================
    # CONVERT TO NUMPY
    # =================================================================

    reconstruction_np = (
        reconstruction
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    target_np = (
        target_cube
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # =================================================================
    # REMOVE CHANNEL DIMENSION
    #
    # The controlled experiment uses one channel.
    # =================================================================

    reconstruction_volume = (
        reconstruction_np[0]
    )

    target_volume = (
        target_np[0]
    )

    mask_volume = (
        mask_np[0]
    )

    # =================================================================
    # CALCULATE METRICS
    # =================================================================

    metrics = calculate_metrics(
        prediction=(
            reconstruction_volume
        ),
        target=(
            target_volume
        ),
        mask=(
            mask_volume
        ),
    )

    # =================================================================
    # RETURN EXPERIMENT RESULT
    # =================================================================

    return {
        "Case_ID": case_id,
        "geological_mode": (
            geological_mode
        ),
        "mask_mode": (
            mask_mode
        ),
        "requested_missing_rate": (
            missing_rate
        ),
        "actual_missing_rate": (
            actual_missing_rate
        ),
        "seed": seed,
        "cube_depth": (
            BASELINE_CUBE_SIZE[0]
        ),
        "cube_height": (
            BASELINE_CUBE_SIZE[1]
        ),
        "cube_width": (
            BASELINE_CUBE_SIZE[2]
        ),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "PSNR": metrics["PSNR"],
        "SNR": metrics["SNR"],
        "SSIM": metrics["SSIM"],
        "Missing_MAE": (
            metrics["Missing_MAE"]
        ),
        "Missing_RMSE": (
            metrics["Missing_RMSE"]
        ),
        "Runtime_seconds": (
            runtime_seconds
        ),
        "Observed_Preservation_Error": (
            preservation_error
        ),
        "Observed_Preservation_Pass": (
            preservation_pass
        ),
        "status": "SUCCESS",
    }


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def calculate_summary(
    results: list[dict],
) -> list[dict]:
    """
    Calculate grouped summary statistics.

    Grouping factors:
        geological mode
        missing mechanism
        requested missing rate
    """

    if not results:
        return []

    groups = {}

    for result in results:

        key = (
            result[
                "geological_mode"
            ],
            result[
                "mask_mode"
            ],
            result[
                "requested_missing_rate"
            ],
        )

        groups.setdefault(
            key,
            [],
        ).append(result)

    summary = []

    for (
        geological_mode,
        mask_mode,
        missing_rate,
    ), group in groups.items():

        def values(
            field: str,
        ):
            return np.asarray(
                [
                    item[field]
                    for item in group
                    if np.isfinite(
                        item[field]
                    )
                ],
                dtype=np.float64,
            )

        mae = values("MAE")
        rmse = values("RMSE")
        psnr = values("PSNR")
        snr = values("SNR")
        ssim = values("SSIM")
        missing_mae = values(
            "Missing_MAE"
        )
        missing_rmse = values(
            "Missing_RMSE"
        )
        runtime = values(
            "Runtime_seconds"
        )

        summary.append(
            {
                "geological_mode": (
                    geological_mode
                ),
                "mask_mode": (
                    mask_mode
                ),
                "requested_missing_rate": (
                    missing_rate
                ),
                "Number_of_Cases": (
                    len(group)
                ),
                "Mean_MAE": (
                    float(np.mean(mae))
                    if len(mae)
                    else float("nan")
                ),
                "Std_MAE": (
                    float(np.std(mae, ddof=1))
                    if len(mae) > 1
                    else 0.0
                ),
                "Mean_RMSE": (
                    float(np.mean(rmse))
                    if len(rmse)
                    else float("nan")
                ),
                "Std_RMSE": (
                    float(np.std(rmse, ddof=1))
                    if len(rmse) > 1
                    else 0.0
                ),
                "Mean_PSNR": (
                    float(np.mean(psnr))
                    if len(psnr)
                    else float("nan")
                ),
                "Std_PSNR": (
                    float(np.std(psnr, ddof=1))
                    if len(psnr) > 1
                    else 0.0
                ),
                "Mean_SNR": (
                    float(np.mean(snr))
                    if len(snr)
                    else float("nan")
                ),
                "Std_SNR": (
                    float(np.std(snr, ddof=1))
                    if len(snr) > 1
                    else 0.0
                ),
                "Mean_SSIM": (
                    float(np.mean(ssim))
                    if len(ssim)
                    else float("nan")
                ),
                "Std_SSIM": (
                    float(np.std(ssim, ddof=1))
                    if len(ssim) > 1
                    else 0.0
                ),
                "Mean_Missing_MAE": (
                    float(np.mean(missing_mae))
                    if len(missing_mae)
                    else float("nan")
                ),
                "Std_Missing_MAE": (
                    float(
                        np.std(
                            missing_mae,
                            ddof=1,
                        )
                    )
                    if len(missing_mae) > 1
                    else 0.0
                ),
                "Mean_Missing_RMSE": (
                    float(np.mean(missing_rmse))
                    if len(missing_rmse)
                    else float("nan")
                ),
                "Std_Missing_RMSE": (
                    float(
                        np.std(
                            missing_rmse,
                            ddof=1,
                        )
                    )
                    if len(missing_rmse) > 1
                    else 0.0
                ),
                "Mean_Runtime_seconds": (
                    float(np.mean(runtime))
                    if len(runtime)
                    else float("nan")
                ),
                "Std_Runtime_seconds": (
                    float(
                        np.std(
                            runtime,
                            ddof=1,
                        )
                    )
                    if len(runtime) > 1
                    else 0.0
                ),
            }
        )

    return summary


# =====================================================================
# CSV WRITER
# =====================================================================

def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """
    Write a list of dictionaries to CSV.
    """

    import csv

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# =====================================================================
# MAIN CONTROLLED EXPERIMENT
# =====================================================================

def main() -> None:
    """
    Execute the Curvelet POCS controlled experiment.
    """

    print()
    print("=" * 72)
    print(
        "CURVELET POCS CONTROLLED-MATRIX EXPERIMENT"
    )
    print("=" * 72)

    # =================================================================
    # BUILD COMPLETE MATRIX
    # =================================================================

    all_cases = (
        build_controlled_cases()
    )

    print(
        f"Full controlled matrix: "
        f"{len(all_cases)} cases"
    )

    if len(all_cases) != 750:
        raise RuntimeError(
            "Controlled matrix must contain exactly "
            "750 cases. "
            f"Generated: {len(all_cases)}"
        )

    # =================================================================
    # SELECT EXECUTION CASES
    # =================================================================

    cases_to_execute = (
        select_cases_for_execution(
            all_cases
        )
    )

    print(
        f"Cases to execute: "
        f"{len(cases_to_execute)}"
    )

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):
        print(
            "Execution mode: FULL 750-CASE MATRIX"
        )
    else:
        print(
            "Execution mode: SMOKE TEST"
        )

    # =================================================================
    # RESULT STORAGE
    # =================================================================

    results = []

    # =================================================================
    # EXECUTE CASES
    # =================================================================

    for index, case in enumerate(
        cases_to_execute,
        start=1,
    ):

        print()
        print(
            "-" * 72
        )

        print(
            f"Case {index}/"
            f"{len(cases_to_execute)}"
        )

        print(
            f"Case_ID: "
            f"{case['Case_ID']}"
        )

        print(
            f"Geology: "
            f"{case['geological_mode']}"
        )

        print(
            f"Mask: "
            f"{case['mask_mode']}"
        )

        print(
            f"Missing rate: "
            f"{case['missing_rate']:.2f}"
        )

        print(
            f"Seed: "
            f"{case['seed']}"
        )

        try:

            result = (
                run_single_experiment(
                    case
                )
            )

            results.append(
                result
            )

            print(
                f"MAE: "
                f"{result['MAE']:.6f}"
            )

            print(
                f"RMSE: "
                f"{result['RMSE']:.6f}"
            )

            print(
                f"PSNR: "
                f"{result['PSNR']:.4f}"
            )

            print(
                f"SSIM: "
                f"{result['SSIM']:.6f}"
            )

            print(
                f"Missing MAE: "
                f"{result['Missing_MAE']:.6f}"
            )

            print(
                f"Runtime: "
                f"{result['Runtime_seconds']:.3f} s"
            )

            print(
                "Observed preservation: "
                "PASS"
            )

        except Exception as error:

            print(
                "STATUS: FAILED"
            )

            print(
                f"Error: {error}"
            )

            results.append(
                {
                    "Case_ID": (
                        case["Case_ID"]
                    ),
                    "geological_mode": (
                        case[
                            "geological_mode"
                        ]
                    ),
                    "mask_mode": (
                        case[
                            "mask_mode"
                        ]
                    ),
                    "requested_missing_rate": (
                        case[
                            "missing_rate"
                        ]
                    ),
                    "actual_missing_rate": (
                        np.nan
                    ),
                    "seed": (
                        case["seed"]
                    ),
                    "cube_depth": (
                        BASELINE_CUBE_SIZE[0]
                    ),
                    "cube_height": (
                        BASELINE_CUBE_SIZE[1]
                    ),
                    "cube_width": (
                        BASELINE_CUBE_SIZE[2]
                    ),
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "PSNR": np.nan,
                    "SNR": np.nan,
                    "SSIM": np.nan,
                    "Missing_MAE": np.nan,
                    "Missing_RMSE": np.nan,
                    "Runtime_seconds": np.nan,
                    "Observed_Preservation_Error": np.nan,
                    "Observed_Preservation_Pass": False,
                    "status": "FAILED",
                    "error": str(error),
                }
            )

    # =================================================================
    # OUTPUT DIRECTORY
    # =================================================================

    report_directory = Path(
        REPORT_DIR
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # WRITE RAW RESULTS
    # =================================================================

    raw_output = (
        report_directory
        / "curvelet_pocs_controlled_matrix.csv"
    )

    write_csv(
        results,
        raw_output,
    )

    print()
    print(
        f"Raw results saved to:\n"
        f"{raw_output}"
    )

    # =================================================================
    # GENERATE SUMMARY
    # =================================================================

    successful_results = [
        result
        for result in results
        if result["status"]
        == "SUCCESS"
    ]

    summary = (
        calculate_summary(
            successful_results
        )
    )

    # =================================================================
    # WRITE SUMMARY
    # =================================================================

    summary_output = (
        report_directory
        / "curvelet_pocs_controlled_summary.csv"
    )

    write_csv(
        summary,
        summary_output,
    )

    print(
        f"Summary saved to:\n"
        f"{summary_output}"
    )

    # =================================================================
    # FINAL VALIDATION
    # =================================================================

    successful_count = len(
        successful_results
    )

    expected_count = len(
        cases_to_execute
    )

    print()
    print("=" * 72)

    if (
        successful_count
        == expected_count
    ):

        print(
            "CURVELET POCS CONTROLLED MATRIX: PASS"
        )

        print(
            f"Successful cases: "
            f"{successful_count}/"
            f"{expected_count}"
        )

    else:

        print(
            "CURVELET POCS CONTROLLED MATRIX: FAIL"
        )

        print(
            f"Successful cases: "
            f"{successful_count}/"
            f"{expected_count}"
        )

    print("=" * 72)
    print()


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()