"""
=================================================================
Compressive Sensing Controlled-Matrix Baseline
=================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Baseline method:
    Compressive Sensing (CS)

Methodological basis:
    Sparse seismic reconstruction in a 3D wavelet domain
    using iterative soft-thresholding and exact data consistency.

Input convention:
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask convention:
    1 -> observed
    0 -> missing

Controlled experimental design:
    Geological modes:
        horizontal
        dipping
        faulted
        folded
        complex
        highly_complex

    Missing-data mechanisms:
        random_voxels
        missing_traces
        missing_inlines
        missing_crosslines
        missing_blocks

    Missing rates:
        10%, 20%, 30%, 40%, 50%

    Random seeds:
        42, 43, 44, 45, 46

    Total:
        6 × 5 × 5 × 5 = 750 cases

For smoke testing:
    CONTROLLED_MATRIX_CASE_LIMIT = 10

For final experiments:
    CONTROLLED_MATRIX_CASE_LIMIT = None

Observed seismic samples are NEVER modified.

This file contains both:
    1. The classical CS reconstruction algorithm.
    2. The controlled experimental evaluation layer.

Author: Ormin Joseph
=================================================================
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "PyWavelets is required for the Compressive Sensing "
        "baseline.\n\n"
        "Install it with:\n"
        "pip install PyWavelets"
    ) from exc


# ==============================================================
# PROJECT CONFIGURATION
# ==============================================================

from utils.config import (
    BASELINE_NUM_SAMPLES,
    BASELINE_CUBE_SIZE,
    OBSERVED_PRESERVATION_TOLERANCE,
    REPORT_DIR,
    CONTROLLED_MATRIX_CASE_LIMIT,
)


# ==============================================================
# CONTROLLED EXPERIMENT FACTORS
# ==============================================================

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


# ==============================================================
# CS PARAMETERS
# ==============================================================

DEFAULT_WAVELET = "db4"

DEFAULT_LEVEL = 3

DEFAULT_ITERATIONS = 12

DEFAULT_THRESHOLD = 0.05

DEFAULT_THRESHOLD_DECAY = 0.90

DEFAULT_TOLERANCE = 1.0e-5


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# ==============================================================
# INPUT VALIDATION
# ==============================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Validate the input seismic cube and observation mask.
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
            f"Received {tuple(corrupted_cube.shape)}."
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received {tuple(mask.shape)}."
        )

    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have "
            "identical shapes.\n"
            f"Cube: {tuple(corrupted_cube.shape)}\n"
            f"Mask: {tuple(mask.shape)}"
        )

    check_finite(
        corrupted_cube,
        "corrupted_cube",
    )

    check_finite(
        mask,
        "mask",
    )

    unique_mask_values = torch.unique(mask)

    for value in unique_mask_values:

        is_zero = torch.isclose(
            value,
            torch.tensor(
                0.0,
                device=value.device,
                dtype=value.dtype,
            ),
        )

        is_one = torch.isclose(
            value,
            torch.tensor(
                1.0,
                device=value.device,
                dtype=value.dtype,
            ),
        )

        if not (is_zero or is_one):
            raise ValueError(
                "mask must contain only 0 and 1."
            )

    if not torch.any(mask == 1):
        raise ValueError(
            "mask contains no observed samples."
        )


# ==============================================================
# PARAMETER VALIDATION
# ==============================================================

def _validate_parameters(
    wavelet: str,
    level: int,
    iterations: int,
    threshold: float,
    threshold_decay: float,
    tolerance: float,
) -> None:
    """
    Validate CS algorithm parameters.
    """

    if not isinstance(
        wavelet,
        str,
    ):
        raise TypeError(
            "wavelet must be a string."
        )

    if not isinstance(
        level,
        int,
    ):
        raise TypeError(
            "level must be an integer."
        )

    if level < 1:
        raise ValueError(
            "level must be >= 1."
        )

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

    try:
        pywt.Wavelet(wavelet)
    except Exception as exc:
        raise ValueError(
            f"Unknown wavelet: {wavelet}"
        ) from exc


# ==============================================================
# WAVELET LEVEL SAFETY
# ==============================================================

def _safe_wavelet_level(
    shape,
    wavelet: str,
    requested_level: int,
) -> int:
    """
    Determine a valid decomposition level for a 3D volume.

    The requested level is capped by PyWavelets' maximum
    permissible level for the smallest spatial dimension.
    """

    wavelet_object = pywt.Wavelet(
        wavelet
    )

    filter_length = (
        wavelet_object.dec_len
    )

    maximum_level = min(
        pywt.dwt_max_level(
            data_len=int(dimension),
            filter_len=filter_length,
        )
        for dimension in shape
    )

    if maximum_level < 1:
        raise ValueError(
            "The supplied volume is too small for "
            f"wavelet '{wavelet}'. "
            f"Volume shape: {tuple(shape)}."
        )

    return min(
        requested_level,
        maximum_level,
    )


# ==============================================================
# SOFT THRESHOLDING
# ==============================================================

def _soft_threshold(
    coefficients,
    threshold: float,
):
    """
    Apply element-wise soft thresholding.

    S_lambda(x)
        = sign(x) * max(|x| - lambda, 0)
    """

    return (
        np.sign(coefficients)
        *
        np.maximum(
            np.abs(coefficients) - threshold,
            0.0,
        )
    )


# ==============================================================
# WAVELET COEFFICIENT THRESHOLDING
# ==============================================================

def _threshold_wavelet_coefficients(
    coefficients,
    threshold: float,
):
    """
    Apply soft thresholding to wavelet detail coefficients.

    Approximation coefficients are preserved.
    """

    thresholded = [
        coefficients[0].copy()
    ]

    for detail_level in coefficients[1:]:

        thresholded_details = {}

        for key, values in detail_level.items():

            thresholded_details[key] = (
                _soft_threshold(
                    values,
                    threshold,
                )
            )

        thresholded.append(
            thresholded_details
        )

    return thresholded


# ==============================================================
# SINGLE WAVELET RECONSTRUCTION
# ==============================================================

def _wavelet_reconstruct(
    volume: np.ndarray,
    wavelet: str,
    level: int,
    threshold: float,
) -> np.ndarray:
    """
    Perform one sparse wavelet reconstruction step.
    """

    safe_level = _safe_wavelet_level(
        volume.shape,
        wavelet,
        level,
    )

    coefficients = pywt.wavedecn(
        volume,
        wavelet=wavelet,
        level=safe_level,
        mode="periodization",
    )

    thresholded = (
        _threshold_wavelet_coefficients(
            coefficients,
            threshold,
        )
    )

    reconstructed = pywt.waverecn(
        thresholded,
        wavelet=wavelet,
        mode="periodization",
    )

    reconstructed = reconstructed[
        :volume.shape[0],
        :volume.shape[1],
        :volume.shape[2],
    ]

    reconstructed = reconstructed.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        reconstructed
    ).all():
        raise FloatingPointError(
            "Wavelet reconstruction produced "
            "NaN or Inf values."
        )

    return reconstructed


# ==============================================================
# SINGLE-CHANNEL COMPRESSIVE SENSING
# ==============================================================

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

    Observed samples are enforced after every iteration.
    """

    corrupted = np.asarray(
        corrupted,
        dtype=np.float32,
    )

    observed_mask = np.asarray(
        observed_mask,
        dtype=np.float32,
    )

    reconstruction = corrupted.copy()

    previous = reconstruction.copy()

    current_threshold = float(
        threshold
    )

    for _ in range(iterations):

        sparse_estimate = (
            _wavelet_reconstruct(
                reconstruction,
                wavelet=wavelet,
                level=level,
                threshold=current_threshold,
            )
        )

        reconstruction = (
            observed_mask
            * corrupted
            +
            (1.0 - observed_mask)
            * sparse_estimate
        )

        if not np.isfinite(
            reconstruction
        ).all():
            raise FloatingPointError(
                "CS reconstruction produced "
                "NaN or Inf values."
            )

        difference = np.linalg.norm(
            reconstruction - previous
        )

        reference = max(
            np.linalg.norm(previous),
            1.0e-12,
        )

        relative_change = (
            difference / reference
        )

        current_threshold *= (
            threshold_decay
        )

        previous = reconstruction.copy()

        if relative_change < tolerance:
            break

    return reconstruction


# ==============================================================
# PUBLIC CS RECONSTRUCTION
# ==============================================================

def compressive_sensing_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    iterations: int = DEFAULT_ITERATIONS,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_decay: float = DEFAULT_THRESHOLD_DECAY,
    tolerance: float = DEFAULT_TOLERANCE,
) -> torch.Tensor:
    """
    Reconstruct a 3D seismic cube using wavelet-domain
    compressive sensing.

    Parameters
    ----------
    corrupted_cube:
        Tensor with shape (C, D, H, W).

    mask:
        Binary tensor with shape (C, D, H, W).

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube with the same shape,
        dtype and device as the input.
    """

    _validate_inputs(
        corrupted_cube,
        mask,
    )

    _validate_parameters(
        wavelet,
        level,
        iterations,
        threshold,
        threshold_decay,
        tolerance,
    )

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

    corrupted_numpy = (
        corrupted_cpu.numpy()
    )

    mask_numpy = (
        mask_cpu.numpy()
    )

    reconstructed_numpy = (
        corrupted_numpy.copy()
    )

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
                tolerance=tolerance,
            )
        )

    # ----------------------------------------------------------
    # FINAL DATA-CONSISTENCY PROJECTION
    # ----------------------------------------------------------

    reconstructed_numpy = (
        mask_numpy
        * corrupted_numpy
        +
        (1.0 - mask_numpy)
        * reconstructed_numpy
    )

    if not np.isfinite(
        reconstructed_numpy
    ).all():
        raise FloatingPointError(
            "Final CS reconstruction contains "
            "NaN or Inf values."
        )

    reconstructed = torch.from_numpy(
        reconstructed_numpy
    )

    if original_dtype != torch.float32:
        reconstructed = reconstructed.to(
            dtype=original_dtype
        )

    reconstructed = reconstructed.to(
        device=original_device
    )

    if reconstructed.shape != (
        corrupted_cube.shape
    ):
        raise RuntimeError(
            "CS reconstruction changed the input shape.\n"
            f"Input: {tuple(corrupted_cube.shape)}\n"
            f"Output: {tuple(reconstructed.shape)}"
        )

    # ----------------------------------------------------------
    # EXACT OBSERVED-DATA PRESERVATION
    # ----------------------------------------------------------

    observed_difference = torch.abs(
        reconstructed[mask == 1]
        -
        corrupted_cube[mask == 1]
    )

    if observed_difference.numel() > 0:

        maximum_difference = (
            observed_difference.max()
        )

        if (
            maximum_difference.item()
            >
            OBSERVED_PRESERVATION_TOLERANCE
        ):
            raise RuntimeError(
                "CS reconstruction failed to preserve "
                "observed seismic samples.\n"
                f"Maximum difference: "
                f"{maximum_difference.item():.6e}"
            )

    return reconstructed


# ==============================================================
# METRIC FUNCTIONS
# ==============================================================

def _mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return torch.mean(
        torch.abs(
            prediction - target
        )
    ).item()


def _rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return torch.sqrt(
        torch.mean(
            (prediction - target) ** 2
        )
    ).item()


def _psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> float:

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse <= 0.0:
        return float("inf")

    return float(
        20.0 * math.log10(
            data_range
        )
        -
        10.0 * math.log10(
            mse
        )
    )


def _snr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    signal_power = torch.mean(
        target ** 2
    ).item()

    noise_power = torch.mean(
        (target - prediction) ** 2
    ).item()

    if noise_power <= 0.0:
        return float("inf")

    if signal_power <= 0.0:
        return float("-inf")

    return float(
        10.0
        *
        math.log10(
            signal_power
            /
            noise_power
        )
    )


def _ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """
    Global SSIM-style comparison using means, variances,
    and covariance.

    This is used only as a controlled-baseline metric.
    """

    x = prediction.float()
    y = target.float()

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    variance_x = torch.var(
        x,
        unbiased=False,
    )

    variance_y = torch.var(
        y,
        unbiased=False,
    )

    covariance = torch.mean(
        (x - mean_x)
        *
        (y - mean_y)
    )

    c1 = (
        0.01 * data_range
    ) ** 2

    c2 = (
        0.03 * data_range
    ) ** 2

    numerator = (
        (2.0 * mean_x * mean_y + c1)
        *
        (2.0 * covariance + c2)
    )

    denominator = (
        (mean_x ** 2 + mean_y ** 2 + c1)
        *
        (variance_x + variance_y + c2)
    )

    if denominator.item() == 0.0:
        return 1.0

    return float(
        numerator.div(
            denominator
        ).item()
    )


def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate global and missing-region reconstruction metrics.
    """

    reconstruction = (
        reconstruction.float()
    )

    target = target.float()

    mask = mask.float()

    missing = (
        mask == 0
    )

    metrics = {
        "MAE": _mae(
            reconstruction,
            target,
        ),
        "RMSE": _rmse(
            reconstruction,
            target,
        ),
        "PSNR": _psnr(
            reconstruction,
            target,
        ),
        "SNR": _snr(
            reconstruction,
            target,
        ),
        "SSIM": _ssim(
            reconstruction,
            target,
        ),
    }

    if torch.any(missing):

        missing_prediction = (
            reconstruction[missing]
        )

        missing_target = (
            target[missing]
        )

        metrics["Missing_MAE"] = (
            torch.mean(
                torch.abs(
                    missing_prediction
                    -
                    missing_target
                )
            ).item()
        )

        metrics["Missing_RMSE"] = (
            torch.sqrt(
                torch.mean(
                    (
                        missing_prediction
                        -
                        missing_target
                    ) ** 2
                )
            ).item()
        )

    else:

        metrics["Missing_MAE"] = 0.0

        metrics["Missing_RMSE"] = 0.0

    return metrics


# ==============================================================
# CONTROLLED CASE GENERATION
# ==============================================================

def build_controlled_cases() -> List[dict]:
    """
    Construct the complete 750-case factorial design.

    6 geology × 5 masks × 5 missing rates × 5 seeds.
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
                            "geological_mode": geological_mode,
                            "mask_mode": mask_mode,
                            "requested_missing_rate": missing_rate,
                            "seed": seed,
                        }
                    )

                    case_id += 1

    return cases


def select_cases_for_execution(
    cases: List[dict],
    case_limit,
) -> List[dict]:
    """
    Select the number of cases specified by config.

    None means execute all cases.
    """

    if case_limit is None:
        return cases

    if not isinstance(
        case_limit,
        int,
    ):
        raise TypeError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "an integer or None."
        )

    if case_limit < 1:
        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            ">= 1 or None."
        )

    return cases[
        :case_limit
    ]


# ==============================================================
# SINGLE CONTROLLED EXPERIMENT
# ==============================================================

def run_single_experiment(
    case: dict,
) -> dict:
    """
    Run one controlled CS experiment.
    """

    from dataset.synthetic_dataset import (
        SyntheticSeismicDataset,
    )

    set_seed(
        case["seed"]
    )

    dataset = SyntheticSeismicDataset(
        num_samples=BASELINE_NUM_SAMPLES,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=case[
            "requested_missing_rate"
        ],
        geological_mode=case[
            "geological_mode"
        ],
        mask_mode=case[
            "mask_mode"
        ],
        seed=case["seed"],
    )

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    del velocity

    if corrupted_cube.ndim != 4:
        raise ValueError(
            "Expected corrupted_cube shape "
            "(C, D, H, W). "
            f"Received {tuple(corrupted_cube.shape)}."
        )

    if target.shape != corrupted_cube.shape:
        raise ValueError(
            "Target and corrupted cube shapes differ."
        )

    if mask.shape != corrupted_cube.shape:
        raise ValueError(
            "Mask and corrupted cube shapes differ."
        )

    check_finite(
        corrupted_cube,
        "corrupted_cube",
    )

    check_finite(
        target,
        "target",
    )

    check_finite(
        mask,
        "mask",
    )

    actual_missing_rate = float(
        torch.mean(
            (mask == 0).float()
        ).item()
    )

    start_time = time.perf_counter()

    reconstruction = (
        compressive_sensing_reconstruction(
            corrupted_cube=corrupted_cube,
            mask=mask,
        )
    )

    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )

    check_finite(
        reconstruction,
        "reconstruction",
    )

    # ----------------------------------------------------------
    # FINAL DATA CONSISTENCY
    # ----------------------------------------------------------

    observed_error = torch.abs(
        reconstruction[mask == 1]
        -
        corrupted_cube[mask == 1]
    )

    if observed_error.numel() > 0:

        observed_preservation_error = float(
            observed_error.max().item()
        )

    else:

        observed_preservation_error = 0.0

    if (
        observed_preservation_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):
        raise RuntimeError(
            "Observed-data preservation tolerance exceeded."
        )

    # ----------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    result = {
        "Case_ID": case["Case_ID"],
        "Method": "Compressive_Sensing",
        "geological_mode": actual_geological_mode,
        "mask_mode": actual_mask_mode,
        "requested_missing_rate": case[
            "requested_missing_rate"
        ],
        "actual_missing_rate": actual_missing_rate,
        "seed": case["seed"],
        "Num_Samples": BASELINE_NUM_SAMPLES,
        "Cube_D": BASELINE_CUBE_SIZE[0],
        "Cube_H": BASELINE_CUBE_SIZE[1],
        "Cube_W": BASELINE_CUBE_SIZE[2],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "PSNR": metrics["PSNR"],
        "SNR": metrics["SNR"],
        "SSIM": metrics["SSIM"],
        "Missing_MAE": metrics[
            "Missing_MAE"
        ],
        "Missing_RMSE": metrics[
            "Missing_RMSE"
        ],
        "Runtime_Seconds": runtime_seconds,
        "Observed_Preservation_Error": (
            observed_preservation_error
        ),
        "Wavelet": DEFAULT_WAVELET,
        "Wavelet_Level": DEFAULT_LEVEL,
        "Iterations": DEFAULT_ITERATIONS,
        "Threshold": DEFAULT_THRESHOLD,
        "Threshold_Decay": DEFAULT_THRESHOLD_DECAY,
        "Tolerance": DEFAULT_TOLERANCE,
        "Status": "PASS",
    }

    return result


# ==============================================================
# SUMMARY STATISTICS
# ==============================================================

def calculate_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate grouped summary statistics.

    Grouping:
        geological mode
        mask mode
        requested missing rate
    """

    group_columns = [
        "geological_mode",
        "mask_mode",
        "requested_missing_rate",
    ]

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "Missing_MAE",
        "Missing_RMSE",
        "Runtime_Seconds",
        "Observed_Preservation_Error",
    ]

    summary = (
        dataframe
        .groupby(
            group_columns,
            dropna=False,
        )[metric_columns]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        "_".join(
            column
        ).strip("_")
        if isinstance(
            column,
            tuple,
        )
        else column
        for column in summary.columns
    ]

    return summary


# ==============================================================
# CSV WRITER
# ==============================================================

def write_results(
    results: List[dict],
    report_directory: Path,
) -> None:
    """
    Write raw and summary CS controlled-matrix results.
    """

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    )

    raw_path = (
        report_directory
        /
        "cs_controlled_matrix.csv"
    )

    summary_path = (
        report_directory
        /
        "cs_controlled_matrix_summary.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False,
    )

    summary = calculate_summary(
        dataframe
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nRaw results saved to:\n"
        f"{raw_path}"
    )

    print(
        f"\nSummary results saved to:\n"
        f"{summary_path}"
    )


# ==============================================================
# MAIN CONTROLLED EXPERIMENT
# ==============================================================

def main() -> None:
    """
    Execute the controlled CS experiment.
    """

    print(
        "\n"
        "=" * 70
    )

    print(
        "COMPRESSIVE SENSING CONTROLLED MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        f"Cube size: {BASELINE_CUBE_SIZE}"
    )

    print(
        f"Samples per case: "
        f"{BASELINE_NUM_SAMPLES}"
    )

    print(
        f"Wavelet: {DEFAULT_WAVELET}"
    )

    print(
        f"Wavelet level: {DEFAULT_LEVEL}"
    )

    print(
        f"Iterations: {DEFAULT_ITERATIONS}"
    )

    print(
        f"Threshold: {DEFAULT_THRESHOLD}"
    )

    print(
        f"Threshold decay: "
        f"{DEFAULT_THRESHOLD_DECAY}"
    )

    print(
        f"Convergence tolerance: "
        f"{DEFAULT_TOLERANCE}"
    )

    # ----------------------------------------------------------
    # BUILD COMPLETE DESIGN
    # ----------------------------------------------------------

    all_cases = (
        build_controlled_cases()
    )

    print(
        f"\nFull controlled matrix: "
        f"{len(all_cases)} cases"
    )

    if len(all_cases) != 750:
        raise RuntimeError(
            "Controlled matrix construction failed. "
            f"Expected 750 cases, got "
            f"{len(all_cases)}."
        )

    # ----------------------------------------------------------
    # SELECT EXECUTION SET
    # ----------------------------------------------------------

    cases_to_execute = (
        select_cases_for_execution(
            all_cases,
            CONTROLLED_MATRIX_CASE_LIMIT,
        )
    )

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):
        execution_mode = (
            "FULL EXPERIMENT"
        )

    else:
        execution_mode = (
            "SMOKE TEST"
        )

    print(
        f"Cases to execute: "
        f"{len(cases_to_execute)}"
    )

    print(
        f"Execution mode: "
        f"{execution_mode}"
    )

    # ----------------------------------------------------------
    # EXECUTE CASES
    # ----------------------------------------------------------

    results = []

    for index, case in enumerate(
        cases_to_execute,
        start=1,
    ):

        print(
            f"\n"
            f"[{index}/"
            f"{len(cases_to_execute)}]"
            f" Case {case['Case_ID']}"
        )

        print(
            f"    Geology: "
            f"{case['geological_mode']}"
        )

        print(
            f"    Mask: "
            f"{case['mask_mode']}"
        )

        print(
            f"    Missing rate: "
            f"{case['requested_missing_rate']:.0%}"
        )

        print(
            f"    Seed: "
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
                f"    MAE: "
                f"{result['MAE']:.6f}"
            )

            print(
                f"    RMSE: "
                f"{result['RMSE']:.6f}"
            )

            print(
                f"    SSIM: "
                f"{result['SSIM']:.6f}"
            )

            print(
                f"    Missing MAE: "
                f"{result['Missing_MAE']:.6f}"
            )

            print(
                f"    Runtime: "
                f"{result['Runtime_Seconds']:.3f} s"
            )

            print(
                "    Status: PASS"
            )

        except Exception as exc:

            print(
                f"    Status: FAIL"
            )

            print(
                f"    Error: {exc}"
            )

    # ----------------------------------------------------------
    # VALIDATE EXECUTION COMPLETION
    # ----------------------------------------------------------

    if len(results) != len(
        cases_to_execute
    ):
        raise RuntimeError(
            "Controlled CS experiment did not "
            "complete all requested cases.\n"
            f"Expected: {len(cases_to_execute)}\n"
            f"Completed: {len(results)}"
        )

    # ----------------------------------------------------------
    # WRITE RESULTS
    # ----------------------------------------------------------

    write_results(
        results,
        Path(REPORT_DIR),
    )

    # ----------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------

    print(
        "\n"
        "=" * 70
    )

    print(
        "CONTROLLED CS EXPERIMENT: PASS"
    )

    print(
        "=" * 70
    )

    print(
        f"Successful cases: "
        f"{len(results)} / "
        f"{len(cases_to_execute)}"
    )

    print(
        f"Full matrix definition: "
        f"{len(all_cases)} cases"
    )

    print(
        f"Execution mode: "
        f"{execution_mode}"
    )

    print(
        "=" * 70
    )


# ==============================================================
# MODULE ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()