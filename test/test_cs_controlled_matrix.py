"""
=========================================================
Controlled Matrix Test — Compressive Sensing Baseline
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the Compressive Sensing (CS) baseline under the
same controlled experimental conditions used for the
other seismic reconstruction baselines.

Controlled experimental matrix
------------------------------
Missing rates:
    10%, 20%, 30%, 40%, 50%

Missing mechanisms:
    random_voxels
    missing_traces
    missing_inlines
    missing_crosslines
    missing_blocks

Geological modes:
    horizontal
    dipping
    faulted
    folded
    complex
    highly_complex

Random seeds:
    42, 43, 44, 45, 46

Total cases:
    6 geological modes
    × 5 missing mechanisms
    × 5 missing rates
    × 5 seeds
    = 750 cases

Metrics
-------
    MAE
    RMSE
    PSNR
    SNR
    SSIM
    Runtime

Additional validation
---------------------
    Shape consistency
    Mask validity
    Input consistency
    Observed-data preservation
    Missing-sample reconstruction
    Finite-value validation
    Failure tracking
    Reproducibility

Outputs
-------
Raw results:
    outputs/synthetic_training/reports/
        cs_controlled_matrix.csv

Summary results:
    outputs/synthetic_training/reports/
        cs_controlled_matrix_summary.csv

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.compressive_sensing import (
    compressive_sensing_reconstruction,
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Synthetic cube size
# ---------------------------------------------------------

CUBE_SIZE = (
    64,
    128,
    128,
)


# ---------------------------------------------------------
# Number of samples per experimental case
# ---------------------------------------------------------

NUM_SAMPLES = 1


# ---------------------------------------------------------
# Controlled missing-data rates
# ---------------------------------------------------------

MISSING_RATES = [
    0.30,
]


# ---------------------------------------------------------
# Controlled missing-data mechanisms
# ---------------------------------------------------------

MASK_MODES = [
    "random_voxels",
    "missing_traces",
]


# ---------------------------------------------------------
# Controlled geological complexity
# ---------------------------------------------------------

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
]


# ---------------------------------------------------------
# Controlled random seeds
# ---------------------------------------------------------

SEEDS = [
    42,
    43,
]


# =========================================================
# COMPRESSIVE SENSING PARAMETERS
# =========================================================

# These parameters are fixed before the controlled
# benchmark and must not be changed between cases.

CS_WAVELET = "db4"

CS_LEVEL = 3

CS_ITERATIONS = 12

CS_THRESHOLD = 0.05

CS_THRESHOLD_DECAY = 0.90

CS_TOLERANCE = 1.0e-5


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

# Use the existing experiment-specific output structure.

OUTPUT_ROOT = Path(
    "outputs"
) / "synthetic_training"

REPORT_DIR = (
    OUTPUT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Raw and summary output files
# ---------------------------------------------------------

RAW_RESULTS_FILE = (
    REPORT_DIR
    / "cs_controlled_matrix.csv"
)

SUMMARY_RESULTS_FILE = (
    REPORT_DIR
    / "cs_controlled_matrix_summary.csv"
)


# =========================================================
# EXPECTED NUMBER OF CASES
# =========================================================

EXPECTED_CASES = (
    len(GEOLOGICAL_MODES)
    * len(MASK_MODES)
    * len(MISSING_RATES)
    * len(SEEDS)
)


# =========================================================
# NUMERICAL VALIDATION
# =========================================================

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6

FINITE_CHECK = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    """

    if isinstance(tensor, torch.Tensor):

        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


# =========================================================
# METRIC HELPER
# =========================================================

def calculate_metrics(
    target,
    reconstruction,
    missing_mask,
):
    """
    Calculate reconstruction metrics.

    Metrics are calculated primarily over the missing
    samples so that the baseline is evaluated on the
    actual reconstruction task.

    Full-volume metrics are also returned for comparison.
    """

    target_np = _to_numpy(
        target
    )

    reconstruction_np = _to_numpy(
        reconstruction
    )

    missing_mask_np = _to_numpy(
        missing_mask
    )

    # -----------------------------------------------------
    # Extract missing samples
    # -----------------------------------------------------

    missing_indices = (
        missing_mask_np == 1
    )

    target_missing = (
        target_np[missing_indices]
    )

    reconstruction_missing = (
        reconstruction_np[missing_indices]
    )

    # -----------------------------------------------------
    # Validate missing data
    # -----------------------------------------------------

    if target_missing.size == 0:

        raise RuntimeError(
            "No missing samples were found."
        )

    # -----------------------------------------------------
    # Missing-sample MAE
    # -----------------------------------------------------

    missing_mae = float(
        np.mean(
            np.abs(
                target_missing
                -
                reconstruction_missing
            )
        )
    )

    # -----------------------------------------------------
    # Missing-sample RMSE
    # -----------------------------------------------------

    missing_rmse = float(
        np.sqrt(
            np.mean(
                (
                    target_missing
                    -
                    reconstruction_missing
                ) ** 2
            )
        )
    )

    # -----------------------------------------------------
    # Full-volume metrics
    # -----------------------------------------------------

    full_mae = float(
        mae(
            target,
            reconstruction,
        )
    )

    full_rmse = float(
        rmse(
            target,
            reconstruction,
        )
    )

    full_psnr = float(
        psnr(
            target,
            reconstruction,
        )
    )

    full_snr = float(
        snr(
            target,
            reconstruction,
        )
    )

    full_ssim = float(
        ssim(
            target,
            reconstruction,
        )
    )

    return {
        "missing_mae": missing_mae,
        "missing_rmse": missing_rmse,
        "mae": full_mae,
        "rmse": full_rmse,
        "psnr": full_psnr,
        "snr": full_snr,
        "ssim": full_ssim,
    }


# =========================================================
# SINGLE EXPERIMENT
# =========================================================

def run_single_experiment(
    geological_mode,
    mask_mode,
    missing_rate,
    seed,
):
    """
    Run one controlled CS experiment.

    One experiment contains:

        dataset generation
        input validation
        CS reconstruction
        output validation
        observed-data preservation
        metric calculation
        runtime measurement
    """

    # =====================================================
    # DATASET CREATION
    # =====================================================

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )

    # -----------------------------------------------------
    # Retrieve first sample
    # -----------------------------------------------------

    sample = dataset[0]

    (
        corrupted,
        target,
        mask,
        velocity_model,
        actual_mask_type,
        actual_geological_mode,
    ) = sample


    # =====================================================
    # BASIC SHAPE VALIDATION
    # =====================================================

    if corrupted.shape != target.shape:

        raise RuntimeError(
            "Corrupted and target tensors "
            "have different shapes."
        )

    if corrupted.shape != mask.shape:

        raise RuntimeError(
            "Corrupted and mask tensors "
            "have different shapes."
        )

    if corrupted.shape != velocity_model.shape:

        raise RuntimeError(
            "Corrupted and velocity tensors "
            "have different shapes."
        )

    # ---------------------------------------------------------
    # Validate spatial cube dimensions
    # ---------------------------------------------------------

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    if tuple(corrupted.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected cube shape: "
            f"{tuple(corrupted.shape)}. "
            f"Expected: {expected_shape}"
        )


    # =====================================================
    # MASK VALIDATION
    # =====================================================

    unique_mask_values = torch.unique(
        mask
    ).detach().cpu().numpy()

    allowed_mask_values = {
        0.0,
        1.0,
    }

    for value in unique_mask_values:

        if float(value) not in allowed_mask_values:

            raise RuntimeError(
                "Mask contains values other "
                "than 0 and 1."
            )


    # =====================================================
    # INPUT CONSISTENCY
    # =====================================================

    expected_corrupted = (
        target * mask
    )

    input_consistency_error = float(
        torch.max(
            torch.abs(
                corrupted
                -
                expected_corrupted
            )
        ).item()
    )

    if (
        input_consistency_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Input consistency validation failed. "
            f"Maximum difference: "
            f"{input_consistency_error:.6e}"
        )


    # =====================================================
    # SAMPLE COUNTS
    # =====================================================

    observed_samples = int(
        torch.sum(
            mask == 1
        ).item()
    )

    missing_samples = int(
        torch.sum(
            mask == 0
        ).item()
    )

    if missing_samples == 0:

        raise RuntimeError(
            "Experimental case contains "
            "no missing samples."
        )


    # =====================================================
    # RUN COMPRESSIVE SENSING
    # =====================================================

    start_time = time.perf_counter()

    reconstruction = (
        compressive_sensing_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            wavelet=CS_WAVELET,
            level=CS_LEVEL,
            iterations=CS_ITERATIONS,
            threshold=CS_THRESHOLD,
            threshold_decay=CS_THRESHOLD_DECAY,
            tolerance=CS_TOLERANCE,
        )
    )

    end_time = time.perf_counter()

    runtime_seconds = (
        end_time
        -
        start_time
    )


    # =====================================================
    # OUTPUT SHAPE VALIDATION
    # =====================================================

    if reconstruction.shape != corrupted.shape:

        raise RuntimeError(
            "CS reconstruction changed "
            "the input shape."
        )


    # =====================================================
    # FINITE-VALUE VALIDATION
    # =====================================================

    if FINITE_CHECK:

        if not torch.isfinite(
            reconstruction
        ).all():

            raise RuntimeError(
                "CS reconstruction contains "
                "NaN or Inf values."
            )


    # =====================================================
    # OBSERVED-DATA PRESERVATION
    # =====================================================

    observed_difference = torch.abs(
        reconstruction[mask == 1]
        -
        corrupted[mask == 1]
    )

    if observed_difference.numel() > 0:

        maximum_observed_difference = float(
            observed_difference.max().item()
        )

    else:

        maximum_observed_difference = 0.0


    if (
        maximum_observed_difference
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed seismic samples "
            "were modified."
        )


    # =====================================================
    # MISSING-SAMPLE RECONSTRUCTION
    # =====================================================

    reconstructed_missing = (
        reconstruction[mask == 0]
    )

    if reconstructed_missing.numel() == 0:

        raise RuntimeError(
            "No missing samples were "
            "reconstructed."
        )

    if not torch.isfinite(
        reconstructed_missing
    ).all():

        raise RuntimeError(
            "Missing-sample reconstruction "
            "contains invalid values."
        )


    # =====================================================
    # METRICS
    # =====================================================

    # Convert the mask so that:
    #
    # 1 = missing
    #
    missing_mask = (
        1.0
        -
        mask
    )

    metrics = calculate_metrics(
        target=target,
        reconstruction=reconstruction,
        missing_mask=missing_mask,
    )


    # =====================================================
    # RESULT RECORD
    # =====================================================

    result = {

        "method":
            "compressive_sensing",

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "actual_geological_mode":
            str(actual_geological_mode),

        "actual_mask_type":
            str(actual_mask_type),

        "requested_missing_rate":
            float(missing_rate),

        "seed":
            int(seed),

        "cube_depth":
            int(CUBE_SIZE[0]),

        "cube_height":
            int(CUBE_SIZE[1]),

        "cube_width":
            int(CUBE_SIZE[2]),

        "observed_samples":
            observed_samples,

        "missing_samples":
            missing_samples,

        "input_consistency_error":
            input_consistency_error,

        "maximum_observed_difference":
            maximum_observed_difference,

        "runtime_seconds":
            runtime_seconds,

        "cs_wavelet":
            CS_WAVELET,

        "cs_level":
            CS_LEVEL,

        "cs_iterations":
            CS_ITERATIONS,

        "cs_threshold":
            CS_THRESHOLD,

        "cs_threshold_decay":
            CS_THRESHOLD_DECAY,

        "cs_tolerance":
            CS_TOLERANCE,

        "missing_mae":
            metrics["missing_mae"],

        "missing_rmse":
            metrics["missing_rmse"],

        "mae":
            metrics["mae"],

        "rmse":
            metrics["rmse"],

        "psnr":
            metrics["psnr"],

        "snr":
            metrics["snr"],

        "ssim":
            metrics["ssim"],

        "status":
            "success",

        "error":
            "",
    }

    return result


# =========================================================
# CSV WRITER
# =========================================================

def write_results(
    results,
    output_file,
):
    """
    Write raw controlled-matrix results
    to CSV.
    """

    if not results:

        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# =========================================================
# SUMMARY GENERATION
# =========================================================

def generate_summary(
    results,
):
    """
    Generate grouped summary statistics.

    Groups are defined by:

        geological mode
        mask mode
        requested missing rate

    Each group contains five seeds.
    """

    successful_results = [
        result
        for result in results
        if result["status"] == "success"
    ]

    grouped = {}

    # -----------------------------------------------------
    # Group results
    # -----------------------------------------------------

    for result in successful_results:

        key = (
            result["geological_mode"],
            result["mask_mode"],
            result["requested_missing_rate"],
        )

        if key not in grouped:

            grouped[key] = []

        grouped[key].append(
            result
        )


    # -----------------------------------------------------
    # Generate summary records
    # -----------------------------------------------------

    summary = []

    for key, group in sorted(
        grouped.items()
    ):

        (
            geological_mode,
            mask_mode,
            missing_rate,
        ) = key

        # -------------------------------------------------
        # Helper for mean and standard deviation
        # -------------------------------------------------

        def values(metric):

            return np.asarray(
                [
                    float(item[metric])
                    for item in group
                ],
                dtype=np.float64,
            )

        # -------------------------------------------------
        # Summary record
        # -------------------------------------------------

        summary.append({

            "method":
                "compressive_sensing",

            "geological_mode":
                geological_mode,

            "mask_mode":
                mask_mode,

            "requested_missing_rate":
                missing_rate,

            "number_of_successful_seeds":
                len(group),

            "missing_mae_mean":
                float(
                    np.mean(
                        values(
                            "missing_mae"
                        )
                    )
                ),

            "missing_mae_std":
                float(
                    np.std(
                        values(
                            "missing_mae"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "missing_rmse_mean":
                float(
                    np.mean(
                        values(
                            "missing_rmse"
                        )
                    )
                ),

            "missing_rmse_std":
                float(
                    np.std(
                        values(
                            "missing_rmse"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "mae_mean":
                float(
                    np.mean(
                        values(
                            "mae"
                        )
                    )
                ),

            "mae_std":
                float(
                    np.std(
                        values(
                            "mae"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "rmse_mean":
                float(
                    np.mean(
                        values(
                            "rmse"
                        )
                    )
                ),

            "rmse_std":
                float(
                    np.std(
                        values(
                            "rmse"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "psnr_mean":
                float(
                    np.mean(
                        values(
                            "psnr"
                        )
                    )
                ),

            "psnr_std":
                float(
                    np.std(
                        values(
                            "psnr"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "snr_mean":
                float(
                    np.mean(
                        values(
                            "snr"
                        )
                    )
                ),

            "snr_std":
                float(
                    np.std(
                        values(
                            "snr"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "ssim_mean":
                float(
                    np.mean(
                        values(
                            "ssim"
                        )
                    )
                ),

            "ssim_std":
                float(
                    np.std(
                        values(
                            "ssim"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,

            "runtime_mean":
                float(
                    np.mean(
                        values(
                            "runtime_seconds"
                        )
                    )
                ),

            "runtime_std":
                float(
                    np.std(
                        values(
                            "runtime_seconds"
                        ),
                        ddof=1,
                    )
                )
                if len(group) > 1
                else 0.0,
        })


    return summary


# =========================================================
# MAIN CONTROLLED MATRIX
# =========================================================

def main():

    print(
        "\n"
        "=========================================================\n"
        "COMPRESSIVE SENSING CONTROLLED MATRIX\n"
        "========================================================="
    )

    print(
        f"\nExpected cases: "
        f"{EXPECTED_CASES}"
    )

    print(
        f"Cube size: "
        f"{CUBE_SIZE}"
    )

    print(
        f"CS wavelet: "
        f"{CS_WAVELET}"
    )

    print(
        f"CS level: "
        f"{CS_LEVEL}"
    )

    print(
        f"CS iterations: "
        f"{CS_ITERATIONS}"
    )

    print(
        f"CS threshold: "
        f"{CS_THRESHOLD}"
    )

    print(
        f"CS threshold decay: "
        f"{CS_THRESHOLD_DECAY}"
    )

    print(
        f"CS tolerance: "
        f"{CS_TOLERANCE}"
    )

    print(
        "\nControlled geological modes:"
    )

    for mode in GEOLOGICAL_MODES:

        print(
            f"  - {mode}"
        )

    print(
        "\nControlled missing mechanisms:"
    )

    for mode in MASK_MODES:

        print(
            f"  - {mode}"
        )

    print(
        "\nControlled missing rates:"
    )

    for rate in MISSING_RATES:

        print(
            f"  - {rate:.0%}"
        )

    print(
        "\nControlled seeds:"
    )

    for seed in SEEDS:

        print(
            f"  - {seed}"
        )


    # =====================================================
    # RESULT STORAGE
    # =====================================================

    results = []

    completed_cases = 0

    successful_cases = 0

    failed_cases = 0


    # =====================================================
    # CONTROLLED EXPERIMENT LOOP
    # =====================================================

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    completed_cases += 1

                    print(
                        "\n"
                        "---------------------------------------------------------"
                    )

                    print(
                        f"Case "
                        f"{completed_cases}/"
                        f"{EXPECTED_CASES}"
                    )

                    print(
                        f"Geology       : "
                        f"{geological_mode}"
                    )

                    print(
                        f"Mask          : "
                        f"{mask_mode}"
                    )

                    print(
                        f"Missing rate  : "
                        f"{missing_rate:.0%}"
                    )

                    print(
                        f"Seed          : "
                        f"{seed}"
                    )

                    # -------------------------------------------------
                    # Run experiment
                    # -------------------------------------------------

                    try:

                        result = (
                            run_single_experiment(
                                geological_mode=geological_mode,
                                mask_mode=mask_mode,
                                missing_rate=missing_rate,
                                seed=seed,
                            )
                        )

                        successful_cases += 1

                        print(
                            "Status        : SUCCESS"
                        )

                        print(
                            f"Missing MAE   : "
                            f"{result['missing_mae']:.6f}"
                        )

                        print(
                            f"Missing RMSE  : "
                            f"{result['missing_rmse']:.6f}"
                        )

                        print(
                            f"MAE           : "
                            f"{result['mae']:.6f}"
                        )

                        print(
                            f"RMSE          : "
                            f"{result['rmse']:.6f}"
                        )

                        print(
                            f"PSNR          : "
                            f"{result['psnr']:.6f} dB"
                        )

                        print(
                            f"SNR           : "
                            f"{result['snr']:.6f} dB"
                        )

                        print(
                            f"SSIM          : "
                            f"{result['ssim']:.6f}"
                        )

                        print(
                            f"Runtime       : "
                            f"{result['runtime_seconds']:.4f} s"
                        )

                    except Exception as exc:

                        failed_cases += 1

                        result = {

                            "method":
                                "compressive_sensing",

                            "geological_mode":
                                geological_mode,

                            "mask_mode":
                                mask_mode,

                            "actual_geological_mode":
                                "",

                            "actual_mask_type":
                                "",

                            "requested_missing_rate":
                                float(missing_rate),

                            "seed":
                                int(seed),

                            "cube_depth":
                                int(CUBE_SIZE[0]),

                            "cube_height":
                                int(CUBE_SIZE[1]),

                            "cube_width":
                                int(CUBE_SIZE[2]),

                            "observed_samples":
                                0,

                            "missing_samples":
                                0,

                            "input_consistency_error":
                                np.nan,

                            "maximum_observed_difference":
                                np.nan,

                            "runtime_seconds":
                                np.nan,

                            "cs_wavelet":
                                CS_WAVELET,

                            "cs_level":
                                CS_LEVEL,

                            "cs_iterations":
                                CS_ITERATIONS,

                            "cs_threshold":
                                CS_THRESHOLD,

                            "cs_threshold_decay":
                                CS_THRESHOLD_DECAY,

                            "cs_tolerance":
                                CS_TOLERANCE,

                            "missing_mae":
                                np.nan,

                            "missing_rmse":
                                np.nan,

                            "mae":
                                np.nan,

                            "rmse":
                                np.nan,

                            "psnr":
                                np.nan,

                            "snr":
                                np.nan,

                            "ssim":
                                np.nan,

                            "status":
                                "failed",

                            "error":
                                str(exc),
                        }

                        print(
                            "Status        : FAILED"
                        )

                        print(
                            f"Error         : "
                            f"{exc}"
                        )

                    results.append(
                        result
                    )


    # =====================================================
    # WRITE RAW RESULTS
    # =====================================================

    write_results(
        results,
        RAW_RESULTS_FILE,
    )


    # =====================================================
    # GENERATE SUMMARY
    # =====================================================

    summary = generate_summary(
        results
    )


    # =====================================================
    # WRITE SUMMARY RESULTS
    # =====================================================

    if summary:

        summary_fieldnames = list(
            summary[0].keys()
        )

        with open(
            SUMMARY_RESULTS_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=summary_fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                summary
            )


    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if completed_cases != EXPECTED_CASES:

        raise RuntimeError(
            "Controlled matrix did not execute "
            "the expected number of cases.\n"
            f"Expected: {EXPECTED_CASES}\n"
            f"Completed: {completed_cases}"
        )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    print(
        "\n"
        "=========================================================\n"
        "COMPRESSIVE SENSING CONTROLLED MATRIX COMPLETE\n"
        "========================================================="
    )

    print(
        f"\nExpected cases : "
        f"{EXPECTED_CASES}"
    )

    print(
        f"Completed cases: "
        f"{completed_cases}"
    )

    print(
        f"Successful     : "
        f"{successful_cases}"
    )

    print(
        f"Failed         : "
        f"{failed_cases}"
    )

    print(
        "\nRaw results:"
    )

    print(
        f"  {RAW_RESULTS_FILE}"
    )

    print(
        "\nSummary results:"
    )

    print(
        f"  {SUMMARY_RESULTS_FILE}"
    )

    print(
        "\n========================================================="
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()