"""
====================================================================
Nearest Neighbor Controlled Experimental Matrix
====================================================================

Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction

Controlled benchmark for the Nearest Neighbor baseline.

Experimental matrix
-------------------
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
    5 missing rates
    × 5 missing mechanisms
    × 6 geological modes
    × 5 seeds
    = 750 cases

Standard cube size
------------------
    (D, H, W) = (64, 128, 128)

Tensor convention
-----------------
    (C, D, H, W)

For this experiment:
    (1, 64, 128, 128)

The reconstruction baseline receives exactly the same
corrupted input and mask used in the controlled experiment.

Observed samples must remain unchanged.

Outputs
-------
Raw results:
    outputs/synthetic_training/reports/
        nearest_neighbor_controlled_matrix.csv

Summary results:
    outputs/synthetic_training/reports/
        nearest_neighbor_controlled_matrix_summary.csv

Author: Ormin Joseph
====================================================================
"""

# ====================================================================
# IMPORTS
# ====================================================================

import csv
import os
import time

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.baseline_nearest_neighbor import (
    nearest_neighbor_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ====================================================================
# EXPERIMENT CONFIGURATION
# ====================================================================

# Standard controlled benchmark cube.
CUBE_SIZE = (64, 128, 128)

# Number of channels.
CHANNELS = 1

# Missing rates.
MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]

# Missing mechanisms.
MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks"
]

# Geological complexity modes.
GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex"
]

# Controlled random seeds.
SEEDS = [
    42,
    43,
    44,
    45,
    46
]

# One synthetic sample is sufficient for each controlled case.
SAMPLES_PER_CASE = 1

# Expected number of experiments.
EXPECTED_CASES = (
    len(MISSING_RATES)
    * len(MASK_MODES)
    * len(GEOLOGICAL_MODES)
    * len(SEEDS)
)

# CPU is used because the nearest-neighbor operation relies on
# SciPy's distance transform.
DEVICE = torch.device("cpu")

# Output directory.
OUTPUT_DIR = os.path.join(
    "outputs",
    "synthetic_training",
    "reports"
)

# Raw result file.
RAW_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "nearest_neighbor_controlled_matrix.csv"
)

# Summary result file.
SUMMARY_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "nearest_neighbor_controlled_matrix_summary.csv"
)


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def to_float(value):
    """
    Convert a metric result into a standard Python float.

    This makes the values safe for CSV writing and statistical
    processing.
    """

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    return float(value)


def ensure_output_directory():
    """
    Create the output directory if it does not already exist.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ====================================================================
# SINGLE CONTROLLED CASE
# ====================================================================

def run_single_case(
        missing_rate,
        mask_mode,
        geological_mode,
        seed
):
    """
    Run one controlled Nearest Neighbor experiment.

    Parameters
    ----------
    missing_rate : float
        Fraction of the seismic cube that is missing.

    mask_mode : str
        Missing-data mechanism.

    geological_mode : str
        Geological complexity.

    seed : int
        Random seed.

    Returns
    -------
    dict
        Results for the controlled experiment.
    """

    # --------------------------------------------------------------
    # Construct exactly one synthetic sample.
    # --------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=SAMPLES_PER_CASE,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed
    )

    # --------------------------------------------------------------
    # Retrieve the single controlled sample.
    #
    # The canonical dataset returns:
    #
    # input_cube
    # target
    # mask
    # velocity
    # mask_type
    # geological_mode
    # --------------------------------------------------------------

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode
    ) = dataset[0]

    # --------------------------------------------------------------
    # Move tensors to CPU.
    # --------------------------------------------------------------

    corrupted_cube = corrupted_cube.to(DEVICE)
    target = target.to(DEVICE)
    mask = mask.to(DEVICE)

    # --------------------------------------------------------------
    # Confirm expected tensor shape.
    # --------------------------------------------------------------

    expected_shape = (
        CHANNELS,
        *CUBE_SIZE
    )

    if tuple(corrupted_cube.shape) != expected_shape:
        raise ValueError(
            "Unexpected corrupted_cube shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(corrupted_cube.shape)}."
        )

    if tuple(target.shape) != expected_shape:
        raise ValueError(
            "Unexpected target shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(target.shape)}."
        )

    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            "Unexpected mask shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(mask.shape)}."
        )

    # --------------------------------------------------------------
    # Verify mask and metadata.
    # --------------------------------------------------------------

    if returned_mask_mode != mask_mode:
        raise ValueError(
            "Dataset returned an unexpected mask mode. "
            f"Expected '{mask_mode}', "
            f"received '{returned_mask_mode}'."
        )

    if returned_geological_mode != geological_mode:
        raise ValueError(
            "Dataset returned an unexpected geological mode. "
            f"Expected '{geological_mode}', "
            f"received '{returned_geological_mode}'."
        )

    # --------------------------------------------------------------
    # Count observed and missing samples.
    # --------------------------------------------------------------

    observed_count = int(
        torch.sum(mask == 1).item()
    )

    missing_count = int(
        torch.sum(mask == 0).item()
    )

    if observed_count == 0:
        raise ValueError(
            "Controlled case contains no observed samples."
        )

    if missing_count == 0:
        raise ValueError(
            "Controlled case contains no missing samples."
        )

    # --------------------------------------------------------------
    # Verify that the corrupted input is consistent with the
    # target and observation mask.
    #
    # Input should equal:
    #
    #       target * mask
    # --------------------------------------------------------------

    input_consistency_error = torch.max(
        torch.abs(
            corrupted_cube -
            (target * mask)
        )
    ).item()

    if input_consistency_error > 1e-6:
        raise ValueError(
            "Corrupted input is inconsistent with "
            "target * mask. "
            f"Maximum difference: "
            f"{input_consistency_error:.6e}"
        )

    # --------------------------------------------------------------
    # Run the Nearest Neighbor reconstruction.
    # --------------------------------------------------------------

    start_time = time.perf_counter()

    reconstruction = nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
    )

    runtime = time.perf_counter() - start_time

    # --------------------------------------------------------------
    # Validate reconstruction shape.
    # --------------------------------------------------------------

    if tuple(reconstruction.shape) != expected_shape:
        raise ValueError(
            "Unexpected reconstruction shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(reconstruction.shape)}."
        )

    # --------------------------------------------------------------
    # Validate finite values.
    # --------------------------------------------------------------

    if not torch.isfinite(reconstruction).all():
        raise ValueError(
            "Nearest Neighbor reconstruction contains "
            "non-finite values."
        )

    # --------------------------------------------------------------
    # Verify exact observed-data preservation.
    # --------------------------------------------------------------

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            -
            corrupted_cube[mask == 1]
        )
    ).item()

    if observed_difference > 1e-6:
        raise ValueError(
            "Observed-data preservation failed. "
            f"Maximum observed difference: "
            f"{observed_difference:.6e}"
        )

    # --------------------------------------------------------------
    # Confirm missing samples were reconstructed.
    # --------------------------------------------------------------

    missing_change = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            -
            corrupted_cube[mask == 0]
        )
    ).item()

    # --------------------------------------------------------------
    # Calculate reconstruction metrics.
    #
    # IMPORTANT:
    # These are the canonical project metric functions.
    # --------------------------------------------------------------

    mae_value = to_float(
        mae(
            reconstruction,
            target
        )
    )

    rmse_value = to_float(
        rmse(
            reconstruction,
            target
        )
    )

    psnr_value = to_float(
        psnr(
            reconstruction,
            target
        )
    )

    snr_value = to_float(
        snr(
            reconstruction,
            target
        )
    )

    ssim_value = to_float(
        ssim(
            reconstruction,
            target
        )
    )

    # --------------------------------------------------------------
    # Calculate metrics specifically on missing samples.
    #
    # This is useful for reconstruction analysis because the
    # observed samples are already known.
    # --------------------------------------------------------------

    missing_reconstruction = reconstruction[
        mask == 0
    ]

    missing_target = target[
        mask == 0
    ]

    missing_mae_value = to_float(
        torch.mean(
            torch.abs(
                missing_reconstruction
                -
                missing_target
            )
        )
    )

    missing_rmse_value = to_float(
        torch.sqrt(
            torch.mean(
                (
                    missing_reconstruction
                    -
                    missing_target
                ) ** 2
            )
        )
    )

    # --------------------------------------------------------------
    # Return standardized experiment result.
    # --------------------------------------------------------------

    return {
        "missing_rate_requested": missing_rate,
        "mask_mode": mask_mode,
        "geological_mode": geological_mode,
        "seed": seed,

        "cube_depth": CUBE_SIZE[0],
        "cube_height": CUBE_SIZE[1],
        "cube_width": CUBE_SIZE[2],

        "observed_samples": observed_count,
        "missing_samples": missing_count,

        "input_consistency_error": input_consistency_error,
        "observed_difference": observed_difference,

        "missing_reconstruction_change": missing_change,

        "missing_mae": missing_mae_value,
        "missing_rmse": missing_rmse_value,

        "mae": mae_value,
        "rmse": rmse_value,
        "psnr": psnr_value,
        "snr": snr_value,
        "ssim": ssim_value,

        "runtime_seconds": runtime,

        "status": "SUCCESS",
        "error": ""
    }


# ====================================================================
# WRITE RAW RESULTS
# ====================================================================

def write_raw_results(results):
    """
    Write all individual controlled experiments to CSV.
    """

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        RAW_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ====================================================================
# SUMMARY STATISTICS
# ====================================================================

def calculate_summary(results):
    """
    Calculate mean and standard deviation of reconstruction
    metrics grouped by:

        missing rate
        mask mode
        geological mode

    across the five random seeds.
    """

    grouped = {}

    for result in results:

        key = (
            result["missing_rate_requested"],
            result["mask_mode"],
            result["geological_mode"]
        )

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(
            result
        )

    summary = []

    metric_names = [
        "missing_mae",
        "missing_rmse",
        "mae",
        "rmse",
        "psnr",
        "snr",
        "ssim",
        "runtime_seconds"
    ]

    for key, group in grouped.items():

        missing_rate = key[0]
        mask_mode = key[1]
        geological_mode = key[2]

        row = {
            "missing_rate_requested": missing_rate,
            "mask_mode": mask_mode,
            "geological_mode": geological_mode,
            "n_seeds": len(group)
        }

        for metric_name in metric_names:

            values = np.asarray(
                [
                    item[metric_name]
                    for item in group
                ],
                dtype=np.float64
            )

            row[
                f"{metric_name}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric_name}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1
                )
            ) if len(values) > 1 else 0.0

        summary.append(row)

    return summary


def write_summary_results(summary):
    """
    Write grouped controlled-matrix statistics to CSV.
    """

    if not summary:
        return

    fieldnames = list(
        summary[0].keys()
    )

    with open(
        SUMMARY_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            summary
        )


# ====================================================================
# MAIN CONTROLLED MATRIX
# ====================================================================

def main():

    # --------------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------------

    ensure_output_directory()

    print("=" * 70)
    print("NEAREST NEIGHBOR CONTROLLED EXPERIMENTAL MATRIX")
    print("=" * 70)

    print()
    print(
        f"Cube size           : {CUBE_SIZE}"
    )

    print(
        f"Missing rates       : {MISSING_RATES}"
    )

    print(
        f"Missing mechanisms  : {MASK_MODES}"
    )

    print(
        f"Geological modes    : {GEOLOGICAL_MODES}"
    )

    print(
        f"Seeds               : {SEEDS}"
    )

    print(
        f"Expected cases      : {EXPECTED_CASES}"
    )

    print()

    # --------------------------------------------------------------
    # Storage for raw experiment results.
    # --------------------------------------------------------------

    results = []

    case_number = 0

    start_matrix_time = time.perf_counter()

    # --------------------------------------------------------------
    # Controlled experimental loops.
    # --------------------------------------------------------------

    for missing_rate in MISSING_RATES:

        for mask_mode in MASK_MODES:

            for geological_mode in GEOLOGICAL_MODES:

                for seed in SEEDS:

                    case_number += 1

                    print(
                        f"Case {case_number}/{EXPECTED_CASES}"
                    )

                    print(
                        f"  Missing rate : "
                        f"{missing_rate:.2f}"
                    )

                    print(
                        f"  Mask mode    : "
                        f"{mask_mode}"
                    )

                    print(
                        f"  Geology      : "
                        f"{geological_mode}"
                    )

                    print(
                        f"  Seed         : "
                        f"{seed}"
                    )

                    try:

                        result = run_single_case(
                            missing_rate=missing_rate,
                            mask_mode=mask_mode,
                            geological_mode=geological_mode,
                            seed=seed
                        )

                        results.append(
                            result
                        )

                        print(
                            f"  Status       : "
                            f"{result['status']}"
                        )

                        print(
                            f"  Missing MAE  : "
                            f"{result['missing_mae']:.6f}"
                        )

                        print(
                            f"  MAE          : "
                            f"{result['mae']:.6f}"
                        )

                        print(
                            f"  RMSE         : "
                            f"{result['rmse']:.6f}"
                        )

                        print(
                            f"  PSNR         : "
                            f"{result['psnr']:.6f} dB"
                        )

                        print(
                            f"  SNR          : "
                            f"{result['snr']:.6f} dB"
                        )

                        print(
                            f"  SSIM         : "
                            f"{result['ssim']:.6f}"
                        )

                        print(
                            f"  Runtime      : "
                            f"{result['runtime_seconds']:.4f} s"
                        )

                    except Exception as error:

                        error_result = {
                            "missing_rate_requested": missing_rate,
                            "mask_mode": mask_mode,
                            "geological_mode": geological_mode,
                            "seed": seed,

                            "cube_depth": CUBE_SIZE[0],
                            "cube_height": CUBE_SIZE[1],
                            "cube_width": CUBE_SIZE[2],

                            "observed_samples": "",
                            "missing_samples": "",

                            "input_consistency_error": "",
                            "observed_difference": "",

                            "missing_reconstruction_change": "",

                            "missing_mae": "",
                            "missing_rmse": "",

                            "mae": "",
                            "rmse": "",
                            "psnr": "",
                            "snr": "",
                            "ssim": "",

                            "runtime_seconds": "",

                            "status": "FAILED",
                            "error": str(error)
                        }

                        results.append(
                            error_result
                        )

                        print(
                            f"  Status       : FAILED"
                        )

                        print(
                            f"  Error        : "
                            f"{error}"
                        )

                    print()

                    # --------------------------------------------------
                    # Save progress after every case.
                    #
                    # This protects the experiment from losing all
                    # results if execution is interrupted.
                    # --------------------------------------------------

                    write_raw_results(
                        results
                    )

    # --------------------------------------------------------------
    # Calculate total runtime.
    # --------------------------------------------------------------

    total_runtime = (
        time.perf_counter()
        -
        start_matrix_time
    )

    # --------------------------------------------------------------
    # Final raw results.
    # --------------------------------------------------------------

    write_raw_results(
        results
    )

    # --------------------------------------------------------------
    # Calculate summary statistics using successful cases only.
    # --------------------------------------------------------------

    successful_results = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]

    failed_results = [
        result
        for result in results
        if result["status"] == "FAILED"
    ]

    summary = calculate_summary(
        successful_results
    )

    write_summary_results(
        summary
    )

    # --------------------------------------------------------------
    # Final report.
    # --------------------------------------------------------------

    print("=" * 70)
    print(
        "NEAREST NEIGHBOR CONTROLLED MATRIX COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Expected cases : {EXPECTED_CASES}"
    )

    print(
        f"Completed cases: {len(results)}"
    )

    print(
        f"Successful     : "
        f"{len(successful_results)}"
    )

    print(
        f"Failed         : "
        f"{len(failed_results)}"
    )

    print(
        f"Total runtime  : "
        f"{total_runtime:.2f} s"
    )

    print()

    print(
        f"Raw results:"
    )

    print(
        f"  {RAW_RESULTS_FILE}"
    )

    print()

    print(
        f"Summary results:"
    )

    print(
        f"  {SUMMARY_RESULTS_FILE}"
    )

    print()

    # --------------------------------------------------------------
    # Overall status.
    # --------------------------------------------------------------

    if (
        len(results) == EXPECTED_CASES
        and
        len(failed_results) == 0
    ):

        print(
            "OVERALL STATUS: PASS"
        )

        print(
            "All controlled Nearest Neighbor experiments "
            "completed successfully."
        )

    else:

        print(
            "OVERALL STATUS: FAIL"
        )

        print(
            "One or more controlled experiments failed."
        )

    print("=" * 70)


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()