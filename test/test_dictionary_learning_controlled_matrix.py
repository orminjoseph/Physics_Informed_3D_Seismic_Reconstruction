"""
=========================================================
Dictionary Learning Baseline - Controlled Matrix Test
=========================================================

Validates the Dictionary Learning reconstruction baseline
across the complete controlled experimental matrix.

Controlled experimental matrix:

    Geological modes : 6
    Missing rates    : 5
    Missing mechanisms: 5
    Random seeds     : 5

Total:

    6 × 5 × 5 × 5 = 750 cases

All experiments use the same:

    Cube size        : 64 × 128 × 128
    Patch size       : 8 × 8 × 8
    Dictionary size  : 64 components
    Alpha            : 1.0
    Maximum iterations: 20
    Batch size       : 64
    Training patches : 2000
    Minimum observed : 0.80

Observed seismic samples are preserved exactly.

Author: Ormin Joseph
=========================================================
"""

import csv
import os
import time
from collections import defaultdict

import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.dictionary_learning import (
    dictionary_learning_reconstruction,
)


# =========================================================
# Experimental configuration
# =========================================================

CUBE_SIZE = (64, 128, 128)

MISSING_RATES = [
    0.30,
]

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
]

MASK_MODES = [
    "random_voxels",
    "missing_traces",
]

SEEDS = [
    42,
    43,
]


# =========================================================
# Dictionary Learning parameters
# =========================================================

PATCH_SIZE = (8, 8, 8)

N_COMPONENTS = 64

ALPHA = 1.0

MAX_ITER = 20

BATCH_SIZE = 64

MAX_TRAINING_PATCHES = 2000

MIN_OBSERVED_FRACTION = 0.80


# =========================================================
# Output configuration
# =========================================================

OUTPUT_DIR = os.path.join(
    "outputs",
    "synthetic_training",
    "reports",
)

RAW_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "dictionary_learning_controlled_matrix.csv",
)

SUMMARY_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "dictionary_learning_controlled_matrix_summary.csv",
)


# =========================================================
# Metric functions
# =========================================================

def mae(prediction, target):
    """
    Mean Absolute Error.
    """

    return torch.mean(
        torch.abs(prediction - target)
    ).item()


def rmse(prediction, target):
    """
    Root Mean Square Error.
    """

    return torch.sqrt(
        torch.mean(
            (prediction - target) ** 2
        )
    ).item()


def psnr(prediction, target, data_range=2.0):
    """
    Peak Signal-to-Noise Ratio.

    Synthetic seismic data are normalized approximately
    to [-1, 1], therefore data_range = 2.
    """

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0.0:
        return float("inf")

    return (
        10.0
        * torch.log10(
            torch.tensor(
                (data_range ** 2) / mse
            )
        )
    ).item()


def snr(prediction, target):
    """
    Signal-to-Noise Ratio.
    """

    signal_power = torch.sum(
        target ** 2
    )

    noise_power = torch.sum(
        (target - prediction) ** 2
    )

    if noise_power.item() == 0.0:
        return float("inf")

    return (
        10.0
        * torch.log10(
            signal_power / noise_power
        )
    ).item()


def ssim_simple(
    prediction,
    target,
    data_range=2.0,
):
    """
    Simple global SSIM-style measure used consistently
    with the validated single-sample Dictionary Learning test.

    The final unified evaluation pipeline should use the
    project's standard SSIM implementation.
    """

    x = prediction.float()
    y = target.float()

    mu_x = torch.mean(x)
    mu_y = torch.mean(y)

    sigma_x = torch.var(
        x,
        unbiased=False,
    )

    sigma_y = torch.var(
        y,
        unbiased=False,
    )

    covariance = torch.mean(
        (x - mu_x)
        * (y - mu_y)
    )

    c1 = (
        0.01 * data_range
    ) ** 2

    c2 = (
        0.03 * data_range
    ) ** 2

    numerator = (
        (2 * mu_x * mu_y + c1)
        * (2 * covariance + c2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + c1)
        * (sigma_x + sigma_y + c2)
    )

    return (
        numerator / denominator
    ).item()


# =========================================================
# Single experiment
# =========================================================

def run_single_case(
    case_number,
    total_cases,
    missing_rate,
    geological_mode,
    mask_mode,
    seed,
):
    """
    Execute one controlled Dictionary Learning experiment.
    """

    print()
    print("=" * 78)
    print(
        f"CASE {case_number}/{total_cases}"
    )
    print("=" * 78)

    print(
        f"Missing rate     : {missing_rate:.2f}"
    )

    print(
        f"Geological mode  : {geological_mode}"
    )

    print(
        f"Mask mode        : {mask_mode}"
    )

    print(
        f"Seed             : {seed}"
    )

    print()


    # =====================================================
    # Create controlled synthetic dataset
    # =====================================================

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        seed=seed,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
    )


    # =====================================================
    # Obtain sample
    # =====================================================

    sample = dataset[0]

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = sample


    # =====================================================
    # Expected shape
    # =====================================================

    expected_shape = (
        1,
        *CUBE_SIZE,
    )


    # =====================================================
    # Shape validation
    # =====================================================

    if tuple(corrupted.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected corrupted input shape: "
            f"{tuple(corrupted.shape)}. "
            f"Expected {expected_shape}."
        )

    if tuple(target.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected target shape: "
            f"{tuple(target.shape)}. "
            f"Expected {expected_shape}."
        )

    if tuple(mask.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected mask shape: "
            f"{tuple(mask.shape)}. "
            f"Expected {expected_shape}."
        )


    # =====================================================
    # Mask validation
    # =====================================================

    unique_mask = torch.unique(mask)

    if not torch.all(
        (unique_mask == 0)
        | (unique_mask == 1)
    ):
        raise RuntimeError(
            "Mask contains values other than 0 and 1."
        )


    # =====================================================
    # Input consistency
    # =====================================================

    expected_corrupted = (
        target * mask
    )

    input_error = torch.max(
        torch.abs(
            corrupted
            - expected_corrupted
        )
    ).item()

    if input_error > 1e-6:
        raise RuntimeError(
            "Corrupted input is inconsistent with "
            "target and mask."
        )


    # =====================================================
    # Count observed and missing samples
    # =====================================================

    observed_samples = int(
        torch.sum(mask == 1).item()
    )

    missing_samples = int(
        torch.sum(mask == 0).item()
    )


    # =====================================================
    # Dictionary Learning reconstruction
    # =====================================================

    start_time = time.perf_counter()

    reconstruction = (
        dictionary_learning_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            patch_size=PATCH_SIZE,
            n_components=N_COMPONENTS,
            alpha=ALPHA,
            max_iter=MAX_ITER,
            batch_size=BATCH_SIZE,
            max_training_patches=MAX_TRAINING_PATCHES,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            random_state=seed,
        )
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )


    # =====================================================
    # Output validation
    # =====================================================

    if reconstruction.shape != corrupted.shape:
        raise RuntimeError(
            "Reconstruction shape mismatch."
        )

    if not torch.isfinite(
        reconstruction
    ).all():

        raise RuntimeError(
            "Reconstruction contains NaN or infinite values."
        )


    # =====================================================
    # Observed-data preservation
    # =====================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            - corrupted[mask == 1]
        )
    ).item()

    if observed_difference > 0.0:
        raise RuntimeError(
            "Observed-data preservation failed."
        )


    # =====================================================
    # Missing-sample reconstruction
    # =====================================================

    zero_filled_error = torch.mean(
        torch.abs(
            corrupted[mask == 0]
            - target[mask == 0]
        )
    ).item()

    reconstructed_error = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - target[mask == 0]
        )
    ).item()

    missing_change = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - corrupted[mask == 0]
        )
    ).item()

    if missing_change <= 0.0:
        raise RuntimeError(
            "Dictionary Learning did not modify "
            "the missing samples."
        )


    # =====================================================
    # Global metrics
    # =====================================================

    metric_mae = mae(
        reconstruction,
        target,
    )

    metric_rmse = rmse(
        reconstruction,
        target,
    )

    metric_psnr = psnr(
        reconstruction,
        target,
    )

    metric_snr = snr(
        reconstruction,
        target,
    )

    metric_ssim = ssim_simple(
        reconstruction,
        target,
    )


    # =====================================================
    # Return experiment result
    # =====================================================

    return {
        "case": case_number,
        "missing_rate": missing_rate,
        "geological_mode": actual_geological_mode,
        "mask_mode": actual_mask_mode,
        "seed": seed,

        "cube_depth": CUBE_SIZE[0],
        "cube_height": CUBE_SIZE[1],
        "cube_width": CUBE_SIZE[2],

        "patch_depth": PATCH_SIZE[0],
        "patch_height": PATCH_SIZE[1],
        "patch_width": PATCH_SIZE[2],

        "n_components": N_COMPONENTS,
        "alpha": ALPHA,
        "max_iter": MAX_ITER,
        "batch_size": BATCH_SIZE,
        "max_training_patches": MAX_TRAINING_PATCHES,
        "min_observed_fraction": MIN_OBSERVED_FRACTION,

        "observed_samples": observed_samples,
        "missing_samples": missing_samples,

        "input_consistency_error": input_error,
        "observed_difference": observed_difference,

        "zero_filled_missing_mae": zero_filled_error,
        "dictionary_missing_mae": reconstructed_error,
        "missing_reconstruction_change": missing_change,

        "mae": metric_mae,
        "rmse": metric_rmse,
        "psnr": metric_psnr,
        "snr": metric_snr,
        "ssim": metric_ssim,

        "runtime_seconds": elapsed_time,

        "status": "SUCCESS",
        "error": "",
    }


# =========================================================
# Main controlled experiment
# =========================================================

def main():

    print("=" * 78)
    print("DICTIONARY LEARNING CONTROLLED MATRIX")
    print("=" * 78)

    print()
    print("Experimental design")
    print("-" * 78)

    print(
        f"Cube size              : {CUBE_SIZE}"
    )

    print(
        f"Missing rates           : {MISSING_RATES}"
    )

    print(
        f"Geological modes       : {len(GEOLOGICAL_MODES)}"
    )

    print(
        f"Missing mechanisms     : {len(MASK_MODES)}"
    )

    print(
        f"Random seeds           : {SEEDS}"
    )

    print(
        f"Dictionary components  : {N_COMPONENTS}"
    )

    print(
        f"Patch size             : {PATCH_SIZE}"
    )

    print(
        f"Maximum iterations     : {MAX_ITER}"
    )

    print()


    # =====================================================
    # Calculate expected number of cases
    # =====================================================

    total_cases = (
        len(MISSING_RATES)
        * len(GEOLOGICAL_MODES)
        * len(MASK_MODES)
        * len(SEEDS)
    )

    print(
        f"Expected total cases : {total_cases}"
    )

    if total_cases != 750:
        raise RuntimeError(
            "Controlled experimental matrix does not "
            "contain exactly 750 cases."
        )


    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


    # =====================================================
    # Storage for raw results
    # =====================================================

    results = []

    successful_cases = 0

    failed_cases = 0


    # =====================================================
    # Global experiment timer
    # =====================================================

    experiment_start = time.perf_counter()


    # =====================================================
    # Case counter
    # =====================================================

    case_number = 0


    # =====================================================
    # Controlled experimental matrix
    # =====================================================

    for geological_mode in GEOLOGICAL_MODES:

        for missing_rate in MISSING_RATES:

            for mask_mode in MASK_MODES:

                for seed in SEEDS:

                    case_number += 1

                    try:

                        result = run_single_case(
                            case_number=case_number,
                            total_cases=total_cases,
                            missing_rate=missing_rate,
                            geological_mode=geological_mode,
                            mask_mode=mask_mode,
                            seed=seed,
                        )

                        results.append(result)

                        successful_cases += 1

                        print(
                            f"Case {case_number}/{total_cases} "
                            f"SUCCESS"
                        )

                        print(
                            f"MAE  : {result['mae']:.6f}"
                        )

                        print(
                            f"RMSE : {result['rmse']:.6f}"
                        )

                        print(
                            f"PSNR : {result['psnr']:.6f} dB"
                        )

                        print(
                            f"SNR  : {result['snr']:.6f} dB"
                        )

                        print(
                            f"SSIM : {result['ssim']:.6f}"
                        )

                        print(
                            f"Runtime : "
                            f"{result['runtime_seconds']:.4f} s"
                        )

                    except Exception as exc:

                        failed_cases += 1

                        error_message = (
                            f"{type(exc).__name__}: {exc}"
                        )

                        print(
                            f"Case {case_number}/{total_cases} "
                            f"FAILED"
                        )

                        print(
                            f"Error: {error_message}"
                        )


                        results.append({
                            "case": case_number,
                            "missing_rate": missing_rate,
                            "geological_mode": geological_mode,
                            "mask_mode": mask_mode,
                            "seed": seed,

                            "cube_depth": CUBE_SIZE[0],
                            "cube_height": CUBE_SIZE[1],
                            "cube_width": CUBE_SIZE[2],

                            "patch_depth": PATCH_SIZE[0],
                            "patch_height": PATCH_SIZE[1],
                            "patch_width": PATCH_SIZE[2],

                            "n_components": N_COMPONENTS,
                            "alpha": ALPHA,
                            "max_iter": MAX_ITER,
                            "batch_size": BATCH_SIZE,
                            "max_training_patches":
                                MAX_TRAINING_PATCHES,
                            "min_observed_fraction":
                                MIN_OBSERVED_FRACTION,

                            "observed_samples": "",
                            "missing_samples": "",

                            "input_consistency_error": "",
                            "observed_difference": "",

                            "zero_filled_missing_mae": "",
                            "dictionary_missing_mae": "",
                            "missing_reconstruction_change": "",

                            "mae": "",
                            "rmse": "",
                            "psnr": "",
                            "snr": "",
                            "ssim": "",

                            "runtime_seconds": "",

                            "status": "FAILED",
                            "error": error_message,
                        })


    # =====================================================
    # Total experiment runtime
    # =====================================================

    total_runtime = (
        time.perf_counter()
        - experiment_start
    )


    # =====================================================
    # Write raw results
    # =====================================================

    fieldnames = [
        "case",
        "missing_rate",
        "geological_mode",
        "mask_mode",
        "seed",

        "cube_depth",
        "cube_height",
        "cube_width",

        "patch_depth",
        "patch_height",
        "patch_width",

        "n_components",
        "alpha",
        "max_iter",
        "batch_size",
        "max_training_patches",
        "min_observed_fraction",

        "observed_samples",
        "missing_samples",

        "input_consistency_error",
        "observed_difference",

        "zero_filled_missing_mae",
        "dictionary_missing_mae",
        "missing_reconstruction_change",

        "mae",
        "rmse",
        "psnr",
        "snr",
        "ssim",

        "runtime_seconds",

        "status",
        "error",
    ]


    with open(
        RAW_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)


    # =====================================================
    # Build summary statistics
    # =====================================================

    grouped_results = defaultdict(list)

    for result in results:

        if result["status"] != "SUCCESS":
            continue

        key = (
            result["geological_mode"],
            result["missing_rate"],
            result["mask_mode"],
        )

        grouped_results[key].append(result)


    summary_rows = []


    for (
        geological_mode,
        missing_rate,
        mask_mode,
    ), group in sorted(grouped_results.items()):

        def mean_metric(name):

            values = [
                float(row[name])
                for row in group
                if row[name] != ""
            ]

            if not values:
                return ""

            return sum(values) / len(values)


        summary_rows.append({
            "geological_mode": geological_mode,
            "missing_rate": missing_rate,
            "mask_mode": mask_mode,

            "n_cases": len(group),

            "successful_cases": len(group),

            "mae_mean": mean_metric("mae"),
            "rmse_mean": mean_metric("rmse"),
            "psnr_mean": mean_metric("psnr"),
            "snr_mean": mean_metric("snr"),
            "ssim_mean": mean_metric("ssim"),

            "runtime_mean_seconds":
                mean_metric("runtime_seconds"),

            "dictionary_missing_mae_mean":
                mean_metric(
                    "dictionary_missing_mae"
                ),

            "zero_filled_missing_mae_mean":
                mean_metric(
                    "zero_filled_missing_mae"
                ),
        })


    # =====================================================
    # Write summary results
    # =====================================================

    summary_fieldnames = [
        "geological_mode",
        "missing_rate",
        "mask_mode",
        "n_cases",
        "successful_cases",

        "mae_mean",
        "rmse_mean",
        "psnr_mean",
        "snr_mean",
        "ssim_mean",

        "runtime_mean_seconds",

        "dictionary_missing_mae_mean",
        "zero_filled_missing_mae_mean",
    ]


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

        writer.writerows(summary_rows)


    # =====================================================
    # Final report
    # =====================================================

    print()
    print("=" * 78)
    print("DICTIONARY LEARNING CONTROLLED MATRIX COMPLETE")
    print("=" * 78)

    print(
        f"Expected cases : {total_cases}"
    )

    print(
        f"Completed cases: {case_number}"
    )

    print(
        f"Successful     : {successful_cases}"
    )

    print(
        f"Failed         : {failed_cases}"
    )

    print(
        f"Total runtime  : "
        f"{total_runtime:.2f} seconds"
    )

    print()

    print(
        f"Raw results:"
    )

    print(
        RAW_RESULTS_FILE
    )

    print()

    print(
        f"Summary results:"
    )

    print(
        SUMMARY_RESULTS_FILE
    )

    print()


    # =====================================================
    # Overall status
    # =====================================================

    if (
        case_number == total_cases
        and successful_cases == total_cases
        and failed_cases == 0
    ):

        print(
            "OVERALL STATUS: PASS"
        )

        print(
            "All controlled Dictionary Learning "
            "experiments completed successfully."
        )

    else:

        print(
            "OVERALL STATUS: FAIL"
        )

        print(
            "One or more controlled Dictionary Learning "
            "experiments failed."
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()