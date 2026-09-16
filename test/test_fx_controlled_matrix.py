"""
======================================================================
CONTROLLED F-X PREDICTION EXPERIMENTAL MATRIX
======================================================================

Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

Purpose
-------
PhD-standard controlled validation of the classical f-x prediction
baseline across:

    Geological complexity:
        6 levels

    Missing-data mechanisms:
        5 mechanisms

    Missing-data percentages:
        10%, 20%, 30%, 40%, 50%

    Independent random seeds:
        5 seeds

Total:
    6 × 5 × 5 × 5 = 750 experimental cases

For every experiment:

    Complete synthetic volume
              |
              v
       Controlled mask
              |
              v
       Incomplete volume
              |
              v
       F-X reconstruction
              |
              v
       Common metrics

Metrics
-------
    MAE
    RMSE
    PSNR
    SNR
    SSIM

Additional experimental quantities
----------------------------------
    Runtime
    Number of missing voxels
    Number of observed voxels
    Input consistency
    Observed-data preservation
    Reconstruction validity
    Seed
    Geological mode
    Missing-data mechanism
    Missing percentage

Important
---------
This script validates the existing f-x implementation.

It does NOT modify the f-x algorithm.

Results are written to:

    outputs/synthetic_training/reports/

======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.fx_prediction import (
    fx_prediction_reconstruction,
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =====================================================================
# CONTROLLED EXPERIMENTAL MATRIX
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
# SYNTHETIC EXPERIMENT CONFIGURATION
# =====================================================================

CUBE_SIZE = (
    64,
    128,
    128,
)


# One synthetic sample is sufficient for each controlled condition
# because this experiment is repeated across independent seeds.
NUM_SAMPLES = 1


# F-X algorithm configuration
FX_PREDICTION_ORDER = 4
FX_ITERATIONS = 2


# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "reports"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_FILE = (
    OUTPUT_DIR
    / "fx_controlled_matrix.csv"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "fx_controlled_matrix_summary.csv"
)


# =====================================================================
# NUMERICAL SETTINGS
# =====================================================================

TOLERANCE = 1.0e-6


# =====================================================================
# HELPER: FINITE VALUE CHECK
# =====================================================================

def tensor_is_finite(tensor):
    """
    Check whether every element of a tensor is finite.
    """

    return bool(
        torch.isfinite(tensor).all().item()
    )


# =====================================================================
# HELPER: METRIC CONVERSION
# =====================================================================

def metric_to_float(value):
    """
    Convert a metric output to a Python float.

    Handles:
        torch.Tensor
        NumPy scalar
        Python numeric
    """

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        return float(value.item())

    return float(value)


# =====================================================================
# RUN ONE EXPERIMENT
# =====================================================================

def run_single_experiment(
    geological_mode,
    mask_mode,
    missing_rate,
    seed,
):
    """
    Run one controlled f-x experiment.

    Returns
    -------
    dict
        Experimental result record.
    """

    # ---------------------------------------------------------------
    # Build exactly one deterministic synthetic sample.
    # ---------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )


    # ---------------------------------------------------------------
    # Retrieve the single sample.
    # ---------------------------------------------------------------

    (
        corrupted,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode,
    ) = dataset[0]


    # ---------------------------------------------------------------
    # Confirm experimental metadata.
    # ---------------------------------------------------------------

    if returned_mask_mode != mask_mode:
        raise RuntimeError(
            "Mask mode mismatch: "
            f"expected={mask_mode}, "
            f"received={returned_mask_mode}"
        )


    if returned_geological_mode != geological_mode:
        raise RuntimeError(
            "Geological mode mismatch: "
            f"expected={geological_mode}, "
            f"received={returned_geological_mode}"
        )


    # ---------------------------------------------------------------
    # Validate tensor shapes.
    # ---------------------------------------------------------------

    if corrupted.shape != target.shape:
        raise RuntimeError(
            "Input/target shape mismatch."
        )


    if corrupted.shape != mask.shape:
        raise RuntimeError(
            "Input/mask shape mismatch."
        )


    if corrupted.shape != velocity.shape:
        raise RuntimeError(
            "Input/velocity shape mismatch."
        )


    # ---------------------------------------------------------------
    # Validate finite values.
    # ---------------------------------------------------------------

    if not tensor_is_finite(corrupted):
        raise RuntimeError(
            "Corrupted input contains non-finite values."
        )


    if not tensor_is_finite(target):
        raise RuntimeError(
            "Target contains non-finite values."
        )


    if not tensor_is_finite(mask):
        raise RuntimeError(
            "Mask contains non-finite values."
        )


    # ---------------------------------------------------------------
    # Verify mask values.
    # ---------------------------------------------------------------

    unique_mask_values = torch.unique(mask)

    allowed_values = {
        0.0,
        1.0,
    }


    for value in unique_mask_values.tolist():

        if float(value) not in allowed_values:

            raise RuntimeError(
                "Mask contains values other than "
                "0 and 1."
            )


    # ---------------------------------------------------------------
    # Verify input consistency.
    #
    # Observed samples must remain equal to the target.
    # ---------------------------------------------------------------

    observed_difference = (
        torch.abs(
            corrupted[mask == 1]
            - target[mask == 1]
        )
    )


    if observed_difference.numel() > 0:

        maximum_observed_difference = float(
            observed_difference.max().item()
        )

    else:

        maximum_observed_difference = 0.0


    if (
        maximum_observed_difference
        > TOLERANCE
    ):

        raise RuntimeError(
            "Observed samples are not identical "
            "to the target."
        )


    # ---------------------------------------------------------------
    # Count observed and missing voxels.
    # ---------------------------------------------------------------

    total_voxels = int(
        mask.numel()
    )


    observed_voxels = int(
        (mask == 1).sum().item()
    )


    missing_voxels = int(
        (mask == 0).sum().item()
    )


    # ---------------------------------------------------------------
    # Verify that the requested missing percentage is represented.
    # ---------------------------------------------------------------

    measured_missing_rate = (
        missing_voxels
        / total_voxels
    )


    # ---------------------------------------------------------------
    # Run f-x reconstruction.
    # ---------------------------------------------------------------

    start_time = time.perf_counter()


    reconstruction = (
        fx_prediction_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            prediction_order=FX_PREDICTION_ORDER,
            iterations=FX_ITERATIONS,
        )
    )


    end_time = time.perf_counter()


    runtime_seconds = (
        end_time
        - start_time
    )


    # ---------------------------------------------------------------
    # Validate reconstruction shape.
    # ---------------------------------------------------------------

    if reconstruction.shape != target.shape:

        raise RuntimeError(
            "F-X reconstruction shape mismatch."
        )


    # ---------------------------------------------------------------
    # Validate reconstruction values.
    # ---------------------------------------------------------------

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "F-X reconstruction contains "
            "non-finite values."
        )


    # ---------------------------------------------------------------
    # Verify exact preservation of observed samples.
    # ---------------------------------------------------------------

    reconstruction_observed_difference = (
        torch.abs(
            reconstruction[mask == 1]
            - target[mask == 1]
        )
    )


    if (
        reconstruction_observed_difference.numel()
        > 0
    ):

        maximum_reconstruction_observed_difference = (
            float(
                reconstruction_observed_difference
                .max()
                .item()
            )
        )

    else:

        maximum_reconstruction_observed_difference = 0.0


    if (
        maximum_reconstruction_observed_difference
        > TOLERANCE
    ):

        raise RuntimeError(
            "F-X reconstruction modified "
            "observed samples."
        )


    # ---------------------------------------------------------------
    # Evaluate only the reconstructed missing samples.
    #
    # This is the scientifically relevant reconstruction error.
    # ---------------------------------------------------------------

    missing_target = target[mask == 0]

    missing_reconstruction = (
        reconstruction[mask == 0]
    )


    if missing_target.numel() == 0:

        raise RuntimeError(
            "No missing samples were generated."
        )


    # ---------------------------------------------------------------
    # Compute reconstruction metrics.
    # ---------------------------------------------------------------

    missing_mae = metric_to_float(
        mae(
            missing_reconstruction,
            missing_target,
        )
    )


    missing_rmse = metric_to_float(
        rmse(
            missing_reconstruction,
            missing_target,
        )
    )


    missing_psnr = metric_to_float(
        psnr(
            missing_reconstruction,
            missing_target,
        )
    )


    missing_snr = metric_to_float(
        snr(
            missing_reconstruction,
            missing_target,
        )
    )


    missing_ssim = metric_to_float(
        ssim(
            missing_reconstruction,
            missing_target,
        )
    )


    # ---------------------------------------------------------------
    # Return complete experimental record.
    # ---------------------------------------------------------------

    return {

        "method":
            "f-x_prediction",

        "seed":
            seed,

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "requested_missing_rate":
            missing_rate,

        "measured_missing_rate":
            measured_missing_rate,

        "cube_depth":
            CUBE_SIZE[0],

        "cube_height":
            CUBE_SIZE[1],

        "cube_width":
            CUBE_SIZE[2],

        "total_voxels":
            total_voxels,

        "observed_voxels":
            observed_voxels,

        "missing_voxels":
            missing_voxels,

        "prediction_order":
            FX_PREDICTION_ORDER,

        "iterations":
            FX_ITERATIONS,

        "runtime_seconds":
            runtime_seconds,

        "input_consistency_error":
            maximum_observed_difference,

        "observed_preservation_error":
            maximum_reconstruction_observed_difference,

        "MAE":
            missing_mae,

        "RMSE":
            missing_rmse,

        "PSNR":
            missing_psnr,

        "SNR":
            missing_snr,

        "SSIM":
            missing_ssim,

        "status":
            "PASS",
    }


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def calculate_summary(records):
    """
    Calculate grouped summary statistics.

    Groups by:

        geological mode
        mask mode
        requested missing rate

    For each group:

        mean
        standard deviation

    are calculated for all reconstruction metrics.
    """

    grouped = {}


    # ---------------------------------------------------------------
    # Group experimental records.
    # ---------------------------------------------------------------

    for record in records:

        key = (
            record["geological_mode"],
            record["mask_mode"],
            record["requested_missing_rate"],
        )


        if key not in grouped:
            grouped[key] = []


        grouped[key].append(record)


    # ---------------------------------------------------------------
    # Build summary records.
    # ---------------------------------------------------------------

    summary_records = []


    metric_names = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "runtime_seconds",
    ]


    for key, group in grouped.items():

        (
            geological_mode,
            mask_mode,
            missing_rate,
        ) = key


        summary = {

            "geological_mode":
                geological_mode,

            "mask_mode":
                mask_mode,

            "requested_missing_rate":
                missing_rate,

            "n_seeds":
                len(group),
        }


        # -----------------------------------------------------------
        # Mean and standard deviation.
        # -----------------------------------------------------------

        for metric_name in metric_names:

            values = np.asarray(
                [
                    float(
                        record[metric_name]
                    )
                    for record in group
                ],
                dtype=np.float64,
            )


            summary[
                f"{metric_name}_mean"
            ] = float(
                values.mean()
            )


            summary[
                f"{metric_name}_std"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )


        summary_records.append(
            summary
        )


    return summary_records


# =====================================================================
# WRITE CSV
# =====================================================================

def write_csv(
    filename,
    records,
):
    """
    Write experiment records to CSV.
    """

    if not records:
        return


    fieldnames = list(
        records[0].keys()
    )


    with open(
        filename,
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
            records
        )


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def main():

    # ---------------------------------------------------------------
    # Expected experiment count.
    # ---------------------------------------------------------------

    expected_experiments = (
        len(GEOLOGICAL_MODES)
        * len(MASK_MODES)
        * len(MISSING_RATES)
        * len(SEEDS)
    )


    print()
    print("=" * 78)
    print(
        "CONTROLLED F-X PREDICTION "
        "EXPERIMENTAL MATRIX"
    )
    print("=" * 78)

    print()
    print(
        f"Cube size       : {CUBE_SIZE}"
    )

    print(
        f"Geological modes: {len(GEOLOGICAL_MODES)}"
    )

    print(
        f"Mask mechanisms : {len(MASK_MODES)}"
    )

    print(
        f"Missing rates   : {len(MISSING_RATES)}"
    )

    print(
        f"Independent seeds: {len(SEEDS)}"
    )

    print(
        f"Expected cases  : {expected_experiments}"
    )

    print()
    print(
        "F-X configuration"
    )

    print(
        f"Prediction order: "
        f"{FX_PREDICTION_ORDER}"
    )

    print(
        f"Iterations      : "
        f"{FX_ITERATIONS}"
    )

    print()
    print(
        "Results directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("=" * 78)


    # ---------------------------------------------------------------
    # Experimental records.
    # ---------------------------------------------------------------

    records = []


    completed = 0
    failed = 0


    # ---------------------------------------------------------------
    # Execute controlled matrix.
    # ---------------------------------------------------------------

    for geological_mode in GEOLOGICAL_MODES:

        print()
        print(
            f"GEology: {geological_mode}"
        )
        print(
            "-" * 78
        )


        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    completed_case = (
                        completed
                        + failed
                        + 1
                    )


                    print(
                        f"[{completed_case:03d}/"
                        f"{expected_experiments:03d}] "
                        f"{geological_mode:15s} | "
                        f"{mask_mode:20s} | "
                        f"{missing_rate:.0%} | "
                        f"seed={seed}",
                        end=" ... ",
                    )


                    try:

                        result = (
                            run_single_experiment(
                                geological_mode=(
                                    geological_mode
                                ),
                                mask_mode=(
                                    mask_mode
                                ),
                                missing_rate=(
                                    missing_rate
                                ),
                                seed=seed,
                            )
                        )


                        records.append(
                            result
                        )


                        completed += 1


                        print(
                            f"PASS | "
                            f"MAE="
                            f"{result['MAE']:.6f} | "
                            f"SSIM="
                            f"{result['SSIM']:.6f} | "
                            f"time="
                            f"{result['runtime_seconds']:.2f}s"
                        )


                    except Exception as error:

                        failed += 1

                        print(
                            "FAIL"
                        )


                        records.append({

                            "method":
                                "f-x_prediction",

                            "seed":
                                seed,

                            "geological_mode":
                                geological_mode,

                            "mask_mode":
                                mask_mode,

                            "requested_missing_rate":
                                missing_rate,

                            "status":
                                "FAIL",

                            "error":
                                str(error),
                        })


    # ---------------------------------------------------------------
    # Write raw experimental results.
    # ---------------------------------------------------------------

    write_csv(
        RESULTS_FILE,
        records,
    )


    # ---------------------------------------------------------------
    # Keep only successful experiments for statistics.
    # ---------------------------------------------------------------

    successful_records = [
        record
        for record in records
        if record.get("status") == "PASS"
    ]


    summary_records = (
        calculate_summary(
            successful_records
        )
    )


    # ---------------------------------------------------------------
    # Write summary.
    # ---------------------------------------------------------------

    write_csv(
        SUMMARY_FILE,
        summary_records,
    )


    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "CONTROLLED F-X MATRIX COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Expected experiments : "
        f"{expected_experiments}"
    )

    print(
        f"Successful experiments: "
        f"{completed}"
    )

    print(
        f"Failed experiments    : "
        f"{failed}"
    )

    print()


    if completed == expected_experiments:

        print(
            "OVERALL STATUS: PASS"
        )

        print()
        print(
            "All controlled f-x experiments "
            "completed successfully."
        )

    else:

        print(
            "OVERALL STATUS: FAIL"
        )

        print()
        print(
            "At least one controlled "
            "experiment failed."
        )


    print()
    print(
        "Raw results:"
    )

    print(
        f"  {RESULTS_FILE}"
    )


    print()
    print(
        "Summary results:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )


    print()
    print("=" * 78)


    # ---------------------------------------------------------------
    # Return failure code to the operating system.
    # ---------------------------------------------------------------

    if failed > 0:
        raise SystemExit(1)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()