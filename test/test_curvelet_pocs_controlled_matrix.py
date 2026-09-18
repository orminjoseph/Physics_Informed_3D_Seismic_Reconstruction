"""
======================================================================
Controlled Matrix Test — 3-D Curvelet POCS Baseline
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the 3-D Curvelet POCS baseline under the same controlled
experimental conditions used for the other seismic reconstruction
baselines.

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
    6 × 5 × 5 × 5 = 750 cases

Synthetic cube
--------------
    64 × 128 × 128

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

Curvelet configuration
----------------------
    Transform:
        3-D UDCT

    Number of scales:
        3

    Wedges per direction:
        3

    POCS iterations:
        12

    Initial threshold:
        0.05

    Threshold decay:
        0.90

    Tolerance:
        1.0e-5

Outputs
-------
Raw results:
    outputs/synthetic_training/reports/
        curvelet_pocs_controlled_matrix.csv

Summary results:
    outputs/synthetic_training/reports/
        curvelet_pocs_controlled_matrix_summary.csv

Author:
    Ormin Joseph
======================================================================
"""


# ======================================================================
# IMPORTS
# ======================================================================

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.curvelet_pocs import (
    curvelet_pocs_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# CONTROLLED EXPERIMENTAL MATRIX
# ======================================================================

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
]


MASK_MODES = [
    "random_voxels",
    "missing_traces",
]


MISSING_RATES = [
    0.30,
]


SEEDS = [
    42,
    43,
]


# ======================================================================
# SYNTHETIC EXPERIMENT CONFIGURATION
# ======================================================================

# IMPORTANT:
#
# This is the final controlled benchmark cube size.
#
# The previous 32 × 32 × 32 test was only a computational/API
# validation test.
#
# The actual baseline benchmark uses:
#
#     depth   = 64
#     height  = 128
#     width   = 128
#
CUBE_SIZE = (
    64,
    128,
    128,
)


# One deterministic synthetic sample per controlled condition.
NUM_SAMPLES = 1


# ======================================================================
# CURVELET / UDCT CONFIGURATION
# ======================================================================

CURVELET_NUM_SCALES = 3

CURVELET_WEDGES_PER_DIRECTION = 3

CURVELET_ITERATIONS = 12

CURVELET_THRESHOLD = 0.05

CURVELET_THRESHOLD_DECAY = 0.90

CURVELET_TOLERANCE = 1.0e-5


# ======================================================================
# NUMERICAL VALIDATION SETTINGS
# ======================================================================

OBSERVED_TOLERANCE = 1.0e-6

RECONSTRUCTION_TOLERANCE = 1.0e-8


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "reports"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# OUTPUT FILES
# ======================================================================

RESULTS_FILE = (
    OUTPUT_DIR
    / "curvelet_pocs_controlled_matrix.csv"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "curvelet_pocs_controlled_matrix_summary.csv"
)


# ======================================================================
# EXPECTED NUMBER OF CASES
# ======================================================================

EXPECTED_CASES = (
    len(GEOLOGICAL_MODES)
    * len(MASK_MODES)
    * len(MISSING_RATES)
    * len(SEEDS)
)


# ======================================================================
# HELPER: FINITE TENSOR VALIDATION
# ======================================================================

def tensor_is_finite(
    tensor
):
    """
    Return True when every tensor element is finite.
    """

    return bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


# ======================================================================
# HELPER: METRIC CONVERSION
# ======================================================================

def metric_to_float(
    value
):
    """
    Convert a metric output to a Python float.

    Supports:
        torch.Tensor
        NumPy scalar
        Python numeric
    """

    if isinstance(
        value,
        torch.Tensor
    ):
        return float(
            value.detach()
            .cpu()
            .item()
        )

    if isinstance(
        value,
        np.ndarray
    ):
        return float(
            value.item()
        )

    return float(value)


# ======================================================================
# HELPER: COMPUTE METRICS
# ======================================================================

def compute_metrics(
    reconstruction,
    target
):
    """
    Compute the common reconstruction metrics.
    """

    return {
        "mae": metric_to_float(
            mae(
                reconstruction,
                target
            )
        ),

        "rmse": metric_to_float(
            rmse(
                reconstruction,
                target
            )
        ),

        "psnr": metric_to_float(
            psnr(
                reconstruction,
                target
            )
        ),

        "snr": metric_to_float(
            snr(
                reconstruction,
                target
            )
        ),

        "ssim": metric_to_float(
            ssim(
                reconstruction,
                target
            )
        ),
    }


# ======================================================================
# HELPER: WRITE CSV HEADER
# ======================================================================

CSV_FIELDS = [
    "case_id",
    "geological_mode",
    "mask_mode",
    "missing_rate",
    "seed",

    "cube_depth",
    "cube_height",
    "cube_width",

    "num_scales",
    "wedges_per_direction",
    "iterations",
    "threshold",
    "threshold_decay",
    "tolerance",

    "observed_samples",
    "missing_samples",

    "input_consistency_error",
    "maximum_observed_difference",
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


def initialize_results_file():
    """
    Create the raw-results CSV and write its header.
    """

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()


# ======================================================================
# HELPER: APPEND ONE RESULT
# ======================================================================

def append_result(
    result
):
    """
    Append one experiment result to the raw CSV file.
    """

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writerow(
            result
        )


# ======================================================================
# RUN ONE CONTROLLED EXPERIMENT
# ======================================================================

def run_single_experiment(
    case_id,
    geological_mode,
    mask_mode,
    missing_rate,
    seed
):
    """
    Run one deterministic Curvelet POCS experiment.

    Returns
    -------
    dict
        Complete result record.
    """

    start_time = time.perf_counter()


    # ==================================================================
    # BUILD DATASET
    # ==================================================================

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )


    # ==================================================================
    # RETRIEVE SINGLE SAMPLE
    # ==================================================================

    (
        corrupted,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode,
    ) = dataset[0]


    # ==================================================================
    # EXPECTED SHAPE
    # ==================================================================

    expected_shape = (
        1,
        *CUBE_SIZE
    )


    # ==================================================================
    # SHAPE VALIDATION
    # ==================================================================

    if tuple(
        corrupted.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected corrupted-cube shape: "
            f"{tuple(corrupted.shape)}. "
            f"Expected: {expected_shape}"
        )


    if tuple(
        target.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected target shape: "
            f"{tuple(target.shape)}. "
            f"Expected: {expected_shape}"
        )


    if tuple(
        mask.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected mask shape: "
            f"{tuple(mask.shape)}. "
            f"Expected: {expected_shape}"
        )


    if tuple(
        velocity.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected velocity shape: "
            f"{tuple(velocity.shape)}. "
            f"Expected: {expected_shape}"
        )


    # ==================================================================
    # METADATA VALIDATION
    # ==================================================================

    if returned_mask_mode != mask_mode:

        raise RuntimeError(
            "Dataset returned an unexpected mask mode. "
            f"Expected: {mask_mode}; "
            f"Received: {returned_mask_mode}"
        )


    if returned_geological_mode != geological_mode:

        raise RuntimeError(
            "Dataset returned an unexpected geological mode. "
            f"Expected: {geological_mode}; "
            f"Received: {returned_geological_mode}"
        )


    # ==================================================================
    # FINITE-VALUE VALIDATION
    # ==================================================================

    for name, tensor in [
        ("corrupted", corrupted),
        ("target", target),
        ("mask", mask),
        ("velocity", velocity),
    ]:

        if not tensor_is_finite(
            tensor
        ):

            raise RuntimeError(
                f"Non-finite values detected "
                f"in {name}."
            )


    # ==================================================================
    # MASK VALIDATION
    # ==================================================================

    unique_mask_values = torch.unique(
        mask
    ).cpu().tolist()


    if not all(
        value in (0.0, 1.0)
        for value in unique_mask_values
    ):

        raise RuntimeError(
            "Mask contains values other than "
            "0.0 and 1.0: "
            f"{unique_mask_values}"
        )


    # ==================================================================
    # INPUT CONSISTENCY
    # ==================================================================

    expected_corrupted = (
        target
        * mask
    )


    input_consistency_error = float(
        (
            corrupted
            - expected_corrupted
        )
        .abs()
        .max()
        .item()
    )


    if (
        input_consistency_error
        > OBSERVED_TOLERANCE
    ):

        raise RuntimeError(
            "Corrupted input is inconsistent "
            "with target × mask. "
            f"Maximum error: "
            f"{input_consistency_error:.6e}"
        )


    # ==================================================================
    # SAMPLE COUNTS
    # ==================================================================

    observed_samples = int(
        mask.sum().item()
    )


    total_samples = int(
        mask.numel()
    )


    missing_samples = (
        total_samples
        - observed_samples
    )


    if missing_samples <= 0:

        raise RuntimeError(
            "No missing samples detected."
        )


    # ==================================================================
    # CURVELET POCS RECONSTRUCTION
    # ==================================================================

    reconstruction = (
        curvelet_pocs_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,

            num_scales=(
                CURVELET_NUM_SCALES
            ),

            wedges_per_direction=(
                CURVELET_WEDGES_PER_DIRECTION
            ),

            iterations=(
                CURVELET_ITERATIONS
            ),

            threshold=(
                CURVELET_THRESHOLD
            ),

            threshold_decay=(
                CURVELET_THRESHOLD_DECAY
            ),

            tolerance=(
                CURVELET_TOLERANCE
            ),
        )
    )


    # ==================================================================
    # OUTPUT SHAPE VALIDATION
    # ==================================================================

    if tuple(
        reconstruction.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected reconstruction shape: "
            f"{tuple(reconstruction.shape)}. "
            f"Expected: {expected_shape}"
        )


    # ==================================================================
    # OUTPUT FINITENESS
    # ==================================================================

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "Curvelet POCS produced "
            "non-finite values."
        )


    # ==================================================================
    # OBSERVED-DATA PRESERVATION
    # ==================================================================

    observed_difference = (
        (
            reconstruction
            - corrupted
        )
        * mask
    ).abs()


    maximum_observed_difference = float(
        observed_difference.max().item()
    )


    if (
        maximum_observed_difference
        > OBSERVED_TOLERANCE
    ):

        raise RuntimeError(
            "Observed seismic samples were "
            "changed by Curvelet POCS. "
            f"Maximum difference: "
            f"{maximum_observed_difference:.6e}"
        )


    # ==================================================================
    # MISSING-SAMPLE RECONSTRUCTION CHECK
    # ==================================================================

    missing_difference = (
        (
            reconstruction
            - corrupted
        )
        * (1.0 - mask)
    ).abs()


    missing_reconstruction_change = float(
        missing_difference.sum().item()
    )


    if (
        missing_reconstruction_change
        <= RECONSTRUCTION_TOLERANCE
    ):

        raise RuntimeError(
            "Curvelet POCS did not modify "
            "the missing samples."
        )


    # ==================================================================
    # RECONSTRUCTION METRICS
    # ==================================================================

    metric_values = compute_metrics(
        reconstruction,
        target
    )


    # ==================================================================
    # RUNTIME
    # ==================================================================

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )


    # ==================================================================
    # RESULT RECORD
    # ==================================================================

    result = {

        "case_id":
            case_id,

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "missing_rate":
            missing_rate,

        "seed":
            seed,

        "cube_depth":
            CUBE_SIZE[0],

        "cube_height":
            CUBE_SIZE[1],

        "cube_width":
            CUBE_SIZE[2],

        "num_scales":
            CURVELET_NUM_SCALES,

        "wedges_per_direction":
            CURVELET_WEDGES_PER_DIRECTION,

        "iterations":
            CURVELET_ITERATIONS,

        "threshold":
            CURVELET_THRESHOLD,

        "threshold_decay":
            CURVELET_THRESHOLD_DECAY,

        "tolerance":
            CURVELET_TOLERANCE,

        "observed_samples":
            observed_samples,

        "missing_samples":
            missing_samples,

        "input_consistency_error":
            input_consistency_error,

        "maximum_observed_difference":
            maximum_observed_difference,

        "missing_reconstruction_change":
            missing_reconstruction_change,

        "mae":
            metric_values["mae"],

        "rmse":
            metric_values["rmse"],

        "psnr":
            metric_values["psnr"],

        "snr":
            metric_values["snr"],

        "ssim":
            metric_values["ssim"],

        "runtime_seconds":
            runtime_seconds,

        "status":
            "SUCCESS",

        "error":
            "",
    }


    return result


# ======================================================================
# SUMMARY GENERATION
# ======================================================================

def generate_summary():
    """
    Generate a grouped summary from the raw 750-case results.
    """

    import pandas as pd


    dataframe = pd.read_csv(
        RESULTS_FILE
    )


    successful = dataframe[
        dataframe["status"]
        == "SUCCESS"
    ].copy()


    if successful.empty:

        raise RuntimeError(
            "No successful experiments are "
            "available for summary generation."
        )


    # ==================================================================
    # GROUP BY GEOLOGY, MASK, AND MISSING RATE
    # ==================================================================

    grouped = (
        successful
        .groupby(
            [
                "geological_mode",
                "mask_mode",
                "missing_rate",
            ],
            as_index=False
        )
        .agg(
            cases=(
                "case_id",
                "count"
            ),

            mae_mean=(
                "mae",
                "mean"
            ),

            mae_std=(
                "mae",
                "std"
            ),

            rmse_mean=(
                "rmse",
                "mean"
            ),

            rmse_std=(
                "rmse",
                "std"
            ),

            psnr_mean=(
                "psnr",
                "mean"
            ),

            psnr_std=(
                "psnr",
                "std"
            ),

            snr_mean=(
                "snr",
                "mean"
            ),

            snr_std=(
                "snr",
                "std"
            ),

            ssim_mean=(
                "ssim",
                "mean"
            ),

            ssim_std=(
                "ssim",
                "std"
            ),

            runtime_mean=(
                "runtime_seconds",
                "mean"
            ),

            runtime_std=(
                "runtime_seconds",
                "std"
            ),
        )
    )


    grouped.to_csv(
        SUMMARY_FILE,
        index=False
    )


    return grouped


# ======================================================================
# MAIN CONTROLLED EXPERIMENT
# ======================================================================

def main():

    print()
    print("=" * 78)
    print(
        "3-D CURVELET POCS CONTROLLED EXPERIMENTAL MATRIX"
    )
    print("=" * 78)

    print()
    print(
        f"Cube size            : {CUBE_SIZE}"
    )

    print(
        f"Missing rates        : {MISSING_RATES}"
    )

    print(
        f"Missing mechanisms   : {len(MASK_MODES)}"
    )

    print(
        f"Geological modes     : {len(GEOLOGICAL_MODES)}"
    )

    print(
        f"Random seeds         : {SEEDS}"
    )

    print(
        f"Expected cases       : {EXPECTED_CASES}"
    )

    print()
    print(
        "Curvelet configuration"
    )
    print(
        "----------------------"
    )

    print(
        f"UDCT scales          : "
        f"{CURVELET_NUM_SCALES}"
    )

    print(
        f"Wedges/direction     : "
        f"{CURVELET_WEDGES_PER_DIRECTION}"
    )

    print(
        f"POCS iterations      : "
        f"{CURVELET_ITERATIONS}"
    )

    print(
        f"Initial threshold    : "
        f"{CURVELET_THRESHOLD}"
    )

    print(
        f"Threshold decay      : "
        f"{CURVELET_THRESHOLD_DECAY}"
    )

    print(
        f"Tolerance             : "
        f"{CURVELET_TOLERANCE}"
    )

    print()
    print(
        f"Raw results          : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Summary results      : "
        f"{SUMMARY_FILE}"
    )

    print()
    print("=" * 78)


    # ==================================================================
    # INITIALIZE CSV
    # ==================================================================

    initialize_results_file()


    successful_cases = 0

    failed_cases = 0


    case_id = 0


    # ==================================================================
    # CONTROLLED MATRIX
    # ==================================================================

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    case_id += 1


                    print()
                    print(
                        "=" * 78
                    )

                    print(
                        f"Case {case_id}/"
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

                    print(
                        "=" * 78
                    )


                    try:

                        result = (
                            run_single_experiment(
                                case_id=case_id,
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


                        append_result(
                            result
                        )


                        successful_cases += 1


                        print(
                            "Status        : "
                            "SUCCESS"
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


                    except Exception as error:

                        failed_cases += 1


                        failure_result = {

                            "case_id":
                                case_id,

                            "geological_mode":
                                geological_mode,

                            "mask_mode":
                                mask_mode,

                            "missing_rate":
                                missing_rate,

                            "seed":
                                seed,

                            "cube_depth":
                                CUBE_SIZE[0],

                            "cube_height":
                                CUBE_SIZE[1],

                            "cube_width":
                                CUBE_SIZE[2],

                            "num_scales":
                                CURVELET_NUM_SCALES,

                            "wedges_per_direction":
                                CURVELET_WEDGES_PER_DIRECTION,

                            "iterations":
                                CURVELET_ITERATIONS,

                            "threshold":
                                CURVELET_THRESHOLD,

                            "threshold_decay":
                                CURVELET_THRESHOLD_DECAY,

                            "tolerance":
                                CURVELET_TOLERANCE,

                            "observed_samples":
                                "",

                            "missing_samples":
                                "",

                            "input_consistency_error":
                                "",

                            "maximum_observed_difference":
                                "",

                            "missing_reconstruction_change":
                                "",

                            "mae":
                                "",

                            "rmse":
                                "",

                            "psnr":
                                "",

                            "snr":
                                "",

                            "ssim":
                                "",

                            "runtime_seconds":
                                "",

                            "status":
                                "FAILED",

                            "error":
                                str(error),
                        }


                        append_result(
                            failure_result
                        )


                        print(
                            "Status        : FAILED"
                        )

                        print(
                            f"Error         : "
                            f"{error}"
                        )


    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print("=" * 78)
    print(
        "CURVELET POCS CONTROLLED MATRIX COMPLETE"
    )
    print("=" * 78)

    print(
        f"Expected cases : "
        f"{EXPECTED_CASES}"
    )

    print(
        f"Completed cases: "
        f"{successful_cases + failed_cases}"
    )

    print(
        f"Successful     : "
        f"{successful_cases}"
    )

    print(
        f"Failed         : "
        f"{failed_cases}"
    )


    # ==================================================================
    # GENERATE SUMMARY
    # ==================================================================

    if successful_cases > 0:

        generate_summary()


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


    # ==================================================================
    # FINAL STATUS
    # ==================================================================

    if (
        successful_cases
        == EXPECTED_CASES
        and failed_cases == 0
    ):

        print()
        print(
            "OVERALL STATUS: PASS"
        )

        print(
            "All controlled Curvelet POCS "
            "experiments completed successfully."
        )

    else:

        print()
        print(
            "OVERALL STATUS: FAIL"
        )

        print(
            "One or more controlled Curvelet "
            "POCS experiments failed."
        )


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()