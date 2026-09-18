"""
=================================================================
Linear Interpolation Controlled Experimental Matrix
=================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Controlled evaluation of the Linear Interpolation baseline
using the standardized synthetic experimental matrix.

Experimental matrix
--------------------

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

    = 750 controlled experiments

Standard seismic cube:
    (D, H, W) = (64, 128, 128)

Tensor convention:
    (C, D, H, W) = (1, 64, 128, 128)

Metrics:
    MAE
    RMSE
    PSNR
    SNR
    SSIM

Additional checks:
    - Input consistency
    - Observed-sample preservation
    - Missing-only reconstruction metrics
    - Finite reconstruction
    - Runtime
    - Reproducibility through deterministic dataset seeds

Output:
    outputs/synthetic_training/reports/

Files:
    linear_interpolation_controlled_matrix.csv
    linear_interpolation_controlled_matrix_summary.csv

Author: Ormin Joseph
=================================================================
"""

import csv
import os
import time

import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.baseline_linear_interpolation import (
    linear_interpolation_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cpu")


# ================================================================
# STANDARD CONTROLLED EXPERIMENTAL MATRIX
# ================================================================

CUBE_SIZE = (64, 128, 128)

MISSING_RATES = [
    0.30,
]

MISSING_MECHANISMS = [
    "random_voxels",
    "missing_traces",
]

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
]

SEEDS = [
    42,
    43,
]


# ================================================================
# EXPECTED NUMBER OF EXPERIMENTAL CASES
# ================================================================

EXPECTED_CASES = (
    len(MISSING_RATES)
    * len(MISSING_MECHANISMS)
    * len(GEOLOGICAL_MODES)
    * len(SEEDS)
)


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIRECTORY = (
    "outputs"
    + os.sep
    + "synthetic_training"
    + os.sep
    + "reports"
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# ================================================================
# OUTPUT FILES
# ================================================================

RAW_RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "linear_interpolation_controlled_matrix.csv"
)

SUMMARY_RESULTS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "linear_interpolation_controlled_matrix_summary.csv"
)


# ================================================================
# UTILITY FUNCTION
# ================================================================

def to_float(value):
    """
    Convert a metric value to a Python float.
    """

    if isinstance(value, torch.Tensor):

        return float(value.detach().cpu().item())

    return float(value)


# ================================================================
# SINGLE CONTROLLED EXPERIMENT
# ================================================================

def run_single_case(
        missing_rate,
        missing_mechanism,
        geological_mode,
        seed
):
    """
    Run one controlled Linear Interpolation experiment.

    Returns
    -------
    dict
        Results for the experimental case.
    """

    # ------------------------------------------------------------
    # Create exactly one synthetic sample.
    # ------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=missing_mechanism,
        seed=seed
    )

    # ------------------------------------------------------------
    # Retrieve the single controlled sample.
    # ------------------------------------------------------------

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        mask_type,
        resolved_geological_mode
    ) = dataset[0]

    # ------------------------------------------------------------
    # Validate tensor shapes.
    # ------------------------------------------------------------

    expected_shape = (
        1,
        CUBE_SIZE[0],
        CUBE_SIZE[1],
        CUBE_SIZE[2]
    )

    if tuple(corrupted_cube.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected corrupted_cube shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(corrupted_cube.shape)}."
        )

    if tuple(target.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected target shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(target.shape)}."
        )

    if tuple(mask.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected mask shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(mask.shape)}."
        )

    # ------------------------------------------------------------
    # Validate metadata.
    # ------------------------------------------------------------

    if mask_type != missing_mechanism:

        raise RuntimeError(
            "Mask mechanism mismatch. "
            f"Expected {missing_mechanism}, "
            f"received {mask_type}."
        )

    if resolved_geological_mode != geological_mode:

        raise RuntimeError(
            "Geological mode mismatch. "
            f"Expected {geological_mode}, "
            f"received {resolved_geological_mode}."
        )

    # ------------------------------------------------------------
    # Move tensors to the standard evaluation device.
    # ------------------------------------------------------------

    corrupted_cube = corrupted_cube.to(DEVICE)
    target = target.to(DEVICE)
    mask = mask.to(DEVICE)

    # ------------------------------------------------------------
    # Verify that the input cube is actually the masked target.
    #
    # The synthetic dataset convention is:
    #
    #     corrupted_cube = target * mask
    # ------------------------------------------------------------

    input_consistency_error = torch.max(
        torch.abs(
            corrupted_cube
            - target * mask
        )
    ).item()

    if input_consistency_error > 1e-6:

        raise RuntimeError(
            "Input consistency check failed. "
            f"Maximum difference: "
            f"{input_consistency_error}"
        )

    # ------------------------------------------------------------
    # Confirm that observed samples are present.
    # ------------------------------------------------------------

    observed = mask == 1
    missing = mask == 0

    if not observed.any():

        raise RuntimeError(
            "Controlled case contains no observed samples."
        )

    if not missing.any():

        raise RuntimeError(
            "Controlled case contains no missing samples."
        )

    # ------------------------------------------------------------
    # Run Linear Interpolation.
    # ------------------------------------------------------------

    start_time = time.perf_counter()

    reconstruction = linear_interpolation_reconstruction(
        corrupted_cube,
        mask
    )

    end_time = time.perf_counter()

    runtime_seconds = (
        end_time - start_time
    )

    # ------------------------------------------------------------
    # Validate reconstruction shape.
    # ------------------------------------------------------------

    if tuple(reconstruction.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected reconstruction shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(reconstruction.shape)}."
        )

    # ------------------------------------------------------------
    # Validate finite reconstruction.
    # ------------------------------------------------------------

    if not torch.isfinite(reconstruction).all():

        raise RuntimeError(
            "Linear interpolation produced "
            "non-finite reconstruction values."
        )

    # ------------------------------------------------------------
    # Verify exact preservation of observed samples.
    # ------------------------------------------------------------

    observed_difference = torch.max(
        torch.abs(
            reconstruction[observed]
            - corrupted_cube[observed]
        )
    ).item()

    if observed_difference > 1e-6:

        raise RuntimeError(
            "Observed seismic samples were modified. "
            f"Maximum difference: "
            f"{observed_difference}"
        )

    # ------------------------------------------------------------
    # Calculate common reconstruction metrics.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Calculate metrics specifically on missing samples.
    #
    # This is particularly important for reconstruction methods
    # because observed samples are already known.
    # ------------------------------------------------------------

    reconstruction_missing = reconstruction[missing]
    target_missing = target[missing]

    missing_mae_value = to_float(
        mae(
            reconstruction_missing,
            target_missing
        )
    )

    missing_rmse_value = to_float(
        rmse(
            reconstruction_missing,
            target_missing
        )
    )

    # ------------------------------------------------------------
    # Return all controlled-case results.
    # ------------------------------------------------------------

    return {
        "missing_rate": missing_rate,
        "missing_mechanism": missing_mechanism,
        "geological_mode": geological_mode,
        "seed": seed,
        "cube_depth": CUBE_SIZE[0],
        "cube_height": CUBE_SIZE[1],
        "cube_width": CUBE_SIZE[2],
        "mae": mae_value,
        "rmse": rmse_value,
        "psnr": psnr_value,
        "snr": snr_value,
        "ssim": ssim_value,
        "missing_mae": missing_mae_value,
        "missing_rmse": missing_rmse_value,
        "runtime_seconds": runtime_seconds,
        "observed_difference": observed_difference,
        "input_consistency_error": input_consistency_error,
        "status": "SUCCESS"
    }


# ================================================================
# CSV FIELD NAMES
# ================================================================

FIELD_NAMES = [
    "missing_rate",
    "missing_mechanism",
    "geological_mode",
    "seed",
    "cube_depth",
    "cube_height",
    "cube_width",
    "mae",
    "rmse",
    "psnr",
    "snr",
    "ssim",
    "missing_mae",
    "missing_rmse",
    "runtime_seconds",
    "observed_difference",
    "input_consistency_error",
    "status"
]


# ================================================================
# MAIN CONTROLLED EXPERIMENT
# ================================================================

def main():

    print(
        "\n"
        "=========================================================\n"
        "LINEAR INTERPOLATION CONTROLLED EXPERIMENTAL MATRIX\n"
        "=========================================================\n"
    )

    print(
        f"Cube size       : {CUBE_SIZE}"
    )

    print(
        f"Missing rates   : {MISSING_RATES}"
    )

    print(
        f"Missing methods : {MISSING_MECHANISMS}"
    )

    print(
        f"Geological modes: {GEOLOGICAL_MODES}"
    )

    print(
        f"Seeds           : {SEEDS}"
    )

    print(
        f"Expected cases  : {EXPECTED_CASES}"
    )

    print(
        f"Output directory: {OUTPUT_DIRECTORY}"
    )

    print(
        "=========================================================\n"
    )

    # ------------------------------------------------------------
    # Prepare result storage.
    # ------------------------------------------------------------

    results = []

    successful_cases = 0
    failed_cases = 0

    case_number = 0

    # ------------------------------------------------------------
    # Create / overwrite raw result CSV.
    # ------------------------------------------------------------

    with open(
        RAW_RESULTS_FILE,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELD_NAMES
        )

        writer.writeheader()

        # --------------------------------------------------------
        # Execute every controlled experimental combination.
        # --------------------------------------------------------

        for missing_rate in MISSING_RATES:

            for missing_mechanism in MISSING_MECHANISMS:

                for geological_mode in GEOLOGICAL_MODES:

                    for seed in SEEDS:

                        case_number += 1

                        print(
                            f"Case {case_number}/{EXPECTED_CASES} | "
                            f"Missing={missing_rate:.2f} | "
                            f"Mechanism={missing_mechanism} | "
                            f"Geology={geological_mode} | "
                            f"Seed={seed}"
                        )

                        try:

                            case_result = run_single_case(
                                missing_rate=missing_rate,
                                missing_mechanism=missing_mechanism,
                                geological_mode=geological_mode,
                                seed=seed
                            )

                            results.append(
                                case_result
                            )

                            writer.writerow(
                                case_result
                            )

                            csv_file.flush()

                            successful_cases += 1

                            print(
                                f"  Status SUCCESS | "
                                f"MAE={case_result['mae']:.6f} | "
                                f"RMSE={case_result['rmse']:.6f} | "
                                f"SSIM={case_result['ssim']:.6f} | "
                                f"Runtime="
                                f"{case_result['runtime_seconds']:.4f}s"
                            )

                        except Exception as error:

                            failed_cases += 1

                            failed_result = {
                                "missing_rate": missing_rate,
                                "missing_mechanism": missing_mechanism,
                                "geological_mode": geological_mode,
                                "seed": seed,
                                "cube_depth": CUBE_SIZE[0],
                                "cube_height": CUBE_SIZE[1],
                                "cube_width": CUBE_SIZE[2],
                                "mae": "",
                                "rmse": "",
                                "psnr": "",
                                "snr": "",
                                "ssim": "",
                                "missing_mae": "",
                                "missing_rmse": "",
                                "runtime_seconds": "",
                                "observed_difference": "",
                                "input_consistency_error": "",
                                "status": f"FAILED: {error}"
                            }

                            writer.writerow(
                                failed_result
                            )

                            csv_file.flush()

                            print(
                                f"  Status FAILED | "
                                f"{error}"
                            )

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================

    if successful_cases > 0:

        metric_names = [
            "mae",
            "rmse",
            "psnr",
            "snr",
            "ssim",
            "missing_mae",
            "missing_rmse",
            "runtime_seconds"
        ]

        summary_rows = []

        for metric_name in metric_names:

            values = [
                float(row[metric_name])
                for row in results
            ]

            mean_value = (
                sum(values)
                / len(values)
            )

            minimum_value = min(values)
            maximum_value = max(values)

            summary_rows.append(
                {
                    "metric": metric_name,
                    "mean": mean_value,
                    "minimum": minimum_value,
                    "maximum": maximum_value,
                    "successful_cases": successful_cases
                }
            )

        # --------------------------------------------------------
        # Write summary CSV.
        # --------------------------------------------------------

        with open(
            SUMMARY_RESULTS_FILE,
            "w",
            newline=""
        ) as csv_file:

            summary_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "metric",
                    "mean",
                    "minimum",
                    "maximum",
                    "successful_cases"
                ]
            )

            summary_writer.writeheader()

            summary_writer.writerows(
                summary_rows
            )

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print(
        "\n"
        "=========================================================\n"
        "LINEAR INTERPOLATION CONTROLLED MATRIX COMPLETE\n"
        "========================================================="
    )

    print(
        f"Expected cases : {EXPECTED_CASES}"
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
        f"\nRaw results:\n{RAW_RESULTS_FILE}"
    )

    print(
        f"Summary results:\n{SUMMARY_RESULTS_FILE}"
    )

    # ------------------------------------------------------------
    # Overall pass/fail condition.
    # ------------------------------------------------------------

    if (
        case_number == EXPECTED_CASES
        and successful_cases == EXPECTED_CASES
        and failed_cases == 0
    ):

        print(
            "\nOVERALL STATUS: PASS"
        )

        print(
            "All controlled Linear Interpolation "
            "experiments completed successfully."
        )

    else:

        print(
            "\nOVERALL STATUS: FAIL"
        )

        print(
            "One or more controlled experiments failed."
        )


# ================================================================
# SCRIPT ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()