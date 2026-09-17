"""
======================================================================
PROPOSED MODEL — 750-CASE CONTROLLED EXPERIMENTAL MATRIX
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This script extends the already validated single controlled
proposed-model experiment to the complete 750-case matrix.

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

Seeds:
    42, 43, 44, 45, 46

Total:
    5 × 5 × 6 × 5 = 750 cases

Cube:
    64 × 128 × 128

IMPORTANT
---------
The uncertainty calculations in this script follow the
already validated Colab controlled single-run implementation.

Author: Ormin Joseph
======================================================================
"""


# =====================================================================
# 1. STANDARD LIBRARY
# =====================================================================

import sys
import time
from pathlib import Path


# =====================================================================
# 2. NUMERICAL / DEEP LEARNING LIBRARIES
# =====================================================================

import numpy as np
import pandas as pd
import torch


# =====================================================================
# 3. PROJECT ROOT
# =====================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =====================================================================
# 4. PROJECT IMPORTS
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from models.network import Network3D

from models.mc_dropout import (
    MCDropout3D
)

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.config import (
    MC_DROPOUT_SAMPLES
)


# =====================================================================
# 5. CONTROLLED EXPERIMENT CONFIGURATION
# =====================================================================

CUBE_SIZE = (
    64,
    128,
    128,
)


MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
]


GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


NUM_SAMPLES = 1


# =====================================================================
# 6. EXPECTED NUMBER OF CASES
# =====================================================================

EXPECTED_CASES = (
    len(MISSING_RATES)
    * len(MASK_MODES)
    * len(GEOLOGICAL_MODES)
    * len(SEEDS)
)


# =====================================================================
# 7. DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# 8. CHECKPOINT
# =====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "checkpoints"
    / "best_model.pth"
)


# =====================================================================
# 9. OUTPUT DIRECTORY
# =====================================================================

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# 10. OUTPUT FILES
# =====================================================================

RAW_RESULTS_FILE = (
    REPORT_DIR
    / "proposed_controlled_matrix.csv"
)

SUMMARY_RESULTS_FILE = (
    REPORT_DIR
    / "proposed_controlled_matrix_summary.csv"
)


# =====================================================================
# 11. RANDOM SEED
# =====================================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 12. LOAD TRAINED PROPOSED MODEL
# =====================================================================

def load_model():

    # ---------------------------------------------------------------
    # Check checkpoint.
    # ---------------------------------------------------------------

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "\nFresh best_model.pth was not found:\n"
            f"{CHECKPOINT}\n"
        )


    # ---------------------------------------------------------------
    # Create the SAME architecture used in the successful
    # single controlled experiment.
    # ---------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    ).to(DEVICE)


    # ---------------------------------------------------------------
    # Load checkpoint.
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    # ---------------------------------------------------------------
    # Check checkpoint structure.
    # ---------------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    # ---------------------------------------------------------------
    # Load trained weights.
    # ---------------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # ---------------------------------------------------------------
    # Evaluation mode.
    # ---------------------------------------------------------------

    model.eval()


    return model


# =====================================================================
# 13. RUN ONE CONTROLLED CASE
# =====================================================================

def run_single_case(
    model,
    missing_rate,
    mask_mode,
    geological_mode,
    seed,
):

    # ================================================================
    # REPRODUCIBILITY
    # ================================================================

    set_seed(seed)


    # ================================================================
    # DATASET
    # ================================================================

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )


    # ================================================================
    # GET CONTROLLED SAMPLE
    # ================================================================

    sample = dataset[0]


    # ---------------------------------------------------------------
    # Current standard dataset returns:
    #
    # input_cube
    # target_cube
    # mask
    # velocity_model
    # mask_type
    # geological_mode
    # ---------------------------------------------------------------

    (
        input_cube,
        target_cube,
        mask,
        velocity_model,
        returned_mask_mode,
        returned_geological_mode,
    ) = sample


    # ================================================================
    # EXPECTED SHAPE
    # ================================================================

    expected_shape = (
        1,
        *CUBE_SIZE
    )


    if tuple(input_cube.shape) != expected_shape:

        raise RuntimeError(
            "Input shape mismatch: "
            f"{tuple(input_cube.shape)}"
        )


    if tuple(target_cube.shape) != expected_shape:

        raise RuntimeError(
            "Target shape mismatch: "
            f"{tuple(target_cube.shape)}"
        )


    if tuple(mask.shape) != expected_shape:

        raise RuntimeError(
            "Mask shape mismatch: "
            f"{tuple(mask.shape)}"
        )


    if tuple(velocity_model.shape) != expected_shape:

        raise RuntimeError(
            "Velocity shape mismatch: "
            f"{tuple(velocity_model.shape)}"
        )


    # ================================================================
    # VERIFY CONTROLLED CONDITIONS
    # ================================================================

    if returned_mask_mode != mask_mode:

        raise RuntimeError(
            "Returned mask mode does not match "
            "requested mask mode."
        )


    if returned_geological_mode != geological_mode:

        raise RuntimeError(
            "Returned geological mode does not match "
            "requested geological mode."
        )


    # ================================================================
    # INPUT CONSISTENCY
    # ================================================================

    input_difference = torch.max(
        torch.abs(
            input_cube[mask == 1]
            - target_cube[mask == 1]
        )
    ).item()


    if input_difference > 1e-6:

        raise RuntimeError(
            "Observed input samples do not match "
            "target samples."
        )


    # ================================================================
    # OBSERVED / MISSING COUNTS
    # ================================================================

    observed_voxels = int(
        torch.sum(
            mask == 1
        ).item()
    )


    missing_voxels = int(
        torch.sum(
            mask == 0
        ).item()
    )


    # ================================================================
    # BATCH DIMENSION
    # ================================================================

    input_batch = (
        input_cube
        .unsqueeze(0)
        .to(DEVICE)
    )


    target_batch = (
        target_cube
        .unsqueeze(0)
        .to(DEVICE)
    )


    mask_batch = (
        mask
        .unsqueeze(0)
        .to(DEVICE)
    )


    # ================================================================
    # MC DROPOUT
    # ================================================================

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_DROPOUT_SAMPLES,
    )


    # ================================================================
    # RUNTIME
    # ================================================================

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    start_time = time.perf_counter()


    with torch.no_grad():

        predictions = (
            mc_dropout.predict(
                input_batch
            )
        )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    runtime = (
        time.perf_counter()
        - start_time
    )


    # ================================================================
    # EXTRACT MC OUTPUTS
    # ================================================================

    reconstruction_samples = (
        predictions[
            "reconstruction_samples"
        ]
    )


    travel_time_samples = (
        predictions[
            "travel_time_samples"
        ]
    )


    log_variance_samples = (
        predictions[
            "log_variance_samples"
        ]
    )


    # ================================================================
    # CREATE UNCERTAINTY ESTIMATOR
    # ================================================================
    #
    # EXACTLY AS IN THE SUCCESSFUL COLAB RUN.
    # ================================================================

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
    )


    # ================================================================
    # ALEATORIC VARIANCE
    # ================================================================

    aleatoric_variance = (
        uncertainty_estimator.aleatoric_variance(
            log_variance_samples
        )
    )


    # ================================================================
    # EPISTEMIC VARIANCE
    # ================================================================

    epistemic_variance = (
        uncertainty_estimator.epistemic_variance(
            reconstruction_samples
        )
    )


    # ================================================================
    # PREDICTIVE VARIANCE
    # ================================================================
    #
    # IMPORTANT:
    #
    # We deliberately use the ORIGINAL validated Colab call:
    #
    # predictive_variance =
    #     estimator.predictive_variance(
    #         log_variance_samples,
    #         reconstruction_samples
    #     )
    #
    # ================================================================

    predictive_variance = (
        uncertainty_estimator.predictive_variance(
            log_variance_samples,
            reconstruction_samples
        )
    )


    # ================================================================
    # PREDICTIVE STANDARD DEVIATION
    # ================================================================

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0
        )
    )


    # ================================================================
    # RECONSTRUCTION MEAN
    # ================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )


    # ================================================================
    # DATA-CONSISTENCY PROJECTION
    # ================================================================
    #
    # EXACTLY AS IN THE SUCCESSFUL COLAB RUN:
    #
    # observed reconstruction =
    #     original observed input
    #
    # ================================================================

    reconstruction_mean = (
        reconstruction_mean.clone()
    )


    reconstruction_mean[
        mask_batch == 1
    ] = input_batch[
        mask_batch == 1
    ]


    # ================================================================
    # SHAPE CHECKS
    # ================================================================

    if reconstruction_mean.shape != expected_shape:

        raise RuntimeError(
            "Reconstruction shape mismatch."
        )


    if aleatoric_variance.shape != expected_shape:

        raise RuntimeError(
            "Aleatoric variance shape mismatch."
        )


    if epistemic_variance.shape != expected_shape:

        raise RuntimeError(
            "Epistemic variance shape mismatch."
        )


    if predictive_variance.shape != expected_shape:

        raise RuntimeError(
            "Predictive variance shape mismatch."
        )


    if predictive_std.shape != expected_shape:

        raise RuntimeError(
            "Predictive standard deviation shape mismatch."
        )


    # ================================================================
    # FINITE-VALUE CHECKS
    # ================================================================

    for name, tensor in {

        "reconstruction":
            reconstruction_mean,

        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,

        "predictive_std":
            predictive_std,

    }.items():

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )


    # ================================================================
    # NON-NEGATIVE VARIANCE CHECKS
    # ================================================================

    if not (
        aleatoric_variance >= 0
    ).all():

        raise RuntimeError(
            "Aleatoric variance contains "
            "negative values."
        )


    if not (
        epistemic_variance >= 0
    ).all():

        raise RuntimeError(
            "Epistemic variance contains "
            "negative values."
        )


    if not (
        predictive_variance >= 0
    ).all():

        raise RuntimeError(
            "Predictive variance contains "
            "negative values."
        )


    if not (
        predictive_std >= 0
    ).all():

        raise RuntimeError(
            "Predictive standard deviation "
            "contains negative values."
        )


    # ================================================================
    # OBSERVED DATA PRESERVATION
    # ================================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction_mean[
                mask_batch == 1
            ]
            - input_batch[
                mask_batch == 1
            ]
        )
    ).item()


    if observed_difference > 1e-6:

        raise RuntimeError(
            "Observed samples were not preserved."
        )


    # ================================================================
    # RECONSTRUCTION METRICS
    # ================================================================

    metric_mae = mae(
        reconstruction_mean,
        target_batch
    ).item()


    metric_rmse = rmse(
        reconstruction_mean,
        target_batch
    ).item()


    metric_psnr = psnr(
        reconstruction_mean,
        target_batch
    ).item()


    metric_snr = snr(
        reconstruction_mean,
        target_batch
    ).item()


    metric_ssim = ssim(
        reconstruction_mean,
        target_batch
    ).item()


    # ================================================================
    # MISSING-ONLY MAE
    # ================================================================

    missing_mae = torch.mean(
        torch.abs(
            reconstruction_mean[
                mask_batch == 0
            ]
            - target_batch[
                mask_batch == 0
            ]
        )
    ).item()


    # ================================================================
    # MISSING-ONLY RMSE
    # ================================================================

    missing_rmse = torch.sqrt(
        torch.mean(
            (
                reconstruction_mean[
                    mask_batch == 0
                ]
                - target_batch[
                    mask_batch == 0
                ]
            ) ** 2
        )
    ).item()


    # ================================================================
    # UNCERTAINTY STATISTICS
    # ================================================================

    mean_aleatoric = (
        aleatoric_variance.mean()
        .item()
    )


    mean_epistemic = (
        epistemic_variance.mean()
        .item()
    )


    mean_predictive = (
        predictive_variance.mean()
        .item()
    )


    mean_predictive_std = (
        predictive_std.mean()
        .item()
    )


    missing_predictive_std = (
        predictive_std[
            mask_batch == 0
        ]
        .mean()
        .item()
    )


    # ================================================================
    # RETURN CASE RESULT
    # ================================================================

    return {

        "missing_rate":
            missing_rate,

        "missing_percentage":
            missing_rate * 100.0,

        "mask_mode":
            mask_mode,

        "geological_mode":
            geological_mode,

        "seed":
            seed,

        "cube_depth":
            CUBE_SIZE[0],

        "cube_height":
            CUBE_SIZE[1],

        "cube_width":
            CUBE_SIZE[2],

        "observed_voxels":
            observed_voxels,

        "missing_voxels":
            missing_voxels,

        "mc_samples":
            MC_DROPOUT_SAMPLES,

        "observed_difference":
            observed_difference,

        "mae":
            metric_mae,

        "rmse":
            metric_rmse,

        "psnr_db":
            metric_psnr,

        "snr_db":
            metric_snr,

        "ssim":
            metric_ssim,

        "missing_mae":
            missing_mae,

        "missing_rmse":
            missing_rmse,

        "mean_aleatoric_variance":
            mean_aleatoric,

        "mean_epistemic_variance":
            mean_epistemic,

        "mean_predictive_variance":
            mean_predictive,

        "mean_predictive_std":
            mean_predictive_std,

        "missing_predictive_std":
            missing_predictive_std,

        "runtime_seconds":
            runtime,

        "status":
            "SUCCESS",
    }


# =====================================================================
# 14. MAIN
# =====================================================================

def main():

    # ================================================================
    # HEADER
    # ================================================================

    print()
    print("=" * 80)
    print(
        "PROPOSED MODEL CONTROLLED MATRIX"
    )
    print("=" * 80)


    print()
    print("Configuration")
    print("-------------")


    print(
        f"Device              : {DEVICE}"
    )


    print(
        f"Cube size           : {CUBE_SIZE}"
    )


    print(
        f"Missing rates       : {MISSING_RATES}"
    )


    print(
        f"Missing mechanisms  : {len(MASK_MODES)}"
    )


    print(
        f"Geological modes    : {len(GEOLOGICAL_MODES)}"
    )


    print(
        f"Seeds               : {SEEDS}"
    )


    print(
        f"MC samples          : "
        f"{MC_DROPOUT_SAMPLES}"
    )


    print(
        f"Expected cases      : "
        f"{EXPECTED_CASES}"
    )


    print()
    print("Checkpoint")
    print("----------")

    print(
        CHECKPOINT
    )


    # ================================================================
    # CHECKPOINT VALIDATION
    # ================================================================

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT}"
        )


    print(
        "Checkpoint exists   : PASS"
    )


    # ================================================================
    # LOAD MODEL ONCE
    # ================================================================

    print()
    print("=" * 80)
    print(
        "LOADING PROPOSED MODEL"
    )
    print("=" * 80)


    model = load_model()


    print(
        "Model loading       : PASS"
    )


    # ================================================================
    # RESULTS
    # ================================================================

    results = []


    # ================================================================
    # CASE COUNTER
    # ================================================================

    case_number = 0


    # ================================================================
    # MATRIX LOOP
    # ================================================================

    for missing_rate in MISSING_RATES:

        for mask_mode in MASK_MODES:

            for geological_mode in GEOLOGICAL_MODES:

                for seed in SEEDS:

                    case_number += 1


                    print()
                    print("-" * 80)


                    print(
                        f"CASE "
                        f"{case_number}/"
                        f"{EXPECTED_CASES}"
                    )


                    print(
                        f"Missing rate      : "
                        f"{missing_rate:.0%}"
                    )


                    print(
                        f"Missing mechanism : "
                        f"{mask_mode}"
                    )


                    print(
                        f"Geological mode   : "
                        f"{geological_mode}"
                    )


                    print(
                        f"Seed              : "
                        f"{seed}"
                    )


                    # =================================================
                    # RUN CASE
                    # =================================================

                    case_start = (
                        time.perf_counter()
                    )


                    try:

                        result = run_single_case(
                            model=model,
                            missing_rate=missing_rate,
                            mask_mode=mask_mode,
                            geological_mode=geological_mode,
                            seed=seed,
                        )


                        result[
                            "case_number"
                        ] = case_number


                        results.append(
                            result
                        )


                        case_runtime = (
                            time.perf_counter()
                            - case_start
                        )


                        print()
                        print(
                            "Status            : "
                            "SUCCESS"
                        )


                        print(
                            f"Missing MAE       : "
                            f"{result['missing_mae']:.6f}"
                        )


                        print(
                            f"MAE               : "
                            f"{result['mae']:.6f}"
                        )


                        print(
                            f"RMSE              : "
                            f"{result['rmse']:.6f}"
                        )


                        print(
                            f"PSNR              : "
                            f"{result['psnr_db']:.6f} dB"
                        )


                        print(
                            f"SNR               : "
                            f"{result['snr_db']:.6f} dB"
                        )


                        print(
                            f"SSIM              : "
                            f"{result['ssim']:.6f}"
                        )


                        print(
                            f"Predictive std    : "
                            f"{result['mean_predictive_std']:.6e}"
                        )


                        print(
                            f"Runtime           : "
                            f"{case_runtime:.4f} s"
                        )


                    except Exception as error:

                        # ------------------------------------------------
                        # Record failure without stopping the complete
                        # experiment.
                        # ------------------------------------------------

                        case_runtime = (
                            time.perf_counter()
                            - case_start
                        )


                        failed_result = {

                            "case_number":
                                case_number,

                            "missing_rate":
                                missing_rate,

                            "missing_percentage":
                                missing_rate * 100.0,

                            "mask_mode":
                                mask_mode,

                            "geological_mode":
                                geological_mode,

                            "seed":
                                seed,

                            "cube_depth":
                                CUBE_SIZE[0],

                            "cube_height":
                                CUBE_SIZE[1],

                            "cube_width":
                                CUBE_SIZE[2],

                            "status":
                                "FAILED",

                            "error":
                                str(error),

                            "runtime_seconds":
                                case_runtime,
                        }


                        results.append(
                            failed_result
                        )


                        print()
                        print(
                            "Status            : "
                            "FAILED"
                        )


                        print(
                            f"Error             : "
                            f"{error}"
                        )


                    # =================================================
                    # SAVE PROGRESS AFTER EVERY CASE
                    # =================================================

                    progress_dataframe = (
                        pd.DataFrame(
                            results
                        )
                    )


                    progress_dataframe.to_csv(
                        RAW_RESULTS_FILE,
                        index=False,
                    )


    # ================================================================
    # FINAL RAW RESULTS
    # ================================================================

    results_dataframe = (
        pd.DataFrame(
            results
        )
    )


    results_dataframe.to_csv(
        RAW_RESULTS_FILE,
        index=False,
    )


    # ================================================================
    # SUCCESS / FAILURE COUNTS
    # ================================================================

    successful_cases = int(
        (
            results_dataframe[
                "status"
            ]
            == "SUCCESS"
        ).sum()
    )


    failed_cases = int(
        (
            results_dataframe[
                "status"
            ]
            == "FAILED"
        ).sum()
    )


    # ================================================================
    # SUMMARY
    # ================================================================

    successful_results = (
        results_dataframe[
            results_dataframe[
                "status"
            ]
            == "SUCCESS"
        ]
        .copy()
    )


    if len(successful_results) > 0:

        summary_dataframe = (
            successful_results
            .groupby(
                [
                    "missing_rate",
                    "missing_percentage",
                    "mask_mode",
                    "geological_mode",
                ],
                as_index=False,
            )
            .agg(

                seed_count=(
                    "seed",
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
                    "psnr_db",
                    "mean"
                ),

                psnr_std=(
                    "psnr_db",
                    "std"
                ),

                snr_mean=(
                    "snr_db",
                    "mean"
                ),

                snr_std=(
                    "snr_db",
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

                missing_mae_mean=(
                    "missing_mae",
                    "mean"
                ),

                missing_mae_std=(
                    "missing_mae",
                    "std"
                ),

                missing_rmse_mean=(
                    "missing_rmse",
                    "mean"
                ),

                missing_rmse_std=(
                    "missing_rmse",
                    "std"
                ),

                aleatoric_variance_mean=(
                    "mean_aleatoric_variance",
                    "mean"
                ),

                epistemic_variance_mean=(
                    "mean_epistemic_variance",
                    "mean"
                ),

                predictive_variance_mean=(
                    "mean_predictive_variance",
                    "mean"
                ),

                predictive_std_mean=(
                    "mean_predictive_std",
                    "mean"
                ),

                missing_predictive_std_mean=(
                    "missing_predictive_std",
                    "mean"
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


    else:

        summary_dataframe = (
            pd.DataFrame()
        )


    # ================================================================
    # SAVE SUMMARY
    # ================================================================

    summary_dataframe.to_csv(
        SUMMARY_RESULTS_FILE,
        index=False,
    )


    # ================================================================
    # FINAL REPORT
    # ================================================================

    print()
    print("=" * 80)
    print(
        "PROPOSED MODEL CONTROLLED MATRIX COMPLETE"
    )
    print("=" * 80)


    print()
    print(
        f"Expected cases     : "
        f"{EXPECTED_CASES}"
    )


    print(
        f"Completed cases    : "
        f"{len(results_dataframe)}"
    )


    print(
        f"Successful cases   : "
        f"{successful_cases}"
    )


    print(
        f"Failed cases       : "
        f"{failed_cases}"
    )


    print()
    print(
        "Raw results:"
    )


    print(
        RAW_RESULTS_FILE
    )


    print()
    print(
        "Summary results:"
    )


    print(
        SUMMARY_RESULTS_FILE
    )


    # ================================================================
    # OVERALL STATUS
    # ================================================================

    if (
        len(results_dataframe)
        == EXPECTED_CASES
        and failed_cases == 0
    ):

        print()
        print("=" * 80)
        print(
            "OVERALL STATUS: PASS"
        )
        print(
            "All controlled proposed-model "
            "experiments completed successfully."
        )
        print("=" * 80)


    else:

        print()
        print("=" * 80)
        print(
            "OVERALL STATUS: REVIEW"
        )
        print(
            "One or more controlled experiments "
            "failed."
        )
        print("=" * 80)


# =====================================================================
# 15. SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()