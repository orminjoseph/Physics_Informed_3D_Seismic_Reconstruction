"""
====================================================================
PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the FROZEN proposed model over the same controlled
experimental matrix used for the baseline methods.

Experimental matrix
-------------------
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
    10%
    20%
    30%
    40%
    50%

Seeds:
    42
    43
    44
    45
    46

Total:
    6 × 5 × 5 × 5 = 750 cases

Important
---------
The neural network is NOT retrained for every case.

A single frozen best_model.pth is evaluated across all 750
controlled cases.

The reconstruction is made data-consistent by restoring the
observed samples exactly after neural-network inference.

Predictive uncertainty:
    Aleatoric variance
    Epistemic variance
    Predictive variance
    Predictive standard deviation

Author: Ormin Joseph
====================================================================
"""

import os
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    MC_DROPOUT_SAMPLES,
)


# ====================================================================
# PROJECT ROOT
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ====================================================================
# CONTROLLED EXPERIMENTAL MATRIX
# ====================================================================

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


# ====================================================================
# CUBE CONFIGURATION
# ====================================================================

CUBE_SIZE = (
    64,
    128,
    128
)

NUM_SAMPLES = 1


# ====================================================================
# MODEL CONFIGURATION
# ====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "checkpoints"
    / "best_model.pth"
)


MC_SAMPLES = MC_DROPOUT_SAMPLES


# ====================================================================
# OUTPUT CONFIGURATION
# ====================================================================

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


RESULTS_FILE = (
    REPORT_DIR
    / "proposed_model_controlled_matrix.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "proposed_model_controlled_matrix_summary.csv"
)


# ====================================================================
# NUMERICAL TOLERANCE
# ====================================================================

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# ====================================================================
# DEVICE
# ====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """
    Set deterministic random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================
# METRIC CONVERSION
# ====================================================================

def metric_to_float(value):
    """
    Convert a metric output to Python float.
    """

    if isinstance(value, torch.Tensor):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    if isinstance(value, np.ndarray):

        return float(
            value.item()
        )

    return float(value)


# ====================================================================
# FINITE CHECK
# ====================================================================

def tensor_is_finite(tensor):
    """
    Return True when all tensor values are finite.
    """

    return bool(
        torch.isfinite(tensor)
        .all()
        .item()
    )


# ====================================================================
# LOAD FROZEN MODEL
# ====================================================================

def load_model():
    """
    Load the frozen production Network3D checkpoint.
    """

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "\nProposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}\n\n"
            "Train the proposed model and create "
            "best_model.pth before running this matrix."
        )

    print()
    print("=" * 70)
    print("LOADING FROZEN PROPOSED MODEL")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Device:")
    print(DEVICE)

    # ------------------------------------------------------------
    # Production architecture
    # ------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    model = model.to(DEVICE)

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------

    model.eval()

    print()
    print("Proposed model loaded successfully.")

    if "best_epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint["best_epoch"]
        )

    if "best_validation_loss" in checkpoint:

        print(
            "Best validation loss:",
            checkpoint["best_validation_loss"]
        )

    return model


# ====================================================================
# RUN ONE CONTROLLED EXPERIMENT
# ====================================================================

def run_single_experiment(
    model,
    geological_mode,
    mask_mode,
    missing_rate,
    seed
):
    """
    Run one controlled proposed-model experiment.
    """

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
        seed=seed
    )

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode
    ) = dataset[0]

    # ================================================================
    # METADATA VALIDATION
    # ================================================================

    if actual_geological_mode != geological_mode:

        raise RuntimeError(
            "Geological mode mismatch: "
            f"expected={geological_mode}, "
            f"received={actual_geological_mode}"
        )

    if actual_mask_mode != mask_mode:

        raise RuntimeError(
            "Mask mode mismatch: "
            f"expected={mask_mode}, "
            f"received={actual_mask_mode}"
        )

    # ================================================================
    # SHAPE VALIDATION
    # ================================================================

    expected_shape = CUBE_SIZE

    if tuple(corrupted.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected corrupted cube shape: "
            f"{tuple(corrupted.shape)}"
        )

    if corrupted.shape != target.shape:

        raise RuntimeError(
            "Corrupted and target shapes differ."
        )

    if corrupted.shape != mask.shape:

        raise RuntimeError(
            "Corrupted and mask shapes differ."
        )

    if corrupted.shape != velocity.shape:

        raise RuntimeError(
            "Corrupted and velocity shapes differ."
        )

    # ================================================================
    # FINITE CHECKS
    # ================================================================

    for name, tensor in {
        "corrupted": corrupted,
        "target": target,
        "mask": mask,
        "velocity": velocity
    }.items():

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    # ================================================================
    # MASK VALIDATION
    # ================================================================

    unique_values = torch.unique(mask)

    for value in unique_values.tolist():

        if float(value) not in {0.0, 1.0}:

            raise RuntimeError(
                "Mask contains values other than "
                "0 and 1."
            )

    # ================================================================
    # INPUT CONSISTENCY
    # ================================================================

    expected_input = target * mask

    input_difference = torch.max(
        torch.abs(
            corrupted
            - expected_input
        )
    ).item()

    if (
        input_difference
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Input consistency check failed: "
            f"{input_difference:.6e}"
        )

    # ================================================================
    # SAMPLE COUNTS
    # ================================================================

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
            "No missing samples detected."
        )

    total_samples = (
        observed_samples
        +
        missing_samples
    )

    measured_missing_rate = (
        missing_samples
        /
        total_samples
    )

    # ================================================================
    # PREPARE NETWORK INPUT
    # ================================================================

    input_batch = (
        corrupted
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ================================================================
    # MC DROPOUT INFERENCE
    # ================================================================

    predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES
    )

    start_time = time.perf_counter()

    predictions = predictor.predict(
        input_batch
    )

    end_time = time.perf_counter()

    runtime_seconds = (
        end_time
        -
        start_time
    )

    # ================================================================
    # REQUIRED MC OUTPUTS
    # ================================================================

    required_keys = {
        "reconstruction_samples",
        "log_variance_samples"
    }

    missing_keys = (
        required_keys
        -
        predictions.keys()
    )

    if missing_keys:

        raise KeyError(
            "Missing MC outputs: "
            f"{missing_keys}"
        )

    reconstruction_samples = (
        predictions[
            "reconstruction_samples"
        ]
    )

    log_variance_samples = (
        predictions[
            "log_variance_samples"
        ]
    )

    # ================================================================
    # MC RECONSTRUCTION MEAN
    # ================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

    # ================================================================
    # UNCERTAINTY DECOMPOSITION
    # ================================================================

    aleatoric_variance = (
        PredictiveUncertaintyEstimator
        .aleatoric_variance(
            log_variance_samples
        )
    )

    epistemic_variance = (
        PredictiveUncertaintyEstimator
        .epistemic_variance(
            reconstruction_samples
        )
    )

    predictive_variance = (
        PredictiveUncertaintyEstimator
        .predictive_variance(
            aleatoric_variance,
            epistemic_variance
        )
    )

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0
        )
    )

    # ================================================================
    # FINITE UNCERTAINTY CHECKS
    # ================================================================

    uncertainty_tensors = {
        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,

        "predictive_std":
            predictive_std
    }

    for name, tensor in (
        uncertainty_tensors.items()
    ):

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    # ================================================================
    # NON-NEGATIVE VARIANCE CHECKS
    # ================================================================

    if (
        aleatoric_variance < 0
    ).any():

        raise RuntimeError(
            "Negative aleatoric variance detected."
        )

    if (
        epistemic_variance < 0
    ).any():

        raise RuntimeError(
            "Negative epistemic variance detected."
        )

    if (
        predictive_variance < 0
    ).any():

        raise RuntimeError(
            "Negative predictive variance detected."
        )

    # ================================================================
    # DATA CONSISTENCY PROJECTION
    # ================================================================
    #
    # The neural network prediction is projected back onto the
    # observed seismic samples.
    #
    # This guarantees:
    #
    #     reconstruction[mask == 1]
    #         =
    #     corrupted[mask == 1]
    #
    # The uncertainty fields are NOT modified.
    #
    # This is an evaluation-time data-consistency operation.
    # ================================================================

    reconstruction = (
        reconstruction_mean
        *
        (1.0 - mask.unsqueeze(0))
        +
        corrupted.unsqueeze(0)
        *
        mask.unsqueeze(0)
    )

    # ================================================================
    # OBSERVED SAMPLE PRESERVATION
    # ================================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction[
                mask.unsqueeze(0) == 1
            ]
            -
            corrupted.unsqueeze(0)[
                mask.unsqueeze(0) == 1
            ]
        )
    ).item()

    if (
        observed_difference
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation failed: "
            f"{observed_difference:.6e}"
        )

    # ================================================================
    # MISSING-DATA METRICS
    # ================================================================

    missing_selector = (
        mask.unsqueeze(0)
        ==
        0
    )

    missing_prediction = (
        reconstruction[
            missing_selector
        ]
    )

    missing_target = (
        target.unsqueeze(0)[
            missing_selector
        ]
    )

    missing_mae = float(
        torch.mean(
            torch.abs(
                missing_prediction
                -
                missing_target
            )
        ).item()
    )

    missing_rmse = float(
        torch.sqrt(
            torch.mean(
                (
                    missing_prediction
                    -
                    missing_target
                )
                ** 2
            )
        ).item()
    )

    # ================================================================
    # GLOBAL METRICS
    # ================================================================

    target_batch = (
        target
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    reconstruction_batch = (
        reconstruction
        .to(DEVICE)
    )

    metric_mae = metric_to_float(
        mae(
            reconstruction_batch,
            target_batch
        )
    )

    metric_rmse = metric_to_float(
        rmse(
            reconstruction_batch,
            target_batch
        )
    )

    metric_psnr = metric_to_float(
        psnr(
            reconstruction_batch,
            target_batch
        )
    )

    metric_snr = metric_to_float(
        snr(
            reconstruction_batch,
            target_batch
        )
    )

    metric_ssim = metric_to_float(
        ssim(
            reconstruction_batch,
            target_batch
        )
    )

    # ================================================================
    # UNCERTAINTY SUMMARY
    # ================================================================

    mean_aleatoric = float(
        aleatoric_variance
        .mean()
        .item()
    )

    mean_epistemic = float(
        epistemic_variance
        .mean()
        .item()
    )

    mean_predictive = float(
        predictive_variance
        .mean()
        .item()
    )

    mean_predictive_std = float(
        predictive_std
        .mean()
        .item()
    )

    # ================================================================
    # UNCERTAINTY ON MISSING LOCATIONS
    # ================================================================

    missing_aleatoric = float(
        aleatoric_variance[
            missing_selector
        ]
        .mean()
        .item()
    )

    missing_epistemic = float(
        epistemic_variance[
            missing_selector
        ]
        .mean()
        .item()
    )

    missing_predictive = float(
        predictive_variance[
            missing_selector
        ]
        .mean()
        .item()
    )

    missing_predictive_std = float(
        predictive_std[
            missing_selector
        ]
        .mean()
        .item()
    )

    # ================================================================
    # RESULT RECORD
    # ================================================================

    return {

        "method":
            "proposed_physics_informed_3d",

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "seed":
            int(seed),

        "requested_missing_rate":
            float(missing_rate),

        "measured_missing_rate":
            float(measured_missing_rate),

        "cube_depth":
            int(CUBE_SIZE[0]),

        "cube_height":
            int(CUBE_SIZE[1]),

        "cube_width":
            int(CUBE_SIZE[2]),

        "observed_samples":
            int(observed_samples),

        "missing_samples":
            int(missing_samples),

        "input_consistency_error":
            float(input_difference),

        "observed_preservation_error":
            float(observed_difference),

        "runtime_seconds":
            float(runtime_seconds),

        "missing_mae":
            missing_mae,

        "missing_rmse":
            missing_rmse,

        "MAE":
            metric_mae,

        "RMSE":
            metric_rmse,

        "PSNR":
            metric_psnr,

        "SNR":
            metric_snr,

        "SSIM":
            metric_ssim,

        "aleatoric_variance_mean":
            mean_aleatoric,

        "epistemic_variance_mean":
            mean_epistemic,

        "predictive_variance_mean":
            mean_predictive,

        "predictive_std_mean":
            mean_predictive_std,

        "missing_aleatoric_variance_mean":
            missing_aleatoric,

        "missing_epistemic_variance_mean":
            missing_epistemic,

        "missing_predictive_variance_mean":
            missing_predictive,

        "missing_predictive_std_mean":
            missing_predictive_std,

        "status":
            "PASS",

        "error":
            ""
    }


# ====================================================================
# SUMMARY STATISTICS
# ====================================================================

def calculate_summary(records):
    """
    Calculate mean and standard deviation grouped by:

        geological_mode
        mask_mode
        requested_missing_rate
    """

    grouped = {}

    for record in records:

        key = (
            record[
                "geological_mode"
            ],

            record[
                "mask_mode"
            ],

            record[
                "requested_missing_rate"
            ]
        )

        grouped.setdefault(
            key,
            []
        ).append(record)

    metric_names = [

        "missing_mae",
        "missing_rmse",

        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",

        "runtime_seconds",

        "aleatoric_variance_mean",
        "epistemic_variance_mean",
        "predictive_variance_mean",
        "predictive_std_mean",

        "missing_aleatoric_variance_mean",
        "missing_epistemic_variance_mean",
        "missing_predictive_variance_mean",
        "missing_predictive_std_mean"
    ]

    summaries = []

    for key, group in grouped.items():

        geological_mode, mask_mode, missing_rate = key

        summary = {

            "geological_mode":
                geological_mode,

            "mask_mode":
                mask_mode,

            "requested_missing_rate":
                missing_rate,

            "n_seeds":
                len(group)
        }

        for metric_name in metric_names:

            values = np.asarray(
                [
                    float(
                        record[
                            metric_name
                        ]
                    )

                    for record in group
                ],

                dtype=np.float64
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

        summaries.append(
            summary
        )

    return summaries


# ====================================================================
# WRITE CSV
# ====================================================================

def write_csv(
    filename,
    records
):
    """
    Write records to CSV.
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
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ====================================================================
# MAIN CONTROLLED MATRIX
# ====================================================================

def main():

    print()
    print("=" * 78)
    print(
        "PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX"
    )
    print("=" * 78)

    expected_cases = (
        len(GEOLOGICAL_MODES)
        *
        len(MASK_MODES)
        *
        len(MISSING_RATES)
        *
        len(SEEDS)
    )

    print()
    print(
        f"Expected cases : {expected_cases}"
    )

    print(
        f"Cube size      : {CUBE_SIZE}"
    )

    print(
        f"Device         : {DEVICE}"
    )

    print(
        f"MC samples     : {MC_SAMPLES}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    # ================================================================
    # LOAD MODEL ONCE
    # ================================================================

    model = load_model()

    # ================================================================
    # RESULTS
    # ================================================================

    records = []

    completed_cases = 0
    successful_cases = 0
    failed_cases = 0

    # ================================================================
    # CONTROLLED MATRIX
    # ================================================================

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    completed_cases += 1

                    print()
                    print(
                        "-" * 78
                    )

                    print(
                        f"Case "
                        f"{completed_cases}/"
                        f"{expected_cases}"
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

                    try:

                        result = (
                            run_single_experiment(
                                model=model,
                                geological_mode=
                                    geological_mode,
                                mask_mode=
                                    mask_mode,
                                missing_rate=
                                    missing_rate,
                                seed=seed
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
                            f"{result['MAE']:.6f}"
                        )

                        print(
                            f"RMSE          : "
                            f"{result['RMSE']:.6f}"
                        )

                        print(
                            f"PSNR          : "
                            f"{result['PSNR']:.6f} dB"
                        )

                        print(
                            f"SNR           : "
                            f"{result['SNR']:.6f} dB"
                        )

                        print(
                            f"SSIM          : "
                            f"{result['SSIM']:.6f}"
                        )

                        print(
                            f"Predictive σ  : "
                            f"{result['predictive_std_mean']:.6e}"
                        )

                        print(
                            f"Runtime       : "
                            f"{result['runtime_seconds']:.4f} s"
                        )

                    except Exception as exc:

                        failed_cases += 1

                        print(
                            "Status        : FAILED"
                        )

                        print(
                            f"Error         : "
                            f"{exc}"
                        )

                        result = {

                            "method":
                                "proposed_physics_informed_3d",

                            "geological_mode":
                                geological_mode,

                            "mask_mode":
                                mask_mode,

                            "seed":
                                int(seed),

                            "requested_missing_rate":
                                float(
                                    missing_rate
                                ),

                            "measured_missing_rate":
                                np.nan,

                            "cube_depth":
                                int(
                                    CUBE_SIZE[0]
                                ),

                            "cube_height":
                                int(
                                    CUBE_SIZE[1]
                                ),

                            "cube_width":
                                int(
                                    CUBE_SIZE[2]
                                ),

                            "observed_samples":
                                0,

                            "missing_samples":
                                0,

                            "input_consistency_error":
                                np.nan,

                            "observed_preservation_error":
                                np.nan,

                            "runtime_seconds":
                                np.nan,

                            "missing_mae":
                                np.nan,

                            "missing_rmse":
                                np.nan,

                            "MAE":
                                np.nan,

                            "RMSE":
                                np.nan,

                            "PSNR":
                                np.nan,

                            "SNR":
                                np.nan,

                            "SSIM":
                                np.nan,

                            "aleatoric_variance_mean":
                                np.nan,

                            "epistemic_variance_mean":
                                np.nan,

                            "predictive_variance_mean":
                                np.nan,

                            "predictive_std_mean":
                                np.nan,

                            "missing_aleatoric_variance_mean":
                                np.nan,

                            "missing_epistemic_variance_mean":
                                np.nan,

                            "missing_predictive_variance_mean":
                                np.nan,

                            "missing_predictive_std_mean":
                                np.nan,

                            "status":
                                "FAILED",

                            "error":
                                str(exc)
                        }

                    records.append(
                        result
                    )

    # ================================================================
    # WRITE RAW RESULTS
    # ================================================================

    write_csv(
        RESULTS_FILE,
        records
    )

    # ================================================================
    # SUMMARY
    # ================================================================

    successful_records = [
        record
        for record in records
        if record["status"] == "PASS"
    ]

    summaries = calculate_summary(
        successful_records
    )

    write_csv(
        SUMMARY_FILE,
        summaries
    )

    # ================================================================
    # FINAL REPORT
    # ================================================================

    print()
    print("=" * 78)
    print(
        "PROPOSED MODEL CONTROLLED MATRIX COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Expected cases : {expected_cases}"
    )

    print(
        f"Completed cases: {completed_cases}"
    )

    print(
        f"Successful     : {successful_cases}"
    )

    print(
        f"Failed         : {failed_cases}"
    )

    print()
    print(
        "Raw results:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        "Summary results:"
    )

    print(
        SUMMARY_FILE
    )

    if (
        completed_cases == expected_cases
        and successful_cases == expected_cases
        and failed_cases == 0
    ):

        print()
        print(
            "OVERALL STATUS: PASS"
        )

        print(
            "All controlled proposed-model "
            "experiments completed successfully."
        )

    else:

        print()
        print(
            "OVERALL STATUS: FAIL"
        )

        print(
            "One or more controlled "
            "experiments failed."
        )


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()