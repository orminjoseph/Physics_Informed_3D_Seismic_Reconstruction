"""
======================================================================
COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Compare the six classical reconstruction baselines and the proposed
Physics-Informed 3D Encoder-Decoder model under identical input
conditions.

Methods
-------
1. Nearest Neighbor
2. Linear Interpolation
3. f-x Prediction
4. Compressive Sensing
5. Curvelet POCS
6. Dictionary Learning
7. Proposed Physics-Informed 3D Encoder-Decoder

Important
---------
This script is NOT the 750-case controlled experimental matrix.

The dedicated controlled-matrix scripts are responsible for the
large-scale 750-case experiments.

This script performs a common side-by-side comparison using exactly
the same:

    target
    corrupted input
    observation mask

for every reconstruction method.

Common metrics
--------------
    MAE
    RMSE
    PSNR
    SNR
    SSIM
    Missing-region MAE
    Missing-region RMSE
    Runtime
    Observed-data preservation error

Proposed-model-specific quantities
----------------------------------
    Aleatoric variance
    Epistemic variance
    Predictive variance
    Predictive standard deviation

Input convention
----------------
    (C, D, H, W)

Mask convention
---------------
    1 = observed
    0 = missing

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.baseline_nearest_neighbor import (
    nearest_neighbor_reconstruction,
)

from evaluation.baselines.baseline_linear_interpolation import (
    linear_interpolation_reconstruction,
)

from evaluation.baselines.fx_prediction import (
    fx_prediction_reconstruction,
)

from evaluation.baselines.compressive_sensing import (
    compressive_sensing_reconstruction,
)

from evaluation.baselines.curvelet_pocs import (
    curvelet_pocs_reconstruction,
)

from evaluation.baselines.dictionary_learning import (
    dictionary_learning_reconstruction,
)

from models.network import Network3D

from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.config import (
    MC_DROPOUT_SAMPLES,
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

# -------------------------------------------------------------
# Standard controlled benchmark cube
# -------------------------------------------------------------

CUBE_SIZE = (
    64,
    128,
    128,
)

# -------------------------------------------------------------
# Single common comparison case
#
# The same target, corrupted input and mask are passed to
# every reconstruction method.
# -------------------------------------------------------------

NUM_SAMPLES = 1

MISSING_RATE = 0.30

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"

SEED = 42


# =====================================================================
# PROPOSED MODEL CONFIGURATION
# =====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "checkpoints"
    / "best_model.pth"
)

MC_SAMPLES = MC_DROPOUT_SAMPLES


# =====================================================================
# OUTPUT CONFIGURATION
# =====================================================================

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_FILE = (
    REPORT_DIR
    / "baseline_comparison.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "baseline_comparison_summary.csv"
)


# =====================================================================
# NUMERICAL TOLERANCE
# =====================================================================

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# =====================================================================
# DEVICE
# =====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):
    """
    Set deterministic random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# METRIC CONVERSION
# =====================================================================

def metric_to_float(value):
    """
    Convert a metric result to a Python float.
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


# =====================================================================
# FINITE CHECK
# =====================================================================

def tensor_is_finite(tensor):
    """
    Return True when all tensor values are finite.
    """

    return bool(
        torch.isfinite(tensor)
        .all()
        .item()
    )


# =====================================================================
# VALIDATE RECONSTRUCTION
# =====================================================================

def validate_reconstruction(
    reconstruction,
    corrupted,
    mask,
):
    """
    Validate the reconstructed cube.

    Checks:
        1. Tensor type
        2. Shape
        3. Finite values
        4. Observed-data preservation
    """

    if not isinstance(
        reconstruction,
        torch.Tensor,
    ):

        raise TypeError(
            "Reconstruction must be a torch.Tensor."
        )

    if (
        reconstruction.shape
        != corrupted.shape
    ):

        raise RuntimeError(
            "Reconstruction shape mismatch. "
            f"Expected {tuple(corrupted.shape)}, "
            f"received {tuple(reconstruction.shape)}."
        )

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "Reconstruction contains NaN or Inf."
        )

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            -
            corrupted[mask == 1]
        )
    ).item()

    if (
        observed_difference
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum difference: "
            f"{observed_difference:.6e}"
        )

    return float(
        observed_difference
    )


# =====================================================================
# CALCULATE COMMON METRICS
# =====================================================================

def calculate_metrics(
    reconstruction,
    target,
    mask,
):
    """
    Calculate common reconstruction metrics.
    """

    # -------------------------------------------------------------
    # Missing-region selection
    # -------------------------------------------------------------

    missing_selector = (
        mask == 0
    )

    missing_prediction = (
        reconstruction[
            missing_selector
        ]
    )

    missing_target = (
        target[
            missing_selector
        ]
    )

    if missing_prediction.numel() == 0:

        raise RuntimeError(
            "No missing samples are available "
            "for missing-region metrics."
        )

    # -------------------------------------------------------------
    # Missing-region MAE
    # -------------------------------------------------------------

    missing_mae = float(
        torch.mean(
            torch.abs(
                missing_prediction
                -
                missing_target
            )
        )
        .item()
    )

    # -------------------------------------------------------------
    # Missing-region RMSE
    # -------------------------------------------------------------

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
        )
        .item()
    )

    # -------------------------------------------------------------
    # Add batch and channel dimensions
    # -------------------------------------------------------------

    reconstruction_batch = (
        reconstruction
        .unsqueeze(0)
        .to(DEVICE)
    )

    target_batch = (
        target
        .unsqueeze(0)
        .to(DEVICE)
    )

    # -------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------

    metric_mae = metric_to_float(
        mae(
            reconstruction_batch,
            target_batch,
        )
    )

    metric_rmse = metric_to_float(
        rmse(
            reconstruction_batch,
            target_batch,
        )
    )

    metric_psnr = metric_to_float(
        psnr(
            reconstruction_batch,
            target_batch,
        )
    )

    metric_snr = metric_to_float(
        snr(
            reconstruction_batch,
            target_batch,
        )
    )

    metric_ssim = metric_to_float(
        ssim(
            reconstruction_batch,
            target_batch,
        )
    )

    return {
        "missing_mae": missing_mae,
        "missing_rmse": missing_rmse,
        "MAE": metric_mae,
        "RMSE": metric_rmse,
        "PSNR": metric_psnr,
        "SNR": metric_snr,
        "SSIM": metric_ssim,
    }


# =====================================================================
# LOAD PROPOSED MODEL
# =====================================================================

def load_proposed_model():
    """
    Load the frozen proposed-model checkpoint.
    """

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "\nProposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}\n"
        )

    print()
    print("=" * 70)
    print("LOADING PROPOSED MODEL")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Device:")
    print(DEVICE)

    # -------------------------------------------------------------
    # Create production architecture
    # -------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    )

    model = model.to(DEVICE)

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print(
        "Proposed model loaded successfully."
    )

    if "best_epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint["best_epoch"],
        )

    return model


# =====================================================================
# RUN CLASSICAL BASELINE
# =====================================================================

def run_classical_method(
    method_name,
    reconstruction_function,
    corrupted,
    mask,
    target,
):
    """
    Run one classical reconstruction baseline.
    """

    print()
    print("-" * 70)
    print(method_name)
    print("-" * 70)

    start_time = time.perf_counter()

    reconstruction = (
        reconstruction_function(
            corrupted,
            mask,
        )
    )

    end_time = time.perf_counter()

    runtime_seconds = (
        end_time
        -
        start_time
    )

    # -------------------------------------------------------------
    # Validate reconstruction
    # -------------------------------------------------------------

    observed_difference = (
        validate_reconstruction(
            reconstruction,
            corrupted,
            mask,
        )
    )

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    result = {

        "method":
            method_name,

        "geological_mode":
            GEOLOGICAL_MODE,

        "mask_mode":
            MASK_MODE,

        "seed":
            int(SEED),

        "requested_missing_rate":
            float(MISSING_RATE),

        "cube_depth":
            int(CUBE_SIZE[0]),

        "cube_height":
            int(CUBE_SIZE[1]),

        "cube_width":
            int(CUBE_SIZE[2]),

        "runtime_seconds":
            float(runtime_seconds),

        "observed_preservation_error":
            float(observed_difference),

        "missing_mae":
            metrics["missing_mae"],

        "missing_rmse":
            metrics["missing_rmse"],

        "MAE":
            metrics["MAE"],

        "RMSE":
            metrics["RMSE"],

        "PSNR":
            metrics["PSNR"],

        "SNR":
            metrics["SNR"],

        "SSIM":
            metrics["SSIM"],

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
            "PASS",

        "error":
            "",
    }

    print(
        f"MAE          : {result['MAE']:.6f}"
    )

    print(
        f"RMSE         : {result['RMSE']:.6f}"
    )

    print(
        f"PSNR         : {result['PSNR']:.6f} dB"
    )

    print(
        f"SNR          : {result['SNR']:.6f} dB"
    )

    print(
        f"SSIM         : {result['SSIM']:.6f}"
    )

    print(
        f"Missing MAE  : "
        f"{result['missing_mae']:.6f}"
    )

    print(
        f"Runtime      : "
        f"{result['runtime_seconds']:.4f} s"
    )

    print(
        f"Observed err : "
        f"{result['observed_preservation_error']:.6e}"
    )

    return result


# =====================================================================
# RUN PROPOSED MODEL
# =====================================================================

def run_proposed_model(
    model,
    corrupted,
    mask,
    target,
):
    """
    Run the proposed Physics-Informed 3D model.
    """

    print()
    print("-" * 70)
    print(
        "Proposed Physics-Informed 3D Model"
    )
    print("-" * 70)

    # -------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------

    input_batch = (
        corrupted
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # -------------------------------------------------------------
    # MC Dropout predictor
    # -------------------------------------------------------------

    predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Required MC outputs
    # -------------------------------------------------------------

    required_keys = {
        "reconstruction_samples",
        "log_variance_samples",
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

    # -------------------------------------------------------------
    # Reconstruction mean
    # -------------------------------------------------------------

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

    # -------------------------------------------------------------
    # Uncertainty decomposition
    # -------------------------------------------------------------

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
            epistemic_variance,
        )
    )

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0,
        )
    )

    # -------------------------------------------------------------
    # Validate uncertainty tensors
    # -------------------------------------------------------------

    uncertainty_tensors = {
        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,

        "predictive_std":
            predictive_std,
    }

    for name, tensor in (
        uncertainty_tensors.items()
    ):

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

        if (
            name.endswith("variance")
            and (tensor < 0).any()
        ):

            raise RuntimeError(
                f"Negative {name} detected."
            )

    # -------------------------------------------------------------
    # Data-consistency projection
    # -------------------------------------------------------------

    reconstruction = (
        reconstruction_mean
        *
        (
            1.0
            -
            mask.unsqueeze(0)
        )
        +
        corrupted.unsqueeze(0)
        *
        mask.unsqueeze(0)
    )

    # Remove the batch dimension.
    reconstruction = (
        reconstruction.squeeze(0)
    )

    # -------------------------------------------------------------
    # Validate reconstruction
    # -------------------------------------------------------------

    observed_difference = (
        validate_reconstruction(
            reconstruction,
            corrupted,
            mask,
        )
    )

    # -------------------------------------------------------------
    # Common metrics
    # -------------------------------------------------------------

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    # -------------------------------------------------------------
    # Missing-region uncertainty
    # -------------------------------------------------------------

    missing_selector = (
        mask.unsqueeze(0)
        ==
        0
    )

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

    # -------------------------------------------------------------
    # Overall uncertainty
    # -------------------------------------------------------------

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

    result = {

        "method":
            "proposed_physics_informed_3d",

        "geological_mode":
            GEOLOGICAL_MODE,

        "mask_mode":
            MASK_MODE,

        "seed":
            int(SEED),

        "requested_missing_rate":
            float(MISSING_RATE),

        "cube_depth":
            int(CUBE_SIZE[0]),

        "cube_height":
            int(CUBE_SIZE[1]),

        "cube_width":
            int(CUBE_SIZE[2]),

        "runtime_seconds":
            float(runtime_seconds),

        "observed_preservation_error":
            float(observed_difference),

        "missing_mae":
            metrics["missing_mae"],

        "missing_rmse":
            metrics["missing_rmse"],

        "MAE":
            metrics["MAE"],

        "RMSE":
            metrics["RMSE"],

        "PSNR":
            metrics["PSNR"],

        "SNR":
            metrics["SNR"],

        "SSIM":
            metrics["SSIM"],

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
            "",
    }

    print(
        f"MAE          : {result['MAE']:.6f}"
    )

    print(
        f"RMSE         : {result['RMSE']:.6f}"
    )

    print(
        f"PSNR         : {result['PSNR']:.6f} dB"
    )

    print(
        f"SNR          : {result['SNR']:.6f} dB"
    )

    print(
        f"SSIM         : {result['SSIM']:.6f}"
    )

    print(
        f"Missing MAE  : "
        f"{result['missing_mae']:.6f}"
    )

    print(
        f"Runtime      : "
        f"{result['runtime_seconds']:.4f} s"
    )

    print(
        f"Predictive σ : "
        f"{result['predictive_std_mean']:.6e}"
    )

    return result


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
# SUMMARY
# =====================================================================

def calculate_summary(
    records,
):
    """
    Calculate summary statistics by method.

    Since this comparison script uses one common test case per
    method, the standard deviation is zero for the current run.

    The dedicated 750-case matrices remain the source of
    multi-case statistical summaries.
    """

    summaries = []

    for record in records:

        summary = {
            "method":
                record["method"],

            "n_cases":
                1,

            "MAE_mean":
                record["MAE"],

            "RMSE_mean":
                record["RMSE"],

            "PSNR_mean":
                record["PSNR"],

            "SNR_mean":
                record["SNR"],

            "SSIM_mean":
                record["SSIM"],

            "missing_mae_mean":
                record["missing_mae"],

            "missing_rmse_mean":
                record["missing_rmse"],

            "runtime_seconds_mean":
                record["runtime_seconds"],

            "observed_preservation_error_max":
                record[
                    "observed_preservation_error"
                ],

            "aleatoric_variance_mean":
                record[
                    "aleatoric_variance_mean"
                ],

            "epistemic_variance_mean":
                record[
                    "epistemic_variance_mean"
                ],

            "predictive_variance_mean":
                record[
                    "predictive_variance_mean"
                ],

            "predictive_std_mean":
                record[
                    "predictive_std_mean"
                ],

            "missing_aleatoric_variance_mean":
                record[
                    "missing_aleatoric_variance_mean"
                ],

            "missing_epistemic_variance_mean":
                record[
                    "missing_epistemic_variance_mean"
                ],

            "missing_predictive_variance_mean":
                record[
                    "missing_predictive_variance_mean"
                ],

            "missing_predictive_std_mean":
                record[
                    "missing_predictive_std_mean"
                ],

            "status":
                record["status"],
        }

        summaries.append(
            summary
        )

    return summaries


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 78)
    print(
        "COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON"
    )
    print("=" * 78)

    print()
    print(
        f"Cube size       : {CUBE_SIZE}"
    )

    print(
        f"Geology         : {GEOLOGICAL_MODE}"
    )

    print(
        f"Mask            : {MASK_MODE}"
    )

    print(
        f"Missing rate    : {MISSING_RATE:.0%}"
    )

    print(
        f"Seed            : {SEED}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"MC samples      : {MC_SAMPLES}"
    )

    # ================================================================
    # REPRODUCIBILITY
    # ================================================================

    set_seed(SEED)

    # ================================================================
    # CREATE ONE COMMON DATA SAMPLE
    # ================================================================

    print()
    print("=" * 78)
    print(
        "CREATING COMMON TEST SAMPLE"
    )
    print("=" * 78)

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
        seed=SEED,
    )

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    # ================================================================
    # DATASET VALIDATION
    # ================================================================

    if actual_geological_mode != GEOLOGICAL_MODE:

        raise RuntimeError(
            "Geological mode mismatch. "
            f"Expected {GEOLOGICAL_MODE}, "
            f"received {actual_geological_mode}."
        )

    if actual_mask_mode != MASK_MODE:

        raise RuntimeError(
            "Mask mode mismatch. "
            f"Expected {MASK_MODE}, "
            f"received {actual_mask_mode}."
        )

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    if tuple(corrupted.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected corrupted shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(corrupted.shape)}."
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
        "velocity": velocity,
    }.items():

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    # ================================================================
    # MASK VALIDATION
    # ================================================================

    unique_mask = torch.unique(mask)

    if not torch.all(
        (unique_mask == 0)
        |
        (unique_mask == 1)
    ):

        raise RuntimeError(
            "Mask must contain only 0 and 1."
        )

    # ================================================================
    # INPUT CONSISTENCY
    # ================================================================

    expected_input = (
        target
        *
        mask
    )

    input_difference = torch.max(
        torch.abs(
            corrupted
            -
            expected_input
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

    print()
    print(
        f"Observed samples : {observed_samples}"
    )

    print(
        f"Missing samples  : {missing_samples}"
    )

    print(
        f"Measured missing : "
        f"{measured_missing_rate:.4f}"
    )

    # ================================================================
    # LOAD PROPOSED MODEL
    # ================================================================

    proposed_model = (
        load_proposed_model()
    )

    # ================================================================
    # RESULT STORAGE
    # ================================================================

    records = []

    # ================================================================
    # CLASSICAL METHODS
    # ================================================================

    classical_methods = [

        (
            "nearest_neighbor",
            nearest_neighbor_reconstruction,
        ),

        (
            "linear_interpolation",
            linear_interpolation_reconstruction,
        ),

        (
            "fx_prediction",
            fx_prediction_reconstruction,
        ),

        (
            "compressive_sensing",
            compressive_sensing_reconstruction,
        ),

        (
            "curvelet_pocs",
            curvelet_pocs_reconstruction,
        ),

        (
            "dictionary_learning",
            dictionary_learning_reconstruction,
        ),
    ]

    # ================================================================
    # RUN CLASSICAL BASELINES
    # ================================================================

    for (
        method_name,
        reconstruction_function,
    ) in classical_methods:

        try:

            result = run_classical_method(
                method_name=method_name,
                reconstruction_function=
                    reconstruction_function,
                corrupted=corrupted,
                mask=mask,
                target=target,
            )

        except Exception as exc:

            print()
            print(
                f"{method_name} FAILED:"
            )

            print(exc)

            result = {

                "method":
                    method_name,

                "geological_mode":
                    GEOLOGICAL_MODE,

                "mask_mode":
                    MASK_MODE,

                "seed":
                    int(SEED),

                "requested_missing_rate":
                    float(MISSING_RATE),

                "cube_depth":
                    int(CUBE_SIZE[0]),

                "cube_height":
                    int(CUBE_SIZE[1]),

                "cube_width":
                    int(CUBE_SIZE[2]),

                "runtime_seconds":
                    np.nan,

                "observed_preservation_error":
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
                    str(exc),
            }

        records.append(
            result
        )

    # ================================================================
    # RUN PROPOSED MODEL
    # ================================================================

    try:

        result = run_proposed_model(
            model=proposed_model,
            corrupted=corrupted,
            mask=mask,
            target=target,
        )

    except Exception as exc:

        print()
        print(
            "proposed_physics_informed_3d FAILED:"
        )

        print(exc)

        result = {

            "method":
                "proposed_physics_informed_3d",

            "geological_mode":
                GEOLOGICAL_MODE,

            "mask_mode":
                MASK_MODE,

            "seed":
                int(SEED),

            "requested_missing_rate":
                float(MISSING_RATE),

            "cube_depth":
                int(CUBE_SIZE[0]),

            "cube_height":
                int(CUBE_SIZE[1]),

            "cube_width":
                int(CUBE_SIZE[2]),

            "runtime_seconds":
                np.nan,

            "observed_preservation_error":
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
                str(exc),
        }

    records.append(
        result
    )

    # ================================================================
    # WRITE RAW RESULTS
    # ================================================================

    write_csv(
        RESULTS_FILE,
        records,
    )

    # ================================================================
    # WRITE SUMMARY
    # ================================================================

    summaries = calculate_summary(
        records
    )

    write_csv(
        SUMMARY_FILE,
        summaries,
    )

    # ================================================================
    # FINAL REPORT
    # ================================================================

    successful = sum(
        record["status"] == "PASS"
        for record in records
    )

    failed = sum(
        record["status"] == "FAILED"
        for record in records
    )

    print()
    print("=" * 78)
    print(
        "SEVEN-METHOD COMPARISON COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Methods evaluated : {len(records)}"
    )

    print(
        f"Successful        : {successful}"
    )

    print(
        f"Failed            : {failed}"
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

    # ================================================================
    # COMPARISON TABLE IN CONSOLE
    # ================================================================

    print()
    print("=" * 78)
    print(
        "COMMON RECONSTRUCTION METRICS"
    )
    print("=" * 78)

    print()

    print(
        f"{'Method':<32}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'PSNR':>10}"
        f"{'SNR':>10}"
        f"{'SSIM':>10}"
    )

    print(
        "-" * 82
    )

    for record in records:

        if record["status"] != "PASS":
            continue

        print(
            f"{record['method']:<32}"
            f"{record['MAE']:>10.6f}"
            f"{record['RMSE']:>10.6f}"
            f"{record['PSNR']:>10.4f}"
            f"{record['SNR']:>10.4f}"
            f"{record['SSIM']:>10.6f}"
        )

    print()

    if failed == 0:

        print(
            "OVERALL STATUS: PASS"
        )

        print(
            "All seven reconstruction methods "
            "completed successfully on the same "
            "input cube and observation mask."
        )

    else:

        print(
            "OVERALL STATUS: FAIL"
        )

        print(
            "One or more reconstruction methods failed. "
            "Inspect the CSV error column."
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()