"""
=====================================================================
FINAL PhD NOISE ROBUSTNESS EVALUATION
=====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

Purpose
-------
Evaluate the robustness of the trained reconstruction model under
increasing levels of additive Gaussian noise.

Experimental design
-------------------

    Configured Dataset
            |
            v
    Existing Corrupted Input
            |
            v
    Add Gaussian Noise to OBSERVED Voxels Only
            |
            v
    Physics-Informed 3D Encoder–Decoder
            |
            v
    Reconstruction
            |
            v
    Quantitative Evaluation

Noise is deliberately NOT added to missing voxels.

This ensures that the experiment evaluates robustness to measurement
noise while preserving the original missing-data pattern.

Metrics
-------

Global:
    MAE
    RMSE
    PSNR
    SNR
    SSIM

Missing region:
    Missing MAE
    Missing RMSE

Observed-data consistency:
    Observed MAE
    Observed RMSE
    Observed Preservation Error

Uncertainty:
    Mean Aleatoric Standard Deviation
    Mean Aleatoric Variance

Important
---------
The current production Predictor provides reconstruction and
aleatoric uncertainty derived from the model's log-variance output.

Epistemic uncertainty is NOT reported by this module because
epistemic uncertainty requires the dedicated stochastic
MC-Dropout evaluation pathway.

Therefore this module does not incorrectly treat deterministic
aleatoric uncertainty as epistemic uncertainty.

Output
------

    <REPORT_DIR>/noise_robustness/

        noise_robustness_detailed.csv
        noise_robustness_summary.csv
        noise_robustness_metadata.json

Current experiment
------------------

With:

    DATASET_MODE = "synthetic"
    EXPERIMENT_NAME = "synthetic_training"

results are written under:

    outputs/synthetic_training/reports/noise_robustness/

=====================================================================
"""

from pathlib import Path
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from dataset.build_dataset import build_dataset

from inference.predictor import Predictor

from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,

    CHECKPOINT_DIR,
    REPORT_DIR,

    DEVICE as CONFIG_DEVICE,

    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,

    MASK_OBSERVED_VALUE,
    MASK_MISSING_VALUE,

    SEISMIC_DATA_RANGE,

    NOISE_ROBUSTNESS_LEVELS,
    NOISE_ROBUSTNESS_NUM_SAMPLES,
    NOISE_ROBUSTNESS_SEED,

    OBSERVED_PRESERVATION_TOLERANCE,
)


# =====================================================================
# PATHS
# =====================================================================

CHECKPOINT_FILE = (
    Path(CHECKPOINT_DIR)
    / "best_model.pth"
)

OUTPUT_DIRECTORY = (
    Path(REPORT_DIR)
    / "noise_robustness"
)

DETAILED_CSV_FILE = (
    OUTPUT_DIRECTORY
    / "noise_robustness_detailed.csv"
)

SUMMARY_CSV_FILE = (
    OUTPUT_DIRECTORY
    / "noise_robustness_summary.csv"
)

METADATA_FILE = (
    OUTPUT_DIRECTORY
    / "noise_robustness_metadata.json"
)


# =====================================================================
# DEVICE
# =====================================================================

def resolve_device():
    """
    Resolve the device according to utils.config.

    DEVICE semantics
    ----------------
    auto
        Use CUDA when available, otherwise CPU.

    cuda
        Require CUDA.

    cpu
        Force CPU.
    """

    if CONFIG_DEVICE == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if CONFIG_DEVICE == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "\nDEVICE='cuda' is configured, "
                "but CUDA is not available."
            )

        return torch.device("cuda")

    return torch.device("cpu")


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_reproducibility_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# TENSOR VALIDATION
# =====================================================================

def validate_tensor(
        tensor,
        name
):
    """
    Validate a tensor before numerical processing.
    """

    if not isinstance(
        tensor,
        torch.Tensor
    ):

        raise TypeError(
            f"{name} must be a torch.Tensor. "
            f"Received: {type(tensor)}"
        )

    if tensor.numel() == 0:

        raise ValueError(
            f"{name} is empty."
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# =====================================================================
# BATCH PREPARATION
# =====================================================================

def prepare_batch(
        tensor,
        name
):
    """
    Convert a single sample into model batch format.

    Individual sample:
        [C, D, H, W]

    Model batch:
        [B, C, D, H, W]
    """

    validate_tensor(
        tensor,
        name
    )

    if tensor.ndim == 4:

        return tensor.unsqueeze(0)

    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        f"{name} must have either four or five dimensions. "
        f"Received shape: {tuple(tensor.shape)}"
    )


# =====================================================================
# DATASET SAMPLE EXTRACTION
# =====================================================================

def extract_dataset_sample(
        sample,
        sample_index
):
    """
    Extract input, target and mask from the current dataset convention.

    Current project convention:

        sample[0] -> input/corrupted cube
        sample[1] -> target cube
        sample[2] -> observation mask
        sample[3] -> velocity
        additional fields may follow
    """

    if not isinstance(
        sample,
        (tuple, list)
    ):

        raise TypeError(
            f"Dataset sample {sample_index} must be a tuple or list."
        )

    if len(sample) < 3:

        raise ValueError(
            f"Dataset sample {sample_index} does not contain "
            "input, target and mask."
        )

    input_cube = prepare_batch(
        sample[0],
        f"input[{sample_index}]"
    )

    target_cube = prepare_batch(
        sample[1],
        f"target[{sample_index}]"
    )

    mask = prepare_batch(
        sample[2],
        f"mask[{sample_index}]"
    )

    if input_cube.shape != target_cube.shape:

        raise ValueError(
            f"\nSample {sample_index}: input and target shapes differ.\n"
            f"Input : {tuple(input_cube.shape)}\n"
            f"Target: {tuple(target_cube.shape)}"
        )

    if input_cube.shape != mask.shape:

        raise ValueError(
            f"\nSample {sample_index}: input and mask shapes differ.\n"
            f"Input: {tuple(input_cube.shape)}\n"
            f"Mask : {tuple(mask.shape)}"
        )

    return (
        input_cube,
        target_cube,
        mask,
    )


# =====================================================================
# MASK VALIDATION
# =====================================================================

def validate_mask(mask):
    """
    Validate that the mask contains the configured observed/missing
    values.
    """

    unique_values = torch.unique(mask)

    valid_values = {
        float(MASK_OBSERVED_VALUE),
        float(MASK_MISSING_VALUE),
    }

    for value in unique_values.tolist():

        if float(value) not in valid_values:

            raise ValueError(
                "\nMask contains an unsupported value.\n"
                f"Value: {value}\n"
                f"Expected: {valid_values}"
            )


# =====================================================================
# ADD GAUSSIAN NOISE
# =====================================================================

def add_gaussian_noise(
        cube,
        mask,
        noise_std
):
    """
    Add Gaussian noise only to observed voxels.

    Missing voxels remain exactly unchanged.

    Parameters
    ----------
    cube : torch.Tensor
        Corrupted seismic cube.

    mask : torch.Tensor
        Observation mask.

    noise_std : float
        Standard deviation of Gaussian noise.
    """

    validate_tensor(
        cube,
        "cube"
    )

    validate_tensor(
        mask,
        "mask"
    )

    validate_mask(
        mask
    )

    if cube.shape != mask.shape:

        raise ValueError(
            "Cube and mask must have identical shapes."
        )

    noise_std = float(
        noise_std
    )

    if not np.isfinite(
        noise_std
    ):

        raise ValueError(
            "noise_std must be finite."
        )

    if noise_std < 0.0:

        raise ValueError(
            "noise_std cannot be negative."
        )

    if noise_std == 0.0:

        return cube.clone()

    # Generate Gaussian noise with the same shape and dtype
    # as the seismic input.
    noise = (
        torch.randn_like(cube)
        * noise_std
    )

    observed_mask = (
        mask == MASK_OBSERVED_VALUE
    )

    noisy_cube = torch.where(
        observed_mask,
        cube + noise,
        cube,
    )

    return noisy_cube


# =====================================================================
# NOISE STATISTICS
# =====================================================================

def calculate_noise_statistics(
        original,
        noisy,
        mask
):
    """
    Measure the actual noise applied to observed voxels.

    Returns
    -------
    dict
        Observed voxel count and measured noise standard deviation.
    """

    observed_mask = (
        mask == MASK_OBSERVED_VALUE
    )

    observed_count = int(
        observed_mask.sum().item()
    )

    if observed_count == 0:

        return {
            "Observed_Voxels": 0,
            "Actual_Noise_STD": 0.0,
        }

    difference = (
        noisy -
        original
    )

    observed_noise = difference[
        observed_mask
    ]

    actual_std = observed_noise.std(
        unbiased=False
    ).item()

    return {
        "Observed_Voxels":
            observed_count,

        "Actual_Noise_STD":
            float(actual_std),
    }


# =====================================================================
# METRIC HELPER
# =====================================================================

def scalar_metric(
        value,
        metric_name
):
    """
    Convert a scalar metric tensor to a validated Python float.
    """

    if isinstance(
        value,
        torch.Tensor
    ):

        value = value.detach().cpu().item()

    value = float(value)

    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"Metric '{metric_name}' produced "
            f"a non-finite value: {value}"
        )

    return value


# =====================================================================
# RECONSTRUCTION METRICS
# =====================================================================

def calculate_global_metrics(
        prediction,
        target
):
    """
    Calculate global reconstruction metrics.
    """

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target shapes must match."
        )

    return {

        "MAE":
            scalar_metric(
                mae(
                    prediction,
                    target
                ),
                "MAE",
            ),

        "RMSE":
            scalar_metric(
                rmse(
                    prediction,
                    target
                ),
                "RMSE",
            ),

        "PSNR":
            scalar_metric(
                psnr(
                    prediction,
                    target
                ),
                "PSNR",
            ),

        "SNR":
            scalar_metric(
                snr(
                    prediction,
                    target
                ),
                "SNR",
            ),

        "SSIM":
            scalar_metric(
                ssim(
                    prediction,
                    target
                ),
                "SSIM",
            ),
    }


# =====================================================================
# REGION METRICS
# =====================================================================

def calculate_region_metrics(
        prediction,
        target,
        mask,
        region_value,
        region_name
):
    """
    Calculate MAE and RMSE within a selected mask region.
    """

    region = (
        mask == region_value
    )

    voxel_count = int(
        region.sum().item()
    )

    if voxel_count == 0:

        return {
            f"{region_name}_Voxels":
                0,

            f"{region_name}_MAE":
                np.nan,

            f"{region_name}_RMSE":
                np.nan,
        }

    prediction_region = prediction[
        region
    ]

    target_region = target[
        region
    ]

    error = (
        prediction_region -
        target_region
    )

    region_mae = (
        torch.abs(error)
        .mean()
        .item()
    )

    region_rmse = (
        torch.sqrt(
            torch.mean(
                error ** 2
            )
        )
        .item()
    )

    return {
        f"{region_name}_Voxels":
            voxel_count,

        f"{region_name}_MAE":
            float(region_mae),

        f"{region_name}_RMSE":
            float(region_rmse),
    }


# =====================================================================
# OBSERVED DATA PRESERVATION
# =====================================================================

def calculate_observed_preservation_error(
        reconstruction,
        input_cube,
        mask
):
    """
    Calculate how much the reconstructed result differs from the
    noisy observed input in observed voxels.

    This verifies the data-consistency behaviour of the reconstruction.
    """

    observed = (
        mask == MASK_OBSERVED_VALUE
    )

    if not observed.any():

        return 0.0

    difference = torch.abs(
        reconstruction -
        input_cube
    )

    preservation_error = difference[
        observed
    ].max().item()

    if not np.isfinite(
        preservation_error
    ):

        raise ValueError(
            "Observed-data preservation error is non-finite."
        )

    return float(
        preservation_error
    )


# =====================================================================
# PREDICTOR INITIALIZATION
# =====================================================================

def initialize_predictor(
        device
):
    """
    Construct Network3D using the centralized architecture settings
    and load best_model.pth.
    """

    if not CHECKPOINT_FILE.is_file():

        raise FileNotFoundError(
            "\nBest model checkpoint was not found:\n"
            f"{CHECKPOINT_FILE}\n\n"
            "Complete training before running noise robustness."
        )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    predictor = Predictor(
        model=model,
        checkpoint=str(
            CHECKPOINT_FILE
        ),
        device=device,
    )

    return predictor


# =====================================================================
# PREDICTOR OUTPUT EXTRACTION
# =====================================================================

def extract_predictor_outputs(
        prediction
):
    """
    Extract reconstruction and aleatoric uncertainty from the
    current production Predictor interface.

    Current production convention:

        reconstruction
        travel_time
        log_variance
        aleatoric_std

    The gallery/robustness module only requires:

        reconstruction
        aleatoric_std
    """

    if not isinstance(
        prediction,
        (tuple, list)
    ):

        raise TypeError(
            "\nPredictor.predict() returned an unsupported object.\n"
            f"Received: {type(prediction)}"
        )

    if len(prediction) < 4:

        raise ValueError(
            "\nPredictor.predict() returned fewer than four outputs.\n"
            f"Number of outputs: {len(prediction)}\n\n"
            "Expected production convention:\n"
            "reconstruction, travel_time, log_variance, aleatoric_std"
        )

    reconstruction = prediction[0]

    aleatoric_std = prediction[3]

    return (
        reconstruction,
        aleatoric_std,
    )


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("NOISE ROBUSTNESS EVALUATION")
    print("=" * 70)

    print()
    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Dataset mode:",
        DATASET_MODE
    )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    set_reproducibility_seed(
        NOISE_ROBUSTNESS_SEED
    )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = resolve_device()

    print(
        "Configured device:",
        CONFIG_DEVICE
    )

    print(
        "Using device:",
        device
    )

    # -----------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_FILE
    )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    print()
    print(
        "Noise levels:",
        NOISE_ROBUSTNESS_LEVELS
    )

    print(
        "Maximum samples:",
        NOISE_ROBUSTNESS_NUM_SAMPLES
    )

    print(
        "Random seed:",
        NOISE_ROBUSTNESS_SEED
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    print()
    print("Loading configured dataset...")

    dataset = build_dataset()

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "The configured dataset is empty."
        )

    if NOISE_ROBUSTNESS_NUM_SAMPLES is None:

        num_test_samples = len(dataset)

    else:

        if NOISE_ROBUSTNESS_NUM_SAMPLES <= 0:

            raise ValueError(
                "NOISE_ROBUSTNESS_NUM_SAMPLES must be "
                "positive or None."
            )

        num_test_samples = min(
            int(
                NOISE_ROBUSTNESS_NUM_SAMPLES
            ),
            len(dataset),
        )

    print(
        "Dataset size:",
        len(dataset)
    )

    print(
        "Samples evaluated:",
        num_test_samples
    )

    # -----------------------------------------------------------------
    # Predictor
    # -----------------------------------------------------------------

    print()
    print("Loading trained model...")

    predictor = initialize_predictor(
        device
    )

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    detailed_results = []

    # =================================================================
    # NOISE LEVEL LOOP
    # =================================================================

    for noise_std in NOISE_ROBUSTNESS_LEVELS:

        noise_std = float(
            noise_std
        )

        print()
        print("-" * 70)

        print(
            f"Testing Gaussian noise σ = "
            f"{noise_std:.4f}"
        )

        print("-" * 70)

        # -------------------------------------------------------------
        # Evaluate each sample
        # -------------------------------------------------------------

        for sample_index in range(
            num_test_samples
        ):

            print(
                f"Sample "
                f"{sample_index + 1}/"
                f"{num_test_samples}"
            )

            # ---------------------------------------------------------
            # Dataset sample
            # ---------------------------------------------------------

            sample = dataset[
                sample_index
            ]

            (
                corrupted,
                target,
                mask,
            ) = extract_dataset_sample(
                sample,
                sample_index
            )

            # ---------------------------------------------------------
            # Add noise only to observed voxels
            # ---------------------------------------------------------

            noisy_input = add_gaussian_noise(
                corrupted,
                mask,
                noise_std
            )

            # ---------------------------------------------------------
            # Verify noise
            # ---------------------------------------------------------

            noise_statistics = (
                calculate_noise_statistics(
                    corrupted,
                    noisy_input,
                    mask,
                )
            )

            # ---------------------------------------------------------
            # Inference
            # ---------------------------------------------------------

            with torch.no_grad():

                prediction = predictor.predict(
                    noisy_input
                )

            (
                reconstruction,
                aleatoric_std,
            ) = extract_predictor_outputs(
                prediction
            )

            # ---------------------------------------------------------
            # Validate prediction outputs
            # ---------------------------------------------------------

            validate_tensor(
                reconstruction,
                "reconstruction"
            )

            validate_tensor(
                aleatoric_std,
                "aleatoric_std"
            )

            # ---------------------------------------------------------
            # Shape normalization
            # ---------------------------------------------------------

            reconstruction = prepare_batch(
                reconstruction,
                "reconstruction"
            )

            aleatoric_std = prepare_batch(
                aleatoric_std,
                "aleatoric_std"
            )

            if reconstruction.shape != target.shape:

                raise ValueError(
                    f"\nSample {sample_index}: reconstruction "
                    "shape differs from target.\n"
                    f"Reconstruction: "
                    f"{tuple(reconstruction.shape)}\n"
                    f"Target: "
                    f"{tuple(target.shape)}"
                )

            if aleatoric_std.shape != target.shape:

                raise ValueError(
                    f"\nSample {sample_index}: aleatoric uncertainty "
                    "shape differs from target.\n"
                    f"Aleatoric STD: "
                    f"{tuple(aleatoric_std.shape)}\n"
                    f"Target: "
                    f"{tuple(target.shape)}"
                )

            # ---------------------------------------------------------
            # Global metrics
            # ---------------------------------------------------------

            global_metrics = calculate_global_metrics(
                reconstruction,
                target
            )

            # ---------------------------------------------------------
            # Missing-region metrics
            # ---------------------------------------------------------

            missing_metrics = calculate_region_metrics(
                reconstruction,
                target,
                mask,
                MASK_MISSING_VALUE,
                "Missing"
            )

            # ---------------------------------------------------------
            # Observed-region metrics
            # ---------------------------------------------------------

            observed_metrics = calculate_region_metrics(
                reconstruction,
                target,
                mask,
                MASK_OBSERVED_VALUE,
                "Observed"
            )

            # ---------------------------------------------------------
            # Data preservation
            # ---------------------------------------------------------

            preservation_error = (
                calculate_observed_preservation_error(
                    reconstruction,
                    noisy_input,
                    mask
                )
            )

            # ---------------------------------------------------------
            # Uncertainty
            # ---------------------------------------------------------

            aleatoric_variance = (
                aleatoric_std ** 2
            )

            mean_aleatoric_std = (
                aleatoric_std
                .mean()
                .item()
            )

            mean_aleatoric_variance = (
                aleatoric_variance
                .mean()
                .item()
            )

            # ---------------------------------------------------------
            # Result record
            # ---------------------------------------------------------

            record = {

                "Experiment":
                    EXPERIMENT_NAME,

                "Dataset_Mode":
                    DATASET_MODE,

                "Sample_ID":
                    sample_index,

                "Requested_Noise_STD":
                    noise_std,

                "Actual_Noise_STD":
                    noise_statistics[
                        "Actual_Noise_STD"
                    ],

                "Observed_Voxels":
                    noise_statistics[
                        "Observed_Voxels"
                    ],

                "MAE":
                    global_metrics[
                        "MAE"
                    ],

                "RMSE":
                    global_metrics[
                        "RMSE"
                    ],

                "PSNR":
                    global_metrics[
                        "PSNR"
                    ],

                "SNR":
                    global_metrics[
                        "SNR"
                    ],

                "SSIM":
                    global_metrics[
                        "SSIM"
                    ],

                "Missing_Voxels":
                    missing_metrics[
                        "Missing_Voxels"
                    ],

                "Missing_MAE":
                    missing_metrics[
                        "Missing_MAE"
                    ],

                "Missing_RMSE":
                    missing_metrics[
                        "Missing_RMSE"
                    ],

                "Observed_MAE":
                    observed_metrics[
                        "Observed_MAE"
                    ],

                "Observed_RMSE":
                    observed_metrics[
                        "Observed_RMSE"
                    ],

                "Observed_Preservation_Error":
                    preservation_error,

                "Aleatoric_STD_Mean":
                    float(
                        mean_aleatoric_std
                    ),

                "Aleatoric_Variance_Mean":
                    float(
                        mean_aleatoric_variance
                    ),
            }

            detailed_results.append(
                record
            )

            # ---------------------------------------------------------
            # Console output
            # ---------------------------------------------------------

            print(
                f"  MAE={record['MAE']:.6f} | "
                f"RMSE={record['RMSE']:.6f} | "
                f"PSNR={record['PSNR']:.4f} | "
                f"SNR={record['SNR']:.4f} | "
                f"SSIM={record['SSIM']:.6f}"
            )

    # =================================================================
    # CREATE DETAILED DATAFRAME
    # =================================================================

    detailed_df = pd.DataFrame(
        detailed_results
    )

    if detailed_df.empty:

        raise RuntimeError(
            "Noise robustness produced no evaluation results."
        )

    # -----------------------------------------------------------------
    # Validate numerical columns
    # -----------------------------------------------------------------

    numerical_columns = [
        column
        for column in detailed_df.columns
        if column not in {
            "Experiment",
            "Dataset_Mode",
        }
    ]

    for column in numerical_columns:

        if not np.isfinite(
            detailed_df[column]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        ).all():

            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )

    # =================================================================
    # SUMMARY BY NOISE LEVEL
    # =================================================================

    summary_df = (
        detailed_df
        .groupby(
            "Requested_Noise_STD",
            as_index=False
        )
        .agg({

            "Actual_Noise_STD":
                "mean",

            "Observed_Voxels":
                "mean",

            "MAE":
                ["mean", "std"],

            "RMSE":
                ["mean", "std"],

            "PSNR":
                ["mean", "std"],

            "SNR":
                ["mean", "std"],

            "SSIM":
                ["mean", "std"],

            "Missing_MAE":
                ["mean", "std"],

            "Missing_RMSE":
                ["mean", "std"],

            "Observed_MAE":
                ["mean", "std"],

            "Observed_RMSE":
                ["mean", "std"],

            "Observed_Preservation_Error":
                "max",

            "Aleatoric_STD_Mean":
                ["mean", "std"],

            "Aleatoric_Variance_Mean":
                ["mean", "std"],
        })
    )

    # -----------------------------------------------------------------
    # Flatten multi-level column names
    # -----------------------------------------------------------------

    flattened_columns = []

    for column in summary_df.columns:

        if isinstance(
            column,
            tuple
        ):

            if column[1]:

                flattened_columns.append(
                    f"{column[0]}_{column[1]}"
                )

            else:

                flattened_columns.append(
                    column[0]
                )

        else:

            flattened_columns.append(
                column
            )

    summary_df.columns = (
        flattened_columns
    )

    # -----------------------------------------------------------------
    # Add experiment metadata
    # -----------------------------------------------------------------

    summary_df.insert(
        0,
        "Experiment",
        EXPERIMENT_NAME
    )

    summary_df.insert(
        1,
        "Dataset_Mode",
        DATASET_MODE
    )

    # =================================================================
    # SAVE DETAILED RESULTS
    # =================================================================

    detailed_df.to_csv(
        DETAILED_CSV_FILE,
        index=False
    )

    # =================================================================
    # SAVE SUMMARY
    # =================================================================

    summary_df.to_csv(
        SUMMARY_CSV_FILE,
        index=False
    )

    # =================================================================
    # SAVE METADATA
    # =================================================================

    metadata = {

        "experiment":
            EXPERIMENT_NAME,

        "dataset_mode":
            DATASET_MODE,

        "checkpoint":
            str(
                CHECKPOINT_FILE
            ),

        "device":
            str(device),

        "configured_device":
            CONFIG_DEVICE,

        "use_attention":
            USE_ATTENTION,

        "use_residual":
            USE_RESIDUAL,

        "use_uncertainty":
            USE_UNCERTAINTY,

        "noise_levels":
            [
                float(level)
                for level
                in NOISE_ROBUSTNESS_LEVELS
            ],

        "number_of_samples":
            num_test_samples,

        "random_seed":
            NOISE_ROBUSTNESS_SEED,

        "mask_observed_value":
            MASK_OBSERVED_VALUE,

        "mask_missing_value":
            MASK_MISSING_VALUE,

        "seismic_data_range":
            SEISMIC_DATA_RANGE,

        "observed_preservation_tolerance":
            OBSERVED_PRESERVATION_TOLERANCE,

        "uncertainty_type":
            "Aleatoric standard deviation and variance",

        "epistemic_uncertainty":
            "Not evaluated in this deterministic robustness module",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # =================================================================
    # FINAL OUTPUT
    # =================================================================

    print()
    print("=" * 70)
    print("NOISE ROBUSTNESS RESULTS")
    print("=" * 70)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("Detailed results:")
    print(
        DETAILED_CSV_FILE
    )

    print()
    print("Summary results:")
    print(
        SUMMARY_CSV_FILE
    )

    print()
    print("Metadata:")
    print(
        METADATA_FILE
    )

    print()
    print("=" * 70)
    print("NOISE ROBUSTNESS EVALUATION COMPLETED")
    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()