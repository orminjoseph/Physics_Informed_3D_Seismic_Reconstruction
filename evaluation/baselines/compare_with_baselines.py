"""
======================================================================
COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Perform a common side-by-side comparison of the six classical
reconstruction baselines and the proposed Physics-Informed
3D Encoder-Decoder model.

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
THIS SCRIPT IS NOT THE 750-CASE CONTROLLED EXPERIMENT.

The dedicated controlled-matrix scripts are responsible for the
750-case experiments.

This script performs ONE common side-by-side comparison in which
all seven methods receive exactly the same:

    corrupted seismic cube
    observation mask
    target cube

Ground truth is used ONLY for post-reconstruction evaluation.

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

Network convention
------------------
    (B, C, D, H, W)

Mask convention
---------------
    1 = observed
    0 = missing

Checkpoint
----------
    outputs/<experiment_name>/checkpoints/best_model.pth

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


# =====================================================================
# DATASET
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset,
)


# =====================================================================
# CLASSICAL BASELINES
#
# These imports point to the final baseline implementations under
# evaluation/baselines/.
# =====================================================================

from evaluation.baselines.baseline_nearest_neighbor_controlled_matrix import (
    nearest_neighbor_reconstruction,
)

from evaluation.baselines.baseline_linear_interpolation_controlled_matrix import (
    linear_interpolation_reconstruction,
)

from evaluation.baselines.fx_controlled_matrix import (
    fx_prediction_reconstruction,
)

from evaluation.baselines.compressive_sensing_controlled_matrix import (
    compressive_sensing_reconstruction,
)

from evaluation.baselines.curvelet_pocs_controlled_matrix import (
    curvelet_pocs_reconstruction,
)

from evaluation.baselines.dictionary_learning_controlled_matrix import (
    dictionary_learning_reconstruction,
)


# =====================================================================
# PROPOSED MODEL
# =====================================================================

from models.network import Network3D

from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)


# =====================================================================
# CENTRALIZED METRICS
# =====================================================================

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =====================================================================
# CENTRALIZED CONFIGURATION
# =====================================================================

from utils.config import (
    BASELINE_NUM_SAMPLES,
    BASELINE_CUBE_SIZE,
    BASELINE_MISSING_RATE,
    BASELINE_GEOLOGICAL_MODE,
    BASELINE_MASK_MODE,
    BASELINE_SEED,

    OBSERVED_PRESERVATION_TOLERANCE,

    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,

    MC_DROPOUT_SAMPLES,

    DEVICE as CONFIG_DEVICE,

    CHECKPOINT_DIR,
    REPORT_DIR as CONFIG_REPORT_DIR,
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# =====================================================================
# COMMON COMPARISON CONFIGURATION
# =====================================================================

CUBE_SIZE = BASELINE_CUBE_SIZE

NUM_SAMPLES = BASELINE_NUM_SAMPLES

MISSING_RATE = BASELINE_MISSING_RATE

GEOLOGICAL_MODE = BASELINE_GEOLOGICAL_MODE

MASK_MODE = BASELINE_MASK_MODE

SEED = BASELINE_SEED


# =====================================================================
# PROPOSED MODEL CHECKPOINT
# =====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / Path(CHECKPOINT_DIR)
    / "best_model.pth"
)


# =====================================================================
# MC-DROPOUT CONFIGURATION
# =====================================================================

MC_SAMPLES = MC_DROPOUT_SAMPLES


# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

REPORT_DIR = (
    PROJECT_ROOT
    / Path(CONFIG_REPORT_DIR)
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# OUTPUT FILES
# =====================================================================

RESULTS_FILE = (
    REPORT_DIR
    / "baseline_comparison.csv"
)


SUMMARY_FILE = (
    REPORT_DIR
    / "baseline_comparison_summary.csv"
)


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device(
    configured_device,
) -> torch.device:
    """
    Resolve the centralized DEVICE configuration.

    Supported values:
        "auto"
        "cpu"
        "cuda"
    """

    if configured_device == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if configured_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' is configured, "
                "but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


DEVICE = resolve_device(
    CONFIG_DEVICE
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =====================================================================
# FINITE TENSOR CHECK
# =====================================================================

def tensor_is_finite(
    tensor: torch.Tensor,
) -> bool:
    """
    Return True when every tensor element is finite.
    """

    return bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


# =====================================================================
# METRIC CONVERSION
# =====================================================================

def metric_to_float(
    value,
) -> float:
    """
    Convert a tensor, NumPy scalar, or Python numeric value
    into a Python float.
    """

    if isinstance(
        value,
        torch.Tensor,
    ):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):

        return float(
            value.item()
        )

    return float(value)


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def validate_common_input(
    corrupted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Validate the common comparison dataset.

    All seven methods receive the same input cube and mask.
    """

    # -----------------------------------------------------------------
    # Tensor types.
    # -----------------------------------------------------------------

    tensors = {
        "corrupted": corrupted,
        "target": target,
        "mask": mask,
        "velocity": velocity,
    }

    for name, tensor in tensors.items():

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

    # -----------------------------------------------------------------
    # Expected 4-D convention.
    # -----------------------------------------------------------------

    for name, tensor in tensors.items():

        if tensor.ndim != 4:

            raise ValueError(
                f"{name} must have shape "
                "(C, D, H, W). "
                f"Received: {tuple(tensor.shape)}"
            )

    # -----------------------------------------------------------------
    # Shape consistency.
    # -----------------------------------------------------------------

    if corrupted.shape != target.shape:

        raise RuntimeError(
            "Corrupted and target shapes differ. "
            f"Corrupted: {tuple(corrupted.shape)}; "
            f"Target: {tuple(target.shape)}"
        )

    if corrupted.shape != mask.shape:

        raise RuntimeError(
            "Corrupted and mask shapes differ."
        )

    if corrupted.shape != velocity.shape:

        raise RuntimeError(
            "Corrupted and velocity shapes differ."
        )

    # -----------------------------------------------------------------
    # Expected benchmark shape.
    # -----------------------------------------------------------------

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    if tuple(corrupted.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected benchmark cube shape. "
            f"Expected: {expected_shape}; "
            f"Received: {tuple(corrupted.shape)}"
        )

    # -----------------------------------------------------------------
    # Finite-value checks.
    # -----------------------------------------------------------------

    for name, tensor in tensors.items():

        if not tensor_is_finite(
            tensor
        ):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    # -----------------------------------------------------------------
    # Binary mask.
    # -----------------------------------------------------------------

    unique_mask = torch.unique(
        mask
    )

    if not torch.all(
        (unique_mask == 0)
        |
        (unique_mask == 1)
    ):

        raise RuntimeError(
            "Mask must contain only 0 and 1. "
            f"Received: {unique_mask.tolist()}"
        )

    # -----------------------------------------------------------------
    # At least one observed and one missing sample.
    # -----------------------------------------------------------------

    observed_count = int(
        (mask == 1).sum().item()
    )

    missing_count = int(
        (mask == 0).sum().item()
    )

    if observed_count == 0:

        raise RuntimeError(
            "The common comparison sample contains "
            "no observed samples."
        )

    if missing_count == 0:

        raise RuntimeError(
            "The common comparison sample contains "
            "no missing samples."
        )


# =====================================================================
# OBSERVED-DATA PRESERVATION
# =====================================================================

def validate_reconstruction(
    reconstruction: torch.Tensor,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Validate one reconstructed cube.

    Checks:
        1. Tensor type
        2. Shape
        3. Finite values
        4. Exact observed-data preservation

    Returns
    -------
    float
        Maximum observed-data preservation error.
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
            f"Expected: {tuple(corrupted.shape)}; "
            f"Received: {tuple(reconstruction.shape)}"
        )

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "Reconstruction contains NaN or Inf."
        )

    observed_selector = (
        mask == 1
    )

    observed_difference = torch.max(
        torch.abs(
            reconstruction[
                observed_selector
            ]
            -
            corrupted[
                observed_selector
            ]
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
            f"{observed_difference:.6e}; "
            f"Tolerance: "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.6e}"
        )

    return float(
        observed_difference
    )


# =====================================================================
# COMMON METRICS
# =====================================================================

def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Calculate common global and missing-region metrics.

    Input convention:
        (C, D, H, W)

    The centralized metrics module is used for global metrics.
    """

    # -----------------------------------------------------------------
    # Missing-region selector.
    # -----------------------------------------------------------------

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

    if (
        missing_prediction.numel()
        == 0
    ):

        raise RuntimeError(
            "No missing samples available "
            "for missing-region metrics."
        )

    # -----------------------------------------------------------------
    # Missing-region MAE.
    # -----------------------------------------------------------------

    missing_mae = float(
        torch.mean(
            torch.abs(
                missing_prediction
                -
                missing_target
            )
        ).item()
    )

    # -----------------------------------------------------------------
    # Missing-region RMSE.
    # -----------------------------------------------------------------

    missing_rmse = float(
        torch.sqrt(
            torch.mean(
                (
                    missing_prediction
                    -
                    missing_target
                ) ** 2
            )
        ).item()
    )

    # -----------------------------------------------------------------
    # Add batch dimension.
    #
    # (C,D,H,W)
    #       ↓
    # (B,C,D,H,W)
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Global MAE.
    # -----------------------------------------------------------------

    metric_mae = metric_to_float(
        mae(
            reconstruction_batch,
            target_batch,
        )
    )

    # -----------------------------------------------------------------
    # Global RMSE.
    # -----------------------------------------------------------------

    metric_rmse = metric_to_float(
        rmse(
            reconstruction_batch,
            target_batch,
        )
    )

    # -----------------------------------------------------------------
    # Global PSNR.
    # -----------------------------------------------------------------

    metric_psnr = metric_to_float(
        psnr(
            reconstruction_batch,
            target_batch,
        )
    )

    # -----------------------------------------------------------------
    # Global SNR.
    # -----------------------------------------------------------------

    metric_snr = metric_to_float(
        snr(
            reconstruction_batch,
            target_batch,
        )
    )

    # -----------------------------------------------------------------
    # Global SSIM.
    # -----------------------------------------------------------------

    metric_ssim = metric_to_float(
        ssim(
            reconstruction_batch,
            target_batch,
        )
    )

    return {

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

        "missing_mae":
            missing_mae,

        "missing_rmse":
            missing_rmse,
    }


# =====================================================================
# LOAD PROPOSED MODEL
# =====================================================================

def load_proposed_model():
    """
    Load the frozen proposed Physics-Informed 3-D model.
    """

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "Proposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}"
        )

    print()
    print("=" * 78)
    print(
        "LOADING PROPOSED MODEL"
    )
    print("=" * 78)

    print()
    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # -----------------------------------------------------------------
    # Construct production architecture.
    # -----------------------------------------------------------------

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    model = model.to(
        DEVICE
    )

    # -----------------------------------------------------------------
    # Load checkpoint.
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print(
        "Proposed model loaded successfully."
    )

    if "best_epoch" in checkpoint:

        print(
            f"Best epoch: "
            f"{checkpoint['best_epoch']}"
        )

    return model


# =====================================================================
# RUN ONE CLASSICAL BASELINE
# =====================================================================

def run_classical_method(
    method_name: str,
    reconstruction_function,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Run one classical baseline using the common corrupted cube
    and common observation mask.
    """

    print()
    print("-" * 78)
    print(
        f"METHOD: {method_name}"
    )
    print("-" * 78)

    start_time = time.perf_counter()

    # -----------------------------------------------------------------
    # IMPORTANT:
    #
    # Ground truth is NOT passed to the reconstruction function.
    # -----------------------------------------------------------------

    reconstruction = (
        reconstruction_function(
            corrupted,
            mask,
        )
    )

    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )

    # -----------------------------------------------------------------
    # Validate reconstruction.
    # -----------------------------------------------------------------

    preservation_error = (
        validate_reconstruction(
            reconstruction,
            corrupted,
            mask,
        )
    )

    # -----------------------------------------------------------------
    # Calculate metrics.
    # -----------------------------------------------------------------

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
            float(preservation_error),

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

        "missing_mae":
            metrics["missing_mae"],

        "missing_rmse":
            metrics["missing_rmse"],

        # ---------------------------------------------------------
        # Classical methods do not produce predictive uncertainty.
        # ---------------------------------------------------------

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
        f"Missing RMSE : "
        f"{result['missing_rmse']:.6f}"
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
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Run the frozen proposed Physics-Informed 3-D model.

    Input:
        corrupted : (C,D,H,W)
        mask      : (C,D,H,W)

    Network input:
        (B,C,D,H,W)

    Ground truth is used only for evaluation.
    """

    print()
    print("-" * 78)
    print(
        "METHOD: proposed_physics_informed_3d"
    )
    print("-" * 78)

    # =================================================================
    # PREPARE NETWORK INPUT
    # =================================================================

    # -------------------------------------------------------------
    # IMPORTANT CORRECTION:
    #
    # corrupted is already (C,D,H,W).
    #
    # Only ONE batch dimension must be added:
    #
    # (C,D,H,W)
    #      ↓
    # (B,C,D,H,W)
    #
    # DO NOT use unsqueeze(0).unsqueeze(0).
    # -------------------------------------------------------------

    input_batch = (
        corrupted
        .unsqueeze(0)
        .to(DEVICE)
    )

    # =================================================================
    # MC-DROPOUT PREDICTOR
    # =================================================================

    predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    # =================================================================
    # MC-DROPOUT INFERENCE
    # =================================================================

    start_time = time.perf_counter()

    predictions = predictor.predict(
        input_batch
    )

    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )

    # =================================================================
    # VALIDATE MC OUTPUT
    # =================================================================

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
            "The MC-Dropout predictor did not return "
            f"the required outputs: {missing_keys}"
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

    # =================================================================
    # VALIDATE RECONSTRUCTION SAMPLES
    # =================================================================

    if not tensor_is_finite(
        reconstruction_samples
    ):

        raise RuntimeError(
            "MC reconstruction samples contain NaN or Inf."
        )

    if not tensor_is_finite(
        log_variance_samples
    ):

        raise RuntimeError(
            "MC log-variance samples contain NaN or Inf."
        )

    # =================================================================
    # MC MEAN RECONSTRUCTION
    # =================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

    # =================================================================
    # ALEATORIC VARIANCE
    # =================================================================

    aleatoric_variance = (
        PredictiveUncertaintyEstimator
        .aleatoric_variance(
            log_variance_samples
        )
    )

    # =================================================================
    # EPISTEMIC VARIANCE
    # =================================================================

    epistemic_variance = (
        PredictiveUncertaintyEstimator
        .epistemic_variance(
            reconstruction_samples
        )
    )

    # =================================================================
    # PREDICTIVE VARIANCE
    # =================================================================

    predictive_variance = (
        PredictiveUncertaintyEstimator
        .predictive_variance(
            aleatoric_variance,
            epistemic_variance,
        )
    )

    # =================================================================
    # PREDICTIVE STANDARD DEVIATION
    # =================================================================

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0,
        )
    )

    # =================================================================
    # UNCERTAINTY VALIDATION
    # =================================================================

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

        if not tensor_is_finite(
            tensor
        ):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

        if (
            "variance" in name
            and
            (tensor < 0).any()
        ):

            raise RuntimeError(
                f"Negative values detected in {name}."
            )

    # =================================================================
    # DATA-CONSISTENCY PROJECTION
    # =================================================================

    # -----------------------------------------------------------------
    # reconstruction_mean:
    #
    #     (B,C,D,H,W)
    #
    # corrupted.unsqueeze(0):
    #
    #     (B,C,D,H,W)
    #
    # mask.unsqueeze(0):
    #
    #     (B,C,D,H,W)
    # -----------------------------------------------------------------

    reconstruction_batch = (
        (
            1.0
            -
            mask.unsqueeze(0).to(
                DEVICE
            )
        )
        *
        reconstruction_mean
        +
        mask.unsqueeze(0).to(
            DEVICE
        )
        *
        corrupted.unsqueeze(0).to(
            DEVICE
        )
    )

    # -----------------------------------------------------------------
    # Remove only the batch dimension.
    #
    # (B,C,D,H,W)
    #       ↓
    # (C,D,H,W)
    # -----------------------------------------------------------------

    reconstruction = (
        reconstruction_batch
        .squeeze(0)
        .detach()
    )

    # =================================================================
    # FINAL OBSERVED-DATA VALIDATION
    # =================================================================

    preservation_error = (
        validate_reconstruction(
            reconstruction,
            corrupted,
            mask,
        )
    )

    # =================================================================
    # COMMON METRICS
    # =================================================================

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    # =================================================================
    # UNCERTAINTY MASK
    # =================================================================

    missing_selector = (
        mask
        .to(DEVICE)
        == 0
    )

    # =================================================================
    # OVERALL UNCERTAINTY
    # =================================================================

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

    # =================================================================
    # MISSING-REGION UNCERTAINTY
    # =================================================================

    if not missing_selector.any():

        raise RuntimeError(
            "No missing samples available for "
            "missing-region uncertainty."
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

    # =================================================================
    # RESULT RECORD
    # =================================================================

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
            float(preservation_error),

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

        "missing_mae":
            metrics["missing_mae"],

        "missing_rmse":
            metrics["missing_rmse"],

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
        f"Missing RMSE : "
        f"{result['missing_rmse']:.6f}"
    )

    print(
        f"Runtime      : "
        f"{result['runtime_seconds']:.4f} s"
    )

    print(
        f"Aleatoric σ² : "
        f"{result['aleatoric_variance_mean']:.6e}"
    )

    print(
        f"Epistemic σ² : "
        f"{result['epistemic_variance_mean']:.6e}"
    )

    print(
        f"Predictive σ²: "
        f"{result['predictive_variance_mean']:.6e}"
    )

    print(
        f"Predictive σ : "
        f"{result['predictive_std_mean']:.6e}"
    )

    print(
        f"Observed err : "
        f"{result['observed_preservation_error']:.6e}"
    )

    return result


# =====================================================================
# FAILED RESULT RECORD
# =====================================================================

def build_failed_result(
    method_name: str,
    error: Exception,
) -> dict:
    """
    Create a standardized failed-result record.
    """

    return {

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

        "missing_mae":
            np.nan,

        "missing_rmse":
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
            str(error),
    }


# =====================================================================
# WRITE CSV
# =====================================================================

def write_csv(
    output_file: Path,
    records: list[dict],
) -> None:
    """
    Write records to CSV.
    """

    if not records:

        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
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
            records
        )


# =====================================================================
# SUMMARY
# =====================================================================

def calculate_summary(
    records: list[dict],
) -> list[dict]:
    """
    Create the side-by-side summary.

    This script intentionally contains one common test case per
    method. Therefore n_cases is normally 1.

    The large-sample statistical analysis belongs to the dedicated
    750-case controlled experiments.
    """

    summaries = []

    for record in records:

        summaries.append(
            {

                "method":
                    record["method"],

                "n_cases":
                    1
                    if record["status"]
                    == "PASS"
                    else 0,

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
                    record[
                        "runtime_seconds"
                    ],

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
        )

    return summaries


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """
    Execute the common seven-method comparison.
    """

    print()
    print("=" * 78)
    print(
        "COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON"
    )
    print("=" * 78)

    # =================================================================
    # CONFIGURATION DISPLAY
    # =================================================================

    print()
    print(
        f"Cube size          : {CUBE_SIZE}"
    )

    print(
        f"Number of samples  : {NUM_SAMPLES}"
    )

    print(
        f"Geological mode    : {GEOLOGICAL_MODE}"
    )

    print(
        f"Mask mode          : {MASK_MODE}"
    )

    print(
        f"Missing rate       : {MISSING_RATE:.0%}"
    )

    print(
        f"Seed               : {SEED}"
    )

    print(
        f"Configured device  : {CONFIG_DEVICE}"
    )

    print(
        f"Resolved device    : {DEVICE}"
    )

    print(
        f"Attention          : {USE_ATTENTION}"
    )

    print(
        f"Residual           : {USE_RESIDUAL}"
    )

    print(
        f"Uncertainty        : {USE_UNCERTAINTY}"
    )

    print(
        f"MC samples         : {MC_SAMPLES}"
    )

    print()
    print(
        "This script evaluates ONE common test case."
    )

    print(
        "The 750-case controlled experiments are "
        "handled separately."
    )

    # =================================================================
    # REPRODUCIBILITY
    # =================================================================

    set_seed(
        SEED
    )

    # =================================================================
    # CREATE COMMON DATASET
    # =================================================================

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

    # =================================================================
    # VALIDATE DATASET METADATA
    # =================================================================

    if (
        actual_geological_mode
        != GEOLOGICAL_MODE
    ):

        raise RuntimeError(
            "Geological mode mismatch. "
            f"Requested: {GEOLOGICAL_MODE}; "
            f"Received: {actual_geological_mode}"
        )

    if (
        actual_mask_mode
        != MASK_MODE
    ):

        raise RuntimeError(
            "Mask mode mismatch. "
            f"Requested: {MASK_MODE}; "
            f"Received: {actual_mask_mode}"
        )

    # =================================================================
    # VALIDATE COMMON INPUT
    # =================================================================

    validate_common_input(
        corrupted,
        target,
        mask,
        velocity,
    )

    # =================================================================
    # INPUT DATA CONSISTENCY
    # =================================================================

    # -----------------------------------------------------------------
    # The synthetic dataset should satisfy:
    #
    # corrupted = target × mask
    #
    # at the observed/missing representation level.
    # -----------------------------------------------------------------

    expected_corrupted = (
        target
        *
        mask
    )

    input_difference = torch.max(
        torch.abs(
            corrupted
            -
            expected_corrupted
        )
    ).item()

    if (
        input_difference
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Common input consistency failed. "
            f"Maximum difference: "
            f"{input_difference:.6e}"
        )

    # =================================================================
    # SAMPLE STATISTICS
    # =================================================================

    observed_samples = int(
        (mask == 1)
        .sum()
        .item()
    )

    missing_samples = int(
        (mask == 0)
        .sum()
        .item()
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
        f"Observed samples  : {observed_samples}"
    )

    print(
        f"Missing samples   : {missing_samples}"
    )

    print(
        f"Measured missing  : "
        f"{measured_missing_rate:.6f}"
    )

    # =================================================================
    # LOAD FROZEN PROPOSED MODEL
    # =================================================================

    proposed_model = (
        load_proposed_model()
    )

    # =================================================================
    # RESULT STORAGE
    # =================================================================

    records = []

    # =================================================================
    # CLASSICAL BASELINES
    # =================================================================

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

    # =================================================================
    # RUN SIX CLASSICAL METHODS
    # =================================================================

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

        except Exception as error:

            print()
            print(
                f"{method_name} FAILED"
            )

            print(
                f"Error: {error}"
            )

            result = (
                build_failed_result(
                    method_name,
                    error,
                )
            )

        records.append(
            result
        )

    # =================================================================
    # RUN PROPOSED MODEL
    # =================================================================

    try:

        result = run_proposed_model(
            model=proposed_model,
            corrupted=corrupted,
            mask=mask,
            target=target,
        )

    except Exception as error:

        print()
        print(
            "proposed_physics_informed_3d FAILED"
        )

        print(
            f"Error: {error}"
        )

        result = (
            build_failed_result(
                "proposed_physics_informed_3d",
                error,
            )
        )

    records.append(
        result
    )

    # =================================================================
    # WRITE RAW RESULTS
    # =================================================================

    write_csv(
        RESULTS_FILE,
        records,
    )

    # =================================================================
    # WRITE SUMMARY
    # =================================================================

    summaries = (
        calculate_summary(
            records
        )
    )

    write_csv(
        SUMMARY_FILE,
        summaries,
    )

    # =================================================================
    # FINAL COUNTS
    # =================================================================

    successful = sum(
        record["status"]
        == "PASS"
        for record in records
    )

    failed = sum(
        record["status"]
        == "FAILED"
        for record in records
    )

    # =================================================================
    # CONSOLE COMPARISON TABLE
    # =================================================================

    print()
    print("=" * 78)
    print(
        "COMMON RECONSTRUCTION METRICS"
    )
    print("=" * 78)

    print()

    print(
        f"{'Method':<36}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'PSNR':>10}"
        f"{'SNR':>10}"
        f"{'SSIM':>10}"
    )

    print(
        "-" * 86
    )

    for record in records:

        if (
            record["status"]
            != "PASS"
        ):

            continue

        print(
            f"{record['method']:<36}"
            f"{record['MAE']:>10.6f}"
            f"{record['RMSE']:>10.6f}"
            f"{record['PSNR']:>10.4f}"
            f"{record['SNR']:>10.4f}"
            f"{record['SSIM']:>10.6f}"
        )

    # =================================================================
    # OUTPUT SUMMARY
    # =================================================================

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

    # =================================================================
    # FINAL STATUS
    # =================================================================

    if failed == 0:

        print()
        print(
            "OVERALL STATUS: PASS"
        )

        print()
        print(
            "All seven methods completed successfully "
            "using the same corrupted cube and "
            "observation mask."
        )

    else:

        print()
        print(
            "OVERALL STATUS: FAIL"
        )

        print()
        print(
            "One or more methods failed."
        )

        print(
            "Inspect the 'error' column in "
            "baseline_comparison.csv."
        )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()