"""
======================================================================
SINGLE CONTROLLED PROPOSED-MODEL VALIDATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty

Controlled case
---------------
Geological mode      : folded
Missing mechanism    : missing_crosslines
Missing rate         : 30%
Seed                 : 42
Cube                 : 64 x 128 x 128

Purpose
-------
Validate the freshly trained proposed model on one controlled
experimental condition before launching the complete 750-case matrix.

======================================================================
"""

import os
import sys
import time
from pathlib import Path

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
from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import PredictiveUncertaintyEstimator

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.config import MC_DROPOUT_SAMPLES


# =====================================================================
# CONTROLLED EXPERIMENT CONFIGURATION
# =====================================================================

CUBE_SIZE = (
    64,
    128,
    128,
)

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"

MISSING_RATE = 0.30

SEED = 42

NUM_SAMPLES = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# CHECKPOINT
# =====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_training"
    / "checkpoints"
    / "best_model.pth"
)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("SINGLE CONTROLLED PROPOSED-MODEL VALIDATION")
    print("=" * 70)

    print()
    print("Configuration")
    print("-------------")
    print(f"Device              : {DEVICE}")
    print(f"Cube size           : {CUBE_SIZE}")
    print(f"Geological mode     : {GEOLOGICAL_MODE}")
    print(f"Missing mechanism   : {MASK_MODE}")
    print(f"Missing rate        : {MISSING_RATE}")
    print(f"Seed                : {SEED}")
    print(f"MC samples          : {MC_DROPOUT_SAMPLES}")

    print()
    print("Checkpoint")
    print("----------")
    print(CHECKPOINT)

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(
            "\nFresh best_model.pth was not found:\n"
            f"{CHECKPOINT}\n\n"
            "Check that the retrained model was copied into "
            "the project outputs/synthetic_training/checkpoints "
            "directory."
        )

    print("Checkpoint exists    : PASS")


    # =================================================================
    # REPRODUCIBILITY
    # =================================================================

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    # =================================================================
    # DATASET
    # =================================================================

    print()
    print("=" * 70)
    print("CREATING CONTROLLED SYNTHETIC SAMPLE")
    print("=" * 70)

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
        seed=SEED,
    )

    input_cube, target_cube, mask, velocity_model = dataset[0]

    print()
    print(f"Input shape          : {tuple(input_cube.shape)}")
    print(f"Target shape         : {tuple(target_cube.shape)}")
    print(f"Mask shape           : {tuple(mask.shape)}")
    print(f"Velocity shape       : {tuple(velocity_model.shape)}")


    # =================================================================
    # SHAPE VALIDATION
    # =================================================================

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    if tuple(input_cube.shape) != expected_shape:
        raise RuntimeError(
            f"Input shape mismatch: "
            f"{tuple(input_cube.shape)} != {expected_shape}"
        )

    if tuple(target_cube.shape) != expected_shape:
        raise RuntimeError(
            f"Target shape mismatch: "
            f"{tuple(target_cube.shape)} != {expected_shape}"
        )

    if tuple(mask.shape) != expected_shape:
        raise RuntimeError(
            f"Mask shape mismatch: "
            f"{tuple(mask.shape)} != {expected_shape}"
        )

    print("Shape validation      : PASS")


    # =================================================================
    # INPUT CONSISTENCY
    # =================================================================

    input_consistency = torch.max(
        torch.abs(
            input_cube[mask == 1]
            - target_cube[mask == 1]
        )
    ).item()

    print(
        f"Observed input error : "
        f"{input_consistency:.6e}"
    )

    if input_consistency > 1e-6:
        raise RuntimeError(
            "Observed input samples do not match target."
        )

    print("Input consistency     : PASS")


    # =================================================================
    # OBSERVED / MISSING COUNTS
    # =================================================================

    observed_count = int(
        torch.sum(mask == 1).item()
    )

    missing_count = int(
        torch.sum(mask == 0).item()
    )

    print()
    print(f"Observed voxels       : {observed_count}")
    print(f"Missing voxels        : {missing_count}")


    # =================================================================
    # MOVE TO DEVICE
    # =================================================================

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


    # =================================================================
    # MODEL
    # =================================================================

    print()
    print("=" * 70)
    print("LOADING NETWORK")
    print("=" * 70)

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    ).to(DEVICE)


    # =================================================================
    # LOAD CHECKPOINT
    # =================================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint loading    : PASS")


    # =================================================================
    # MC DROPOUT
    # =================================================================

    print()
    print("=" * 70)
    print("MC DROPOUT PREDICTION")
    print("=" * 70)

    model.eval()

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_DROPOUT_SAMPLES,
    )

    start_time = time.perf_counter()

    predictions = mc_dropout.predict(
        input_batch
    )

    runtime = (
        time.perf_counter()
        - start_time
    )


    # =================================================================
    # EXTRACT MC OUTPUTS
    # =================================================================

    reconstruction_samples = (
        predictions["reconstruction_samples"]
    )

    travel_time_samples = (
        predictions["travel_time_samples"]
    )

    log_variance_samples = (
        predictions["log_variance_samples"]
    )

    print()
    print(
        "Reconstruction samples :",
        tuple(reconstruction_samples.shape)
    )

    print(
        "Travel-time samples    :",
        tuple(travel_time_samples.shape)
    )

    print(
        "Log-variance samples  :",
        tuple(log_variance_samples.shape)
    )


    # =================================================================
    # UNCERTAINTY DECOMPOSITION
    # =================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )

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


    # =================================================================
    # DATA-CONSISTENCY PROJECTION
    # =================================================================

    reconstruction_mean = (
        reconstruction_mean.clone()
    )

    reconstruction_mean[
        mask_batch == 1
    ] = target_batch[
        mask_batch == 1
    ]


    # =================================================================
    # FINITE CHECK
    # =================================================================

    tensors_to_check = {
        "reconstruction": reconstruction_mean,
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "predictive_variance": predictive_variance,
        "predictive_std": predictive_std,
    }

    for name, tensor in tensors_to_check.items():

        if not torch.isfinite(tensor).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    print()
    print("Finite-value checks   : PASS")


    # =================================================================
    # OBSERVED DATA PRESERVATION
    # =================================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction_mean[mask_batch == 1]
            - target_batch[mask_batch == 1]
        )
    ).item()

    print(
        f"Observed-data error   : "
        f"{observed_difference:.6e}"
    )

    if observed_difference > 1e-6:

        raise RuntimeError(
            "Observed samples were not preserved."
        )

    print("Observed preservation : PASS")


    # =================================================================
    # METRICS
    # =================================================================

    metric_mae = mae(
        reconstruction_mean,
        target_batch,
    ).item()

    metric_rmse = rmse(
        reconstruction_mean,
        target_batch,
    ).item()

    metric_psnr = psnr(
        reconstruction_mean,
        target_batch,
    ).item()

    metric_snr = snr(
        reconstruction_mean,
        target_batch,
    ).item()

    metric_ssim = ssim(
        reconstruction_mean,
        target_batch,
    ).item()


    # =================================================================
    # MISSING-ONLY MAE
    # =================================================================

    missing_mae = torch.mean(
        torch.abs(
            reconstruction_mean[mask_batch == 0]
            - target_batch[mask_batch == 0]
        )
    ).item()


    # =================================================================
    # UNCERTAINTY SUMMARIES
    # =================================================================

    mean_aleatoric = (
        aleatoric_variance.mean().item()
    )

    mean_epistemic = (
        epistemic_variance.mean().item()
    )

    mean_predictive = (
        predictive_variance.mean().item()
    )

    mean_predictive_std = (
        predictive_std.mean().item()
    )

    missing_predictive_std = (
        predictive_std[mask_batch == 0]
        .mean()
        .item()
    )


    # =================================================================
    # RESULTS
    # =================================================================

    print()
    print("=" * 70)
    print("CONTROLLED VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(f"MAE                    : {metric_mae:.6f}")
    print(f"RMSE                   : {metric_rmse:.6f}")
    print(f"PSNR                   : {metric_psnr:.6f} dB")
    print(f"SNR                    : {metric_snr:.6f} dB")
    print(f"SSIM                   : {metric_ssim:.6f}")

    print()
    print(
        f"Missing-only MAE      : "
        f"{missing_mae:.6f}"
    )

    print()
    print(
        f"Mean aleatoric var    : "
        f"{mean_aleatoric:.6e}"
    )

    print(
        f"Mean epistemic var    : "
        f"{mean_epistemic:.6e}"
    )

    print(
        f"Mean predictive var   : "
        f"{mean_predictive:.6e}"
    )

    print(
        f"Mean predictive std   : "
        f"{mean_predictive_std:.6e}"
    )

    print(
        f"Missing predictive std: "
        f"{missing_predictive_std:.6e}"
    )

    print(
        f"Runtime               : "
        f"{runtime:.4f} s"
    )


    # =================================================================
    # FINAL STATUS
    # =================================================================

    print()
    print("=" * 70)
    print("SINGLE CONTROLLED PROPOSED-MODEL TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()