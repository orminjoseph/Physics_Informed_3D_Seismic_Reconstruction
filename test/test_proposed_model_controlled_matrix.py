"""
====================================================================
Focused Test — Proposed Physics-Informed 3D Encoder–Decoder Model
====================================================================

Purpose
-------
Focused validation test for the proposed-model reconstruction
implementation used by the controlled experimental matrix.

This test is NOT the 750-case controlled experiment.

The full controlled experiment remains in:

    evaluation/baselines/proposed_model_controlled_matrix.py

This test validates ONE deterministic representative case.

Validation includes
-------------------
1. Dataset generation
2. Tensor shapes
3. Metadata consistency
4. Input/target consistency on observed samples
5. Missing-data existence
6. Proposed model loading
7. MC-Dropout inference
8. Aleatoric uncertainty
9. Epistemic uncertainty
10. Predictive uncertainty
11. Reconstruction validity
12. Data-consistency projection
13. Exact observed-data preservation
14. Missing-region reconstruction
15. Reconstruction metrics
16. Uncertainty non-negativity
17. Reproducibility

Author: Ormin Joseph
====================================================================
"""

# ==================================================================
# 1. IMPORTS
# ==================================================================

import random

import numpy as np
import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from models.network import Network3D

from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import PredictiveUncertaintyEstimator

from utils.config import (
    BASELINE_CUBE_SIZE,
    BASELINE_GEOLOGICAL_MODE,
    BASELINE_MASK_MODE,
    BASELINE_MISSING_RATE,
    BASELINE_SEED,
    MC_DROPOUT_SAMPLES,
    OBSERVED_PRESERVATION_TOLERANCE,
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# ==================================================================
# 2. DEVICE RESOLUTION
# ==================================================================

def resolve_device():
    """
    Resolve the computational device.

    The project configuration may use:

        DEVICE = "auto"
        DEVICE = "cpu"
        DEVICE = "cuda"

    The focused test uses CUDA when explicitly available and
    otherwise falls back to CPU.
    """

    # --------------------------------------------------------------
    # Read the configured device if available.
    # --------------------------------------------------------------

    try:

        from utils.config import DEVICE

    except ImportError:

        DEVICE = "auto"

    # --------------------------------------------------------------
    # Automatic device selection.
    # --------------------------------------------------------------

    if DEVICE == "auto":

        if torch.cuda.is_available():

            return torch.device("cuda")

        return torch.device("cpu")

    # --------------------------------------------------------------
    # Explicit CPU selection.
    # --------------------------------------------------------------

    if DEVICE == "cpu":

        return torch.device("cpu")

    # --------------------------------------------------------------
    # Explicit CUDA selection.
    # --------------------------------------------------------------

    if DEVICE == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' was requested, but CUDA is not available."
            )

        return torch.device("cuda")

    # --------------------------------------------------------------
    # Reject unsupported device specifications.
    # --------------------------------------------------------------

    raise ValueError(
        f"Unsupported DEVICE setting: {DEVICE}"
    )


# ==================================================================
# 3. RANDOM-SEED CONTROL
# ==================================================================

def set_seed(seed):
    """
    Set all relevant random seeds for deterministic testing.
    """

    # Python random generator.
    random.seed(seed)

    # NumPy random generator.
    np.random.seed(seed)

    # PyTorch CPU generator.
    torch.manual_seed(seed)

    # PyTorch CUDA generators.
    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ==================================================================
# 4. ASSERTION HELPER
# ==================================================================

def check(condition, message):
    """
    Raise a clear error when a validation condition fails.
    """

    if not condition:

        raise AssertionError(message)


# ==================================================================
# 5. LOAD PROPOSED MODEL
# ==================================================================

def load_proposed_model(device):
    """
    Construct the proposed Physics-Informed 3D Encoder–Decoder
    network and load the trained best checkpoint.

    The checkpoint path is obtained from the centralized
    configuration rather than being hard-coded.
    """

    # --------------------------------------------------------------
    # Import the centralized checkpoint directory.
    # --------------------------------------------------------------

    from pathlib import Path

    from utils.config import CHECKPOINT_DIR

    # --------------------------------------------------------------
    # Construct checkpoint path.
    # --------------------------------------------------------------

    checkpoint_path = (
        Path(CHECKPOINT_DIR) /
        "best_model.pth"
    )

    # --------------------------------------------------------------
    # Verify checkpoint existence.
    # --------------------------------------------------------------

    check(
        checkpoint_path.exists(),
        (
            "Proposed-model checkpoint was not found:\n"
            f"{checkpoint_path}\n\n"
            "Train the model first or verify CHECKPOINT_DIR "
            "in utils/config.py."
        )
    )

    # --------------------------------------------------------------
    # Construct the same architecture used by the proposed model.
    # --------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    )

    # --------------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------------
    # Extract model state dictionary.
    # --------------------------------------------------------------

    check(
        "model_state_dict" in checkpoint,
        (
            "The checkpoint does not contain "
            "'model_state_dict'."
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------------
    # Move model to selected device.
    # --------------------------------------------------------------

    model = model.to(device)

    # --------------------------------------------------------------
    # Evaluation mode.
    # --------------------------------------------------------------

    model.eval()

    return model


# ==================================================================
# 6. RUN PROPOSED MODEL ON ONE CASE
# ==================================================================

def run_proposed_model_case(model, device):
    """
    Run one deterministic representative case through the proposed
    reconstruction model and uncertainty pipeline.
    """

    # --------------------------------------------------------------
    # Generate exactly one synthetic sample.
    # --------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=BASELINE_MISSING_RATE,
        geological_mode=BASELINE_GEOLOGICAL_MODE,
        mask_mode=BASELINE_MASK_MODE,
        seed=BASELINE_SEED,
    )

    # --------------------------------------------------------------
    # Verify dataset length.
    # --------------------------------------------------------------

    check(
        len(dataset) == 1,
        "Focused proposed-model test must contain exactly one sample."
    )

    # --------------------------------------------------------------
    # Retrieve sample.
    # --------------------------------------------------------------

    sample = dataset[0]

    # --------------------------------------------------------------
    # Expected dataset structure:
    #
    # corrupted
    # target
    # mask
    # velocity
    # actual_mask_mode
    # actual_geological_mode
    # --------------------------------------------------------------

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = sample

    # --------------------------------------------------------------
    # Expected shape:
    #
    # [C, D, H, W]
    #
    # For this project:
    #
    # [1, 64, 128, 128]
    # --------------------------------------------------------------

    expected_shape = (
        1,
        BASELINE_CUBE_SIZE[0],
        BASELINE_CUBE_SIZE[1],
        BASELINE_CUBE_SIZE[2],
    )

    check(
        tuple(corrupted.shape) == expected_shape,
        (
            "Corrupted cube has incorrect shape: "
            f"{tuple(corrupted.shape)}; "
            f"expected {expected_shape}"
        )
    )

    check(
        tuple(target.shape) == expected_shape,
        (
            "Target cube has incorrect shape: "
            f"{tuple(target.shape)}; "
            f"expected {expected_shape}"
        )
    )

    check(
        tuple(mask.shape) == expected_shape,
        (
            "Mask has incorrect shape: "
            f"{tuple(mask.shape)}; "
            f"expected {expected_shape}"
        )
    )

    check(
        tuple(velocity.shape) == expected_shape,
        (
            "Velocity has incorrect shape: "
            f"{tuple(velocity.shape)}; "
            f"expected {expected_shape}"
        )
    )

    # --------------------------------------------------------------
    # Verify metadata.
    # --------------------------------------------------------------

    check(
        actual_mask_mode == BASELINE_MASK_MODE,
        (
            "Mask-mode mismatch: "
            f"expected {BASELINE_MASK_MODE}, "
            f"received {actual_mask_mode}"
        )
    )

    check(
        actual_geological_mode == BASELINE_GEOLOGICAL_MODE,
        (
            "Geological-mode mismatch: "
            f"expected {BASELINE_GEOLOGICAL_MODE}, "
            f"received {actual_geological_mode}"
        )
    )

    # --------------------------------------------------------------
    # Verify finite input data.
    # --------------------------------------------------------------

    check(
        torch.isfinite(corrupted).all().item(),
        "Corrupted seismic cube contains NaN or Inf values."
    )

    check(
        torch.isfinite(target).all().item(),
        "Target seismic cube contains NaN or Inf values."
    )

    check(
        torch.isfinite(mask).all().item(),
        "Mask contains NaN or Inf values."
    )

    check(
        torch.isfinite(velocity).all().item(),
        "Velocity contains NaN or Inf values."
    )

    # --------------------------------------------------------------
    # Verify mask values.
    # --------------------------------------------------------------

    unique_mask_values = torch.unique(mask)

    for value in unique_mask_values:

        check(
            float(value) in (0.0, 1.0),
            (
                "Mask contains an unexpected value: "
                f"{float(value)}"
            )
        )

    # --------------------------------------------------------------
    # Identify observed and missing regions.
    # --------------------------------------------------------------

    observed = mask == 1

    missing = mask == 0

    observed_count = int(observed.sum().item())

    missing_count = int(missing.sum().item())

    # --------------------------------------------------------------
    # Both regions must exist.
    # --------------------------------------------------------------

    check(
        observed_count > 0,
        "No observed samples exist in the test case."
    )

    check(
        missing_count > 0,
        "No missing samples exist in the test case."
    )

    # --------------------------------------------------------------
    # Verify corrupted input equals target at observed locations.
    # --------------------------------------------------------------

    observed_difference = torch.abs(
        corrupted[observed] -
        target[observed]
    )

    check(
        torch.max(observed_difference).item()
        <= OBSERVED_PRESERVATION_TOLERANCE,
        (
            "Corrupted input does not preserve the target "
            "at observed locations."
        )
    )

    # --------------------------------------------------------------
    # Move tensors to the selected device.
    # --------------------------------------------------------------

    corrupted_device = corrupted.to(device)

    target_device = target.to(device)

    mask_device = mask.to(device)

    velocity_device = velocity.to(device)

    # --------------------------------------------------------------
    # Add batch dimension.
    #
    # [C,D,H,W]
    #
    # becomes
    #
    # [B,C,D,H,W]
    # --------------------------------------------------------------

    corrupted_batch = corrupted_device.unsqueeze(0)

    target_batch = target_device.unsqueeze(0)

    mask_batch = mask_device.unsqueeze(0)

    velocity_batch = velocity_device.unsqueeze(0)

    # --------------------------------------------------------------
    # Verify final model input shape.
    # --------------------------------------------------------------

    check(
        corrupted_batch.ndim == 5,
        (
            "Proposed model input must be 5-dimensional "
            "[B,C,D,H,W]."
        )
    )

    # --------------------------------------------------------------
    # Create MC-Dropout predictor.
    # --------------------------------------------------------------

    mc_predictor = MCDropout3D(
        model,
        num_samples=MC_DROPOUT_SAMPLES,
    )

    # --------------------------------------------------------------
    # Run Monte Carlo inference.
    # --------------------------------------------------------------

    with torch.no_grad():

        (
            reconstruction_samples,
            travel_time_samples,
            log_variance_samples,
        ) = mc_predictor(
            corrupted_batch,
            mask_batch,
            velocity_batch,
        )

    # --------------------------------------------------------------
    # Verify MC-Dropout sample count.
    # --------------------------------------------------------------

    check(
        reconstruction_samples.shape[0]
        == MC_DROPOUT_SAMPLES,
        (
            "Incorrect number of reconstruction samples: "
            f"{reconstruction_samples.shape[0]}; "
            f"expected {MC_DROPOUT_SAMPLES}"
        )
    )

    check(
        log_variance_samples.shape[0]
        == MC_DROPOUT_SAMPLES,
        (
            "Incorrect number of log-variance samples: "
            f"{log_variance_samples.shape[0]}; "
            f"expected {MC_DROPOUT_SAMPLES}"
        )
    )

    # --------------------------------------------------------------
    # Verify reconstruction sample shape.
    # --------------------------------------------------------------

    expected_batch_shape = (
        1,
        1,
        BASELINE_CUBE_SIZE[0],
        BASELINE_CUBE_SIZE[1],
        BASELINE_CUBE_SIZE[2],
    )

    check(
        tuple(reconstruction_samples.shape[1:])
        == expected_batch_shape,
        (
            "Unexpected reconstruction-sample shape: "
            f"{tuple(reconstruction_samples.shape)}"
        )
    )

    # --------------------------------------------------------------
    # Construct predictive uncertainty estimator.
    # --------------------------------------------------------------

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
    )

    # --------------------------------------------------------------
    # Calculate aleatoric uncertainty.
    # --------------------------------------------------------------

    with torch.no_grad():

        aleatoric_variance = (
            uncertainty_estimator.aleatoric_variance(
                log_variance_samples
            )
        )

    # --------------------------------------------------------------
    # Calculate epistemic uncertainty.
    # --------------------------------------------------------------

    with torch.no_grad():

        epistemic_variance = (
            uncertainty_estimator.epistemic_variance(
                reconstruction_samples
            )
        )

    # --------------------------------------------------------------
    # Calculate predictive uncertainty.
    #
    # IMPORTANT:
    # This follows the uncertainty calculation already validated
    # in the project's proposed-model implementation.
    # --------------------------------------------------------------

    with torch.no_grad():

        predictive_variance = (
            uncertainty_estimator.predictive_variance(
                log_variance_samples,
                reconstruction_samples,
            )
        )

    # --------------------------------------------------------------
    # Convert predictive variance to predictive standard deviation.
    # --------------------------------------------------------------

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0,
        )
    )

    # --------------------------------------------------------------
    # Mean reconstruction across MC samples.
    # --------------------------------------------------------------

    reconstruction = (
        reconstruction_samples.mean(dim=0)
    )

    # --------------------------------------------------------------
    # Data-consistency projection.
    #
    # Observed seismic samples are restored exactly from the
    # corrupted input.
    # --------------------------------------------------------------

    reconstruction = torch.where(
        mask_batch == 1,
        corrupted_batch,
        reconstruction,
    )

    # --------------------------------------------------------------
    # Validate reconstruction shape.
    # --------------------------------------------------------------

    check(
        tuple(reconstruction.shape)
        == expected_batch_shape,
        (
            "Final reconstruction has incorrect shape: "
            f"{tuple(reconstruction.shape)}"
        )
    )

    # --------------------------------------------------------------
    # Validate finite reconstruction.
    # --------------------------------------------------------------

    check(
        torch.isfinite(reconstruction).all().item(),
        "Final reconstruction contains NaN or Inf values."
    )

    # --------------------------------------------------------------
    # Validate uncertainty tensors.
    # --------------------------------------------------------------

    check(
        torch.isfinite(aleatoric_variance).all().item(),
        "Aleatoric variance contains NaN or Inf values."
    )

    check(
        torch.isfinite(epistemic_variance).all().item(),
        "Epistemic variance contains NaN or Inf values."
    )

    check(
        torch.isfinite(predictive_variance).all().item(),
        "Predictive variance contains NaN or Inf values."
    )

    check(
        torch.isfinite(predictive_std).all().item(),
        "Predictive standard deviation contains NaN or Inf values."
    )

    # --------------------------------------------------------------
    # Variances must not be negative.
    # --------------------------------------------------------------

    check(
        torch.min(aleatoric_variance).item() >= -1e-8,
        "Aleatoric variance contains negative values."
    )

    check(
        torch.min(epistemic_variance).item() >= -1e-8,
        "Epistemic variance contains negative values."
    )

    check(
        torch.min(predictive_variance).item() >= -1e-8,
        "Predictive variance contains negative values."
    )

    # --------------------------------------------------------------
    # Predictive standard deviation must be non-negative.
    # --------------------------------------------------------------

    check(
        torch.min(predictive_std).item() >= -1e-8,
        "Predictive standard deviation contains negative values."
    )

    # --------------------------------------------------------------
    # Check observed-data preservation after reconstruction.
    # --------------------------------------------------------------

    observed_reconstruction_error = torch.abs(
        reconstruction[mask_batch == 1]
        -
        corrupted_batch[mask_batch == 1]
    )

    max_observed_error = (
        torch.max(observed_reconstruction_error)
        .item()
    )

    check(
        max_observed_error
        <= OBSERVED_PRESERVATION_TOLERANCE,
        (
            "Observed-data preservation failed. "
            f"Maximum error = {max_observed_error:.12e}; "
            f"allowed tolerance = "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.12e}"
        )
    )

    # --------------------------------------------------------------
    # Calculate reconstruction metrics.
    # --------------------------------------------------------------

    reconstruction_cpu = reconstruction.detach().cpu()

    target_cpu = target_batch.detach().cpu()

    mae_value = mae(
        reconstruction_cpu,
        target_cpu,
    )

    rmse_value = rmse(
        reconstruction_cpu,
        target_cpu,
    )

    psnr_value = psnr(
        reconstruction_cpu,
        target_cpu,
    )

    snr_value = snr(
        reconstruction_cpu,
        target_cpu,
    )

    ssim_value = ssim(
        reconstruction_cpu,
        target_cpu,
    )

    # --------------------------------------------------------------
    # Calculate missing-region metrics.
    # --------------------------------------------------------------

    missing_mask_batch = (
        mask_batch == 0
    )

    missing_difference = torch.abs(
        reconstruction[missing_mask_batch]
        -
        target_batch[missing_mask_batch]
    )

    missing_mae_value = (
        missing_difference.mean().item()
    )

    missing_rmse_value = torch.sqrt(
        torch.mean(
            (
                reconstruction[missing_mask_batch]
                -
                target_batch[missing_mask_batch]
            ) ** 2
        )
    ).item()

    # --------------------------------------------------------------
    # Calculate uncertainty summaries.
    # --------------------------------------------------------------

    predictive_std_mean = (
        predictive_std.mean().item()
    )

    predictive_std_missing_mean = (
        predictive_std[
            missing_mask_batch
        ]
        .mean()
        .item()
    )

    aleatoric_variance_mean = (
        aleatoric_variance.mean().item()
    )

    epistemic_variance_mean = (
        epistemic_variance.mean().item()
    )

    predictive_variance_mean = (
        predictive_variance.mean().item()
    )

    # --------------------------------------------------------------
    # Verify that the missing-region reconstruction actually differs
    # from the corrupted input where missing values exist.
    #
    # This prevents a trivial "do nothing" reconstruction from
    # passing the test.
    # --------------------------------------------------------------

    missing_input_difference = torch.abs(
        reconstruction[missing_mask_batch]
        -
        corrupted_batch[missing_mask_batch]
    )

    missing_changed_count = int(
        (
            missing_input_difference > 1e-8
        ).sum().item()
    )

    check(
        missing_changed_count > 0,
        (
            "The proposed model did not modify any missing-region "
            "samples."
        )
    )

    # --------------------------------------------------------------
    # Return diagnostic information.
    # --------------------------------------------------------------

    return {
        "mae": float(mae_value),
        "rmse": float(rmse_value),
        "psnr": float(psnr_value),
        "snr": float(snr_value),
        "ssim": float(ssim_value),
        "missing_mae": float(missing_mae_value),
        "missing_rmse": float(missing_rmse_value),
        "observed_count": observed_count,
        "missing_count": missing_count,
        "max_observed_error": max_observed_error,
        "aleatoric_variance_mean": (
            aleatoric_variance_mean
        ),
        "epistemic_variance_mean": (
            epistemic_variance_mean
        ),
        "predictive_variance_mean": (
            predictive_variance_mean
        ),
        "predictive_std_mean": (
            predictive_std_mean
        ),
        "predictive_std_missing_mean": (
            predictive_std_missing_mean
        ),
    }


# ==================================================================
# 7. REPRODUCIBILITY TEST
# ==================================================================

def test_reproducibility(model, device):
    """
    Verify that resetting the random seed produces the same
    synthetic input and reconstruction.
    """

    # --------------------------------------------------------------
    # First run.
    # --------------------------------------------------------------

    set_seed(BASELINE_SEED)

    dataset_1 = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=BASELINE_MISSING_RATE,
        geological_mode=BASELINE_GEOLOGICAL_MODE,
        mask_mode=BASELINE_MASK_MODE,
        seed=BASELINE_SEED,
    )

    sample_1 = dataset_1[0]

    corrupted_1 = sample_1[0].to(device)
    target_1 = sample_1[1].to(device)
    mask_1 = sample_1[2].to(device)
    velocity_1 = sample_1[3].to(device)

    input_1 = corrupted_1.unsqueeze(0)
    mask_batch_1 = mask_1.unsqueeze(0)
    velocity_batch_1 = velocity_1.unsqueeze(0)

    mc_predictor_1 = MCDropout3D(
        model,
        num_samples=MC_DROPOUT_SAMPLES,
    )

    with torch.no_grad():

        (
            reconstruction_samples_1,
            _,
            _,
        ) = mc_predictor_1(
            input_1,
            mask_batch_1,
            velocity_batch_1,
        )

    reconstruction_1 = (
        reconstruction_samples_1.mean(dim=0)
    )

    reconstruction_1 = torch.where(
        mask_batch_1 == 1,
        input_1,
        reconstruction_1,
    )

    # --------------------------------------------------------------
    # Second run.
    # --------------------------------------------------------------

    set_seed(BASELINE_SEED)

    dataset_2 = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=BASELINE_MISSING_RATE,
        geological_mode=BASELINE_GEOLOGICAL_MODE,
        mask_mode=BASELINE_MASK_MODE,
        seed=BASELINE_SEED,
    )

    sample_2 = dataset_2[0]

    corrupted_2 = sample_2[0].to(device)
    target_2 = sample_2[1].to(device)
    mask_2 = sample_2[2].to(device)
    velocity_2 = sample_2[3].to(device)

    input_2 = corrupted_2.unsqueeze(0)
    mask_batch_2 = mask_2.unsqueeze(0)
    velocity_batch_2 = velocity_2.unsqueeze(0)

    mc_predictor_2 = MCDropout3D(
        model,
        num_samples=MC_DROPOUT_SAMPLES,
    )

    with torch.no_grad():

        (
            reconstruction_samples_2,
            _,
            _,
        ) = mc_predictor_2(
            input_2,
            mask_batch_2,
            velocity_batch_2,
        )

    reconstruction_2 = (
        reconstruction_samples_2.mean(dim=0)
    )

    reconstruction_2 = torch.where(
        mask_batch_2 == 1,
        input_2,
        reconstruction_2,
    )

    # --------------------------------------------------------------
    # Compare generated input.
    # --------------------------------------------------------------

    check(
        torch.equal(
            corrupted_1.cpu(),
            corrupted_2.cpu(),
        ),
        "Synthetic dataset is not reproducible."
    )

    # --------------------------------------------------------------
    # Compare target.
    # --------------------------------------------------------------

    check(
        torch.equal(
            target_1.cpu(),
            target_2.cpu(),
        ),
        "Synthetic target is not reproducible."
    )

    # --------------------------------------------------------------
    # Compare masks.
    # --------------------------------------------------------------

    check(
        torch.equal(
            mask_1.cpu(),
            mask_2.cpu(),
        ),
        "Synthetic mask is not reproducible."
    )

    # --------------------------------------------------------------
    # Compare velocity.
    # --------------------------------------------------------------

    check(
        torch.equal(
            velocity_1.cpu(),
            velocity_2.cpu(),
        ),
        "Synthetic velocity model is not reproducible."
    )

    # --------------------------------------------------------------
    # Compare model reconstruction.
    # --------------------------------------------------------------

    reconstruction_difference = torch.max(
        torch.abs(
            reconstruction_1 -
            reconstruction_2
        )
    ).item()

    check(
        reconstruction_difference
        <= 1e-6,
        (
            "Proposed-model reconstruction is not reproducible. "
            f"Maximum difference = "
            f"{reconstruction_difference:.12e}"
        )
    )


# ==================================================================
# 8. MAIN TEST
# ==================================================================

def main():
    """
    Execute the complete focused proposed-model test.
    """

    print("=" * 70)

    print(
        "FOCUSED TEST — PROPOSED MODEL"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------------
    # Display test configuration.
    # --------------------------------------------------------------

    print(
        f"Cube size:              {BASELINE_CUBE_SIZE}"
    )

    print(
        f"Missing rate:           {BASELINE_MISSING_RATE}"
    )

    print(
        f"Geological mode:        {BASELINE_GEOLOGICAL_MODE}"
    )

    print(
        f"Mask mode:              {BASELINE_MASK_MODE}"
    )

    print(
        f"Seed:                    {BASELINE_SEED}"
    )

    print(
        f"MC-Dropout samples:     {MC_DROPOUT_SAMPLES}"
    )

    # --------------------------------------------------------------
    # Resolve device.
    # --------------------------------------------------------------

    device = resolve_device()

    print(
        f"Device:                  {device}"
    )

    print()

    # --------------------------------------------------------------
    # Set deterministic seed.
    # --------------------------------------------------------------

    set_seed(BASELINE_SEED)

    # --------------------------------------------------------------
    # Load trained proposed model.
    # --------------------------------------------------------------

    print(
        "Loading proposed-model checkpoint..."
    )

    model = load_proposed_model(device)

    print(
        "Checkpoint loaded successfully."
    )

    print()

    # --------------------------------------------------------------
    # Run reconstruction and uncertainty validation.
    # --------------------------------------------------------------

    print(
        "Running focused proposed-model case..."
    )

    results = run_proposed_model_case(
        model,
        device,
    )

    print(
        "Proposed-model reconstruction passed."
    )

    print()

    # --------------------------------------------------------------
    # Reproducibility validation.
    # --------------------------------------------------------------

    print(
        "Testing reproducibility..."
    )

    test_reproducibility(
        model,
        device,
    )

    print(
        "Reproducibility passed."
    )

    print()

    # --------------------------------------------------------------
    # Display results.
    # --------------------------------------------------------------

    print("-" * 70)

    print(
        "RECONSTRUCTION RESULTS"
    )

    print("-" * 70)

    print(
        f"MAE:                         "
        f"{results['mae']:.6f}"
    )

    print(
        f"RMSE:                        "
        f"{results['rmse']:.6f}"
    )

    print(
        f"PSNR:                        "
        f"{results['psnr']:.6f}"
    )

    print(
        f"SNR:                         "
        f"{results['snr']:.6f}"
    )

    print(
        f"SSIM:                        "
        f"{results['ssim']:.6f}"
    )

    print(
        f"Missing-region MAE:         "
        f"{results['missing_mae']:.6f}"
    )

    print(
        f"Missing-region RMSE:        "
        f"{results['missing_rmse']:.6f}"
    )

    print()

    print("-" * 70)

    print(
        "UNCERTAINTY RESULTS"
    )

    print("-" * 70)

    print(
        f"Mean aleatoric variance:     "
        f"{results['aleatoric_variance_mean']:.6f}"
    )

    print(
        f"Mean epistemic variance:     "
        f"{results['epistemic_variance_mean']:.6f}"
    )

    print(
        f"Mean predictive variance:    "
        f"{results['predictive_variance_mean']:.6f}"
    )

    print(
        f"Mean predictive std:         "
        f"{results['predictive_std_mean']:.6f}"
    )

    print(
        f"Missing predictive std:     "
        f"{results['predictive_std_missing_mean']:.6f}"
    )

    print()

    print("-" * 70)

    print(
        "DATA-CONSISTENCY RESULTS"
    )

    print("-" * 70)

    print(
        f"Observed samples:            "
        f"{results['observed_count']}"
    )

    print(
        f"Missing samples:             "
        f"{results['missing_count']}"
    )

    print(
        f"Maximum observed error:      "
        f"{results['max_observed_error']:.12e}"
    )

    print()

    # --------------------------------------------------------------
    # Final PASS.
    # --------------------------------------------------------------

    print("=" * 70)

    print(
        "PASS — Proposed-model focused test completed successfully."
    )

    print("=" * 70)


# ==================================================================
# 9. PYTHON ENTRY POINT
# ==================================================================

if __name__ == "__main__":

    main()