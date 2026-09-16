"""
=========================================================
Dictionary Learning Baseline - Single Sample Test
=========================================================

Validates the dictionary-learning reconstruction baseline
using one controlled synthetic seismic sample.

Author: Ormin Joseph
=========================================================
"""

import time

import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.dictionary_learning import (
    dictionary_learning_reconstruction,
)


# =========================================================
# Configuration
# =========================================================

CUBE_SIZE = (64, 128, 128)

MISSING_RATE = 0.30

SEED = 42

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"


# Dictionary-learning parameters
PATCH_SIZE = (8, 8, 8)

N_COMPONENTS = 64

ALPHA = 1.0

MAX_ITER = 20

BATCH_SIZE = 64

MAX_TRAINING_PATCHES = 2000

MIN_OBSERVED_FRACTION = 0.80


# =========================================================
# Metric functions
# =========================================================

def mae(prediction, target):
    """
    Mean Absolute Error.
    """

    return torch.mean(
        torch.abs(prediction - target)
    ).item()


def rmse(prediction, target):
    """
    Root Mean Square Error.
    """

    return torch.sqrt(
        torch.mean(
            (prediction - target) ** 2
        )
    ).item()


def psnr(prediction, target, data_range=2.0):
    """
    Peak Signal-to-Noise Ratio.

    Data are normalized approximately to [-1, 1],
    therefore the expected data range is 2.
    """

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0.0:
        return float("inf")

    return (
        10.0
        * torch.log10(
            torch.tensor(
                (data_range ** 2) / mse
            )
        )
    ).item()


def snr(prediction, target):
    """
    Signal-to-Noise Ratio.
    """

    signal_power = torch.sum(
        target ** 2
    )

    noise_power = torch.sum(
        (target - prediction) ** 2
    )

    if noise_power.item() == 0.0:
        return float("inf")

    return (
        10.0
        * torch.log10(
            signal_power / noise_power
        )
    ).item()


def ssim_simple(
    prediction,
    target,
    data_range=2.0,
):
    """
    Simple global SSIM-style measure used for the
    baseline validation test.

    The final unified evaluation pipeline should use
    the project's standard SSIM implementation.
    """

    x = prediction.float()
    y = target.float()

    mu_x = torch.mean(x)
    mu_y = torch.mean(y)

    sigma_x = torch.var(
        x,
        unbiased=False,
    )

    sigma_y = torch.var(
        y,
        unbiased=False,
    )

    covariance = torch.mean(
        (x - mu_x) * (y - mu_y)
    )

    c1 = (
        0.01 * data_range
    ) ** 2

    c2 = (
        0.03 * data_range
    ) ** 2

    numerator = (
        (2 * mu_x * mu_y + c1)
        * (2 * covariance + c2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + c1)
        * (sigma_x + sigma_y + c2)
    )

    return (
        numerator / denominator
    ).item()


# =========================================================
# Main test
# =========================================================

def main():

    print("=" * 78)
    print("DICTIONARY LEARNING BASELINE - SINGLE SAMPLE TEST")
    print("=" * 78)

    print()
    print("Configuration")
    print("-" * 78)

    print(
        f"Cube size              : {CUBE_SIZE}"
    )

    print(
        f"Missing rate           : {MISSING_RATE}"
    )

    print(
        f"Seed                   : {SEED}"
    )

    print(
        f"Geological mode        : {GEOLOGICAL_MODE}"
    )

    print(
        f"Mask mode              : {MASK_MODE}"
    )

    print(
        f"Patch size             : {PATCH_SIZE}"
    )

    print(
        f"Dictionary components  : {N_COMPONENTS}"
    )

    print(
        f"Alpha                  : {ALPHA}"
    )

    print(
        f"Maximum iterations     : {MAX_ITER}"
    )

    print()


    # =====================================================
    # Create synthetic dataset
    # =====================================================

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        seed=SEED,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
    )


    # =====================================================
    # Obtain one sample
    # =====================================================

    sample = dataset[0]

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = sample


    # =====================================================
    # Display sample information
    # =====================================================

    print("Sample Information")
    print("-" * 78)

    print(
        f"Input shape      : {tuple(corrupted.shape)}"
    )

    print(
        f"Target shape     : {tuple(target.shape)}"
    )

    print(
        f"Mask shape       : {tuple(mask.shape)}"
    )

    print(
        f"Velocity shape   : {tuple(velocity.shape)}"
    )

    print(
        f"Mask type        : {actual_mask_mode}"
    )

    print(
        f"Geological mode  : {actual_geological_mode}"
    )

    print()


    # =====================================================
    # Shape validation
    # =====================================================

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    if tuple(corrupted.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected input shape: "
            f"{tuple(corrupted.shape)}. "
            f"Expected: {expected_shape}"
        )

    if tuple(target.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected target shape: "
            f"{tuple(target.shape)}. "
            f"Expected: {expected_shape}"
        )

    if tuple(mask.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected mask shape: "
            f"{tuple(mask.shape)}. "
            f"Expected: {expected_shape}"
        )

    print("Shape validation : PASS")


    # =====================================================
    # Mask validation
    # =====================================================

    unique_mask = torch.unique(mask)

    print(
        f"Mask values      : {unique_mask.tolist()}"
    )

    if not torch.all(
        (unique_mask == 0)
        | (unique_mask == 1)
    ):
        raise RuntimeError(
            "Mask contains values other than 0 and 1."
        )

    print("Mask validation  : PASS")


    # =====================================================
    # Input consistency
    # =====================================================

    expected_corrupted = (
        target * mask
    )

    input_error = torch.max(
        torch.abs(
            corrupted
            - expected_corrupted
        )
    ).item()

    print(
        f"Input consistency error : {input_error:.6e}"
    )

    if input_error > 1e-6:
        raise RuntimeError(
            "Corrupted input is inconsistent with "
            "target and mask."
        )

    print(
        "Input consistency       : PASS"
    )


    # =====================================================
    # Count observed and missing samples
    # =====================================================

    observed_samples = int(
        torch.sum(mask == 1).item()
    )

    missing_samples = int(
        torch.sum(mask == 0).item()
    )

    print(
        f"Observed samples : {observed_samples}"
    )

    print(
        f"Missing samples  : {missing_samples}"
    )

    print()


    # =====================================================
    # Run dictionary learning reconstruction
    # =====================================================

    print(
        "Running Dictionary Learning reconstruction..."
    )

    start_time = time.perf_counter()

    reconstruction = (
        dictionary_learning_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            patch_size=PATCH_SIZE,
            n_components=N_COMPONENTS,
            alpha=ALPHA,
            max_iter=MAX_ITER,
            batch_size=BATCH_SIZE,
            max_training_patches=MAX_TRAINING_PATCHES,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            random_state=SEED,
        )
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    print(
        "Dictionary Learning reconstruction completed."
    )

    print()


    # =====================================================
    # Output validation
    # =====================================================

    if reconstruction.shape != corrupted.shape:
        raise RuntimeError(
            "Reconstruction shape mismatch."
        )

    print(
        "Output shape validation : PASS"
    )


    # =====================================================
    # Finite-value validation
    # =====================================================

    if not torch.isfinite(
        reconstruction
    ).all():

        raise RuntimeError(
            "Reconstruction contains NaN or infinite values."
        )

    print(
        "Finite-value validation : PASS"
    )


    # =====================================================
    # Observed-data preservation
    # =====================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            - corrupted[mask == 1]
        )
    ).item()

    print(
        "Maximum observed-data difference : "
        f"{observed_difference:.6e}"
    )

    if observed_difference > 0.0:
        raise RuntimeError(
            "Observed-data preservation failed."
        )

    print(
        "Observed-data preservation : PASS"
    )


    # =====================================================
    # Missing-sample reconstruction validation
    # =====================================================

    zero_filled_error = torch.mean(
        torch.abs(
            corrupted[mask == 0]
            - target[mask == 0]
        )
    ).item()

    reconstructed_error = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - target[mask == 0]
        )
    ).item()

    print(
        f"Zero-filled missing MAE       : "
        f"{zero_filled_error:.6f}"
    )

    print(
        f"Dictionary missing MAE        : "
        f"{reconstructed_error:.6f}"
    )

    missing_change = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - corrupted[mask == 0]
        )
    ).item()

    print(
        f"Missing-sample reconstruction change : "
        f"{missing_change:.6e}"
    )

    if missing_change <= 0.0:
        raise RuntimeError(
            "Dictionary Learning did not modify "
            "the missing samples."
        )

    print(
        "Missing-sample reconstruction : PASS"
    )


    # =====================================================
    # Global metrics
    # =====================================================

    metric_mae = mae(
        reconstruction,
        target,
    )

    metric_rmse = rmse(
        reconstruction,
        target,
    )

    metric_psnr = psnr(
        reconstruction,
        target,
    )

    metric_snr = snr(
        reconstruction,
        target,
    )

    metric_ssim = ssim_simple(
        reconstruction,
        target,
    )


    # =====================================================
    # Display metrics
    # =====================================================

    print()
    print("Reconstruction Metrics")
    print("-" * 78)

    print(
        f"MAE  : {metric_mae:.6f}"
    )

    print(
        f"RMSE : {metric_rmse:.6f}"
    )

    print(
        f"PSNR : {metric_psnr:.6f} dB"
    )

    print(
        f"SNR  : {metric_snr:.6f} dB"
    )

    print(
        f"SSIM : {metric_ssim:.6f}"
    )

    print(
        f"Runtime : {elapsed_time:.4f} s"
    )


    # =====================================================
    # Reproducibility test
    # =====================================================

    print()
    print(
        "Running reproducibility test..."
    )

    reconstruction_repeat = (
        dictionary_learning_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            patch_size=PATCH_SIZE,
            n_components=N_COMPONENTS,
            alpha=ALPHA,
            max_iter=MAX_ITER,
            batch_size=BATCH_SIZE,
            max_training_patches=MAX_TRAINING_PATCHES,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            random_state=SEED,
        )
    )

    reproducibility_difference = torch.max(
        torch.abs(
            reconstruction
            - reconstruction_repeat
        )
    ).item()

    print(
        "Reconstruction reproducibility "
        f"difference : {reproducibility_difference:.6e}"
    )

    if reproducibility_difference > 1e-6:
        raise RuntimeError(
            "Dictionary Learning is not reproducible."
        )

    print(
        "Reproducibility : PASS"
    )


    # =====================================================
    # Final status
    # =====================================================

    print()
    print("=" * 78)
    print(
        "DICTIONARY LEARNING BASELINE VALIDATION COMPLETE"
    )
    print("=" * 78)

    print(
        "Overall status : PASS"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()