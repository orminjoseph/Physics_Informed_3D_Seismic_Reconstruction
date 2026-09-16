"""
=============================================================
Test: f-x Prediction Baseline
=============================================================

Validates the independent classical f-x prediction baseline.

Checks
------
1. Synthetic dataset compatibility
2. Output shape
3. Numerical validity
4. Observed-data preservation
5. Reconstruction of missing samples
6. Reproducibility
7. Basic reconstruction metrics

Author: Ormin Joseph
=============================================================
"""

import torch

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.fx_prediction import (
    fx_prediction_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_SAMPLES = 1

PATCH_SIZE = (
    64,
    128,
    128
)

MISSING_PROBABILITY = 0.30


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("F-X PREDICTION BASELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Set deterministic seed.
    # --------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    # --------------------------------------------------------
    # Create synthetic dataset.
    # --------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    print()
    print(
        f"Dataset samples : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Obtain one deterministic sample.
    # --------------------------------------------------------

    (
        corrupted,
        target,
        mask,
        velocity_model,
        mask_type,
        geological_mode,
    ) = dataset[0]

    print(
        "Input shape      :",
        tuple(corrupted.shape)
    )

    print(
        "Target shape     :",
        tuple(target.shape)
    )

    print(
        "Mask shape       :",
        tuple(mask.shape)
    )

    # ========================================================
    # F-X RECONSTRUCTION
    # ========================================================

    print()
    print(
        "Running f-x prediction..."
    )

    reconstruction = (
        fx_prediction_reconstruction(
            corrupted,
            mask,
            prediction_order=4,
            iterations=2
        )
    )

    # ========================================================
    # SHAPE VALIDATION
    # ========================================================

    assert (
        reconstruction.shape
        ==
        corrupted.shape
    )

    print(
        "Output shape validation : PASSED"
    )

    # ========================================================
    # NUMERICAL VALIDATION
    # ========================================================

    assert torch.isfinite(
        reconstruction
    ).all()

    print(
        "Finite-value validation : PASSED"
    )

    # ========================================================
    # OBSERVED SAMPLE PRESERVATION
    # ========================================================

    observed_difference = (
        reconstruction[mask == 1]
        -
        corrupted[mask == 1]
    ).abs().max().item()

    print(
        "Maximum observed-data "
        f"difference : {observed_difference:.6e}"
    )

    assert (
        observed_difference
        <
        1e-6
    )

    print(
        "Observed-data preservation : PASSED"
    )

    # ========================================================
    # MISSING SAMPLE CHECK
    # ========================================================

    missing_count = int(
        (mask == 0).sum().item()
    )

    print(
        "Missing samples :",
        missing_count
    )

    assert (
        missing_count > 0
    )

    missing_values = (
        reconstruction[mask == 0]
    )

    assert torch.isfinite(
        missing_values
    ).all()

    print(
        "Missing-sample reconstruction : PASSED"
    )

    # ========================================================
    # RECONSTRUCTION METRICS
    # ========================================================

    prediction = (
        reconstruction.unsqueeze(0)
    )

    reference = (
        target.unsqueeze(0)
    )

    mae_value = mae(
        prediction,
        reference
    ).item()

    rmse_value = rmse(
        prediction,
        reference
    ).item()

    psnr_value = psnr(
        prediction,
        reference
    ).item()

    snr_value = snr(
        prediction,
        reference
    ).item()

    ssim_value = ssim(
        prediction,
        reference
    ).item()

    print()
    print("=" * 70)
    print("F-X RECONSTRUCTION METRICS")
    print("=" * 70)

    print(
        f"MAE  : {mae_value:.6f}"
    )

    print(
        f"RMSE : {rmse_value:.6f}"
    )

    print(
        f"PSNR : {psnr_value:.6f} dB"
    )

    print(
        f"SNR  : {snr_value:.6f} dB"
    )

    print(
        f"SSIM : {ssim_value:.6f}"
    )

    # ========================================================
    # REPRODUCIBILITY TEST
    # ========================================================

    reconstruction_2 = (
        fx_prediction_reconstruction(
            corrupted,
            mask,
            prediction_order=4,
            iterations=2
        )
    )

    reproducibility_error = (
        reconstruction
        -
        reconstruction_2
    ).abs().max().item()

    print()
    print(
        "Reproducibility difference : "
        f"{reproducibility_error:.6e}"
    )

    assert (
        reproducibility_error
        <
        1e-7
    )

    print(
        "Reproducibility : PASSED"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ALL F-X PREDICTION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()