"""
======================================================================
CURVELET POCS RECONSTRUCTION VALIDATION
======================================================================

Implementation-level validation of the 3-D Curvelet POCS baseline.

This test is performed before the full controlled experimental matrix.

Validation includes:
    1. Input shape validation
    2. Mask validation
    3. Reconstruction execution
    4. Output shape validation
    5. Finite-value validation
    6. Exact observed-data preservation
    7. Missing-sample reconstruction
    8. Reconstruction metrics
    9. Reproducibility

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import sys
from pathlib import Path

import numpy as np
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

from evaluation.baselines.curvelet_pocs import (
    curvelet_pocs_reconstruction,
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =====================================================================
# TEST CONFIGURATION
# =====================================================================

CUBE_SIZE = (
    32,
    32,
    32,
)

NUM_SAMPLES = 1

MISSING_RATE = 0.30

SEED = 42

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"

# ---------------------------------------------------------------------
# Curvelet POCS parameters.
#
# These are deliberately modest for the implementation-level test.
# The same parameters will later be frozen for the controlled matrix.
# ---------------------------------------------------------------------

CURVELET_NUM_SCALES = 3

CURVELET_WEDGES_PER_DIRECTION = 3

CURVELET_ITERATIONS = 12

CURVELET_THRESHOLD = 0.05

CURVELET_THRESHOLD_DECAY = 0.90

CURVELET_TOLERANCE = 1.0e-5


# =====================================================================
# MAIN TEST
# =====================================================================

def main():

    print("=" * 70)
    print("3-D CURVELET POCS RECONSTRUCTION VALIDATION")
    print("=" * 70)

    # ================================================================
    # CONFIGURATION
    # ================================================================

    print("\nTest Configuration")
    print("------------------")

    print(
        f"Cube size              : {CUBE_SIZE}"
    )

    print(
        f"Missing rate           : "
        f"{MISSING_RATE}"
    )

    print(
        f"Seed                   : {SEED}"
    )

    print(
        f"Geological mode        : "
        f"{GEOLOGICAL_MODE}"
    )

    print(
        f"Mask mode              : "
        f"{MASK_MODE}"
    )

    print(
        f"UDCT scales            : "
        f"{CURVELET_NUM_SCALES}"
    )

    print(
        f"Wedges per direction  : "
        f"{CURVELET_WEDGES_PER_DIRECTION}"
    )

    print(
        f"POCS iterations        : "
        f"{CURVELET_ITERATIONS}"
    )

    print(
        f"Initial threshold      : "
        f"{CURVELET_THRESHOLD}"
    )

    print(
        f"Threshold decay        : "
        f"{CURVELET_THRESHOLD_DECAY}"
    )

    # ================================================================
    # REPRODUCIBILITY
    # ================================================================

    torch.manual_seed(SEED)

    np.random.seed(SEED)

    # ================================================================
    # CREATE SYNTHETIC DATASET
    # ================================================================

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        seed=SEED,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
    )

    # ================================================================
    # GET ONE CONTROLLED SAMPLE
    # ================================================================

    sample = dataset[0]

    (
        corrupted,
        target,
        mask,
        velocity_model,
        actual_mask_type,
        actual_geological_mode,
    ) = sample

    # ================================================================
    # DISPLAY SAMPLE INFORMATION
    # ================================================================

    print("\nSample Information")
    print("-------------------")

    print(
        f"Input shape      : "
        f"{tuple(corrupted.shape)}"
    )

    print(
        f"Target shape     : "
        f"{tuple(target.shape)}"
    )

    print(
        f"Mask shape       : "
        f"{tuple(mask.shape)}"
    )

    print(
        f"Velocity shape   : "
        f"{tuple(velocity_model.shape)}"
    )

    print(
        f"Mask type        : "
        f"{actual_mask_type}"
    )

    print(
        f"Geological mode  : "
        f"{actual_geological_mode}"
    )

    # ================================================================
    # SHAPE VALIDATION
    # ================================================================

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

    print(
        "\nShape validation : PASS"
    )

    # ================================================================
    # MASK VALIDATION
    # ================================================================

    unique_mask_values = torch.unique(mask)

    print(
        f"Mask values      : "
        f"{unique_mask_values.tolist()}"
    )

    if not torch.all(
        (unique_mask_values == 0)
        | (unique_mask_values == 1)
    ):

        raise RuntimeError(
            "Mask is not binary."
        )

    print(
        "Mask validation  : PASS"
    )

    # ================================================================
    # INPUT CONSISTENCY
    #
    # The corrupted volume must contain the target values at observed
    # locations and zeros at missing locations.
    # ================================================================

    input_consistency_error = (
        (
            corrupted
            - target * mask
        )
        .abs()
        .max()
        .item()
    )

    print(
        f"\nInput consistency error : "
        f"{input_consistency_error:.6e}"
    )

    if input_consistency_error > 1.0e-6:

        raise RuntimeError(
            "Input consistency validation failed."
        )

    print(
        "Input consistency       : PASS"
    )

    # ================================================================
    # COUNT OBSERVED / MISSING SAMPLES
    # ================================================================

    observed_samples = int(
        mask.sum().item()
    )

    total_samples = int(
        mask.numel()
    )

    missing_samples = (
        total_samples
        - observed_samples
    )

    print(
        f"\nObserved samples : "
        f"{observed_samples}"
    )

    print(
        f"Missing samples  : "
        f"{missing_samples}"
    )

    if missing_samples == 0:

        raise RuntimeError(
            "Test sample contains no missing samples."
        )

    # ================================================================
    # CURVELET POCS RECONSTRUCTION
    # ================================================================

    print(
        "\nRunning Curvelet POCS reconstruction..."
    )

    reconstructed = (
        curvelet_pocs_reconstruction(
            corrupted_cube=corrupted,
            mask=mask,
            num_scales=CURVELET_NUM_SCALES,
            wedges_per_direction=(
                CURVELET_WEDGES_PER_DIRECTION
            ),
            iterations=CURVELET_ITERATIONS,
            threshold=CURVELET_THRESHOLD,
            threshold_decay=(
                CURVELET_THRESHOLD_DECAY
            ),
            tolerance=CURVELET_TOLERANCE,
        )
    )

    print(
        "Curvelet POCS reconstruction completed."
    )

    # ================================================================
    # OUTPUT SHAPE
    # ================================================================

    if reconstructed.shape != target.shape:

        raise RuntimeError(
            "Reconstruction shape does not match "
            "target shape.\n"
            f"Reconstruction: "
            f"{tuple(reconstructed.shape)}\n"
            f"Target: "
            f"{tuple(target.shape)}"
        )

    print(
        "Output shape validation : PASS"
    )

    # ================================================================
    # FINITE VALUE VALIDATION
    # ================================================================

    if not torch.isfinite(
        reconstructed
    ).all():

        raise RuntimeError(
            "Reconstruction contains NaN "
            "or infinite values."
        )

    print(
        "Finite-value validation : PASS"
    )

    # ================================================================
    # OBSERVED-DATA PRESERVATION
    # ================================================================

    observed_difference = (
        (
            reconstructed
            - corrupted
        )
        * mask
    ).abs().max().item()

    print(
        f"Maximum observed-data difference : "
        f"{observed_difference:.6e}"
    )

    if observed_difference > 1.0e-6:

        raise RuntimeError(
            "Observed seismic samples were not "
            "preserved exactly."
        )

    print(
        "Observed-data preservation : PASS"
    )

    # ================================================================
    # MISSING-SAMPLE RECONSTRUCTION
    # ================================================================

    missing_difference_from_input = (
        (
            reconstructed
            - corrupted
        )
        * (1.0 - mask)
    ).abs().sum().item()

    print(
        f"Missing-sample reconstruction change : "
        f"{missing_difference_from_input:.6e}"
    )

    if missing_difference_from_input <= 1.0e-8:

        raise RuntimeError(
            "Reconstruction did not modify "
            "missing samples."
        )

    print(
        "Missing-sample reconstruction : PASS"
    )

    # ================================================================
    # METRICS
    # ================================================================

    reconstructed_for_metrics = (
        reconstructed
        .unsqueeze(0)
        if reconstructed.ndim == 4
        else reconstructed
    )

    target_for_metrics = (
        target
        .unsqueeze(0)
        if target.ndim == 4
        else target
    )

    metric_mae = mae(
        reconstructed_for_metrics,
        target_for_metrics,
    )

    metric_rmse = rmse(
        reconstructed_for_metrics,
        target_for_metrics,
    )

    metric_psnr = psnr(
        reconstructed_for_metrics,
        target_for_metrics,
    )

    metric_snr = snr(
        reconstructed_for_metrics,
        target_for_metrics,
    )

    metric_ssim = ssim(
        reconstructed_for_metrics,
        target_for_metrics,
    )

    print("\nReconstruction Metrics")
    print("----------------------")

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

    # ================================================================
    # REPRODUCIBILITY TEST
    # ================================================================

    print(
        "\nRunning reproducibility test..."
    )

    torch.manual_seed(SEED)

    np.random.seed(SEED)

    dataset_2 = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        seed=SEED,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
    )

    sample_2 = dataset_2[0]

    corrupted_2 = sample_2[0]

    target_2 = sample_2[1]

    mask_2 = sample_2[2]

    reconstructed_2 = (
        curvelet_pocs_reconstruction(
            corrupted_cube=corrupted_2,
            mask=mask_2,
            num_scales=CURVELET_NUM_SCALES,
            wedges_per_direction=(
                CURVELET_WEDGES_PER_DIRECTION
            ),
            iterations=CURVELET_ITERATIONS,
            threshold=CURVELET_THRESHOLD,
            threshold_decay=(
                CURVELET_THRESHOLD_DECAY
            ),
            tolerance=CURVELET_TOLERANCE,
        )
    )

    reconstruction_difference = (
        reconstructed
        - reconstructed_2
    ).abs().max().item()

    print(
        f"Reconstruction reproducibility difference : "
        f"{reconstruction_difference:.6e}"
    )

    if reconstruction_difference > 1.0e-6:

        raise RuntimeError(
            "Curvelet POCS reconstruction is not "
            "reproducible."
        )

    print(
        "Reproducibility : PASS"
    )

    # ================================================================
    # FINAL STATUS
    # ================================================================

    print()
    print("=" * 70)
    print(
        "3-D CURVELET POCS VALIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        "Overall status : PASS"
    )

    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()