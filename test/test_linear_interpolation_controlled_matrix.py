"""
=================================================================
Linear Interpolation Baseline Test
=================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Focused validation test for the Linear Interpolation seismic
reconstruction baseline.

This file is a SOFTWARE TEST only.

It does NOT execute the 750-case controlled experimental matrix.

The full controlled experimental implementation is located at:

    evaluation/baselines/linear_interpolation_controlled_matrix.py

This focused test verifies:

    1. Synthetic dataset generation
    2. Expected tensor shapes
    3. Dataset metadata consistency
    4. Input consistency
    5. Presence of observed samples
    6. Presence of missing samples
    7. Linear Interpolation execution
    8. Reconstruction shape
    9. Finite reconstruction values
    10. Exact observed-data preservation
    11. Modification of missing samples
    12. Reconstruction metrics
    13. Dataset reproducibility
    14. Reconstruction reproducibility

Representative test case
------------------------

Cube:
    (D, H, W) = (64, 128, 128)

Tensor convention:
    (C, D, H, W) = (1, 64, 128, 128)

Missing rate:
    30%

Missing mechanism:
    missing_crosslines

Geological mode:
    folded

Random seed:
    42

Author: Ormin Joseph
=================================================================
"""

import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset

from evaluation.baselines.baseline_linear_interpolation_controlled_matrix import (
    linear_interpolation_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ================================================================
# TEST CONFIGURATION
# ================================================================

DEVICE = torch.device("cpu")

CUBE_SIZE = (64, 128, 128)

MISSING_RATE = 0.30

MISSING_MECHANISM = "missing_crosslines"

GEOLOGICAL_MODE = "folded"

SEED = 42


# ================================================================
# UTILITY FUNCTION
# ================================================================

def to_float(value):
    """
    Convert a metric value to a standard Python float.
    """

    if isinstance(value, torch.Tensor):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    return float(value)


# ================================================================
# MAIN TEST
# ================================================================

def main():

    print(
        "\n"
        "=========================================================\n"
        "LINEAR INTERPOLATION BASELINE TEST\n"
        "=========================================================\n"
    )

    print(
        "This is a focused software test.\n"
        "It does NOT execute the 750-case controlled matrix.\n"
    )

    print(
        f"Cube size        : {CUBE_SIZE}"
    )

    print(
        f"Missing rate     : {MISSING_RATE}"
    )

    print(
        f"Missing mechanism: {MISSING_MECHANISM}"
    )

    print(
        f"Geological mode  : {GEOLOGICAL_MODE}"
    )

    print(
        f"Random seed      : {SEED}"
    )

    print(
        "=========================================================\n"
    )


    # ============================================================
    # 1. CREATE SYNTHETIC DATASET
    # ============================================================

    print(
        "[1/12] Creating deterministic synthetic dataset..."
    )

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MISSING_MECHANISM,
        seed=SEED
    )

    print(
        "      PASS - Dataset created."
    )


    # ============================================================
    # 2. RETRIEVE TEST SAMPLE
    # ============================================================

    print(
        "[2/12] Retrieving test sample..."
    )

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        mask_type,
        resolved_geological_mode
    ) = dataset[0]

    print(
        "      PASS - Test sample retrieved."
    )


    # ============================================================
    # 3. VALIDATE EXPECTED SHAPES
    # ============================================================

    print(
        "[3/12] Validating tensor shapes..."
    )

    expected_shape = (
        1,
        CUBE_SIZE[0],
        CUBE_SIZE[1],
        CUBE_SIZE[2]
    )

    if tuple(corrupted_cube.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected corrupted_cube shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(corrupted_cube.shape)}."
        )

    if tuple(target.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected target shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(target.shape)}."
        )

    if tuple(mask.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected mask shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(mask.shape)}."
        )

    print(
        f"      PASS - Tensor shape = {expected_shape}"
    )


    # ============================================================
    # 4. VALIDATE DATASET METADATA
    # ============================================================

    print(
        "[4/12] Validating dataset metadata..."
    )

    if mask_type != MISSING_MECHANISM:

        raise RuntimeError(
            "Mask mechanism mismatch. "
            f"Expected '{MISSING_MECHANISM}', "
            f"received '{mask_type}'."
        )

    if resolved_geological_mode != GEOLOGICAL_MODE:

        raise RuntimeError(
            "Geological mode mismatch. "
            f"Expected '{GEOLOGICAL_MODE}', "
            f"received '{resolved_geological_mode}'."
        )

    print(
        "      PASS - Dataset metadata is correct."
    )


    # ============================================================
    # 5. MOVE TENSORS TO TEST DEVICE
    # ============================================================

    print(
        "[5/12] Moving tensors to test device..."
    )

    corrupted_cube = corrupted_cube.to(DEVICE)

    target = target.to(DEVICE)

    mask = mask.to(DEVICE)

    print(
        f"      PASS - Device = {DEVICE}"
    )


    # ============================================================
    # 6. VALIDATE INPUT CONSISTENCY
    # ============================================================

    print(
        "[6/12] Checking corrupted-input consistency..."
    )

    expected_corrupted = target * mask

    input_consistency_error = torch.max(
        torch.abs(
            corrupted_cube
            - expected_corrupted
        )
    ).item()

    if input_consistency_error > 1e-6:

        raise RuntimeError(
            "Input consistency check failed. "
            f"Maximum difference = "
            f"{input_consistency_error:.10e}"
        )

    print(
        "      PASS - Corrupted input is consistent with target × mask."
    )


    # ============================================================
    # 7. VERIFY OBSERVED AND MISSING SAMPLES
    # ============================================================

    print(
        "[7/12] Checking observed and missing samples..."
    )

    observed = mask == 1

    missing = mask == 0

    observed_count = int(
        observed.sum().item()
    )

    missing_count = int(
        missing.sum().item()
    )

    if observed_count == 0:

        raise RuntimeError(
            "No observed samples were found."
        )

    if missing_count == 0:

        raise RuntimeError(
            "No missing samples were found."
        )

    print(
        f"      Observed samples: {observed_count}"
    )

    print(
        f"      Missing samples : {missing_count}"
    )

    print(
        "      PASS - Both observed and missing samples exist."
    )


    # ============================================================
    # 8. RUN LINEAR INTERPOLATION
    # ============================================================

    print(
        "[8/12] Running Linear Interpolation..."
    )

    reconstruction = linear_interpolation_reconstruction(
        corrupted_cube,
        mask
    )

    print(
        "      PASS - Linear Interpolation executed."
    )


    # ============================================================
    # 9. VALIDATE RECONSTRUCTION
    # ============================================================

    print(
        "[9/12] Validating reconstruction..."
    )

    if tuple(reconstruction.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected reconstruction shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(reconstruction.shape)}."
        )

    if not torch.isfinite(reconstruction).all():

        raise RuntimeError(
            "Linear Interpolation produced "
            "non-finite reconstruction values."
        )

    print(
        f"      PASS - Reconstruction shape = "
        f"{tuple(reconstruction.shape)}"
    )

    print(
        "      PASS - Reconstruction contains only finite values."
    )


    # ============================================================
    # 10. VERIFY OBSERVED-DATA PRESERVATION
    # ============================================================

    print(
        "[10/12] Checking observed-data preservation..."
    )

    observed_difference = torch.max(
        torch.abs(
            reconstruction[observed]
            - corrupted_cube[observed]
        )
    ).item()

    if observed_difference > 1e-6:

        raise RuntimeError(
            "Observed seismic samples were modified. "
            f"Maximum difference = "
            f"{observed_difference:.10e}"
        )

    print(
        f"      Maximum observed difference = "
        f"{observed_difference:.10e}"
    )

    print(
        "      PASS - Observed seismic samples preserved exactly."
    )


    # ============================================================
    # 11. VERIFY MISSING-SAMPLE RECONSTRUCTION
    # ============================================================

    print(
        "[11/12] Checking missing-region reconstruction..."
    )

    missing_difference = torch.max(
        torch.abs(
            reconstruction[missing]
            - corrupted_cube[missing]
        )
    ).item()

    if missing_difference == 0.0:

        raise RuntimeError(
            "Linear Interpolation did not modify any missing "
            "samples. Reconstruction may not have been applied."
        )

    print(
        f"      Maximum missing-region change = "
        f"{missing_difference:.10e}"
    )

    print(
        "      PASS - Missing samples were reconstructed."
    )


    # ============================================================
    # 12. CALCULATE RECONSTRUCTION METRICS
    # ============================================================

    print(
        "[12/12] Calculating reconstruction metrics..."
    )

    mae_value = to_float(
        mae(
            reconstruction,
            target
        )
    )

    rmse_value = to_float(
        rmse(
            reconstruction,
            target
        )
    )

    psnr_value = to_float(
        psnr(
            reconstruction,
            target
        )
    )

    snr_value = to_float(
        snr(
            reconstruction,
            target
        )
    )

    ssim_value = to_float(
        ssim(
            reconstruction,
            target
        )
    )

    missing_mae_value = to_float(
        mae(
            reconstruction[missing],
            target[missing]
        )
    )

    missing_rmse_value = to_float(
        rmse(
            reconstruction[missing],
            target[missing]
        )
    )

    print(
        "\n"
        "      Reconstruction Metrics\n"
        "      -----------------------"
    )

    print(
        f"      MAE          : {mae_value:.6f}"
    )

    print(
        f"      RMSE         : {rmse_value:.6f}"
    )

    print(
        f"      PSNR         : {psnr_value:.6f}"
    )

    print(
        f"      SNR          : {snr_value:.6f}"
    )

    print(
        f"      SSIM         : {ssim_value:.6f}"
    )

    print(
        f"      Missing MAE  : {missing_mae_value:.6f}"
    )

    print(
        f"      Missing RMSE : {missing_rmse_value:.6f}"
    )


    # ============================================================
    # REPRODUCIBILITY TEST
    # ============================================================

    print(
        "\n"
        "=========================================================\n"
        "REPRODUCIBILITY VALIDATION\n"
        "========================================================="
    )

    # ------------------------------------------------------------
    # Recreate the dataset with exactly the same configuration.
    # ------------------------------------------------------------

    dataset_repeat = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MISSING_MECHANISM,
        seed=SEED
    )

    (
        corrupted_repeat,
        target_repeat,
        mask_repeat,
        velocity_repeat,
        mask_type_repeat,
        geological_mode_repeat
    ) = dataset_repeat[0]

    corrupted_repeat = corrupted_repeat.to(DEVICE)

    target_repeat = target_repeat.to(DEVICE)

    mask_repeat = mask_repeat.to(DEVICE)

    # ------------------------------------------------------------
    # Compare generated datasets.
    # ------------------------------------------------------------

    dataset_difference = max(
        torch.max(
            torch.abs(
                corrupted_cube
                - corrupted_repeat
            )
        ).item(),

        torch.max(
            torch.abs(
                target
                - target_repeat
            )
        ).item(),

        torch.max(
            torch.abs(
                mask
                - mask_repeat
            )
        ).item()
    )

    if dataset_difference > 1e-6:

        raise RuntimeError(
            "Dataset reproducibility check failed. "
            f"Maximum difference = "
            f"{dataset_difference:.10e}"
        )

    print(
        f"Dataset reproducibility difference = "
        f"{dataset_difference:.10e}"
    )

    print(
        "PASS - Dataset generation is deterministic."
    )


    # ------------------------------------------------------------
    # Repeat Linear Interpolation.
    # ------------------------------------------------------------

    reconstruction_repeat = (
        linear_interpolation_reconstruction(
            corrupted_repeat,
            mask_repeat
        )
    )

    reconstruction_difference = torch.max(
        torch.abs(
            reconstruction
            - reconstruction_repeat
        )
    ).item()

    if reconstruction_difference > 1e-6:

        raise RuntimeError(
            "Linear Interpolation reproducibility check failed. "
            f"Maximum difference = "
            f"{reconstruction_difference:.10e}"
        )

    print(
        f"Reconstruction reproducibility difference = "
        f"{reconstruction_difference:.10e}"
    )

    print(
        "PASS - Linear Interpolation is deterministic."
    )


    # ============================================================
    # FINAL STATUS
    # ============================================================

    print(
        "\n"
        "=========================================================\n"
        "LINEAR INTERPOLATION BASELINE TEST COMPLETE\n"
        "=========================================================\n"
    )

    print(
        "OVERALL STATUS: PASS"
    )

    print(
        "\nThe Linear Interpolation baseline passed the "
        "focused software validation test."
    )

    print(
        "\nThe full 750-case controlled experiment remains "
        "separate under:"
    )

    print(
        "evaluation/baselines/"
        "linear_interpolation_controlled_matrix.py"
    )

    print(
        "\n=========================================================\n"
    )


# ================================================================
# SCRIPT ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()