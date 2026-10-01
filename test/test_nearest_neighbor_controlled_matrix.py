"""
====================================================================
Nearest Neighbor Baseline Test
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Focused software validation test for the Nearest Neighbor seismic
reconstruction baseline.

This file does NOT execute the 750-case controlled experimental
matrix.

The full controlled experimental implementation remains under:

    evaluation/baselines/nearest_neighbor_controlled_matrix.py

This focused test verifies:

    1. Synthetic dataset generation
    2. Expected tensor shapes
    3. Dataset metadata
    4. Input consistency
    5. Observed and missing samples
    6. Nearest Neighbor reconstruction
    7. Reconstruction shape
    8. Finite reconstruction values
    9. Exact observed-data preservation
    10. Missing-region reconstruction
    11. Reconstruction metrics
    12. Dataset reproducibility
    13. Reconstruction reproducibility

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

Device:
    CPU

Author: Ormin Joseph
====================================================================
"""


# ====================================================================
# IMPORTS
# ====================================================================

import torch

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.baseline_nearest_neighbor_controlled_matrix import (
    nearest_neighbor_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ====================================================================
# TEST CONFIGURATION
# ====================================================================

# --------------------------------------------------------------
# Nearest Neighbor uses the CPU because the implementation relies
# on SciPy's distance-transform operation.
# --------------------------------------------------------------

DEVICE = torch.device("cpu")


# --------------------------------------------------------------
# Standard project test cube.
# --------------------------------------------------------------

CUBE_SIZE = (
    64,
    128,
    128
)


# --------------------------------------------------------------
# Number of seismic channels.
# --------------------------------------------------------------

CHANNELS = 1


# --------------------------------------------------------------
# Representative missing-data rate.
# --------------------------------------------------------------

MISSING_RATE = 0.30


# --------------------------------------------------------------
# Representative missing-data mechanism.
# --------------------------------------------------------------

MISSING_MECHANISM = "missing_crosslines"


# --------------------------------------------------------------
# Representative geological model.
# --------------------------------------------------------------

GEOLOGICAL_MODE = "folded"


# --------------------------------------------------------------
# Deterministic random seed.
# --------------------------------------------------------------

SEED = 42


# --------------------------------------------------------------
# Observed-data preservation tolerance.
#
# This corresponds to the project's controlled-experiment
# preservation requirement.
# --------------------------------------------------------------

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# ====================================================================
# UTILITY FUNCTION
# ====================================================================

def to_float(value):
    """
    Convert a metric result into a standard Python float.

    Parameters
    ----------
    value : torch.Tensor or numeric
        Metric value.

    Returns
    -------
    float
        Python floating-point value.
    """

    if isinstance(value, torch.Tensor):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    return float(value)


# ====================================================================
# MAIN TEST
# ====================================================================

def main():

    print(
        "\n"
        "============================================================\n"
        "NEAREST NEIGHBOR BASELINE TEST\n"
        "============================================================\n"
    )

    print(
        "Focused software validation test.\n"
    )

    print(
        "This test does NOT execute the 750-case "
        "controlled experimental matrix.\n"
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
        f"Device           : {DEVICE}"
    )

    print(
        "============================================================\n"
    )


    # ================================================================
    # 1. CREATE SYNTHETIC DATASET
    # ================================================================

    print(
        "[1/13] Creating deterministic synthetic dataset..."
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
        "      PASS - Synthetic dataset created."
    )


    # ================================================================
    # 2. RETRIEVE TEST SAMPLE
    # ================================================================

    print(
        "[2/13] Retrieving test sample..."
    )

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode
    ) = dataset[0]

    print(
        "      PASS - Test sample retrieved."
    )


    # ================================================================
    # 3. VALIDATE TENSOR SHAPES
    # ================================================================

    print(
        "[3/13] Validating tensor shapes..."
    )

    expected_shape = (
        CHANNELS,
        *CUBE_SIZE
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


    # ================================================================
    # 4. VALIDATE DATASET METADATA
    # ================================================================

    print(
        "[4/13] Validating dataset metadata..."
    )

    if returned_mask_mode != MISSING_MECHANISM:

        raise RuntimeError(
            "Dataset returned an unexpected mask mode. "
            f"Expected '{MISSING_MECHANISM}', "
            f"received '{returned_mask_mode}'."
        )

    if returned_geological_mode != GEOLOGICAL_MODE:

        raise RuntimeError(
            "Dataset returned an unexpected geological mode. "
            f"Expected '{GEOLOGICAL_MODE}', "
            f"received '{returned_geological_mode}'."
        )

    print(
        "      PASS - Dataset metadata is correct."
    )


    # ================================================================
    # 5. MOVE TENSORS TO CPU
    # ================================================================

    print(
        "[5/13] Moving tensors to evaluation device..."
    )

    corrupted_cube = corrupted_cube.to(DEVICE)

    target = target.to(DEVICE)

    mask = mask.to(DEVICE)

    print(
        f"      PASS - Device = {DEVICE}"
    )


    # ================================================================
    # 6. VALIDATE FINITE INPUT DATA
    # ================================================================

    print(
        "[6/13] Checking input tensors for finite values..."
    )

    if not torch.isfinite(corrupted_cube).all():

        raise RuntimeError(
            "Corrupted seismic cube contains "
            "non-finite values."
        )

    if not torch.isfinite(target).all():

        raise RuntimeError(
            "Target seismic cube contains "
            "non-finite values."
        )

    if not torch.isfinite(mask).all():

        raise RuntimeError(
            "Observation mask contains "
            "non-finite values."
        )

    print(
        "      PASS - Input tensors contain finite values."
    )


    # ================================================================
    # 7. VALIDATE INPUT CONSISTENCY
    # ================================================================

    print(
        "[7/13] Checking corrupted-input consistency..."
    )

    # --------------------------------------------------------------
    # The synthetic dataset follows:
    #
    #     corrupted_cube = target * mask
    #
    # Observed locations therefore contain the original target
    # values, while missing locations are zeroed.
    # --------------------------------------------------------------

    expected_corrupted = (
        target * mask
    )

    input_consistency_error = torch.max(
        torch.abs(
            corrupted_cube
            -
            expected_corrupted
        )
    ).item()

    if input_consistency_error > OBSERVED_PRESERVATION_TOLERANCE:

        raise RuntimeError(
            "Input consistency check failed. "
            f"Maximum difference = "
            f"{input_consistency_error:.10e}"
        )

    print(
        f"      Maximum input difference = "
        f"{input_consistency_error:.10e}"
    )

    print(
        "      PASS - Corrupted input is consistent "
        "with target × mask."
    )


    # ================================================================
    # 8. VERIFY OBSERVED AND MISSING SAMPLES
    # ================================================================

    print(
        "[8/13] Checking observed and missing samples..."
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
            "No observed seismic samples were found."
        )

    if missing_count == 0:

        raise RuntimeError(
            "No missing seismic samples were found."
        )

    print(
        f"      Observed samples = {observed_count}"
    )

    print(
        f"      Missing samples  = {missing_count}"
    )

    print(
        "      PASS - Both observed and missing samples exist."
    )


    # ================================================================
    # 9. RUN NEAREST NEIGHBOR RECONSTRUCTION
    # ================================================================

    print(
        "[9/13] Running Nearest Neighbor reconstruction..."
    )

    reconstruction = nearest_neighbor_reconstruction(
        corrupted_cube,
        mask
    )

    print(
        "      PASS - Nearest Neighbor reconstruction executed."
    )


    # ================================================================
    # 10. VALIDATE RECONSTRUCTION
    # ================================================================

    print(
        "[10/13] Validating reconstructed seismic cube..."
    )

    # --------------------------------------------------------------
    # Check output shape.
    # --------------------------------------------------------------

    if tuple(reconstruction.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected reconstruction shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(reconstruction.shape)}."
        )

    # --------------------------------------------------------------
    # Check finite values.
    # --------------------------------------------------------------

    if not torch.isfinite(reconstruction).all():

        raise RuntimeError(
            "Nearest Neighbor reconstruction contains "
            "non-finite values."
        )

    print(
        f"      PASS - Reconstruction shape = "
        f"{tuple(reconstruction.shape)}"
    )

    print(
        "      PASS - Reconstruction contains only finite values."
    )


    # ================================================================
    # 11. VERIFY OBSERVED-DATA PRESERVATION
    # ================================================================

    print(
        "[11/13] Checking observed-data preservation..."
    )

    observed_difference = torch.max(
        torch.abs(
            reconstruction[observed]
            -
            corrupted_cube[observed]
        )
    ).item()

    if observed_difference > OBSERVED_PRESERVATION_TOLERANCE:

        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum difference = "
            f"{observed_difference:.10e}"
        )

    print(
        f"      Maximum observed difference = "
        f"{observed_difference:.10e}"
    )

    print(
        "      PASS - Observed seismic samples "
        "were preserved exactly."
    )


    # ================================================================
    # VERIFY MISSING-REGION RECONSTRUCTION
    # ================================================================

    print(
        "[12/13] Checking missing-region reconstruction..."
    )

    missing_reconstruction = reconstruction[
        missing
    ]

    missing_corrupted = corrupted_cube[
        missing
    ]

    missing_change = torch.mean(
        torch.abs(
            missing_reconstruction
            -
            missing_corrupted
        )
    ).item()

    # --------------------------------------------------------------
    # A zero change would indicate that the missing values were not
    # modified by the reconstruction algorithm.
    # --------------------------------------------------------------

    if missing_change <= 0.0:

        raise RuntimeError(
            "Nearest Neighbor reconstruction did not modify "
            "the missing region."
        )

    print(
        f"      Mean missing-region change = "
        f"{missing_change:.10e}"
    )

    print(
        "      PASS - Missing samples were reconstructed."
    )


    # ================================================================
    # 13. CALCULATE RECONSTRUCTION METRICS
    # ================================================================

    print(
        "[13/13] Calculating reconstruction metrics..."
    )

    # --------------------------------------------------------------
    # Global metrics.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Missing-region metrics.
    #
    # These metrics isolate the actual reconstruction task from
    # the already-observed samples.
    # --------------------------------------------------------------

    missing_target = target[
        missing
    ]

    missing_mae_value = to_float(
        torch.mean(
            torch.abs(
                missing_reconstruction
                -
                missing_target
            )
        )
    )

    missing_rmse_value = to_float(
        torch.sqrt(
            torch.mean(
                (
                    missing_reconstruction
                    -
                    missing_target
                ) ** 2
            )
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
        f"      PSNR         : {psnr_value:.6f} dB"
    )

    print(
        f"      SNR          : {snr_value:.6f} dB"
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


    # ================================================================
    # REPRODUCIBILITY TEST
    # ================================================================

    print(
        "\n"
        "============================================================\n"
        "REPRODUCIBILITY VALIDATION\n"
        "============================================================"
    )

    # --------------------------------------------------------------
    # Recreate the same synthetic dataset using exactly the same
    # configuration and random seed.
    # --------------------------------------------------------------

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
        returned_mask_mode_repeat,
        returned_geological_mode_repeat
    ) = dataset_repeat[0]

    corrupted_repeat = corrupted_repeat.to(DEVICE)

    target_repeat = target_repeat.to(DEVICE)

    mask_repeat = mask_repeat.to(DEVICE)

    # --------------------------------------------------------------
    # Compare corrupted cubes.
    # --------------------------------------------------------------

    corrupted_difference = torch.max(
        torch.abs(
            corrupted_cube
            -
            corrupted_repeat
        )
    ).item()

    # --------------------------------------------------------------
    # Compare targets.
    # --------------------------------------------------------------

    target_difference = torch.max(
        torch.abs(
            target
            -
            target_repeat
        )
    ).item()

    # --------------------------------------------------------------
    # Compare masks.
    # --------------------------------------------------------------

    mask_difference = torch.max(
        torch.abs(
            mask
            -
            mask_repeat
        )
    ).item()

    # --------------------------------------------------------------
    # Use the largest difference as the overall reproducibility
    # measure.
    # --------------------------------------------------------------

    dataset_difference = max(
        corrupted_difference,
        target_difference,
        mask_difference
    )

    if dataset_difference > OBSERVED_PRESERVATION_TOLERANCE:

        raise RuntimeError(
            "Dataset reproducibility check failed. "
            f"Maximum difference = "
            f"{dataset_difference:.10e}"
        )

    print(
        f"      Dataset reproducibility difference = "
        f"{dataset_difference:.10e}"
    )

    print(
        "      PASS - Dataset generation is deterministic."
    )


    # --------------------------------------------------------------
    # Repeat Nearest Neighbor reconstruction.
    # --------------------------------------------------------------

    reconstruction_repeat = (
        nearest_neighbor_reconstruction(
            corrupted_repeat,
            mask_repeat
        )
    )

    # --------------------------------------------------------------
    # Compare reconstructions.
    # --------------------------------------------------------------

    reconstruction_difference = torch.max(
        torch.abs(
            reconstruction
            -
            reconstruction_repeat
        )
    ).item()

    if reconstruction_difference > OBSERVED_PRESERVATION_TOLERANCE:

        raise RuntimeError(
            "Nearest Neighbor reproducibility check failed. "
            f"Maximum difference = "
            f"{reconstruction_difference:.10e}"
        )

    print(
        f"      Reconstruction reproducibility difference = "
        f"{reconstruction_difference:.10e}"
    )

    print(
        "      PASS - Nearest Neighbor reconstruction is deterministic."
    )


    # ================================================================
    # FINAL STATUS
    # ================================================================

    print(
        "\n"
        "============================================================\n"
        "NEAREST NEIGHBOR BASELINE TEST COMPLETE\n"
        "============================================================\n"
    )

    print(
        "OVERALL STATUS: PASS"
    )

    print(
        "\nThe Nearest Neighbor baseline passed the "
        "focused software validation test."
    )

    print(
        "\nThe full controlled experimental implementation remains:"
    )

    print(
        "evaluation/baselines/"
        "nearest_neighbor_controlled_matrix.py"
    )

    print(
        "\n============================================================\n"
    )


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()