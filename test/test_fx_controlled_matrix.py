"""
======================================================================
F-X PREDICTION BASELINE TEST
======================================================================

Focused validation test for the classical f-x prediction
seismic reconstruction baseline.

Purpose
-------
This test validates ONLY the f-x prediction reconstruction
implementation.

It does NOT execute the 750-case controlled experimental matrix.

The complete controlled experimental matrix belongs to:

    evaluation/baselines/
        fx_controlled_matrix.py

This focused test validates:

    1. Synthetic dataset generation
    2. Tensor shapes
    3. Dataset metadata
    4. Tensor finiteness
    5. Mask validity
    6. Input consistency
    7. F-X reconstruction
    8. Reconstruction shape
    9. Reconstruction finiteness
   10. Observed-data preservation
   11. Missing-region reconstruction
   12. Reconstruction metrics
   13. Dataset reproducibility
   14. F-X reconstruction reproducibility

Test configuration
------------------
Cube size       : 64 × 128 × 128
Missing rate    : 30%
Geological mode : folded
Mask mechanism  : missing_crosslines
Seed            : 42

F-X configuration
-----------------
Prediction order : 4
Iterations       : 2

Author: Ormin Joseph
======================================================================
"""


# =====================================================================
# IMPORTS
# =====================================================================

import numpy as np
import torch


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.fx_controlled_matrix import (
    fx_prediction_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# =====================================================================
# TEST CONFIGURATION
# =====================================================================

CUBE_SIZE = (
    64,
    128,
    128,
)

MISSING_RATE = 0.30

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"

SEED = 42


# =====================================================================
# F-X CONFIGURATION
# =====================================================================

FX_PREDICTION_ORDER = 4

FX_ITERATIONS = 2


# =====================================================================
# NUMERICAL VALIDATION
# =====================================================================

OBSERVED_TOLERANCE = 1.0e-6


# =====================================================================
# HELPER: FINITE TENSOR VALIDATION
# =====================================================================

def tensor_is_finite(
    tensor
):
    """
    Return True when every tensor element is finite.
    """

    return bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


# =====================================================================
# HELPER: METRIC CONVERSION
# =====================================================================

def metric_to_float(
    value
):
    """
    Convert a metric result to a Python float.

    Supports:

        torch.Tensor
        NumPy scalar
        Python numeric values
    """

    if isinstance(
        value,
        torch.Tensor
    ):

        return float(
            value.detach()
            .cpu()
            .item()
        )


    if isinstance(
        value,
        np.ndarray
    ):

        return float(
            value.item()
        )


    return float(
        value
    )


# =====================================================================
# HELPER: COMPUTE METRICS
# =====================================================================

def compute_metrics(
    reconstruction,
    target
):
    """
    Compute the project's standard reconstruction metrics.
    """

    return {

        "MAE":
            metric_to_float(
                mae(
                    reconstruction,
                    target
                )
            ),

        "RMSE":
            metric_to_float(
                rmse(
                    reconstruction,
                    target
                )
            ),

        "PSNR":
            metric_to_float(
                psnr(
                    reconstruction,
                    target
                )
            ),

        "SNR":
            metric_to_float(
                snr(
                    reconstruction,
                    target
                )
            ),

        "SSIM":
            metric_to_float(
                ssim(
                    reconstruction,
                    target
                )
            ),
    }


# =====================================================================
# MAIN TEST
# =====================================================================

def main():

    print()
    print("=" * 70)
    print(
        "F-X PREDICTION BASELINE TEST"
    )
    print("=" * 70)

    print()
    print(
        "This is a focused single-baseline validation test."
    )

    print(
        "It does NOT execute the 750-case controlled matrix."
    )

    print()


    # =================================================================
    # DISPLAY TEST CONFIGURATION
    # =================================================================

    print(
        "Test configuration"
    )

    print(
        "-" * 70
    )

    print(
        f"Cube size             : {CUBE_SIZE}"
    )

    print(
        f"Missing rate          : {MISSING_RATE}"
    )

    print(
        f"Geological mode       : {GEOLOGICAL_MODE}"
    )

    print(
        f"Mask mode             : {MASK_MODE}"
    )

    print(
        f"Seed                  : {SEED}"
    )

    print()

    print(
        "F-X configuration"
    )

    print(
        f"Prediction order      : "
        f"{FX_PREDICTION_ORDER}"
    )

    print(
        f"Iterations            : "
        f"{FX_ITERATIONS}"
    )

    print()


    # =================================================================
    # CREATE SYNTHETIC DATASET
    # =================================================================

    print(
        "Creating synthetic dataset..."
    )

    dataset = SyntheticSeismicDataset(

        num_samples=1,

        cube_size=CUBE_SIZE,

        missing_probability=MISSING_RATE,

        geological_mode=GEOLOGICAL_MODE,

        mask_mode=MASK_MODE,

        seed=SEED,
    )


    # =================================================================
    # RETRIEVE SINGLE SAMPLE
    # =================================================================

    (
        corrupted,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode,
    ) = dataset[0]


    # =================================================================
    # EXPECTED SHAPE
    # =================================================================

    expected_shape = (
        1,
        *CUBE_SIZE
    )


    # =================================================================
    # TEST 1: CORRUPTED INPUT SHAPE
    # =================================================================

    if tuple(
        corrupted.shape
    ) != expected_shape:

        raise RuntimeError(
            "F-X test failed: "
            f"corrupted input shape is "
            f"{tuple(corrupted.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: corrupted input shape"
    )


    # =================================================================
    # TEST 2: TARGET SHAPE
    # =================================================================

    if tuple(
        target.shape
    ) != expected_shape:

        raise RuntimeError(
            "F-X test failed: "
            f"target shape is "
            f"{tuple(target.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: target shape"
    )


    # =================================================================
    # TEST 3: MASK SHAPE
    # =================================================================

    if tuple(
        mask.shape
    ) != expected_shape:

        raise RuntimeError(
            "F-X test failed: "
            f"mask shape is "
            f"{tuple(mask.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: mask shape"
    )


    # =================================================================
    # TEST 4: VELOCITY SHAPE
    # =================================================================

    if tuple(
        velocity.shape
    ) != expected_shape:

        raise RuntimeError(
            "F-X test failed: "
            f"velocity shape is "
            f"{tuple(velocity.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: velocity shape"
    )


    # =================================================================
    # TEST 5: DATASET METADATA
    # =================================================================

    if returned_mask_mode != MASK_MODE:

        raise RuntimeError(
            "F-X test failed: "
            f"expected mask mode '{MASK_MODE}', "
            f"received '{returned_mask_mode}'."
        )


    if returned_geological_mode != GEOLOGICAL_MODE:

        raise RuntimeError(
            "F-X test failed: "
            f"expected geological mode "
            f"'{GEOLOGICAL_MODE}', "
            f"received '{returned_geological_mode}'."
        )


    print(
        "PASS: dataset metadata"
    )


    # =================================================================
    # TEST 6: FINITE INPUT VALUES
    # =================================================================

    for name, tensor in [

        ("corrupted", corrupted),

        ("target", target),

        ("mask", mask),

        ("velocity", velocity),

    ]:

        if not tensor_is_finite(
            tensor
        ):

            raise RuntimeError(
                "F-X test failed: "
                f"{name} contains NaN or "
                f"infinite values."
            )


    print(
        "PASS: input tensors contain finite values"
    )


    # =================================================================
    # TEST 7: MASK VALUES
    # =================================================================

    unique_mask = torch.unique(
        mask
    )


    valid_mask = torch.all(
        (unique_mask == 0)
        |
        (unique_mask == 1)
    )


    if not bool(
        valid_mask.item()
    ):

        raise RuntimeError(
            "F-X test failed: "
            "mask contains values other than "
            "0 and 1."
        )


    print(
        "PASS: mask contains only 0 and 1"
    )


    # =================================================================
    # TEST 8: INPUT CONSISTENCY
    # =================================================================

    expected_corrupted = (
        target
        * mask
    )


    input_difference = torch.max(
        torch.abs(
            corrupted
            - expected_corrupted
        )
    ).item()


    if (
        input_difference
        > OBSERVED_TOLERANCE
    ):

        raise RuntimeError(
            "F-X test failed: "
            "corrupted input is inconsistent "
            "with target × mask."
        )


    print(
        "PASS: corrupted input consistency"
    )

    print(
        f"      Maximum input difference: "
        f"{input_difference:.6e}"
    )


    # =================================================================
    # COUNT OBSERVED AND MISSING SAMPLES
    # =================================================================

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


    print()

    print(
        f"Observed samples       : "
        f"{observed_samples}"
    )

    print(
        f"Missing samples        : "
        f"{missing_samples}"
    )


    # =================================================================
    # TEST 9: BOTH REGIONS MUST EXIST
    # =================================================================

    if observed_samples == 0:

        raise RuntimeError(
            "F-X test failed: "
            "there are no observed samples."
        )


    if missing_samples == 0:

        raise RuntimeError(
            "F-X test failed: "
            "there are no missing samples."
        )


    print(
        "PASS: observed and missing regions exist"
    )


    # =================================================================
    # RUN F-X RECONSTRUCTION
    # =================================================================

    print()

    print(
        "Running f-x prediction reconstruction..."
    )


    reconstruction = (
        fx_prediction_reconstruction(

            corrupted_cube=corrupted,

            mask=mask,

            prediction_order=(
                FX_PREDICTION_ORDER
            ),

            iterations=(
                FX_ITERATIONS
            ),
        )
    )


    # =================================================================
    # TEST 10: RECONSTRUCTION SHAPE
    # =================================================================

    if tuple(
        reconstruction.shape
    ) != expected_shape:

        raise RuntimeError(
            "F-X test failed: "
            f"reconstruction shape is "
            f"{tuple(reconstruction.shape)}, "
            f"expected {expected_shape}."
        )


    print(
        "PASS: reconstruction shape"
    )


    # =================================================================
    # TEST 11: RECONSTRUCTION FINITENESS
    # =================================================================

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "F-X test failed: "
            "reconstruction contains NaN "
            "or infinite values."
        )


    print(
        "PASS: reconstruction contains finite values"
    )


    # =================================================================
    # TEST 12: OBSERVED-DATA PRESERVATION
    # =================================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            - corrupted[mask == 1]
        )
    ).item()


    if (
        observed_difference
        > OBSERVED_TOLERANCE
    ):

        raise RuntimeError(
            "F-X test failed: "
            f"observed-data preservation error "
            f"is {observed_difference:.6e}, "
            f"which exceeds the tolerance "
            f"of {OBSERVED_TOLERANCE:.6e}."
        )


    print(
        "PASS: observed seismic samples preserved"
    )

    print(
        f"      Maximum observed difference: "
        f"{observed_difference:.6e}"
    )


    # =================================================================
    # MISSING-REGION EXTRACTION
    # =================================================================

    missing_target = (
        target[mask == 0]
    )


    missing_reconstruction = (
        reconstruction[mask == 0]
    )


    if missing_target.numel() == 0:

        raise RuntimeError(
            "F-X test failed: "
            "no missing samples are available "
            "for reconstruction evaluation."
        )


    # =================================================================
    # TEST 13: MISSING REGION WAS MODIFIED
    # =================================================================

    missing_change = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - corrupted[mask == 0]
        )
    ).item()


    if missing_change <= 0.0:

        raise RuntimeError(
            "F-X test failed: "
            "reconstruction did not modify "
            "the missing region."
        )


    print(
        "PASS: missing region was reconstructed"
    )

    print(
        f"      Mean missing-region change: "
        f"{missing_change:.6f}"
    )


    # =================================================================
    # ZERO-FILLED BASELINE
    # =================================================================

    zero_filled_missing_mae = torch.mean(
        torch.abs(
            corrupted[mask == 0]
            - target[mask == 0]
        )
    ).item()


    # =================================================================
    # MISSING-REGION F-X ERROR
    # =================================================================

    missing_mae = metric_to_float(
        mae(
            missing_reconstruction,
            missing_target
        )
    )


    missing_rmse = metric_to_float(
        rmse(
            missing_reconstruction,
            missing_target
        )
    )


    missing_psnr = metric_to_float(
        psnr(
            missing_reconstruction,
            missing_target
        )
    )


    missing_snr = metric_to_float(
        snr(
            missing_reconstruction,
            missing_target
        )
    )


    missing_ssim = metric_to_float(
        ssim(
            missing_reconstruction,
            missing_target
        )
    )


    # =================================================================
    # GLOBAL METRICS
    # =================================================================

    print()

    print(
        "Computing global reconstruction metrics..."
    )


    metrics = compute_metrics(
        reconstruction,
        target
    )


    # =================================================================
    # TEST 14: METRICS ARE VALID
    # =================================================================

    for metric_name, metric_value in metrics.items():

        if metric_name in [
            "PSNR",
            "SNR",
        ]:

            continue


        if not np.isfinite(
            metric_value
        ):

            raise RuntimeError(
                "F-X test failed: "
                f"{metric_name} is not finite."
            )


    print(
        "PASS: reconstruction metrics computed"
    )


    # =================================================================
    # DISPLAY RESULTS
    # =================================================================

    print()

    print("=" * 70)

    print(
        "F-X PREDICTION RESULTS"
    )

    print("=" * 70)


    print(
        f"Global MAE                : "
        f"{metrics['MAE']:.6f}"
    )


    print(
        f"Global RMSE               : "
        f"{metrics['RMSE']:.6f}"
    )


    print(
        f"Global PSNR               : "
        f"{metrics['PSNR']:.6f} dB"
    )


    print(
        f"Global SNR                : "
        f"{metrics['SNR']:.6f} dB"
    )


    print(
        f"Global SSIM               : "
        f"{metrics['SSIM']:.6f}"
    )


    print()

    print(
        f"Missing-region MAE        : "
        f"{missing_mae:.6f}"
    )


    print(
        f"Missing-region RMSE       : "
        f"{missing_rmse:.6f}"
    )


    print(
        f"Missing-region PSNR       : "
        f"{missing_psnr:.6f} dB"
    )


    print(
        f"Missing-region SNR        : "
        f"{missing_snr:.6f}"
    )


    print(
        f"Missing-region SSIM       : "
        f"{missing_ssim:.6f}"
    )


    print()

    print(
        f"Zero-filled missing MAE   : "
        f"{zero_filled_missing_mae:.6f}"
    )


    print(
        f"Missing-region change     : "
        f"{missing_change:.6f}"
    )


    print(
        f"Observed preservation     : "
        f"{observed_difference:.6e}"
    )


    # =================================================================
    # DATASET REPRODUCIBILITY TEST
    # =================================================================

    print()

    print(
        "Testing deterministic dataset generation..."
    )


    dataset_repeat = (
        SyntheticSeismicDataset(

            num_samples=1,

            cube_size=CUBE_SIZE,

            missing_probability=MISSING_RATE,

            geological_mode=GEOLOGICAL_MODE,

            mask_mode=MASK_MODE,

            seed=SEED,
        )
    )


    (
        corrupted_repeat,
        target_repeat,
        mask_repeat,
        velocity_repeat,
        mask_mode_repeat,
        geological_mode_repeat,
    ) = dataset_repeat[0]


    # =================================================================
    # TEST 15: DATASET REPRODUCIBILITY
    # =================================================================

    if not torch.equal(
        corrupted,
        corrupted_repeat
    ):

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset generation produced "
            "different corrupted data."
        )


    if not torch.equal(
        target,
        target_repeat
    ):

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset generation produced "
            "different targets."
        )


    if not torch.equal(
        mask,
        mask_repeat
    ):

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset generation produced "
            "different masks."
        )


    if not torch.equal(
        velocity,
        velocity_repeat
    ):

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset generation produced "
            "different velocity models."
        )


    if mask_mode_repeat != MASK_MODE:

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset returned a "
            "different mask mode."
        )


    if geological_mode_repeat != GEOLOGICAL_MODE:

        raise RuntimeError(
            "F-X test failed: "
            "repeated dataset returned a "
            "different geological mode."
        )


    print(
        "PASS: dataset generation is reproducible"
    )


    # =================================================================
    # F-X RECONSTRUCTION REPRODUCIBILITY
    # =================================================================

    print(
        "Testing deterministic f-x reconstruction..."
    )


    reconstruction_repeat = (
        fx_prediction_reconstruction(

            corrupted_cube=corrupted_repeat,

            mask=mask_repeat,

            prediction_order=(
                FX_PREDICTION_ORDER
            ),

            iterations=(
                FX_ITERATIONS
            ),
        )
    )


    # =================================================================
    # TEST 16: RECONSTRUCTION REPRODUCIBILITY
    # =================================================================

    reconstruction_difference = torch.max(
        torch.abs(
            reconstruction
            - reconstruction_repeat
        )
    ).item()


    if (
        reconstruction_difference
        > OBSERVED_TOLERANCE
    ):

        raise RuntimeError(
            "F-X test failed: "
            "repeated f-x reconstruction is not "
            "deterministic within the configured "
            "tolerance."
        )


    print(
        "PASS: f-x reconstruction is reproducible"
    )


    print(
        f"      Maximum repeated reconstruction "
        f"difference: "
        f"{reconstruction_difference:.6e}"
    )


    # =================================================================
    # FINAL STATUS
    # =================================================================

    print()

    print("=" * 70)

    print(
        "F-X PREDICTION BASELINE TEST: PASS"
    )

    print("=" * 70)

    print()

    print(
        "The f-x prediction baseline passed the "
        "focused single-case validation."
    )

    print()

    print(
        "The complete 750-case controlled experiment "
        "remains separate under:"
    )

    print(
        "evaluation/baselines/"
        "fx_controlled_matrix.py"
    )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()