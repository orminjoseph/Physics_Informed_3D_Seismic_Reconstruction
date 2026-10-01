"""
=========================================================
Dictionary Learning Baseline Test
=========================================================

Focused validation test for the Dictionary Learning
seismic reconstruction baseline.

Purpose
-------
This test validates ONLY the Dictionary Learning
reconstruction implementation.

It does NOT execute the 750-case controlled experimental
matrix.

The complete controlled experimental matrix belongs to:

    evaluation/baselines/
        dictionary_learning_controlled_matrix.py

This focused test validates:

    1. Synthetic dataset generation
    2. Tensor shapes
    3. Tensor finiteness
    4. Mask validity
    5. Input consistency
    6. Dictionary Learning reconstruction
    7. Reconstruction shape
    8. Reconstruction finiteness
    9. Exact observed-data preservation
   10. Modification of missing samples
   11. Reconstruction metrics
   12. Deterministic dataset generation
   13. Deterministic Dictionary Learning reconstruction

Test configuration
------------------
Cube size       : 64 × 128 × 128
Missing rate    : 30%
Geology         : folded
Mask mechanism  : missing_crosslines
Seed            : 42

Dictionary Learning configuration
----------------------------------
Patch size              : 8 × 8 × 8
Dictionary components   : 64
Alpha                   : 1.0
Maximum iterations      : 20
Batch size              : 64
Training patches        : 2000
Minimum observed        : 0.80

Observed seismic samples must be preserved exactly.

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import torch

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.dictionary_learning_controlled_matrix import (
    dictionary_learning_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# =========================================================
# TEST CONFIGURATION
# =========================================================

CUBE_SIZE = (
    64,
    128,
    128,
)

MISSING_RATE = 0.30

GEOLOGICAL_MODE = "folded"

MASK_MODE = "missing_crosslines"

SEED = 42


# =========================================================
# DICTIONARY LEARNING CONFIGURATION
# =========================================================

PATCH_SIZE = (
    8,
    8,
    8,
)

N_COMPONENTS = 64

ALPHA = 1.0

MAX_ITER = 20

BATCH_SIZE = 64

MAX_TRAINING_PATCHES = 2000

MIN_OBSERVED_FRACTION = 0.80


# =========================================================
# NUMERICAL VALIDATION
# =========================================================

OBSERVED_TOLERANCE = 1.0e-6


# =========================================================
# HELPER: FINITE TENSOR VALIDATION
# =========================================================

def tensor_is_finite(
    tensor
):
    """
    Return True when every element of the tensor is finite.
    """

    return bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


# =========================================================
# HELPER: METRIC CONVERSION
# =========================================================

def metric_to_float(
    value
):
    """
    Convert a metric result into a Python float.

    Supports:
        torch.Tensor
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

    return float(value)


# =========================================================
# HELPER: COMPUTE METRICS
# =========================================================

def compute_metrics(
    reconstruction,
    target
):
    """
    Compute the project's standard reconstruction metrics.
    """

    return {
        "MAE": metric_to_float(
            mae(
                reconstruction,
                target
            )
        ),

        "RMSE": metric_to_float(
            rmse(
                reconstruction,
                target
            )
        ),

        "PSNR": metric_to_float(
            psnr(
                reconstruction,
                target
            )
        ),

        "SNR": metric_to_float(
            snr(
                reconstruction,
                target
            )
        ),

        "SSIM": metric_to_float(
            ssim(
                reconstruction,
                target
            )
        ),
    }


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print("=" * 70)
    print(
        "DICTIONARY LEARNING BASELINE TEST"
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


    # =====================================================
    # DISPLAY TEST CONFIGURATION
    # =====================================================

    print("Test configuration")
    print("-" * 70)

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


    # =====================================================
    # CREATE SYNTHETIC DATASET
    # =====================================================

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


    # =====================================================
    # RETRIEVE SAMPLE
    # =====================================================

    (
        corrupted,
        target,
        mask,
        velocity,
        returned_mask_mode,
        returned_geological_mode,
    ) = dataset[0]


    # =====================================================
    # EXPECTED TENSOR SHAPE
    # =====================================================

    expected_shape = (
        1,
        *CUBE_SIZE
    )


    # =====================================================
    # TEST 1: CORRUPTED INPUT SHAPE
    # =====================================================

    if tuple(
        corrupted.shape
    ) != expected_shape:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"corrupted shape is "
            f"{tuple(corrupted.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: corrupted input shape"
    )


    # =====================================================
    # TEST 2: TARGET SHAPE
    # =====================================================

    if tuple(
        target.shape
    ) != expected_shape:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"target shape is "
            f"{tuple(target.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: target shape"
    )


    # =====================================================
    # TEST 3: MASK SHAPE
    # =====================================================

    if tuple(
        mask.shape
    ) != expected_shape:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"mask shape is "
            f"{tuple(mask.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: mask shape"
    )


    # =====================================================
    # TEST 4: VELOCITY SHAPE
    # =====================================================

    if tuple(
        velocity.shape
    ) != expected_shape:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"velocity shape is "
            f"{tuple(velocity.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: velocity shape"
    )


    # =====================================================
    # TEST 5: DATASET METADATA
    # =====================================================

    if returned_mask_mode != MASK_MODE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"expected mask mode '{MASK_MODE}', "
            f"received '{returned_mask_mode}'."
        )

    if returned_geological_mode != GEOLOGICAL_MODE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"expected geological mode "
            f"'{GEOLOGICAL_MODE}', "
            f"received '{returned_geological_mode}'."
        )

    print(
        "PASS: dataset metadata"
    )


    # =====================================================
    # TEST 6: FINITE INPUT VALUES
    # =====================================================

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
                "Dictionary Learning test failed: "
                f"{name} contains NaN or infinite values."
            )

    print(
        "PASS: input tensors contain finite values"
    )


    # =====================================================
    # TEST 7: MASK VALUES
    # =====================================================

    unique_mask = torch.unique(
        mask
    )

    valid_mask = torch.all(
        (unique_mask == 0)
        | (unique_mask == 1)
    )

    if not bool(
        valid_mask.item()
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "mask contains values other than 0 and 1."
        )

    print(
        "PASS: mask contains only 0 and 1"
    )


    # =====================================================
    # TEST 8: INPUT CONSISTENCY
    # =====================================================

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

    if input_difference > OBSERVED_TOLERANCE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "corrupted input is inconsistent with "
            "target × mask."
        )

    print(
        "PASS: corrupted input consistency"
    )

    print(
        f"      Maximum input difference: "
        f"{input_difference:.6e}"
    )


    # =====================================================
    # OBSERVED / MISSING SAMPLE COUNTS
    # =====================================================

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
        f"Observed samples       : {observed_samples}"
    )

    print(
        f"Missing samples        : {missing_samples}"
    )


    # =====================================================
    # TEST 9: BOTH REGIONS MUST EXIST
    # =====================================================

    if observed_samples == 0:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "there are no observed samples."
        )

    if missing_samples == 0:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "there are no missing samples."
        )

    print(
        "PASS: observed and missing regions exist"
    )


    # =====================================================
    # DICTIONARY LEARNING RECONSTRUCTION
    # =====================================================

    print()
    print(
        "Running Dictionary Learning reconstruction..."
    )

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


    # =====================================================
    # TEST 10: RECONSTRUCTION SHAPE
    # =====================================================

    if tuple(
        reconstruction.shape
    ) != expected_shape:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"reconstruction shape is "
            f"{tuple(reconstruction.shape)}, "
            f"expected {expected_shape}."
        )

    print(
        "PASS: reconstruction shape"
    )


    # =====================================================
    # TEST 11: RECONSTRUCTION FINITENESS
    # =====================================================

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "reconstruction contains NaN or "
            "infinite values."
        )

    print(
        "PASS: reconstruction contains finite values"
    )


    # =====================================================
    # TEST 12: OBSERVED-DATA PRESERVATION
    # =====================================================

    observed_difference = torch.max(
        torch.abs(
            reconstruction[mask == 1]
            - corrupted[mask == 1]
        )
    ).item()

    if observed_difference > OBSERVED_TOLERANCE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            f"observed-data preservation error is "
            f"{observed_difference:.6e}, "
            f"which exceeds the tolerance of "
            f"{OBSERVED_TOLERANCE:.6e}."
        )

    print(
        "PASS: observed seismic samples preserved"
    )

    print(
        f"      Maximum observed difference: "
        f"{observed_difference:.6e}"
    )


    # =====================================================
    # TEST 13: MISSING REGION MUST BE PROCESSED
    # =====================================================

    missing_change = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - corrupted[mask == 0]
        )
    ).item()

    if missing_change <= 0.0:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "reconstruction did not modify the "
            "missing region."
        )

    print(
        "PASS: missing region was reconstructed"
    )

    print(
        f"      Mean missing-region change: "
        f"{missing_change:.6f}"
    )


    # =====================================================
    # BASELINE: ZERO-FILLED MISSING REGION
    # =====================================================

    zero_filled_missing_mae = torch.mean(
        torch.abs(
            corrupted[mask == 0]
            - target[mask == 0]
        )
    ).item()


    # =====================================================
    # DICTIONARY LEARNING MISSING-REGION ERROR
    # =====================================================

    dictionary_missing_mae = torch.mean(
        torch.abs(
            reconstruction[mask == 0]
            - target[mask == 0]
        )
    ).item()


    # =====================================================
    # GLOBAL METRICS
    # =====================================================

    print()
    print(
        "Computing reconstruction metrics..."
    )

    metrics = compute_metrics(
        reconstruction,
        target
    )


    # =====================================================
    # TEST 14: METRICS MUST BE FINITE
    # =====================================================

    for metric_name, metric_value in metrics.items():

        if metric_name in [
            "PSNR",
            "SNR",
        ]:
            continue

        if not torch.isfinite(
            torch.tensor(metric_value)
        ).item():

            raise RuntimeError(
                "Dictionary Learning test failed: "
                f"{metric_name} is not finite."
            )

    print(
        "PASS: reconstruction metrics computed"
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "DICTIONARY LEARNING RESULTS"
    )
    print("=" * 70)

    print(
        f"MAE                       : "
        f"{metrics['MAE']:.6f}"
    )

    print(
        f"RMSE                      : "
        f"{metrics['RMSE']:.6f}"
    )

    print(
        f"PSNR                      : "
        f"{metrics['PSNR']:.6f} dB"
    )

    print(
        f"SNR                       : "
        f"{metrics['SNR']:.6f} dB"
    )

    print(
        f"SSIM                      : "
        f"{metrics['SSIM']:.6f}"
    )

    print(
        f"Zero-filled missing MAE  : "
        f"{zero_filled_missing_mae:.6f}"
    )

    print(
        f"Dictionary missing MAE   : "
        f"{dictionary_missing_mae:.6f}"
    )

    print(
        f"Missing-region change    : "
        f"{missing_change:.6f}"
    )

    print(
        f"Observed preservation    : "
        f"{observed_difference:.6e}"
    )


    # =====================================================
    # REPRODUCIBILITY TEST
    # =====================================================

    print()
    print(
        "Testing deterministic dataset generation..."
    )

    dataset_repeat = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
        seed=SEED,
    )

    (
        corrupted_repeat,
        target_repeat,
        mask_repeat,
        velocity_repeat,
        mask_mode_repeat,
        geological_mode_repeat,
    ) = dataset_repeat[0]


    # =====================================================
    # TEST 15: DATASET REPRODUCIBILITY
    # =====================================================

    if not torch.equal(
        corrupted,
        corrupted_repeat
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset generation produced "
            "different corrupted data."
        )

    if not torch.equal(
        target,
        target_repeat
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset generation produced "
            "different targets."
        )

    if not torch.equal(
        mask,
        mask_repeat
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset generation produced "
            "different masks."
        )

    if not torch.equal(
        velocity,
        velocity_repeat
    ):

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset generation produced "
            "different velocity models."
        )

    if mask_mode_repeat != MASK_MODE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset returned a different "
            "mask mode."
        )

    if geological_mode_repeat != GEOLOGICAL_MODE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated dataset returned a different "
            "geological mode."
        )

    print(
        "PASS: dataset generation is reproducible"
    )


    # =====================================================
    # RECONSTRUCTION REPRODUCIBILITY
    # =====================================================

    print(
        "Testing deterministic Dictionary Learning "
        "reconstruction..."
    )

    reconstruction_repeat = (
        dictionary_learning_reconstruction(
            corrupted_cube=corrupted_repeat,
            mask=mask_repeat,
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


    # =====================================================
    # TEST 16: RECONSTRUCTION REPRODUCIBILITY
    # =====================================================

    reconstruction_difference = torch.max(
        torch.abs(
            reconstruction
            - reconstruction_repeat
        )
    ).item()

    if reconstruction_difference > OBSERVED_TOLERANCE:

        raise RuntimeError(
            "Dictionary Learning test failed: "
            "repeated reconstruction is not "
            "deterministic within the configured tolerance."
        )

    print(
        "PASS: Dictionary Learning reconstruction "
        "is reproducible"
    )

    print(
        f"      Maximum repeated reconstruction "
        f"difference: {reconstruction_difference:.6e}"
    )


    # =====================================================
    # FINAL TEST STATUS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "DICTIONARY LEARNING BASELINE TEST: PASS"
    )
    print("=" * 70)

    print()
    print(
        "The Dictionary Learning baseline passed the "
        "focused single-case validation."
    )

    print()
    print(
        "The 750-case controlled experiment remains "
        "separate under:"
    )

    print(
        "evaluation/baselines/"
        "dictionary_learning_controlled_matrix.py"
    )

    print()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()