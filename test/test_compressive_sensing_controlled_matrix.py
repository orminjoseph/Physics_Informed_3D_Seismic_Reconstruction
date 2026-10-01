"""
=========================================================
Compressive Sensing Baseline Test
=========================================================

Independent validation of the Compressive Sensing (CS)
baseline for 3D seismic reconstruction.

This test validates:

    1. Synthetic dataset generation
    2. Tensor shape validation
    3. Mask validation
    4. Input consistency
    5. CS reconstruction
    6. Finite-value validation
    7. Observed-data preservation
    8. Missing-sample reconstruction
    9. Reproducibility
   10. Reconstruction metrics

IMPORTANT
---------

This file is ONLY an implementation/integration test.

It does NOT run the 750-case controlled experiment.

The 750-case scientific experiment is implemented separately
in:

    evaluation/baselines/compressive_sensing_controlled_matrix.py

This test calls only the actual Compressive Sensing
reconstruction function.

Author: Ormin Joseph
=========================================================
"""


# =========================================================
# IMPORTS
# =========================================================

import torch


# ---------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)


# ---------------------------------------------------------
# Actual Compressive Sensing reconstruction algorithm
# ---------------------------------------------------------
#
# IMPORTANT:
# This must point to the file containing the actual CS
# reconstruction function.
#
# It must NOT import the controlled 750-case experiment.
#

from evaluation.baselines.compressive_sensing_controlled_matrix import (
    compressive_sensing_reconstruction
)


# ---------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ---------------------------------------------------------
# Centralized project configuration
# ---------------------------------------------------------

from utils.config import (
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,
    SEED,
    OBSERVED_PRESERVATION_TOLERANCE
)


# =========================================================
# TEST SETTINGS
# =========================================================

# Synthetic seismic cube size.
#
# This comes directly from utils/config.py.
CUBE_SIZE = SYNTHETIC_PATCH_SIZE


# Missing-data probability.
#
# This also comes directly from utils/config.py.
MISSING_PROBABILITY = (
    SYNTHETIC_MISSING_PROBABILITY
)


# Reproducibility seed.
TEST_SEED = SEED


# Maximum permitted difference between observed input
# samples and reconstructed observed samples.
PRESERVATION_TOLERANCE = (
    OBSERVED_PRESERVATION_TOLERANCE
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def maximum_difference(
        tensor_a,
        tensor_b
):
    """
    Calculate the maximum absolute difference between
    two tensors.

    This function is used for reproducibility and
    consistency validation.
    """

    difference = torch.abs(
        tensor_a - tensor_b
    )

    # Protect against an empty tensor.
    if difference.numel() == 0:
        return 0.0

    return difference.max().item()


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print()
    print("=" * 70)
    print("COMPRESSIVE SENSING BASELINE TEST")
    print("=" * 70)


    # =====================================================
    # TEST CONFIGURATION
    # =====================================================

    print()
    print("Cube size           :", CUBE_SIZE)
    print(
        "Missing probability :",
        MISSING_PROBABILITY
    )
    print(
        "Test seed           :",
        TEST_SEED
    )
    print(
        "Preservation tol.   :",
        PRESERVATION_TOLERANCE
    )


    # =====================================================
    # 1. BUILD SYNTHETIC DATASET
    # =====================================================

    print()
    print("-" * 70)
    print("1. BUILDING SYNTHETIC TEST SAMPLE")
    print("-" * 70)


    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_PROBABILITY,
        seed=TEST_SEED
    )


    print(
        "Dataset samples:",
        len(dataset)
    )


    # The test requires exactly one sample.
    assert len(dataset) == 1, (
        "The CS test dataset must contain exactly "
        "one sample."
    )


    # =====================================================
    # 2. LOAD SAMPLE
    # =====================================================

    (
        corrupted,
        target,
        mask,
        velocity_model,
        mask_type,
        geological_mode
    ) = dataset[0]


    print()
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

    print(
        "Velocity shape   :",
        tuple(velocity_model.shape)
    )

    print(
        "Mask type        :",
        mask_type
    )

    print(
        "Geological mode  :",
        geological_mode
    )


    # =====================================================
    # 3. TENSOR SHAPE VALIDATION
    # =====================================================

    print()
    print("-" * 70)
    print("2. VALIDATING TENSOR SHAPES")
    print("-" * 70)


    # Corrupted input and target must have the same shape.
    assert corrupted.shape == target.shape, (
        "Corrupted input and target shapes differ."
    )


    # Corrupted input and mask must have the same shape.
    assert corrupted.shape == mask.shape, (
        "Corrupted input and mask shapes differ."
    )


    # Velocity model must match the seismic tensor shape.
    assert corrupted.shape == velocity_model.shape, (
        "Corrupted input and velocity-model shapes differ."
    )


    # The project convention is:
    #
    # [C, D, H, W]
    #
    assert corrupted.ndim == 4, (
        "Expected seismic tensor with shape [C,D,H,W]."
    )


    print(
        "Tensor shape validation : PASSED"
    )


    # =====================================================
    # 4. FINITE-VALUE VALIDATION
    # =====================================================

    print()
    print("-" * 70)
    print("3. VALIDATING INPUT VALUES")
    print("-" * 70)


    # Check corrupted seismic data.
    assert torch.isfinite(
        corrupted
    ).all(), (
        "Corrupted input contains NaN or Inf."
    )


    # Check target seismic data.
    assert torch.isfinite(
        target
    ).all(), (
        "Target contains NaN or Inf."
    )


    # Check mask.
    assert torch.isfinite(
        mask
    ).all(), (
        "Mask contains NaN or Inf."
    )


    # Check velocity model.
    assert torch.isfinite(
        velocity_model
    ).all(), (
        "Velocity model contains NaN or Inf."
    )


    print(
        "Input finite-value validation : PASSED"
    )


    # =====================================================
    # 5. MASK VALIDATION
    # =====================================================

    print()
    print("-" * 70)
    print("4. VALIDATING OBSERVATION MASK")
    print("-" * 70)


    unique_mask = torch.unique(
        mask
    )


    print(
        "Mask values:",
        unique_mask.tolist()
    )


    # The project convention is:
    #
    # 1 = observed
    # 0 = missing
    #
    assert torch.all(
        (mask == 0) | (mask == 1)
    ), (
        "Mask contains values other than 0 and 1."
    )


    print(
        "Mask validation : PASSED"
    )


    # =====================================================
    # 6. COUNT OBSERVED AND MISSING SAMPLES
    # =====================================================

    print()
    print("-" * 70)
    print("5. CHECKING OBSERVED AND MISSING SAMPLES")
    print("-" * 70)


    observed_locations = (
        mask == 1
    )


    missing_locations = (
        mask == 0
    )


    number_observed = (
        observed_locations.sum().item()
    )


    number_missing = (
        missing_locations.sum().item()
    )


    print(
        "Observed samples :",
        number_observed
    )


    print(
        "Missing samples  :",
        number_missing
    )


    # Both regions must exist for this test.
    assert number_observed > 0, (
        "Test sample contains no observed samples."
    )


    assert number_missing > 0, (
        "Test sample contains no missing samples."
    )


    print(
        "Observed/missing sample validation : PASSED"
    )


    # =====================================================
    # 7. INPUT CONSISTENCY
    # =====================================================

    print()
    print("-" * 70)
    print("6. VALIDATING INPUT CONSISTENCY")
    print("-" * 70)


    # At observed locations, the corrupted input should
    # contain the original target seismic values.

    observed_input_difference = torch.abs(
        corrupted[observed_locations]
        -
        target[observed_locations]
    )


    if observed_input_difference.numel() > 0:

        maximum_observed_input_difference = (
            observed_input_difference.max().item()
        )

    else:

        maximum_observed_input_difference = 0.0


    print(
        "Maximum observed-input difference :",
        f"{maximum_observed_input_difference:.6e}"
    )


    assert (
        maximum_observed_input_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "Observed input samples do not match "
        "the target samples."
    )


    print(
        "Input consistency : PASSED"
    )


    # =====================================================
    # 8. RUN COMPRESSIVE SENSING
    # =====================================================

    print()
    print("-" * 70)
    print("7. RUNNING COMPRESSIVE SENSING")
    print("-" * 70)


    # Use clones so that accidental in-place operations
    # inside the baseline cannot modify the original
    # test tensors.

    corrupted_for_cs = (
        corrupted.clone()
    )


    mask_for_cs = (
        mask.clone()
    )


    # Run ONLY the actual CS reconstruction algorithm.
    #
    # This is one reconstruction test.
    #
    # No 750-case loop is performed here.

    prediction = (
        compressive_sensing_reconstruction(
            corrupted_for_cs,
            mask_for_cs
        )
    )


    print(
        "CS reconstruction completed."
    )


    # =====================================================
    # 9. OUTPUT SHAPE VALIDATION
    # =====================================================

    print()
    print("-" * 70)
    print("8. VALIDATING RECONSTRUCTION OUTPUT SHAPE")
    print("-" * 70)


    assert prediction.shape == corrupted.shape, (
        "CS reconstruction changed the input shape."
    )


    print(
        "Output shape validation : PASSED"
    )


    # =====================================================
    # 10. OUTPUT FINITE-VALUE VALIDATION
    # =====================================================

    print()
    print("-" * 70)
    print("9. VALIDATING RECONSTRUCTION VALUES")
    print("-" * 70)


    assert torch.isfinite(
        prediction
    ).all(), (
        "CS reconstruction contains NaN or Inf."
    )


    print(
        "Reconstruction finite-value validation : PASSED"
    )


    # =====================================================
    # 11. OBSERVED-DATA PRESERVATION
    # =====================================================

    print()
    print("-" * 70)
    print("10. VALIDATING OBSERVED-DATA PRESERVATION")
    print("-" * 70)


    observed_difference = torch.abs(
        prediction[observed_locations]
        -
        corrupted[observed_locations]
    )


    if observed_difference.numel() > 0:

        maximum_observed_difference = (
            observed_difference.max().item()
        )

    else:

        maximum_observed_difference = 0.0


    print(
        "Maximum observed-data difference :",
        f"{maximum_observed_difference:.6e}"
    )


    assert (
        maximum_observed_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "CS reconstruction modified observed "
        "seismic samples."
    )


    print(
        "Observed-data preservation : PASSED"
    )


    # =====================================================
    # 12. MISSING-SAMPLE RECONSTRUCTION
    # =====================================================

    print()
    print("-" * 70)
    print("11. VALIDATING MISSING-SAMPLE RECONSTRUCTION")
    print("-" * 70)


    missing_prediction = (
        prediction[missing_locations]
    )


    assert missing_prediction.numel() > 0, (
        "No missing samples are available for testing."
    )


    assert torch.isfinite(
        missing_prediction
    ).all(), (
        "Missing-sample reconstruction contains "
        "NaN or Inf."
    )


    print(
        "Missing-sample reconstruction : PASSED"
    )


    # =====================================================
    # 13. ZERO-FILLED INPUT MAE
    # =====================================================

    print()
    print("-" * 70)
    print("12. CALCULATING ZERO-FILLED INPUT ERROR")
    print("-" * 70)


    zero_filled_mae = (
        mae(
            corrupted,
            target
        ).item()
    )


    print(
        "Zero-filled input MAE :",
        f"{zero_filled_mae:.6f}"
    )


    # =====================================================
    # 14. CS MISSING-REGION MAE
    # =====================================================

    cs_missing_mae = (
        mae(
            prediction[missing_locations],
            target[missing_locations]
        ).item()
    )


    print(
        "CS missing-sample MAE :",
        f"{cs_missing_mae:.6f}"
    )


    # =====================================================
    # 15. FULL RECONSTRUCTION METRICS
    # =====================================================

    print()
    print("-" * 70)
    print("13. CS RECONSTRUCTION METRICS")
    print("-" * 70)


    cs_mae = (
        mae(
            prediction,
            target
        ).item()
    )


    cs_rmse = (
        rmse(
            prediction,
            target
        ).item()
    )


    cs_psnr = (
        psnr(
            prediction,
            target
        ).item()
    )


    cs_snr = (
        snr(
            prediction,
            target
        ).item()
    )


    cs_ssim = (
        ssim(
            prediction.unsqueeze(0),
            target.unsqueeze(0)
        ).item()
    )


    print(
        "MAE  :",
        f"{cs_mae:.6f}"
    )


    print(
        "RMSE :",
        f"{cs_rmse:.6f}"
    )


    print(
        "PSNR :",
        f"{cs_psnr:.6f} dB"
    )


    print(
        "SNR  :",
        f"{cs_snr:.6f} dB"
    )


    print(
        "SSIM :",
        f"{cs_ssim:.6f}"
    )


    # =====================================================
    # 16. DATASET REPRODUCIBILITY
    # =====================================================

    print()
    print("-" * 70)
    print("14. DATASET REPRODUCIBILITY TEST")
    print("-" * 70)


    # Generate exactly the same synthetic sample again
    # using the same seed.

    dataset_repeat = (
        SyntheticSeismicDataset(
            num_samples=1,
            cube_size=CUBE_SIZE,
            missing_probability=MISSING_PROBABILITY,
            seed=TEST_SEED
        )
    )


    (
        corrupted_repeat,
        target_repeat,
        mask_repeat,
        velocity_repeat,
        mask_type_repeat,
        geological_mode_repeat
    ) = dataset_repeat[0]


    # -----------------------------------------------------
    # Compare generated input
    # -----------------------------------------------------

    input_difference = (
        maximum_difference(
            corrupted,
            corrupted_repeat
        )
    )


    # Compare target.
    target_difference = (
        maximum_difference(
            target,
            target_repeat
        )
    )


    # Compare mask.
    mask_difference = (
        maximum_difference(
            mask,
            mask_repeat
        )
    )


    # Compare velocity model.
    velocity_difference = (
        maximum_difference(
            velocity_model,
            velocity_repeat
        )
    )


    print(
        "Input reproducibility difference :",
        f"{input_difference:.6e}"
    )


    print(
        "Target reproducibility difference :",
        f"{target_difference:.6e}"
    )


    print(
        "Mask reproducibility difference :",
        f"{mask_difference:.6e}"
    )


    print(
        "Velocity reproducibility difference :",
        f"{velocity_difference:.6e}"
    )


    # -----------------------------------------------------
    # Reproducibility assertions
    # -----------------------------------------------------

    assert (
        input_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "Synthetic input is not reproducible."
    )


    assert (
        target_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "Synthetic target is not reproducible."
    )


    assert (
        mask_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "Synthetic mask is not reproducible."
    )


    assert (
        velocity_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "Synthetic velocity model is not reproducible."
    )


    assert (
        mask_type
        ==
        mask_type_repeat
    ), (
        "Mask type is not reproducible."
    )


    assert (
        geological_mode
        ==
        geological_mode_repeat
    ), (
        "Geological mode is not reproducible."
    )


    print(
        "Dataset reproducibility : PASSED"
    )


    # =====================================================
    # 17. CS RECONSTRUCTION REPRODUCIBILITY
    # =====================================================

    print()
    print("-" * 70)
    print("15. CS RECONSTRUCTION REPRODUCIBILITY TEST")
    print("-" * 70)


    # Run the same CS algorithm again using the
    # reproducibly generated input.

    prediction_repeat = (
        compressive_sensing_reconstruction(
            corrupted_repeat.clone(),
            mask_repeat.clone()
        )
    )


    prediction_difference = (
        maximum_difference(
            prediction,
            prediction_repeat
        )
    )


    print(
        "CS reproducibility difference :",
        f"{prediction_difference:.6e}"
    )


    assert (
        prediction_difference
        <= PRESERVATION_TOLERANCE
    ), (
        "CS reconstruction is not reproducible."
    )


    print(
        "CS reconstruction reproducibility : PASSED"
    )


    # =====================================================
    # 18. FINAL VALIDATION
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL CS BASELINE VALIDATION")
    print("=" * 70)


    # Final output-shape check.
    assert prediction.shape == target.shape


    # Final finite-value check.
    assert torch.isfinite(
        prediction
    ).all()


    # Final observed-data preservation check.
    assert (
        maximum_observed_difference
        <= PRESERVATION_TOLERANCE
    )


    # Final reproducibility check.
    assert (
        prediction_difference
        <= PRESERVATION_TOLERANCE
    )


    print()
    print(
        "CS implementation validation : PASSED"
    )


    print()
    print(
        "ALL COMPRESSIVE SENSING BASELINE "
        "TESTS PASSED"
    )


    print()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()