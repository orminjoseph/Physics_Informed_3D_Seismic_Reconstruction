"""
=========================================================
Compressive Sensing Baseline Test
=========================================================

Independent validation of the Compressive Sensing (CS)
baseline for 3D seismic reconstruction.

Validation includes:

    1. Dataset generation
    2. Tensor shape validation
    3. Mask validation
    4. Input consistency
    5. CS reconstruction
    6. Finite-value validation
    7. Observed-data preservation
    8. Missing-sample reconstruction
    9. Reproducibility
   10. Reconstruction metrics

This is an implementation/integration test.

It is NOT the final 750-case scientific benchmark.

Author: Ormin Joseph
=========================================================
"""

import torch

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from evaluation.baselines.compressive_sensing import (
    compressive_sensing_reconstruction
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,
    SEED
)


# =========================================================
# TEST SETTINGS
# =========================================================

CUBE_SIZE = SYNTHETIC_PATCH_SIZE

MISSING_PROBABILITY = (
    SYNTHETIC_MISSING_PROBABILITY
)

TEST_SEED = SEED


# =========================================================
# HELPER
# =========================================================

def maximum_difference(
        tensor_a,
        tensor_b
):
    """
    Return the maximum absolute difference between
    two tensors.
    """

    return torch.max(
        torch.abs(
            tensor_a - tensor_b
        )
    ).item()


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print()
    print("=" * 65)
    print("COMPRESSIVE SENSING BASELINE TEST")
    print("=" * 65)

    print()
    print("Cube size           :", CUBE_SIZE)
    print(
        "Missing probability :",
        MISSING_PROBABILITY
    )
    print(
        "Seed                :",
        TEST_SEED
    )

    # =====================================================
    # 1. BUILD SYNTHETIC DATASET
    # =====================================================

    print()
    print("-" * 65)
    print("1. BUILDING SYNTHETIC TEST SAMPLE")
    print("-" * 65)

    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=(
            MISSING_PROBABILITY
        ),
        seed=TEST_SEED
    )

    print(
        "Dataset samples:",
        len(dataset)
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
    # 3. SHAPE VALIDATION
    # =====================================================

    assert corrupted.shape == (
        target.shape
    ), (
        "Input and target shapes differ."
    )

    assert corrupted.shape == (
        mask.shape
    ), (
        "Input and mask shapes differ."
    )

    assert corrupted.shape == (
        velocity_model.shape
    ), (
        "Input and velocity shapes differ."
    )

    print(
        "Shape validation : PASSED"
    )

    # =====================================================
    # 4. MASK VALIDATION
    # =====================================================

    unique_mask = torch.unique(
        mask
    )

    print(
        "Mask values      :",
        unique_mask.tolist()
    )

    assert torch.all(
        (mask == 0) |
        (mask == 1)
    ), (
        "Mask contains values other than 0 and 1."
    )

    print(
        "Mask validation  : PASSED"
    )

    # =====================================================
    # 5. INPUT CONSISTENCY
    # =====================================================

    # The dataset convention is:
    #
    # corrupted = target * mask

    reconstructed_input = (
        target * mask
    )

    consistency_error = (
        maximum_difference(
            corrupted,
            reconstructed_input
        )
    )

    print()
    print(
        "Input consistency error :",
        f"{consistency_error:.6e}"
    )

    assert consistency_error <= 1.0e-6, (
        "Input consistency validation failed."
    )

    print(
        "Input consistency : PASSED"
    )

    # =====================================================
    # 6. COUNT MISSING SAMPLES
    # =====================================================

    missing_locations = (
        mask == 0
    )

    observed_locations = (
        mask == 1
    )

    number_missing = (
        missing_locations.sum().item()
    )

    number_observed = (
        observed_locations.sum().item()
    )

    print()
    print(
        "Observed samples :",
        number_observed
    )

    print(
        "Missing samples  :",
        number_missing
    )

    assert number_missing > 0, (
        "Test sample contains no missing samples."
    )

    # =====================================================
    # 7. RUN COMPRESSIVE SENSING
    # =====================================================

    print()
    print("-" * 65)
    print("2. RUNNING COMPRESSIVE SENSING")
    print("-" * 65)

    prediction = (
        compressive_sensing_reconstruction(
            corrupted,
            mask
        )
    )

    print(
        "CS reconstruction completed."
    )

    # =====================================================
    # 8. OUTPUT SHAPE VALIDATION
    # =====================================================

    assert prediction.shape == (
        corrupted.shape
    ), (
        "CS reconstruction changed the input shape."
    )

    print(
        "Output shape validation : PASSED"
    )

    # =====================================================
    # 9. FINITE VALUE VALIDATION
    # =====================================================

    assert torch.isfinite(
        prediction
    ).all(), (
        "CS reconstruction contains NaN or Inf."
    )

    print(
        "Finite-value validation : PASSED"
    )

    # =====================================================
    # 10. OBSERVED DATA PRESERVATION
    # =====================================================

    observed_difference = (
        torch.abs(
            prediction[observed_locations]
            -
            corrupted[observed_locations]
        )
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
        <= 1.0e-6
    ), (
        "Observed seismic samples were modified."
    )

    print(
        "Observed-data preservation : PASSED"
    )

    # =====================================================
    # 11. MISSING-SAMPLE RECONSTRUCTION
    # =====================================================

    missing_prediction = (
        prediction[missing_locations]
    )

    assert missing_prediction.numel() > 0, (
        "No missing samples available for testing."
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
    # 12. ZERO-FILLED INPUT ERROR
    # =====================================================

    zero_filled_mae = mae(
        corrupted,
        target
    ).item()

    print()
    print(
        "Zero-filled input MAE :",
        f"{zero_filled_mae:.6f}"
    )

    # =====================================================
    # 13. CS ERROR ON MISSING SAMPLES
    # =====================================================

    cs_missing_mae = mae(
        prediction[missing_locations],
        target[missing_locations]
    ).item()

    print(
        "CS missing-sample MAE :",
        f"{cs_missing_mae:.6f}"
    )

    # =====================================================
    # 14. FULL RECONSTRUCTION METRICS
    # =====================================================

    print()
    print("-" * 65)
    print("3. CS RECONSTRUCTION METRICS")
    print("-" * 65)

    cs_mae = mae(
        prediction,
        target
    ).item()

    cs_rmse = rmse(
        prediction,
        target
    ).item()

    cs_psnr = psnr(
        prediction,
        target
    ).item()

    cs_snr = snr(
        prediction,
        target
    ).item()

    cs_ssim = ssim(
        prediction.unsqueeze(0),
        target.unsqueeze(0)
    ).item()

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
    # 15. REPRODUCIBILITY
    # =====================================================

    print()
    print("-" * 65)
    print("4. REPRODUCIBILITY TEST")
    print("-" * 65)

    dataset_repeat = (
        SyntheticSeismicDataset(
            num_samples=1,
            cube_size=CUBE_SIZE,
            missing_probability=(
                MISSING_PROBABILITY
            ),
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
    # Dataset reproducibility
    # -----------------------------------------------------

    input_difference = maximum_difference(
        corrupted,
        corrupted_repeat
    )

    target_difference = maximum_difference(
        target,
        target_repeat
    )

    mask_difference = maximum_difference(
        mask,
        mask_repeat
    )

    velocity_difference = maximum_difference(
        velocity_model,
        velocity_repeat
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

    assert input_difference <= 1.0e-6
    assert target_difference <= 1.0e-6
    assert mask_difference <= 1.0e-6
    assert velocity_difference <= 1.0e-6

    assert (
        mask_type == mask_type_repeat
    )

    assert (
        geological_mode
        ==
        geological_mode_repeat
    )

    print(
        "Dataset reproducibility : PASSED"
    )

    # -----------------------------------------------------
    # CS reproducibility
    # -----------------------------------------------------

    prediction_repeat = (
        compressive_sensing_reconstruction(
            corrupted_repeat,
            mask_repeat
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

    assert prediction_difference <= 1.0e-6

    print(
        "CS reproducibility : PASSED"
    )

    # =====================================================
    # 16. FINAL VALIDATION
    # =====================================================

    print()
    print("=" * 65)
    print("FINAL CS BASELINE VALIDATION")
    print("=" * 65)

    assert torch.isfinite(
        prediction
    ).all()

    assert prediction.shape == (
        target.shape
    )

    assert (
        maximum_observed_difference
        <= 1.0e-6
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