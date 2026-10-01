"""
======================================================================
Curvelet POCS Baseline Test
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Independently validate the 3-D Curvelet POCS baseline
implementation for seismic data reconstruction.

This test validates:

    1. Synthetic dataset generation
    2. Tensor shape consistency
    3. Mask validity
    4. Input consistency
    5. Finite-value validation
    6. Curvelet POCS reconstruction
    7. Reconstruction output shape
    8. Reconstruction numerical stability
    9. Observed-data preservation
   10. Missing-sample reconstruction
   11. Reconstruction metrics
   12. Dataset reproducibility
   13. Curvelet reconstruction reproducibility

IMPORTANT
---------
This file is ONLY an implementation/integration test.

It does NOT run the 750-case controlled experiment.

The full controlled Curvelet POCS experiment is implemented
separately under:

    evaluation/baselines/curvelet_pocs_controlled_matrix.py

The present test executes ONE Curvelet POCS reconstruction
on ONE deterministic synthetic seismic sample.

Author:
    Ormin Joseph
======================================================================
"""


# ======================================================================
# IMPORTS
# ======================================================================

from __future__ import annotations

import numpy as np
import torch


# ======================================================================
# SYNTHETIC DATASET
# ======================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)


# ======================================================================
# ACTUAL CURVELET POCS IMPLEMENTATION
# ======================================================================
#
# IMPORTANT:
#
# This import must point to the file containing the actual
# Curvelet POCS reconstruction function.
#
# It must NOT import:
#
#     curvelet_pocs_controlled_matrix
#
# because that file belongs to the 750-case experiment.
#

from evaluation.baselines.curvelet_pocs_controlled_matrix import (
    curvelet_pocs_reconstruction
)


# ======================================================================
# RECONSTRUCTION METRICS
# ======================================================================

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# ======================================================================
# CENTRALIZED PROJECT CONFIGURATION
# ======================================================================

from utils.config import (
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,
    SEED,
    OBSERVED_PRESERVATION_TOLERANCE
)


# ======================================================================
# TEST CONFIGURATION
# ======================================================================

# Use the centralized synthetic cube size.
CUBE_SIZE = SYNTHETIC_PATCH_SIZE


# Use the centralized missing-data probability.
MISSING_PROBABILITY = (
    SYNTHETIC_MISSING_PROBABILITY
)


# Use the centralized project seed.
TEST_SEED = SEED


# Use the centralized observed-data preservation tolerance.
OBSERVED_TOLERANCE = (
    OBSERVED_PRESERVATION_TOLERANCE
)


# ======================================================================
# CURVELET POCS TEST CONFIGURATION
# ======================================================================
#
# These parameters define ONE Curvelet POCS test.
#
# They are not a controlled experimental matrix.
#

CURVELET_NUM_SCALES = 3

CURVELET_WEDGES_PER_DIRECTION = 3

CURVELET_ITERATIONS = 12

CURVELET_THRESHOLD = 0.05

CURVELET_THRESHOLD_DECAY = 0.90

CURVELET_TOLERANCE = 1.0e-5


# ======================================================================
# RECONSTRUCTION-CHANGE TOLERANCE
# ======================================================================
#
# This is used only to detect whether Curvelet POCS actually
# changed the missing region.
#

RECONSTRUCTION_CHANGE_TOLERANCE = 1.0e-8


# ======================================================================
# HELPER: MAXIMUM ABSOLUTE DIFFERENCE
# ======================================================================

def maximum_difference(
        tensor_a,
        tensor_b
):
    """
    Calculate the maximum absolute difference between
    two tensors.

    Parameters
    ----------
    tensor_a : torch.Tensor
        First tensor.

    tensor_b : torch.Tensor
        Second tensor.

    Returns
    -------
    float
        Maximum absolute difference.
    """

    difference = torch.abs(
        tensor_a - tensor_b
    )

    if difference.numel() == 0:
        return 0.0

    return float(
        difference.max().item()
    )


# ======================================================================
# HELPER: METRIC CONVERSION
# ======================================================================

def metric_to_float(
        value
):
    """
    Convert a metric result to a Python float.

    Supports:
        torch.Tensor
        NumPy scalar
        Python numeric values.
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

    return float(value)


# ======================================================================
# HELPER: COMPUTE METRICS
# ======================================================================

def compute_metrics(
        reconstruction,
        target
):
    """
    Calculate the standard seismic reconstruction metrics.
    """

    return {

        "mae": metric_to_float(
            mae(
                reconstruction,
                target
            )
        ),

        "rmse": metric_to_float(
            rmse(
                reconstruction,
                target
            )
        ),

        "psnr": metric_to_float(
            psnr(
                reconstruction,
                target
            )
        ),

        "snr": metric_to_float(
            snr(
                reconstruction,
                target
            )
        ),

        "ssim": metric_to_float(
            ssim(
                reconstruction,
                target
            )
        ),
    }


# ======================================================================
# MAIN TEST
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("CURVELET POCS BASELINE TEST")
    print("=" * 70)


    # ==================================================================
    # TEST CONFIGURATION
    # ==================================================================

    print()
    print("-" * 70)
    print("TEST CONFIGURATION")
    print("-" * 70)

    print(
        "Cube size           :",
        CUBE_SIZE
    )

    print(
        "Missing probability :",
        MISSING_PROBABILITY
    )

    print(
        "Test seed           :",
        TEST_SEED
    )

    print(
        "Observed tolerance  :",
        OBSERVED_TOLERANCE
    )

    print(
        "Curvelet scales     :",
        CURVELET_NUM_SCALES
    )

    print(
        "Wedges/direction    :",
        CURVELET_WEDGES_PER_DIRECTION
    )

    print(
        "POCS iterations     :",
        CURVELET_ITERATIONS
    )

    print(
        "Initial threshold   :",
        CURVELET_THRESHOLD
    )

    print(
        "Threshold decay     :",
        CURVELET_THRESHOLD_DECAY
    )

    print(
        "POCS tolerance      :",
        CURVELET_TOLERANCE
    )


    # ==================================================================
    # 1. BUILD SYNTHETIC DATASET
    # ==================================================================

    print()
    print("-" * 70)
    print("1. BUILDING SYNTHETIC TEST DATASET")
    print("-" * 70)


    dataset = SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_PROBABILITY,
        seed=TEST_SEED
    )


    assert len(dataset) == 1, (
        "Curvelet POCS test dataset must contain "
        "exactly one sample."
    )


    print(
        "Dataset size :",
        len(dataset)
    )


    # ==================================================================
    # 2. LOAD TEST SAMPLE
    # ==================================================================

    print()
    print("-" * 70)
    print("2. LOADING SYNTHETIC TEST SAMPLE")
    print("-" * 70)


    (
        corrupted,
        target,
        mask,
        velocity_model,
        mask_type,
        geological_mode
    ) = dataset[0]


    print(
        "Corrupted shape   :",
        tuple(corrupted.shape)
    )

    print(
        "Target shape      :",
        tuple(target.shape)
    )

    print(
        "Mask shape        :",
        tuple(mask.shape)
    )

    print(
        "Velocity shape    :",
        tuple(velocity_model.shape)
    )

    print(
        "Mask type         :",
        mask_type
    )

    print(
        "Geological mode   :",
        geological_mode
    )


    # ==================================================================
    # 3. SHAPE VALIDATION
    # ==================================================================

    print()
    print("-" * 70)
    print("3. VALIDATING TENSOR SHAPES")
    print("-" * 70)


    # Expected convention:
    #
    # [C, D, H, W]
    #

    assert corrupted.ndim == 4, (
        "Corrupted seismic data must have "
        "shape [C,D,H,W]."
    )


    expected_shape = (
        1,
        *CUBE_SIZE
    )


    assert tuple(
        corrupted.shape
    ) == expected_shape, (
        f"Unexpected corrupted shape: "
        f"{tuple(corrupted.shape)}. "
        f"Expected: {expected_shape}."
    )


    assert tuple(
        target.shape
    ) == expected_shape, (
        f"Unexpected target shape: "
        f"{tuple(target.shape)}. "
        f"Expected: {expected_shape}."
    )


    assert tuple(
        mask.shape
    ) == expected_shape, (
        f"Unexpected mask shape: "
        f"{tuple(mask.shape)}. "
        f"Expected: {expected_shape}."
    )


    assert tuple(
        velocity_model.shape
    ) == expected_shape, (
        f"Unexpected velocity-model shape: "
        f"{tuple(velocity_model.shape)}. "
        f"Expected: {expected_shape}."
    )


    print(
        "Tensor shape validation : PASSED"
    )


    # ==================================================================
    # 4. FINITE-VALUE VALIDATION
    # ==================================================================

    print()
    print("-" * 70)
    print("4. VALIDATING INPUT NUMERICAL VALUES")
    print("-" * 70)


    for name, tensor in [
        ("corrupted", corrupted),
        ("target", target),
        ("mask", mask),
        ("velocity_model", velocity_model),
    ]:

        assert torch.isfinite(
            tensor
        ).all(), (
            f"{name} contains NaN or Inf."
        )


    print(
        "Input finite-value validation : PASSED"
    )


    # ==================================================================
    # 5. MASK VALIDATION
    # ==================================================================

    print()
    print("-" * 70)
    print("5. VALIDATING OBSERVATION MASK")
    print("-" * 70)


    unique_mask_values = (
        torch.unique(mask)
        .detach()
        .cpu()
        .tolist()
    )


    print(
        "Mask values :",
        unique_mask_values
    )


    # Project convention:
    #
    #     1 = observed
    #     0 = missing
    #

    assert torch.all(
        (mask == 0) | (mask == 1)
    ), (
        "Mask contains values other than 0 and 1."
    )


    print(
        "Mask validation : PASSED"
    )


    # ==================================================================
    # 6. CHECK OBSERVED AND MISSING SAMPLES
    # ==================================================================

    print()
    print("-" * 70)
    print("6. CHECKING OBSERVED AND MISSING SAMPLES")
    print("-" * 70)


    observed_locations = (
        mask == 1
    )


    missing_locations = (
        mask == 0
    )


    observed_samples = int(
        observed_locations.sum().item()
    )


    missing_samples = int(
        missing_locations.sum().item()
    )


    print(
        "Observed samples :",
        observed_samples
    )

    print(
        "Missing samples  :",
        missing_samples
    )


    assert observed_samples > 0, (
        "No observed samples are available."
    )


    assert missing_samples > 0, (
        "No missing samples are available."
    )


    print(
        "Observed/missing sample validation : PASSED"
    )


    # ==================================================================
    # 7. INPUT CONSISTENCY
    # ==================================================================

    print()
    print("-" * 70)
    print("7. VALIDATING INPUT CONSISTENCY")
    print("-" * 70)


    # At observed locations, the corrupted seismic
    # input must equal the target seismic data.

    observed_input_difference = torch.abs(
        corrupted[observed_locations]
        -
        target[observed_locations]
    )


    if observed_input_difference.numel() > 0:

        maximum_input_difference = float(
            observed_input_difference.max().item()
        )

    else:

        maximum_input_difference = 0.0


    print(
        "Maximum observed-input difference :",
        f"{maximum_input_difference:.6e}"
    )


    assert (
        maximum_input_difference
        <= OBSERVED_TOLERANCE
    ), (
        "Observed samples in corrupted input "
        "do not match target samples."
    )


    print(
        "Input consistency : PASSED"
    )


    # ==================================================================
    # 8. RUN CURVELET POCS
    # ==================================================================

    print()
    print("-" * 70)
    print("8. RUNNING CURVELET POCS RECONSTRUCTION")
    print("-" * 70)


    # Clone the inputs before reconstruction.
    #
    # This protects the original test tensors from
    # accidental in-place modification.

    corrupted_for_curvelet = (
        corrupted.clone()
    )


    mask_for_curvelet = (
        mask.clone()
    )


    # --------------------------------------------------------------
    # Execute the actual Curvelet POCS implementation.
    #
    # Only ONE reconstruction is performed.
    #
    # There is NO 750-case loop here.
    # --------------------------------------------------------------

    reconstruction = (
        curvelet_pocs_reconstruction(
            corrupted_cube=(
                corrupted_for_curvelet
            ),

            mask=(
                mask_for_curvelet
            ),

            num_scales=(
                CURVELET_NUM_SCALES
            ),

            wedges_per_direction=(
                CURVELET_WEDGES_PER_DIRECTION
            ),

            iterations=(
                CURVELET_ITERATIONS
            ),

            threshold=(
                CURVELET_THRESHOLD
            ),

            threshold_decay=(
                CURVELET_THRESHOLD_DECAY
            ),

            tolerance=(
                CURVELET_TOLERANCE
            ),
        )
    )


    print(
        "Curvelet POCS reconstruction completed."
    )


    # ==================================================================
    # 9. OUTPUT SHAPE VALIDATION
    # ==================================================================

    print()
    print("-" * 70)
    print("9. VALIDATING RECONSTRUCTION OUTPUT SHAPE")
    print("-" * 70)


    assert tuple(
        reconstruction.shape
    ) == expected_shape, (
        f"Unexpected reconstruction shape: "
        f"{tuple(reconstruction.shape)}. "
        f"Expected: {expected_shape}."
    )


    print(
        "Reconstruction shape validation : PASSED"
    )


    # ==================================================================
    # 10. OUTPUT FINITE-VALUE VALIDATION
    # ==================================================================

    print()
    print("-" * 70)
    print("10. VALIDATING RECONSTRUCTION VALUES")
    print("-" * 70)


    assert torch.isfinite(
        reconstruction
    ).all(), (
        "Curvelet POCS reconstruction contains "
        "NaN or Inf."
    )


    print(
        "Reconstruction finite-value validation : PASSED"
    )


    # ==================================================================
    # 11. OBSERVED-DATA PRESERVATION
    # ==================================================================

    print()
    print("-" * 70)
    print("11. VALIDATING OBSERVED-DATA PRESERVATION")
    print("-" * 70)


    observed_difference = torch.abs(
        reconstruction[observed_locations]
        -
        corrupted[observed_locations]
    )


    if observed_difference.numel() > 0:

        maximum_observed_difference = float(
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
        <= OBSERVED_TOLERANCE
    ), (
        "Curvelet POCS modified observed "
        "seismic samples."
    )


    print(
        "Observed-data preservation : PASSED"
    )


    # ==================================================================
    # 12. MISSING-REGION RECONSTRUCTION
    # ==================================================================

    print()
    print("-" * 70)
    print("12. VALIDATING MISSING-REGION RECONSTRUCTION")
    print("-" * 70)


    # Calculate how much the reconstruction differs from
    # the original corrupted input inside the missing region.

    missing_difference = torch.abs(
        reconstruction[missing_locations]
        -
        corrupted[missing_locations]
    )


    if missing_difference.numel() > 0:

        missing_reconstruction_change = float(
            missing_difference.sum().item()
        )

    else:

        missing_reconstruction_change = 0.0


    print(
        "Total missing-region reconstruction change :",
        f"{missing_reconstruction_change:.6e}"
    )


    # A reconstruction method should produce values in
    # the missing region rather than leaving the region
    # completely unchanged from the corrupted input.

    assert (
        missing_reconstruction_change
        > RECONSTRUCTION_CHANGE_TOLERANCE
    ), (
        "Curvelet POCS did not modify the missing region."
    )


    print(
        "Missing-region reconstruction : PASSED"
    )


    # ==================================================================
    # 13. ZERO-FILLED INPUT MAE
    # ==================================================================

    print()
    print("-" * 70)
    print("13. CALCULATING ZERO-FILLED INPUT ERROR")
    print("-" * 70)


    zero_filled_mae = metric_to_float(
        mae(
            corrupted,
            target
        )
    )


    print(
        "Zero-filled input MAE :",
        f"{zero_filled_mae:.6f}"
    )


    # ==================================================================
    # 14. MISSING-REGION MAE
    # ==================================================================

    curvelet_missing_mae = metric_to_float(
        mae(
            reconstruction[missing_locations],
            target[missing_locations]
        )
    )


    print(
        "Curvelet missing-region MAE :",
        f"{curvelet_missing_mae:.6f}"
    )


    # ==================================================================
    # 15. FULL RECONSTRUCTION METRICS
    # ==================================================================

    print()
    print("-" * 70)
    print("14. CURVELET POCS RECONSTRUCTION METRICS")
    print("-" * 70)


    metrics = compute_metrics(
        reconstruction,
        target
    )


    print(
        "MAE  :",
        f"{metrics['mae']:.6f}"
    )


    print(
        "RMSE :",
        f"{metrics['rmse']:.6f}"
    )


    print(
        "PSNR :",
        f"{metrics['psnr']:.6f} dB"
    )


    print(
        "SNR  :",
        f"{metrics['snr']:.6f} dB"
    )


    print(
        "SSIM :",
        f"{metrics['ssim']:.6f}"
    )


    # ==================================================================
    # 16. DATASET REPRODUCIBILITY
    # ==================================================================

    print()
    print("-" * 70)
    print("15. DATASET REPRODUCIBILITY TEST")
    print("-" * 70)


    # Generate the same synthetic sample again using
    # exactly the same configuration and seed.

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


    # --------------------------------------------------------------
    # Compare input
    # --------------------------------------------------------------

    input_reproducibility_difference = (
        maximum_difference(
            corrupted,
            corrupted_repeat
        )
    )


    # --------------------------------------------------------------
    # Compare target
    # --------------------------------------------------------------

    target_reproducibility_difference = (
        maximum_difference(
            target,
            target_repeat
        )
    )


    # --------------------------------------------------------------
    # Compare mask
    # --------------------------------------------------------------

    mask_reproducibility_difference = (
        maximum_difference(
            mask,
            mask_repeat
        )
    )


    # --------------------------------------------------------------
    # Compare velocity model
    # --------------------------------------------------------------

    velocity_reproducibility_difference = (
        maximum_difference(
            velocity_model,
            velocity_repeat
        )
    )


    print(
        "Input difference :",
        f"{input_reproducibility_difference:.6e}"
    )

    print(
        "Target difference :",
        f"{target_reproducibility_difference:.6e}"
    )

    print(
        "Mask difference :",
        f"{mask_reproducibility_difference:.6e}"
    )

    print(
        "Velocity difference :",
        f"{velocity_reproducibility_difference:.6e}"
    )


    # --------------------------------------------------------------
    # Reproducibility assertions
    # --------------------------------------------------------------

    assert (
        input_reproducibility_difference
        <= OBSERVED_TOLERANCE
    ), (
        "Synthetic input is not reproducible."
    )


    assert (
        target_reproducibility_difference
        <= OBSERVED_TOLERANCE
    ), (
        "Synthetic target is not reproducible."
    )


    assert (
        mask_reproducibility_difference
        <= OBSERVED_TOLERANCE
    ), (
        "Synthetic mask is not reproducible."
    )


    assert (
        velocity_reproducibility_difference
        <= OBSERVED_TOLERANCE
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


    # ==================================================================
    # 17. CURVELET RECONSTRUCTION REPRODUCIBILITY
    # ==================================================================

    print()
    print("-" * 70)
    print("16. CURVELET POCS REPRODUCIBILITY TEST")
    print("-" * 70)


    # Run the exact same Curvelet POCS configuration again.

    reconstruction_repeat = (
        curvelet_pocs_reconstruction(
            corrupted_cube=(
                corrupted_repeat.clone()
            ),

            mask=(
                mask_repeat.clone()
            ),

            num_scales=(
                CURVELET_NUM_SCALES
            ),

            wedges_per_direction=(
                CURVELET_WEDGES_PER_DIRECTION
            ),

            iterations=(
                CURVELET_ITERATIONS
            ),

            threshold=(
                CURVELET_THRESHOLD
            ),

            threshold_decay=(
                CURVELET_THRESHOLD_DECAY
            ),

            tolerance=(
                CURVELET_TOLERANCE
            ),
        )
    )


    # Compare the two reconstructions.

    reconstruction_reproducibility_difference = (
        maximum_difference(
            reconstruction,
            reconstruction_repeat
        )
    )


    print(
        "Reconstruction difference :",
        f"{reconstruction_reproducibility_difference:.6e}"
    )


    assert (
        reconstruction_reproducibility_difference
        <= OBSERVED_TOLERANCE
    ), (
        "Curvelet POCS reconstruction is not "
        "reproducible."
    )


    print(
        "Curvelet POCS reproducibility : PASSED"
    )


    # ==================================================================
    # 18. FINAL VALIDATION
    # ==================================================================

    print()
    print("=" * 70)
    print("FINAL CURVELET POCS BASELINE VALIDATION")
    print("=" * 70)


    # Final shape validation.
    assert reconstruction.shape == target.shape


    # Final finite-value validation.
    assert torch.isfinite(
        reconstruction
    ).all()


    # Final observed-data preservation.
    assert (
        maximum_observed_difference
        <= OBSERVED_TOLERANCE
    )


    # Final reproducibility.
    assert (
        reconstruction_reproducibility_difference
        <= OBSERVED_TOLERANCE
    )


    print()
    print(
        "Curvelet POCS implementation validation : PASSED"
    )


    print()
    print(
        "ALL CURVELET POCS BASELINE TESTS PASSED"
    )


    print()


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()