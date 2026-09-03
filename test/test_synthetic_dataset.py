"""
=========================================================
Synthetic 3D Seismic Dataset Test
=========================================================

Tests the complete SyntheticSeismicDataset including:

    1. Dataset length
    2. Tensor shapes
    3. Tensor data types
    4. Finite values
    5. Sampling mask
    6. Input/mask consistency
    7. Missing voxel fraction
    8. Velocity model
    9. Mask type tracking
   10. Geological mode tracking
   11. Explicit mask modes
   12. Random mask mode
   13. All dataset samples
   14. PyTorch DataLoader

Author: Ormin Joseph
=========================================================
"""

import torch

from torch.utils.data import DataLoader

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)


# =====================================================
# CONFIGURATION
# =====================================================

CUBE_SIZE = (16, 32, 32)

NUM_SAMPLES = 5

MISSING_PROBABILITY = 0.30


# =====================================================
# VALID MASK TYPES
# =====================================================

VALID_MASK_TYPES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks"
]


# =====================================================
# VALID GEOLOGICAL MODES
# =====================================================

VALID_GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex"
]


# =====================================================
# DATASET CREATION
# =====================================================

def create_dataset(
    mask_mode="random"
):
    """
    Create a small synthetic dataset for testing.
    """

    return SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_PROBABILITY,
        geological_mode="random",
        mask_mode=mask_mode
    )


# =====================================================
# TEST DATASET LENGTH
# =====================================================

def test_dataset_length():

    print()
    print("Testing Dataset Length")

    dataset = create_dataset()

    assert len(dataset) == NUM_SAMPLES

    print(
        f"Dataset Size: {len(dataset)}"
    )

    print(
        "Dataset Length Test: PASSED"
    )


# =====================================================
# TEST SAMPLE SHAPES
# =====================================================

def test_sample_shapes():

    print()
    print("Testing Sample Shapes")

    dataset = create_dataset()

    (
        input_cube,
        target_cube,
        mask,
        velocity,
        mask_type,
        geological_mode
    ) = dataset[0]

    expected_shape = torch.Size(
        (
            1,
            *CUBE_SIZE
        )
    )

    print(
        f"Input Shape       : "
        f"{input_cube.shape}"
    )

    print(
        f"Target Shape      : "
        f"{target_cube.shape}"
    )

    print(
        f"Mask Shape        : "
        f"{mask.shape}"
    )

    print(
        f"Velocity Shape    : "
        f"{velocity.shape}"
    )

    print(
        f"Mask Type         : "
        f"{mask_type}"
    )

    print(
        f"Geological Mode   : "
        f"{geological_mode}"
    )

    assert input_cube.shape == expected_shape

    assert target_cube.shape == expected_shape

    assert mask.shape == expected_shape

    assert velocity.shape == expected_shape

    print(
        "Sample Shape Test: PASSED"
    )


# =====================================================
# TEST TENSOR TYPES
# =====================================================

def test_tensor_types():

    print()
    print("Testing Tensor Types")

    dataset = create_dataset()

    (
        input_cube,
        target_cube,
        mask,
        velocity,
        _,
        _
    ) = dataset[0]

    assert input_cube.dtype == torch.float32

    assert target_cube.dtype == torch.float32

    assert mask.dtype == torch.float32

    assert velocity.dtype == torch.float32

    print(
        "Tensor Type Test: PASSED"
    )


# =====================================================
# TEST FINITE VALUES
# =====================================================

def test_finite_values():

    print()
    print("Testing Finite Values")

    dataset = create_dataset()

    (
        input_cube,
        target_cube,
        mask,
        velocity,
        _,
        _
    ) = dataset[0]

    assert torch.isfinite(
        input_cube
    ).all()

    assert torch.isfinite(
        target_cube
    ).all()

    assert torch.isfinite(
        mask
    ).all()

    assert torch.isfinite(
        velocity
    ).all()

    print(
        "Finite Value Test: PASSED"
    )


# =====================================================
# TEST SAMPLING MASK
# =====================================================

def test_sampling_mask():

    print()
    print("Testing Sampling Mask")

    dataset = create_dataset()

    (
        _,
        _,
        mask,
        _,
        _,
        _
    ) = dataset[0]

    unique_values = torch.unique(
        mask
    )

    print(
        f"Mask Values: "
        f"{unique_values.tolist()}"
    )

    assert torch.isfinite(
        mask
    ).all()

    assert torch.all(
        (unique_values == 0.0)
        |
        (unique_values == 1.0)
    )

    print(
        "Mask Value Test: PASSED"
    )


# =====================================================
# TEST INPUT / MASK CONSISTENCY
# =====================================================

def test_input_mask_consistency():

    print()
    print(
        "Testing Input and Mask Consistency"
    )

    dataset = create_dataset()

    (
        input_cube,
        target_cube,
        mask,
        _,
        _,
        _
    ) = dataset[0]

    expected_input = (
        target_cube * mask
    )

    assert torch.allclose(
        input_cube,
        expected_input,
        atol=1.0e-6
    )

    print(
        "Input/Mask Consistency Test: PASSED"
    )


# =====================================================
# TEST MISSING VOXEL FRACTION
# =====================================================

def test_missing_voxels():

    print()
    print("Testing Missing Seismic Voxels")

    dataset = create_dataset(
        mask_mode="random_voxels"
    )

    (
        _,
        _,
        mask,
        _,
        mask_type,
        _
    ) = dataset[0]

    assert mask_type == "random_voxels"

    missing_voxels = torch.sum(
        mask == 0.0
    ).item()

    total_voxels = mask.numel()

    missing_fraction = (
        missing_voxels
        /
        total_voxels
    )

    print(
        f"Mask Type       : "
        f"{mask_type}"
    )

    print(
        f"Missing Voxels  : "
        f"{missing_voxels}"
    )

    print(
        f"Total Voxels    : "
        f"{total_voxels}"
    )

    print(
        f"Missing Fraction: "
        f"{missing_fraction:.4f}"
    )

    # -------------------------------------------------
    # Allow stochastic variation.
    # -------------------------------------------------

    tolerance = 0.05

    assert abs(
        missing_fraction
        -
        MISSING_PROBABILITY
    ) < tolerance

    print(
        "Missing Voxel Test: PASSED"
    )


# =====================================================
# TEST VELOCITY MODEL
# =====================================================

def test_velocity_model():

    print()
    print("Testing Velocity Model")

    dataset = create_dataset()

    (
        _,
        _,
        _,
        velocity,
        _,
        _
    ) = dataset[0]

    minimum_velocity = (
        velocity.min().item()
    )

    maximum_velocity = (
        velocity.max().item()
    )

    print(
        f"Velocity Minimum : "
        f"{minimum_velocity:.2f}"
    )

    print(
        f"Velocity Maximum : "
        f"{maximum_velocity:.2f}"
    )

    assert torch.isfinite(
        velocity
    ).all()

    assert torch.all(
        velocity > 0
    )

    assert minimum_velocity >= 1800.0

    assert maximum_velocity <= 3500.0

    print(
        "Velocity Model Test: PASSED"
    )


# =====================================================
# TEST MASK TYPE TRACKING
# =====================================================

def test_mask_type_tracking():

    print()
    print("Testing Mask Type Tracking")

    dataset = create_dataset(
        mask_mode="random"
    )

    # -------------------------------------------------
    # Verify storage length.
    # -------------------------------------------------

    assert len(
        dataset.mask_types
    ) == NUM_SAMPLES

    # -------------------------------------------------
    # Verify stored values.
    # -------------------------------------------------

    for mask_type in dataset.mask_types:

        assert mask_type in VALID_MASK_TYPES

    # -------------------------------------------------
    # Verify returned metadata.
    # -------------------------------------------------

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        returned_mask_type = sample[4]

        assert returned_mask_type == (
            dataset.mask_types[index]
        )

    print()
    print(
        "Stored Mask Types:"
    )

    for index, mask_type in enumerate(
        dataset.mask_types
    ):

        print(
            f"Sample {index + 1}: "
            f"{mask_type}"
        )

    print(
        "Mask Type Tracking Test: PASSED"
    )


# =====================================================
# TEST GEOLOGICAL MODE TRACKING
# =====================================================

def test_geological_mode_tracking():

    print()
    print(
        "Testing Geological Mode Tracking"
    )

    dataset = create_dataset()

    # -------------------------------------------------
    # Verify storage length.
    # -------------------------------------------------

    assert len(
        dataset.geological_modes
    ) == NUM_SAMPLES

    # -------------------------------------------------
    # Verify stored geological modes.
    # -------------------------------------------------

    for mode in dataset.geological_modes:

        assert mode in VALID_GEOLOGICAL_MODES

    # -------------------------------------------------
    # Verify returned metadata.
    # -------------------------------------------------

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        returned_mode = sample[5]

        assert returned_mode == (
            dataset.geological_modes[index]
        )

    print()
    print(
        "Stored Geological Modes:"
    )

    for index, mode in enumerate(
        dataset.geological_modes
    ):

        print(
            f"Sample {index + 1}: "
            f"{mode}"
        )

    print(
        "Geological Mode Tracking Test: PASSED"
    )


# =====================================================
# TEST EXPLICIT MASK MODES
# =====================================================

def test_explicit_mask_modes():

    print()
    print(
        "Testing Explicit Mask Modes"
    )

    for mask_type in VALID_MASK_TYPES:

        dataset = create_dataset(
            mask_mode=mask_type
        )

        for index in range(
            len(dataset)
        ):

            sample = dataset[index]

            returned_mask_type = sample[4]

            assert returned_mask_type == (
                mask_type
            )

        print(
            f"{mask_type}: PASSED"
        )

    print(
        "Explicit Mask Mode Test: PASSED"
    )


# =====================================================
# TEST RANDOM MASK MODE
# =====================================================

def test_random_mask_mode():

    print()
    print(
        "Testing Random Mask Mode"
    )

    dataset = create_dataset(
        mask_mode="random"
    )

    observed_types = set(
        dataset.mask_types
    )

    print(
        f"Observed Mask Types: "
        f"{sorted(observed_types)}"
    )

    # -------------------------------------------------
    # At least one valid mask type must occur.
    # -------------------------------------------------

    assert len(
        observed_types
    ) >= 1

    # -------------------------------------------------
    # Every generated type must be valid.
    # -------------------------------------------------

    for mask_type in observed_types:

        assert mask_type in VALID_MASK_TYPES

    print(
        "Random Mask Mode Test: PASSED"
    )


# =====================================================
# TEST ALL DATASET SAMPLES
# =====================================================

def test_all_samples():

    print()
    print(
        "Testing All Dataset Samples"
    )

    dataset = create_dataset()

    expected_shape = torch.Size(
        (
            1,
            *CUBE_SIZE
        )
    )

    for index in range(
        len(dataset)
    ):

        (
            input_cube,
            target_cube,
            mask,
            velocity,
            mask_type,
            geological_mode
        ) = dataset[index]

        # -------------------------------------------------
        # Shape checks.
        # -------------------------------------------------

        assert input_cube.shape == expected_shape

        assert target_cube.shape == expected_shape

        assert mask.shape == expected_shape

        assert velocity.shape == expected_shape

        # -------------------------------------------------
        # Metadata checks.
        # -------------------------------------------------

        assert mask_type in VALID_MASK_TYPES

        assert geological_mode in (
            VALID_GEOLOGICAL_MODES
        )

        # -------------------------------------------------
        # Tensor checks.
        # -------------------------------------------------

        assert input_cube.dtype == torch.float32

        assert target_cube.dtype == torch.float32

        assert mask.dtype == torch.float32

        assert velocity.dtype == torch.float32

        assert torch.isfinite(
            input_cube
        ).all()

        assert torch.isfinite(
            target_cube
        ).all()

        assert torch.isfinite(
            mask
        ).all()

        assert torch.isfinite(
            velocity
        ).all()

        # -------------------------------------------------
        # Input/mask consistency.
        # -------------------------------------------------

        assert torch.allclose(
            input_cube,
            target_cube * mask,
            atol=1.0e-6
        )

    print(
        "All Dataset Samples Test: PASSED"
    )


# =====================================================
# TEST PYTORCH DATALOADER
# =====================================================

def test_dataloader():

    print()
    print(
        "Testing PyTorch DataLoader"
    )

    dataset = create_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    (
        batch_input,
        batch_target,
        batch_mask,
        batch_velocity,
        batch_mask_type,
        batch_geological_mode
    ) = next(
        iter(dataloader)
    )

    # -------------------------------------------------
    # Display batch information.
    # -------------------------------------------------

    print(
        f"Batch Input Shape       : "
        f"{batch_input.shape}"
    )

    print(
        f"Batch Target Shape      : "
        f"{batch_target.shape}"
    )

    print(
        f"Batch Mask Shape        : "
        f"{batch_mask.shape}"
    )

    print(
        f"Batch Velocity Shape    : "
        f"{batch_velocity.shape}"
    )

    print(
        f"Batch Mask Types        : "
        f"{list(batch_mask_type)}"
    )

    print(
        f"Batch Geological Modes  : "
        f"{list(batch_geological_mode)}"
    )

    # =================================================
    # EXPECTED BATCH SHAPE
    # =================================================

    expected_batch_shape = torch.Size(
        (
            2,
            1,
            *CUBE_SIZE
        )
    )

    # -------------------------------------------------
    # Tensor shape checks.
    # -------------------------------------------------

    assert batch_input.shape == (
        expected_batch_shape
    )

    assert batch_target.shape == (
        expected_batch_shape
    )

    assert batch_mask.shape == (
        expected_batch_shape
    )

    assert batch_velocity.shape == (
        expected_batch_shape
    )

    # -------------------------------------------------
    # Metadata checks.
    # -------------------------------------------------

    assert len(
        batch_mask_type
    ) == 2

    assert len(
        batch_geological_mode
    ) == 2

    for mask_type in batch_mask_type:

        assert mask_type in VALID_MASK_TYPES

    for mode in batch_geological_mode:

        assert mode in VALID_GEOLOGICAL_MODES

    # -------------------------------------------------
    # Verify tensor data types.
    # -------------------------------------------------

    assert batch_input.dtype == torch.float32

    assert batch_target.dtype == torch.float32

    assert batch_mask.dtype == torch.float32

    assert batch_velocity.dtype == torch.float32

    print(
        "DataLoader Test: PASSED"
    )


# =====================================================
# MAIN TEST
# =====================================================

def main():

    print()
    print("=" * 60)

    print(
        "TESTING SYNTHETIC 3D SEISMIC DATASET"
    )

    print("=" * 60)

    # -------------------------------------------------
    # Basic dataset tests.
    # -------------------------------------------------

    test_dataset_length()

    test_sample_shapes()

    test_tensor_types()

    test_finite_values()

    # -------------------------------------------------
    # Sampling tests.
    # -------------------------------------------------

    test_sampling_mask()

    test_input_mask_consistency()

    test_missing_voxels()

    # -------------------------------------------------
    # Physics-related dataset test.
    # -------------------------------------------------

    test_velocity_model()

    # -------------------------------------------------
    # Metadata tests.
    # -------------------------------------------------

    test_mask_type_tracking()

    test_geological_mode_tracking()

    # -------------------------------------------------
    # Mask configuration tests.
    # -------------------------------------------------

    test_explicit_mask_modes()

    test_random_mask_mode()

    # -------------------------------------------------
    # Complete dataset test.
    # -------------------------------------------------

    test_all_samples()

    # -------------------------------------------------
    # DataLoader integration.
    # -------------------------------------------------

    test_dataloader()

    # =================================================
    # FINAL RESULT
    # =================================================

    print()
    print("=" * 60)

    print(
        "SYNTHETIC SEISMIC DATASET TEST: PASSED"
    )

    print("=" * 60)


# =====================================================
# RUN TEST
# =====================================================

if __name__ == "__main__":

    main()