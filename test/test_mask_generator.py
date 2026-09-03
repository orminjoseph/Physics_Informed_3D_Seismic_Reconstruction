"""
=========================================================
Test 3D Seismic Sampling Mask Generator
=========================================================

Tests the binary sampling masks used to simulate
incomplete 3D seismic acquisition.

Author: Ormin Joseph
=========================================================
"""

import random

import torch

from dataset.mask_generator import (
    SeismicMaskGenerator
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def print_mask_statistics(
        mask,
        mask_name
):
    """
    Print useful statistics for a generated mask.
    """

    total_voxels = mask.numel()

    observed_voxels = (
        mask == 1.0
    ).sum().item()

    missing_voxels = (
        mask == 0.0
    ).sum().item()

    missing_fraction = (
        missing_voxels
        /
        total_voxels
    )

    print()

    print(mask_name)

    print(
        "Shape            :",
        mask.shape
    )

    print(
        "Observed Voxels  :",
        observed_voxels
    )

    print(
        "Missing Voxels   :",
        missing_voxels
    )

    print(
        "Missing Fraction :",
        f"{missing_fraction:.4f}"
    )


# =========================================================
# BASIC MASK VALIDATION
# =========================================================

def validate_mask(
        mask,
        expected_shape
):
    """
    Perform basic validation common to all mask types.
    """

    # -----------------------------------------------------
    # Tensor type
    # -----------------------------------------------------

    assert isinstance(
        mask,
        torch.Tensor
    )

    # -----------------------------------------------------
    # Tensor shape
    # -----------------------------------------------------

    assert (
        tuple(mask.shape)
        ==
        expected_shape
    )

    # -----------------------------------------------------
    # Tensor dtype
    # -----------------------------------------------------

    assert (
        mask.dtype
        ==
        torch.float32
    )

    # -----------------------------------------------------
    # Finite values
    # -----------------------------------------------------

    assert torch.isfinite(
        mask
    ).all()

    # -----------------------------------------------------
    # Binary values only
    # -----------------------------------------------------

    unique_values = torch.unique(
        mask
    )

    assert torch.all(
        (
            unique_values == 0.0
        )
        |
        (
            unique_values == 1.0
        )
    )


# =========================================================
# TEST RANDOM VOXEL MASK
# =========================================================

def test_random_voxels(
        generator,
        expected_shape,
        missing_probability
):
    """
    Test randomly missing individual seismic voxels.
    """

    print()

    print("=" * 60)

    print(
        "Testing Random Voxel Mask"
    )

    print("=" * 60)

    mask = generator.generate(
        mask_type="random_voxels"
    )

    validate_mask(
        mask,
        expected_shape
    )

    missing_fraction = (
        (mask == 0.0).float().mean().item()
    )

    print_mask_statistics(
        mask,
        "Random Voxel Mask Statistics"
    )

    # -----------------------------------------------------
    # The Bernoulli mask is stochastic.
    #
    # Therefore, allow a reasonable tolerance around the
    # requested missing probability.
    # -----------------------------------------------------

    tolerance = 0.10

    assert abs(
        missing_fraction
        -
        missing_probability
    ) < tolerance

    print()

    print(
        "Random Voxel Mask Test: PASSED"
    )


# =========================================================
# TEST MISSING TRACES
# =========================================================

def test_missing_traces(
        generator,
        expected_shape,
        missing_probability
):
    """
    Test removal of complete seismic traces.
    """

    print()

    print("=" * 60)

    print(
        "Testing Missing Trace Mask"
    )

    print("=" * 60)

    mask = generator.generate(
        mask_type="missing_traces"
    )

    validate_mask(
        mask,
        expected_shape
    )

    # -----------------------------------------------------
    # Each spatial trace must be either:
    #
    # completely observed
    #
    # or completely missing.
    # -----------------------------------------------------

    trace_sums = mask.sum(
        dim=1
    )

    depth = expected_shape[1]

    assert torch.all(
        (
            trace_sums == 0.0
        )
        |
        (
            trace_sums == depth
        )
    )

    missing_fraction = (
        (mask == 0.0).float().mean().item()
    )

    print_mask_statistics(
        mask,
        "Missing Trace Mask Statistics"
    )

    tolerance = 0.05

    assert abs(
        missing_fraction
        -
        missing_probability
    ) <= tolerance

    print()

    print(
        "Missing Trace Mask Test: PASSED"
    )


# =========================================================
# TEST MISSING INLINE SECTIONS
# =========================================================

def test_missing_inlines(
        generator,
        expected_shape,
        missing_probability
):
    """
    Test removal of complete inline sections.
    """

    print()

    print("=" * 60)

    print(
        "Testing Missing Inline Mask"
    )

    print("=" * 60)

    mask = generator.generate(
        mask_type="missing_inlines"
    )

    validate_mask(
        mask,
        expected_shape
    )

    # -----------------------------------------------------
    # Every inline must be either:
    #
    # completely observed
    #
    # or completely missing.
    # -----------------------------------------------------

    inline_sums = mask.sum(
        dim=(
            1,
            3
        )
    )

    depth = expected_shape[1]

    width = expected_shape[3]

    complete_inline_size = (
        depth
        *
        width
    )

    assert torch.all(
        (
            inline_sums == 0.0
        )
        |
        (
            inline_sums
            ==
            complete_inline_size
        )
    )

    missing_fraction = (
        (mask == 0.0).float().mean().item()
    )

    print_mask_statistics(
        mask,
        "Missing Inline Mask Statistics"
    )

    tolerance = 0.05

    assert abs(
        missing_fraction
        -
        missing_probability
    ) <= tolerance

    print()

    print(
        "Missing Inline Mask Test: PASSED"
    )


# =========================================================
# TEST MISSING CROSSLINE SECTIONS
# =========================================================

def test_missing_crosslines(
        generator,
        expected_shape,
        missing_probability
):
    """
    Test removal of complete crossline sections.
    """

    print()

    print("=" * 60)

    print(
        "Testing Missing Crossline Mask"
    )

    print("=" * 60)

    mask = generator.generate(
        mask_type="missing_crosslines"
    )

    validate_mask(
        mask,
        expected_shape
    )

    # -----------------------------------------------------
    # Every crossline must be either:
    #
    # completely observed
    #
    # or completely missing.
    # -----------------------------------------------------

    crossline_sums = mask.sum(
        dim=(
            1,
            2
        )
    )

    depth = expected_shape[1]

    height = expected_shape[2]

    complete_crossline_size = (
        depth
        *
        height
    )

    assert torch.all(
        (
            crossline_sums == 0.0
        )
        |
        (
            crossline_sums
            ==
            complete_crossline_size
        )
    )

    missing_fraction = (
        (mask == 0.0).float().mean().item()
    )

    print_mask_statistics(
        mask,
        "Missing Crossline Mask Statistics"
    )

    tolerance = 0.05

    assert abs(
        missing_fraction
        -
        missing_probability
    ) <= tolerance

    print()

    print(
        "Missing Crossline Mask Test: PASSED"
    )


# =========================================================
# TEST MISSING BLOCKS
# =========================================================

def test_missing_blocks(
        generator,
        expected_shape,
        missing_probability
):
    """
    Test removal of a contiguous 3D block.
    """

    print()

    print("=" * 60)

    print(
        "Testing Missing Block Mask"
    )

    print("=" * 60)

    mask = generator.generate(
        mask_type="missing_blocks"
    )

    validate_mask(
        mask,
        expected_shape
    )

    missing_fraction = (
        (mask == 0.0).float().mean().item()
    )

    print_mask_statistics(
        mask,
        "Missing Block Mask Statistics"
    )

    # -----------------------------------------------------
    # A block is constructed approximately according to
    # the cube-root scaling rule.
    #
    # Allow some tolerance because integer rounding of
    # block dimensions changes the exact block volume.
    # -----------------------------------------------------

    tolerance = 0.10

    assert abs(
        missing_fraction
        -
        missing_probability
    ) < tolerance

    # -----------------------------------------------------
    # Ensure that missing data actually exist.
    # -----------------------------------------------------

    assert (
        mask == 0.0
    ).any()

    print()

    print(
        "Missing Block Mask Test: PASSED"
    )


# =========================================================
# TEST RANDOM MASK SELECTION
# =========================================================

def test_random_mask_selection(
        generator,
        expected_shape
):
    """
    Test automatic random selection of mask types.
    """

    print()

    print("=" * 60)

    print(
        "Testing Random Mask Selection"
    )

    print("=" * 60)

    for index in range(10):

        mask = generator.generate(
            mask_type="random"
        )

        validate_mask(
            mask,
            expected_shape
        )

        missing_fraction = (
            (mask == 0.0)
            .float()
            .mean()
            .item()
        )

        print(
            f"Random Mask "
            f"{index + 1:02d}: "
            f"Missing Fraction = "
            f"{missing_fraction:.4f}"
        )

    print()

    print(
        "Random Mask Selection Test: PASSED"
    )


# =========================================================
# TEST INVALID MASK TYPE
# =========================================================

def test_invalid_mask_type(
        generator
):
    """
    Ensure unsupported mask types raise an error.
    """

    print()

    print("=" * 60)

    print(
        "Testing Invalid Mask Type"
    )

    print("=" * 60)

    try:

        generator.generate(
            mask_type="invalid_mask_type"
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:

        print(
            "Invalid Mask Type Test: PASSED"
        )


# =========================================================
# TEST REPRODUCIBILITY
# =========================================================

def test_reproducibility():

    """
    Test reproducibility when random seeds are controlled.
    """

    print()

    print("=" * 60)

    print(
        "Testing Mask Reproducibility"
    )

    print("=" * 60)

    cube_size = (
        16,
        32,
        32
    )

    missing_probability = 0.30

    # -----------------------------------------------------
    # Set identical random seeds.
    # -----------------------------------------------------

    torch.manual_seed(
        42
    )

    random.seed(
        42
    )

    generator_1 = (
        SeismicMaskGenerator(
            cube_size=cube_size,
            missing_probability=(
                missing_probability
            )
        )
    )

    mask_1 = generator_1.generate(
        mask_type="random_voxels"
    )

    # -----------------------------------------------------
    # Reset seeds.
    # -----------------------------------------------------

    torch.manual_seed(
        42
    )

    random.seed(
        42
    )

    generator_2 = (
        SeismicMaskGenerator(
            cube_size=cube_size,
            missing_probability=(
                missing_probability
            )
        )
    )

    mask_2 = generator_2.generate(
        mask_type="random_voxels"
    )

    assert torch.equal(
        mask_1,
        mask_2
    )

    print(
        "Mask Reproducibility Test: PASSED"
    )


# =========================================================
# MAIN TEST FUNCTION
# =========================================================

def test_mask_generator():

    print()

    print("=" * 60)

    print(
        "TESTING 3D SEISMIC SAMPLING MASK GENERATOR"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Test configuration
    # -----------------------------------------------------

    cube_size = (
        16,
        32,
        32
    )

    missing_probability = 0.30

    expected_shape = (
        1,
        *cube_size
    )

    generator = (
        SeismicMaskGenerator(
            cube_size=cube_size,
            missing_probability=(
                missing_probability
            )
        )
    )

    # -----------------------------------------------------
    # Run individual tests
    # -----------------------------------------------------

    test_random_voxels(
        generator,
        expected_shape,
        missing_probability
    )

    test_missing_traces(
        generator,
        expected_shape,
        missing_probability
    )

    test_missing_inlines(
        generator,
        expected_shape,
        missing_probability
    )

    test_missing_crosslines(
        generator,
        expected_shape,
        missing_probability
    )

    test_missing_blocks(
        generator,
        expected_shape,
        missing_probability
    )

    test_random_mask_selection(
        generator,
        expected_shape
    )

    test_invalid_mask_type(
        generator
    )

    test_reproducibility()

    print()

    print("=" * 60)

    print(
        "SEISMIC MASK GENERATOR TEST: PASSED"
    )

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

def main():

    test_mask_generator()


if __name__ == "__main__":

    main()