"""
=========================================================
Synthetic Sample Physical Validation
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Validate the complete synthetic sample produced by:

    SyntheticSeismicDataset

The test verifies:

    1. tensor dimensions
    2. tensor data types
    3. finite numerical values
    4. non-trivial seismic target
    5. binary sampling mask
    6. missing-data fraction
    7. input-target-mask consistency
    8. observed-data preservation
    9. missing-data zeroing
   10. velocity positivity
   11. velocity physical bounds
   12. geological-mode validity
   13. mask-type validity
   14. reproducibility
   15. geological-mode coverage
   16. missing-data mechanism coverage

Important
---------
This test validates internal consistency of the synthetic
dataset.

It does NOT claim that the synthetic seismic generator is
a full numerical solution of the acoustic or elastic wave
equation.

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import gc

import torch

from dataset.synthetic_dataset import SyntheticSeismicDataset


# =========================================================
# CONFIGURATION
# =========================================================

CUBE_SIZE = (64, 128, 128)

MISSING_PROBABILITY = 0.30

SEED = 42

NUM_SAMPLES = 6

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 5000.0

TOLERANCE = 1e-6


# =========================================================
# EXPECTED GEOLOGICAL MODES
# =========================================================

GEOLOGICAL_MODES = (
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
)


# =========================================================
# EXPECTED MASK TYPES
# =========================================================

MASK_TYPES = (
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def check(condition, message):
    """
    Raise an assertion error if a validation condition fails.
    """

    if not condition:
        raise AssertionError(message)


# =========================================================
# DATASET FACTORY
# =========================================================

def create_dataset(
    geological_mode,
    mask_mode,
    seed=SEED,
):
    """
    Create a deterministic synthetic dataset.
    """

    return SyntheticSeismicDataset(
        num_samples=1,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_PROBABILITY,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )


# =========================================================
# SINGLE SAMPLE VALIDATION
# =========================================================

def validate_single_sample(
    sample,
    geological_mode,
    mask_mode,
):
    """
    Validate one complete synthetic seismic sample.
    """

    (
        input_cube,
        target,
        mask,
        velocity,
        returned_mask_type,
        returned_geological_mode,
    ) = sample

    # =====================================================
    # EXPECTED SHAPE
    # =====================================================

    expected_shape = (
        1,
        *CUBE_SIZE,
    )

    print()
    print("-" * 70)

    print(
        "Expected tensor shape :",
        expected_shape
    )

    print(
        "Input shape           :",
        tuple(input_cube.shape)
    )

    print(
        "Target shape          :",
        tuple(target.shape)
    )

    print(
        "Mask shape            :",
        tuple(mask.shape)
    )

    print(
        "Velocity shape        :",
        tuple(velocity.shape)
    )

    # -----------------------------------------------------
    # Shape validation
    # -----------------------------------------------------

    check(
        tuple(input_cube.shape) == expected_shape,
        "Input cube has incorrect shape.",
    )

    check(
        tuple(target.shape) == expected_shape,
        "Target cube has incorrect shape.",
    )

    check(
        tuple(mask.shape) == expected_shape,
        "Mask has incorrect shape.",
    )

    check(
        tuple(velocity.shape) == expected_shape,
        "Velocity model has incorrect shape.",
    )

    # =====================================================
    # DATA TYPE VALIDATION
    # =====================================================

    print(
        "Input dtype           :",
        input_cube.dtype
    )

    print(
        "Target dtype          :",
        target.dtype
    )

    print(
        "Mask dtype            :",
        mask.dtype
    )

    print(
        "Velocity dtype        :",
        velocity.dtype
    )

    check(
        input_cube.dtype == torch.float32,
        "Input cube must be float32.",
    )

    check(
        target.dtype == torch.float32,
        "Target cube must be float32.",
    )

    check(
        mask.dtype == torch.float32,
        "Mask must be float32.",
    )

    check(
        velocity.dtype == torch.float32,
        "Velocity model must be float32.",
    )

    # =====================================================
    # FINITE VALUE VALIDATION
    # =====================================================

    check(
        torch.isfinite(input_cube).all().item(),
        "Input cube contains NaN or Inf.",
    )

    check(
        torch.isfinite(target).all().item(),
        "Target cube contains NaN or Inf.",
    )

    check(
        torch.isfinite(mask).all().item(),
        "Mask contains NaN or Inf.",
    )

    check(
        torch.isfinite(velocity).all().item(),
        "Velocity model contains NaN or Inf.",
    )

    print("Finite values        : PASS")

    # =====================================================
    # TARGET VARIABILITY
    # =====================================================

    target_min = float(target.min())
    target_max = float(target.max())
    target_mean = float(target.mean())
    target_std = float(target.std())

    print(
        f"Target minimum       : {target_min:.6f}"
    )

    print(
        f"Target maximum       : {target_max:.6f}"
    )

    print(
        f"Target mean          : {target_mean:.6f}"
    )

    print(
        f"Target std           : {target_std:.6f}"
    )

    check(
        target_std > TOLERANCE,
        "Synthetic seismic target has no meaningful spatial variability.",
    )

    check(
        abs(target_max - target_min) > TOLERANCE,
        "Synthetic seismic target is effectively constant.",
    )

    print("Target variability   : PASS")

    # =====================================================
    # MASK VALIDATION
    # =====================================================

    unique_mask_values = torch.unique(mask)

    print(
        "Mask unique values   :",
        unique_mask_values.tolist()
    )

    check(
        torch.all(
            (unique_mask_values == 0.0)
            |
            (unique_mask_values == 1.0)
        ).item(),
        "Mask is not binary.",
    )

    print("Binary mask          : PASS")

    # =====================================================
    # OBSERVED / MISSING FRACTIONS
    # =====================================================

    observed_fraction = float(
        mask.mean()
    )

    missing_fraction = (
        1.0 - observed_fraction
    )

    print(
        f"Observed fraction    : {observed_fraction:.6f}"
    )

    print(
        f"Missing fraction     : {missing_fraction:.6f}"
    )

    # -----------------------------------------------------
    # The configured probability is a target probability,
    # not necessarily an exact fraction for structured masks.
    # -----------------------------------------------------

    check(
        missing_fraction > 0.0,
        "No missing seismic samples were generated.",
    )

    check(
        observed_fraction > 0.0,
        "No observed seismic samples remain.",
    )

    print("Sampling fractions   : PASS")

    # =====================================================
    # INPUT = TARGET × MASK
    # =====================================================

    expected_input = (
        target * mask
    )

    max_difference = float(
        torch.max(
            torch.abs(
                input_cube
                - expected_input
            )
        )
    )

    print(
        f"Input consistency error : "
        f"{max_difference:.10e}"
    )

    check(
        max_difference <= TOLERANCE,
        "Input cube is not equal to target × mask.",
    )

    print("Input-target consistency : PASS")

    # =====================================================
    # MISSING DATA MUST BE ZERO
    # =====================================================

    missing_locations = (
        mask == 0.0
    )

    if torch.any(
        missing_locations
    ):

        maximum_missing_value = float(
            torch.abs(
                input_cube[
                    missing_locations
                ]
            ).max()
        )

        print(
            f"Maximum missing input value : "
            f"{maximum_missing_value:.10e}"
        )

        check(
            maximum_missing_value <= TOLERANCE,
            "Missing locations contain non-zero input values.",
        )

    print("Missing samples zeroed : PASS")

    # =====================================================
    # OBSERVED DATA MUST BE PRESERVED
    # =====================================================

    observed_locations = (
        mask == 1.0
    )

    if torch.any(
        observed_locations
    ):

        observed_difference = float(
            torch.max(
                torch.abs(
                    input_cube[
                        observed_locations
                    ]
                    -
                    target[
                        observed_locations
                    ]
                )
            )
        )

        print(
            f"Observed-data error : "
            f"{observed_difference:.10e}"
        )

        check(
            observed_difference <= TOLERANCE,
            "Observed seismic samples were modified.",
        )

    print("Observed samples preserved : PASS")

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    velocity_min = float(
        velocity.min()
    )

    velocity_max = float(
        velocity.max()
    )

    velocity_mean = float(
        velocity.mean()
    )

    velocity_std = float(
        velocity.std()
    )

    print(
        f"Velocity minimum     : "
        f"{velocity_min:.3f} m/s"
    )

    print(
        f"Velocity maximum     : "
        f"{velocity_max:.3f} m/s"
    )

    print(
        f"Velocity mean        : "
        f"{velocity_mean:.3f} m/s"
    )

    print(
        f"Velocity std         : "
        f"{velocity_std:.3f} m/s"
    )

    check(
        velocity_min >= VELOCITY_MIN,
        "Velocity falls below the physical lower bound.",
    )

    check(
        velocity_max <= VELOCITY_MAX,
        "Velocity exceeds the physical upper bound.",
    )

    check(
        velocity_min > 0.0,
        "Velocity must be strictly positive.",
    )

    check(
        velocity_std > 0.0,
        "Velocity model has no spatial variability.",
    )

    print("Velocity validation  : PASS")

    # =====================================================
    # METADATA VALIDATION
    # =====================================================

    print(
        "Returned mask type  :",
        returned_mask_type
    )

    print(
        "Returned geology    :",
        returned_geological_mode
    )

    check(
        returned_mask_type == mask_mode,
        "Returned mask type does not match requested mask mode.",
    )

    check(
        returned_geological_mode == geological_mode,
        "Returned geological mode does not match requested mode.",
    )

    check(
        returned_mask_type in MASK_TYPES,
        "Invalid mask type returned.",
    )

    check(
        returned_geological_mode in GEOLOGICAL_MODES,
        "Invalid geological mode returned.",
    )

    print("Metadata validation  : PASS")

    # =====================================================
    # COMPLETE SAMPLE STATUS
    # =====================================================

    print()
    print(
        "COMPLETE SYNTHETIC SAMPLE: PASS"
    )


# =========================================================
# GEOLOGICAL MODE TEST
# =========================================================

def test_geological_modes():
    """
    Validate all geological scenarios.
    """

    print()
    print("=" * 70)
    print("TEST 1: GEOLOGICAL SCENARIO VALIDATION")
    print("=" * 70)

    for mode in GEOLOGICAL_MODES:

        print()
        print(
            f"TESTING GEOLOGICAL MODE: "
            f"{mode.upper()}"
        )

        dataset = create_dataset(
            geological_mode=mode,
            mask_mode="random_voxels",
            seed=SEED,
        )

        sample = dataset[0]

        validate_single_sample(
            sample=sample,
            geological_mode=mode,
            mask_mode="random_voxels",
        )

        del dataset
        del sample

        gc.collect()

    print()
    print(
        "ALL GEOLOGICAL SCENARIOS: PASS"
    )


# =========================================================
# MASK TYPE TEST
# =========================================================

def test_mask_types():
    """
    Validate all missing-data mechanisms.
    """

    print()
    print("=" * 70)
    print("TEST 2: MISSING-DATA MECHANISM VALIDATION")
    print("=" * 70)

    for mask_type in MASK_TYPES:

        print()
        print(
            f"TESTING MASK TYPE: "
            f"{mask_type.upper()}"
        )

        dataset = create_dataset(
            geological_mode="complex",
            mask_mode=mask_type,
            seed=SEED,
        )

        sample = dataset[0]

        validate_single_sample(
            sample=sample,
            geological_mode="complex",
            mask_mode=mask_type,
        )

        del dataset
        del sample

        gc.collect()

    print()
    print(
        "ALL MISSING-DATA MECHANISMS: PASS"
    )


# =========================================================
# REPRODUCIBILITY TEST
# =========================================================

def test_reproducibility():
    """
    Verify deterministic sample generation.
    """

    print()
    print("=" * 70)
    print("TEST 3: FULL SAMPLE REPRODUCIBILITY")
    print("=" * 70)

    dataset_a = create_dataset(
        geological_mode="highly_complex",
        mask_mode="missing_blocks",
        seed=SEED,
    )

    dataset_b = create_dataset(
        geological_mode="highly_complex",
        mask_mode="missing_blocks",
        seed=SEED,
    )

    sample_a = dataset_a[0]
    sample_b = dataset_b[0]

    # -----------------------------------------------------
    # Tensor equality
    # -----------------------------------------------------

    print(
        "Input identical       :",
        torch.equal(
            sample_a[0],
            sample_b[0],
        )
    )

    print(
        "Target identical      :",
        torch.equal(
            sample_a[1],
            sample_b[1],
        )
    )

    print(
        "Mask identical        :",
        torch.equal(
            sample_a[2],
            sample_b[2],
        )
    )

    print(
        "Velocity identical    :",
        torch.equal(
            sample_a[3],
            sample_b[3],
        )
    )

    print(
        "Mask type identical   :",
        sample_a[4] == sample_b[4]
    )

    print(
        "Geology identical     :",
        sample_a[5] == sample_b[5]
    )

    check(
        torch.equal(
            sample_a[0],
            sample_b[0],
        ),
        "Input reproducibility failed.",
    )

    check(
        torch.equal(
            sample_a[1],
            sample_b[1],
        ),
        "Target reproducibility failed.",
    )

    check(
        torch.equal(
            sample_a[2],
            sample_b[2],
        ),
        "Mask reproducibility failed.",
    )

    check(
        torch.equal(
            sample_a[3],
            sample_b[3],
        ),
        "Velocity reproducibility failed.",
    )

    check(
        sample_a[4] == sample_b[4],
        "Mask-type reproducibility failed.",
    )

    check(
        sample_a[5] == sample_b[5],
        "Geological-mode reproducibility failed.",
    )

    print()
    print(
        "FULL SAMPLE REPRODUCIBILITY: PASS"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SYNTHETIC SEISMIC SAMPLE PHYSICAL VALIDATION")
    print("=" * 70)

    print(
        f"Cube size            : {CUBE_SIZE}"
    )

    print(
        f"Missing probability  : "
        f"{MISSING_PROBABILITY:.2f}"
    )

    print(
        f"Velocity bounds      : "
        f"{VELOCITY_MIN:.0f}–{VELOCITY_MAX:.0f} m/s"
    )

    # =====================================================
    # TEST 1
    # =====================================================

    test_geological_modes()

    # =====================================================
    # TEST 2
    # =====================================================

    test_mask_types()

    # =====================================================
    # TEST 3
    # =====================================================

    test_reproducibility()

    # =====================================================
    # FINAL STATUS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "ALL SYNTHETIC SAMPLE PHYSICAL "
        "VALIDATION TESTS PASSED"
    )
    print("=" * 70)