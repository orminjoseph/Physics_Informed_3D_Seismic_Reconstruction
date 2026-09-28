"""
=========================================================
Dataset Pipeline Compatibility Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Tests the complete dataset pipeline:

    build_dataset()
          |
          v
    split_dataset()
          |
          +------> training dataset
          |
          +------> validation dataset
          |
          v
    create_dataloader()
          |
          +------> training DataLoader
          |
          +------> validation DataLoader

The test verifies:

1. Dataset construction
2. Dataset size
3. Train/validation split
4. Split reproducibility
5. No train/validation overlap
6. DataLoader construction
7. Batch dimensions
8. Finite numerical values
9. Mask values
10. Basic data consistency
11. Configuration consistency

This test does NOT train the neural network.

Author: Ormin Joseph
=========================================================
"""

# ---------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------

import sys
from pathlib import Path

import torch


# ---------------------------------------------------------
# 2. MAKE PROJECT ROOT AVAILABLE
# ---------------------------------------------------------

# Get the directory containing this test file.
TEST_DIR = Path(__file__).resolve().parent

# The project root is one level above the test directory.
PROJECT_ROOT = TEST_DIR.parent

# Add the project root to Python's import path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# 3. PROJECT IMPORTS
# ---------------------------------------------------------

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_NUM_SAMPLES,
    SYNTHETIC_MISSING_PROBABILITY,
    VALIDATION_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    SEED,
)

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader


# ---------------------------------------------------------
# 4. TEST HELPER FUNCTIONS
# ---------------------------------------------------------

def print_section(title):
    """
    Print a clearly separated test section.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(condition, success_message, failure_message):
    """
    Print PASS or FAIL depending on the condition.
    """

    if condition:
        print(f"[PASS] {success_message}")
    else:
        print(f"[FAIL] {failure_message}")

    return condition


# ---------------------------------------------------------
# 5. START TEST
# ---------------------------------------------------------

print_section("DATASET PIPELINE COMPATIBILITY TEST")

print(f"Dataset mode       : {DATASET_MODE}")
print(f"Experiment name    : {EXPERIMENT_NAME}")
print(f"Random seed        : {SEED}")
print(f"Validation split   : {VALIDATION_SPLIT}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Num workers        : {NUM_WORKERS}")
print(f"Pin memory         : {PIN_MEMORY}")
print(f"Persistent workers : {PERSISTENT_WORKERS}")


# ---------------------------------------------------------
# 6. BUILD COMPLETE DATASET
# ---------------------------------------------------------

print_section("TEST 1: BUILD DATASET")

try:

    # Construct the complete dataset using the configured
    # DATASET_MODE.
    dataset = build_dataset()

    print(f"Dataset type      : {type(dataset).__name__}")
    print(f"Dataset size      : {len(dataset)}")

    dataset_ok = len(dataset) >= 2

    check(
        dataset_ok,
        f"Dataset successfully created with {len(dataset)} samples.",
        "Dataset must contain at least two samples.",
    )

except Exception as error:

    print("[FAIL] Dataset construction failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 7. VERIFY EXPECTED SYNTHETIC DATASET SIZE
# ---------------------------------------------------------

print_section("TEST 2: VERIFY DATASET CONFIGURATION")

if DATASET_MODE.lower() == "synthetic":

    expected_size = SYNTHETIC_NUM_SAMPLES

    check(
        len(dataset) == expected_size,
        (
            f"Synthetic dataset size matches configuration: "
            f"{len(dataset)} samples."
        ),
        (
            f"Synthetic dataset size mismatch. "
            f"Expected {expected_size}, got {len(dataset)}."
        ),
    )

    print(
        f"Configured synthetic patch size : "
        f"{SYNTHETIC_PATCH_SIZE}"
    )

    print(
        f"Configured missing probability : "
        f"{SYNTHETIC_MISSING_PROBABILITY}"
    )

else:

    print(
        "Dataset is not synthetic. "
        "Skipping synthetic-only size check."
    )


# ---------------------------------------------------------
# 8. SPLIT DATASET
# ---------------------------------------------------------

print_section("TEST 3: TRAIN/VALIDATION SPLIT")

try:

    # Split the complete dataset into training and validation
    # subsets using the configured VALIDATION_SPLIT and SEED.
    train_dataset, validation_dataset = split_dataset(dataset)

    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(validation_dataset)}")

    split_size_ok = (
        len(train_dataset) + len(validation_dataset)
        == len(dataset)
    )

    check(
        split_size_ok,
        "Training + validation sizes equal the complete dataset size.",
        "Training + validation sizes do not equal dataset size.",
    )

    non_empty_ok = (
        len(train_dataset) > 0
        and len(validation_dataset) > 0
    )

    check(
        non_empty_ok,
        "Both training and validation datasets contain samples.",
        "Training or validation dataset is empty.",
    )

except Exception as error:

    print("[FAIL] Dataset splitting failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 9. VERIFY NO TRAIN/VALIDATION OVERLAP
# ---------------------------------------------------------

print_section("TEST 4: TRAIN/VALIDATION OVERLAP")

# random_split() creates torch.utils.data.Subset objects.
# Their original dataset indices are stored in .indices.

if hasattr(train_dataset, "indices") and hasattr(
    validation_dataset, "indices"
):

    train_indices = set(train_dataset.indices)
    validation_indices = set(validation_dataset.indices)

    overlap = train_indices.intersection(validation_indices)

    check(
        len(overlap) == 0,
        "Training and validation sets have no overlapping indices.",
        f"Data leakage detected. Overlapping indices: {overlap}",
    )

    print(f"Training index count   : {len(train_indices)}")
    print(f"Validation index count : {len(validation_indices)}")

else:

    print(
        "[WARNING] Could not inspect Subset indices directly."
    )


# ---------------------------------------------------------
# 10. TEST SPLIT REPRODUCIBILITY
# ---------------------------------------------------------

print_section("TEST 5: SPLIT REPRODUCIBILITY")

try:

    # Run the split again using the same dataset and same
    # configured random seed.
    train_dataset_2, validation_dataset_2 = split_dataset(dataset)

    reproducible_train = (
        list(train_dataset.indices)
        == list(train_dataset_2.indices)
    )

    reproducible_validation = (
        list(validation_dataset.indices)
        == list(validation_dataset_2.indices)
    )

    check(
        reproducible_train,
        "Training split is reproducible with the configured seed.",
        "Training split changed between identical split operations.",
    )

    check(
        reproducible_validation,
        "Validation split is reproducible with the configured seed.",
        "Validation split changed between identical split operations.",
    )

except Exception as error:

    print("[FAIL] Split reproducibility test failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 11. CREATE TRAINING DATALOADER
# ---------------------------------------------------------

print_section("TEST 6: TRAINING DATALOADER")

try:

    # Create the training DataLoader using configuration
    # defaults.
    train_loader = create_dataloader(
        train_dataset,
        shuffle=True,
    )

    print(f"Training batches : {len(train_loader)}")

    train_loader_ok = len(train_loader) > 0

    check(
        train_loader_ok,
        "Training DataLoader created successfully.",
        "Training DataLoader contains no batches.",
    )

except Exception as error:

    print("[FAIL] Training DataLoader construction failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 12. CREATE VALIDATION DATALOADER
# ---------------------------------------------------------

print_section("TEST 7: VALIDATION DATALOADER")

try:

    # Validation data must not be shuffled.
    validation_loader = create_dataloader(
        validation_dataset,
        shuffle=False,
    )

    print(f"Validation batches : {len(validation_loader)}")

    validation_loader_ok = len(validation_loader) > 0

    check(
        validation_loader_ok,
        "Validation DataLoader created successfully.",
        "Validation DataLoader contains no batches.",
    )

except Exception as error:

    print("[FAIL] Validation DataLoader construction failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 13. INSPECT FIRST TRAINING BATCH
# ---------------------------------------------------------

print_section("TEST 8: TRAINING BATCH INSPECTION")

try:

    # Retrieve the first training batch.
    train_batch = next(iter(train_loader))

    print(f"Batch type : {type(train_batch).__name__}")

    # The current SyntheticSeismicDataset returns:
    #
    # input_cube
    # target
    # mask
    # velocity
    # mask_type
    # geological_mode
    #
    # Therefore the default PyTorch collate operation should
    # produce six batch elements.
    if isinstance(train_batch, (tuple, list)):

        print(f"Number of batch elements : {len(train_batch)}")

        check(
            len(train_batch) == 6,
            "Training batch contains the expected six elements.",
            (
                "Unexpected number of batch elements. "
                f"Expected 6, got {len(train_batch)}."
            ),
        )

        input_batch = train_batch[0]
        target_batch = train_batch[1]
        mask_batch = train_batch[2]
        velocity_batch = train_batch[3]
        mask_type_batch = train_batch[4]
        geological_mode_batch = train_batch[5]

    else:

        raise TypeError(
            "Expected the DataLoader batch to be a tuple or list."
        )

except Exception as error:

    print("[FAIL] Training batch inspection failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 14. PRINT BATCH SHAPES
# ---------------------------------------------------------

print_section("TEST 9: BATCH SHAPE CHECK")

print(f"Input shape       : {input_batch.shape}")
print(f"Target shape      : {target_batch.shape}")
print(f"Mask shape        : {mask_batch.shape}")
print(f"Velocity shape    : {velocity_batch.shape}")

print(f"Mask type         : {mask_type_batch}")
print(f"Geological mode   : {geological_mode_batch}")


# ---------------------------------------------------------
# 15. VERIFY 3D SEISMIC TENSOR FORMAT
# ---------------------------------------------------------

# The expected DataLoader tensor convention is:
#
# [B, C, D, H, W]
#
# where:
#
# B = batch
# C = channel
# D = depth
# H = height / crossline
# W = width / inline

expected_dimensions = 5

shape_checks = [
    ("input", input_batch),
    ("target", target_batch),
    ("mask", mask_batch),
    ("velocity", velocity_batch),
]

for tensor_name, tensor in shape_checks:

    check(
        tensor.ndim == expected_dimensions,
        (
            f"{tensor_name} has the expected 5-D format "
            f"[B,C,D,H,W]: {tuple(tensor.shape)}"
        ),
        (
            f"{tensor_name} does not have the expected 5-D "
            f"format. Shape: {tuple(tensor.shape)}"
        ),
    )


# ---------------------------------------------------------
# 16. VERIFY INPUT/TARGET/MASK SHAPE CONSISTENCY
# ---------------------------------------------------------

print_section("TEST 10: TENSOR SHAPE CONSISTENCY")

check(
    input_batch.shape == target_batch.shape,
    "Input and target shapes are identical.",
    "Input and target shapes do not match.",
)

check(
    input_batch.shape == mask_batch.shape,
    "Input and mask shapes are identical.",
    "Input and mask shapes do not match.",
)

check(
    input_batch.shape == velocity_batch.shape,
    "Input and velocity shapes are identical.",
    "Input and velocity shapes do not match.",
)


# ---------------------------------------------------------
# 17. VERIFY FINITE VALUES
# ---------------------------------------------------------

print_section("TEST 11: NUMERICAL FINITENESS")

tensor_checks = [
    ("input", input_batch),
    ("target", target_batch),
    ("mask", mask_batch),
    ("velocity", velocity_batch),
]

for tensor_name, tensor in tensor_checks:

    finite = torch.isfinite(tensor).all().item()

    check(
        finite,
        f"{tensor_name} contains only finite values.",
        f"{tensor_name} contains NaN or infinite values.",
    )


# ---------------------------------------------------------
# 18. VERIFY MASK VALUES
# ---------------------------------------------------------

print_section("TEST 12: MASK VALIDATION")

unique_mask_values = torch.unique(mask_batch)

print(f"Unique mask values: {unique_mask_values.tolist()}")

# The current configuration defines:
#
# observed = 1
# missing  = 0

mask_is_binary = torch.all(
    (unique_mask_values == 0)
    | (unique_mask_values == 1)
).item()

check(
    mask_is_binary,
    "Mask contains only 0 and 1 values.",
    "Mask contains values other than 0 and 1.",
)


# ---------------------------------------------------------
# 19. VERIFY MISSING DATA EXISTS
# ---------------------------------------------------------

missing_fraction = (
    (mask_batch == 0)
    .float()
    .mean()
    .item()
)

observed_fraction = (
    (mask_batch == 1)
    .float()
    .mean()
    .item()
)

print(f"Observed fraction : {observed_fraction:.6f}")
print(f"Missing fraction  : {missing_fraction:.6f}")

check(
    missing_fraction > 0.0,
    "Missing seismic data is present in the batch.",
    "No missing seismic data detected.",
)


# ---------------------------------------------------------
# 20. VERIFY SYNTHETIC MISSING RATE
# ---------------------------------------------------------

if DATASET_MODE.lower() == "synthetic":

    expected_missing = SYNTHETIC_MISSING_PROBABILITY

    # We do not require exact equality because the mask is
    # generated voxel-by-voxel and therefore has sampling
    # variation.
    tolerance = 0.05

    check(
        abs(missing_fraction - expected_missing) <= tolerance,
        (
            f"Measured missing fraction {missing_fraction:.4f} "
            f"is within ±{tolerance:.2f} of the configured "
            f"value {expected_missing:.4f}."
        ),
        (
            f"Measured missing fraction {missing_fraction:.4f} "
            f"differs substantially from configured value "
            f"{expected_missing:.4f}."
        ),
    )


# ---------------------------------------------------------
# 21. VERIFY DATA CONSISTENCY
# ---------------------------------------------------------

print_section("TEST 13: DATA CONSISTENCY")

# Observed seismic samples should be preserved in the input.
#
# input = target * mask
#
# Therefore:
#
# target * mask == input

expected_input = target_batch * mask_batch

consistency_error = torch.abs(
    input_batch - expected_input
).max().item()

print(
    f"Maximum input/target-mask consistency error: "
    f"{consistency_error:.10e}"
)

check(
    consistency_error < 1e-6,
    "Input is consistent with target × mask.",
    (
        "Input is not consistent with target × mask. "
        f"Maximum error = {consistency_error:.10e}"
    ),
)


# ---------------------------------------------------------
# 22. VERIFY VALIDATION BATCH
# ---------------------------------------------------------

print_section("TEST 14: VALIDATION BATCH")

try:

    validation_batch = next(iter(validation_loader))

    print(
        f"Validation batch type : "
        f"{type(validation_batch).__name__}"
    )

    check(
        isinstance(validation_batch, (tuple, list)),
        "Validation batch is a tuple/list as expected.",
        "Validation batch has an unexpected type.",
    )

    if isinstance(validation_batch, (tuple, list)):

        print(
            f"Validation batch elements : "
            f"{len(validation_batch)}"
        )

        check(
            len(validation_batch) == 6,
            "Validation batch contains six elements.",
            (
                "Validation batch does not contain six elements."
            ),
        )

        validation_input = validation_batch[0]
        validation_target = validation_batch[1]
        validation_mask = validation_batch[2]

        print(
            f"Validation input shape   : "
            f"{validation_input.shape}"
        )

        print(
            f"Validation target shape  : "
            f"{validation_target.shape}"
        )

        print(
            f"Validation mask shape    : "
            f"{validation_mask.shape}"
        )

        check(
            validation_input.shape
            == validation_target.shape
            == validation_mask.shape,
            "Validation input, target, and mask shapes match.",
            "Validation tensor shapes do not match.",
        )

except Exception as error:

    print("[FAIL] Validation batch test failed.")
    print(f"Error: {error}")

    raise


# ---------------------------------------------------------
# 23. FINAL TEST SUMMARY
# ---------------------------------------------------------

print_section("DATASET PIPELINE TEST COMPLETED")

print("The dataset pipeline has been exercised through:")

print("  1. build_dataset()")
print("  2. split_dataset()")
print("  3. create_dataloader()")
print("  4. training batch retrieval")
print("  5. validation batch retrieval")
print("  6. tensor shape validation")
print("  7. numerical finiteness validation")
print("  8. mask validation")
print("  9. missing-rate validation")
print(" 10. data consistency validation")
print(" 11. split reproducibility validation")
print(" 12. train/validation leakage validation")

print()
print("If all required checks show [PASS], the dataset pipeline")
print("is ready for integration with the Trainer.")
print()
print("No neural-network training was performed by this test.")
print("=" * 70)