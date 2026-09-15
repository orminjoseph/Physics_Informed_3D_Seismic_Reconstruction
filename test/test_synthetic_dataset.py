"""
=========================================================
Synthetic Dataset Validation Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Tests:
1. Dataset creation
2. Sample generation
3. Tensor shapes
4. Mask validity
5. Velocity validity
6. Geological-mode validity
7. Reproducibility
8. DataLoader compatibility

=========================================================
"""

import torch
from torch.utils.data import DataLoader

from dataset.synthetic_dataset import SyntheticSeismicDataset


def test_dataset():

    print("\n" + "=" * 70)
    print("SYNTHETIC DATASET VALIDATION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Create dataset
    # -----------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=10,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    print("\n[1] Dataset creation")
    print("    Number of samples:", len(dataset))

    assert len(dataset) == 10

    print("    PASS")

    # -----------------------------------------------------
    # 2. Generate one sample
    # -----------------------------------------------------

    print("\n[2] Generating sample 0...")

    sample = dataset[0]

    (
        input_cube,
        target_cube,
        mask,
        velocity,
        mask_type,
        geological_mode,
    ) = sample

    print("    Input shape:   ", input_cube.shape)
    print("    Target shape:  ", target_cube.shape)
    print("    Mask shape:    ", mask.shape)
    print("    Velocity shape:", velocity.shape)

    print("    Mask type:     ", mask_type)
    print("    Geological mode:", geological_mode)

    # -----------------------------------------------------
    # 3. Check tensor dimensions
    # -----------------------------------------------------

    print("\n[3] Checking tensor dimensions")

    expected_shape = (1, 64, 128, 128)

    assert input_cube.shape == expected_shape
    assert target_cube.shape == expected_shape
    assert mask.shape == expected_shape
    assert velocity.shape == expected_shape

    print("    PASS")

    # -----------------------------------------------------
    # 4. Check tensor types
    # -----------------------------------------------------

    print("\n[4] Checking tensor data types")

    assert input_cube.dtype == torch.float32
    assert target_cube.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert velocity.dtype == torch.float32

    print("    PASS")

    # -----------------------------------------------------
    # 5. Check mask values
    # -----------------------------------------------------

    print("\n[5] Checking mask values")

    unique_mask_values = torch.unique(mask)

    print("    Unique mask values:", unique_mask_values.tolist())

    assert torch.all((mask == 0) | (mask == 1))

    print("    PASS")

    # -----------------------------------------------------
    # 6. Check input-mask relationship
    # -----------------------------------------------------

    print("\n[6] Checking input = target × mask")

    reconstructed_input = target_cube * mask

    assert torch.allclose(
        input_cube,
        reconstructed_input,
        atol=1e-6
    )

    print("    PASS")

    # -----------------------------------------------------
    # 7. Check velocity validity
    # -----------------------------------------------------

    print("\n[7] Checking velocity model")

    print("    Minimum velocity:", velocity.min().item())
    print("    Maximum velocity:", velocity.max().item())
    print("    Mean velocity:   ", velocity.mean().item())

    assert torch.isfinite(velocity).all()

    assert velocity.min() >= 1500.0
    assert velocity.max() <= 5000.0

    print("    PASS")

    # -----------------------------------------------------
    # 8. Check seismic target validity
    # -----------------------------------------------------

    print("\n[8] Checking seismic target")

    print("    Minimum amplitude:", target_cube.min().item())
    print("    Maximum amplitude:", target_cube.max().item())
    print("    Mean amplitude:   ", target_cube.mean().item())

    assert torch.isfinite(target_cube).all()

    print("    PASS")

    # -----------------------------------------------------
    # 9. Check geological mode
    # -----------------------------------------------------

    print("\n[9] Checking geological mode")

    valid_modes = {
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    }

    assert geological_mode in valid_modes

    print("    Mode:", geological_mode)
    print("    PASS")

    # -----------------------------------------------------
    # 10. Check mask type
    # -----------------------------------------------------

    print("\n[10] Checking mask type")

    valid_masks = {
        "random_voxels",
        "missing_traces",
        "missing_inlines",
        "missing_crosslines",
        "missing_blocks",
    }

    assert mask_type in valid_masks

    print("    Mask:", mask_type)
    print("    PASS")

    # -----------------------------------------------------
    # 11. Reproducibility test
    # -----------------------------------------------------

    print("\n[11] Testing reproducibility")

    dataset_a = SyntheticSeismicDataset(
        num_samples=3,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    dataset_b = SyntheticSeismicDataset(
        num_samples=3,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    sample_a = dataset_a[0]
    sample_b = dataset_b[0]

    assert torch.equal(sample_a[0], sample_b[0])
    assert torch.equal(sample_a[1], sample_b[1])
    assert torch.equal(sample_a[2], sample_b[2])
    assert torch.equal(sample_a[3], sample_b[3])

    assert sample_a[4] == sample_b[4]
    assert sample_a[5] == sample_b[5]

    print("    Input identical:     True")
    print("    Target identical:    True")
    print("    Mask identical:      True")
    print("    Velocity identical:  True")
    print("    Mask type identical: True")
    print("    Geological mode identical: True")

    print("    PASS")

    # -----------------------------------------------------
    # 12. DataLoader test
    # -----------------------------------------------------

    print("\n[12] Testing DataLoader compatibility")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    (
        batch_input,
        batch_target,
        batch_mask,
        batch_velocity,
        batch_mask_type,
        batch_geological_mode,
    ) = batch

    print("    Batch input shape:   ", batch_input.shape)
    print("    Batch target shape:  ", batch_target.shape)
    print("    Batch mask shape:    ", batch_mask.shape)
    print("    Batch velocity shape:", batch_velocity.shape)

    assert batch_input.shape == (1, 1, 64, 128, 128)
    assert batch_target.shape == (1, 1, 64, 128, 128)
    assert batch_mask.shape == (1, 1, 64, 128, 128)
    assert batch_velocity.shape == (1, 1, 64, 128, 128)

    print("    PASS")

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL SYNTHETIC DATASET TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_dataset()