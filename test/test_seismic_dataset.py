"""
=========================================================
Test Unified Seismic Dataset
=========================================================

Verifies:

• Synthetic seismic cube generation
• Patch extraction
• Mask generation
• Input/target tensor creation
• Correct tensor shapes
=========================================================
"""

import torch

from dataset.seismic_dataset import SeismicDataset


def test_seismic_dataset():

    # --------------------------------------------------
    # Create synthetic seismic dataset
    # --------------------------------------------------

    dataset = SeismicDataset(
        synthetic=True
    )

    # --------------------------------------------------
    # Verify dataset contains patches
    # --------------------------------------------------

    assert len(dataset) > 0

    # --------------------------------------------------
    # Load first sample
    # --------------------------------------------------

    sample = dataset[0]

    # --------------------------------------------------
    # Verify dictionary keys
    # --------------------------------------------------

    assert "input" in sample
    assert "target" in sample
    assert "mask" in sample
    assert "position" in sample

    # --------------------------------------------------
    # Extract tensors
    # --------------------------------------------------

    input_patch = sample["input"]

    target_patch = sample["target"]

    mask = sample["mask"]

    # --------------------------------------------------
    # Verify PyTorch tensors
    # --------------------------------------------------

    assert isinstance(
        input_patch,
        torch.Tensor
    )

    assert isinstance(
        target_patch,
        torch.Tensor
    )

    assert isinstance(
        mask,
        torch.Tensor
    )

    # --------------------------------------------------
    # Verify dimensions
    #
    # Expected:
    #
    # [1, D, H, W]
    # --------------------------------------------------

    assert input_patch.ndim == 4

    assert target_patch.ndim == 4

    assert mask.ndim == 4

    # --------------------------------------------------
    # Verify identical shapes
    # --------------------------------------------------

    assert input_patch.shape == target_patch.shape

    assert input_patch.shape == mask.shape

    # --------------------------------------------------
    # Verify channel dimension
    # --------------------------------------------------

    assert input_patch.shape[0] == 1

    # --------------------------------------------------
    # Verify mask values
    # --------------------------------------------------

    unique_mask_values = torch.unique(
        mask
    )

    for value in unique_mask_values:

        assert value.item() in (
            0.0,
            1.0
        )

    # --------------------------------------------------
    # Verify missing traces actually exist
    # --------------------------------------------------

    assert torch.sum(
        mask == 0
    ).item() > 0

    # --------------------------------------------------
    # Verify input is zero wherever
    # mask is zero
    # --------------------------------------------------

    assert torch.all(
        input_patch[mask == 0] == 0
    )

    # --------------------------------------------------
    # Print diagnostic information
    # --------------------------------------------------

    print()

    print(
        "Number of Patches :",
        len(dataset)
    )

    print(
        "Input Shape       :",
        input_patch.shape
    )

    print(
        "Target Shape      :",
        target_patch.shape
    )

    print(
        "Mask Shape        :",
        mask.shape
    )

    print(
        "Input Min         :",
        input_patch.min().item()
    )

    print(
        "Input Max         :",
        input_patch.max().item()
    )

    print(
        "Target Min        :",
        target_patch.min().item()
    )

    print(
        "Target Max        :",
        target_patch.max().item()
    )

    print(
        "Missing Voxels    :",
        torch.sum(mask == 0).item()
    )

    print()