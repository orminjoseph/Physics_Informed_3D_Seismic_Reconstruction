# test/test_dataloader.py

import torch

from dataset.build_dataset import build_dataset
from torch.utils.data import DataLoader


def test_dataloader():

    print()
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Build the selected dataset
    # -----------------------------------------------------

    dataset = build_dataset()

    print()
    print("Dataset Type:")
    print(type(dataset))

    print()
    print("Dataset Size:")
    print(len(dataset))

    # -----------------------------------------------------
    # Create DataLoader
    # -----------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # -----------------------------------------------------
    # Get one batch
    # -----------------------------------------------------

    batch = next(iter(loader))

    # -----------------------------------------------------
    # Dataset returns:
    #
    # input
    # target
    # mask
    # velocity
    #
    # -----------------------------------------------------

    input_cube, target, mask, velocity = batch

    print()
    print("Input Batch Shape:")
    print(input_cube.shape)

    print()
    print("Target Batch Shape:")
    print(target.shape)

    print()
    print("Mask Batch Shape:")
    print(mask.shape)

    print()
    print("Velocity Batch Shape:")
    print(velocity.shape)

    # -----------------------------------------------------
    # Validate tensors
    # -----------------------------------------------------

    assert isinstance(input_cube, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(velocity, torch.Tensor)

    # -----------------------------------------------------
    # Validate batch dimension
    # -----------------------------------------------------

    assert input_cube.shape[0] == 2
    assert target.shape[0] == 2
    assert mask.shape[0] == 2
    assert velocity.shape[0] == 2

    # -----------------------------------------------------
    # Validate that all spatial dimensions agree
    # -----------------------------------------------------

    assert input_cube.shape == target.shape
    assert input_cube.shape == mask.shape
    assert input_cube.shape == velocity.shape

    # -----------------------------------------------------
    # Validate finite values
    # -----------------------------------------------------

    assert torch.isfinite(input_cube).all()
    assert torch.isfinite(target).all()
    assert torch.isfinite(mask).all()
    assert torch.isfinite(velocity).all()

    # -----------------------------------------------------
    # Validate mask
    # -----------------------------------------------------

    assert mask.min() >= 0
    assert mask.max() <= 1

    print()
    print("DATALOADER TEST PASSED")