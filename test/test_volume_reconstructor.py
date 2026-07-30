"""
============================================================
Testing Volume Reconstructor
============================================================

Author: Ormin Joseph
============================================================
"""

import torch

from models.network import Network3D
from inference.mc_dropout_predictor import MCDropoutPredictor
from inference.volume_reconstructor import VolumeReconstructor

from utils.helpers import get_device
from utils.config import PATCH_SIZE, PATCH_STRIDE


def test_volume_reconstructor():

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTOR TEST")
    print("=" * 60)

    device = get_device()

    model = Network3D()

    checkpoint_path = (
        "outputs/"
        "experiment_20260729_201904/"
        "checkpoints/"
        "best_model.pth"
    )

    predictor = MCDropoutPredictor(
        model=model,
        checkpoint=checkpoint_path,
        device=device
    )

    reconstructor = VolumeReconstructor(
        predictor=predictor,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE
    )

    volume = torch.randn(
        64,
        128,
        128
    ).numpy()

    reconstructed_patches = (
        reconstructor.reconstruct(volume)
    )

    print()
    print(
        "Number of reconstructed patches :",
        len(reconstructed_patches)
    )

    print(
        "Patch Shape :",
        reconstructed_patches[0].shape
    )

    print()
    print(
        "Volume Reconstructor Test: PASSED"
    )


if __name__ == "__main__":

    test_volume_reconstructor()