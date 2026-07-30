"""
============================================================
Testing Volume Reconstructor V2
============================================================

Author: Ormin Joseph
============================================================
"""

import numpy as np

from models.network import Network3D

from inference.mc_dropout_predictor import (
    MCDropoutPredictor
)

from inference.volume_reconstructor import (
    VolumeReconstructor
)

from utils.helpers import get_device
from utils.config import PATCH_SIZE
from utils.config import PATCH_STRIDE


def test_volume_reconstructor_v2():

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTOR V2 TEST")
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

    volume = np.random.randn(
        64,
        128,
        128
    ).astype(np.float32)

    reconstructed_volume = (
        reconstructor.reconstruct(volume)
    )

    print()
    print(
        "Original Volume Shape      :",
        volume.shape
    )

    print(
        "Reconstructed Volume Shape :",
        reconstructed_volume.shape
    )

    print()
    print(
        "Volume Reconstruction V2: PASSED"
    )


if __name__ == "__main__":

    test_volume_reconstructor_v2()