"""
============================================================
VOLUME RECONSTRUCTOR V3 TEST
============================================================

Tests full-volume seismic reconstruction using the
authoritative MC Dropout uncertainty pipeline.

Pipeline:

    Network3D
        |
        v
    MCDropout3D
        |
        v
    VolumeReconstructor

Author: Ormin Joseph
============================================================
"""

import os

import numpy as np
import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D

from inference.volume_reconstructor import VolumeReconstructor

from utils.helpers import get_device
from utils.config import (
    EXPERIMENT_NAME,
    SYNTHETIC_PATCH_SIZE,
    MC_DROPOUT_SAMPLES,
)

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = os.path.join(
    "outputs",
    EXPERIMENT_NAME,
    "checkpoints",
    "best_model.pth"
)

MC_SAMPLES = MC_DROPOUT_SAMPLES

# Current synthetic patch configuration.
PATCH_SIZE = tuple(SYNTHETIC_PATCH_SIZE)

# No separate synthetic stride is currently defined.
# Use non-overlapping patches for this test.
PATCH_STRIDE = PATCH_SIZE


# ============================================================
# TEST
# ============================================================

def test_volume_reconstructor_v3():

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTOR V3 TEST")
    print("=" * 60)

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print()
    print("Device:")
    print(device)

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = Network3D()

    checkpoint_data = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # --------------------------------------------------------
    # Support the project's checkpoint format.
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint_data:

        model.load_state_dict(
            checkpoint_data["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint_data
        )

    model = model.to(device)

    print()
    print("Model loaded successfully.")

    # ========================================================
    # MC DROPOUT ENGINE
    # ========================================================

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES
    )

    print()
    print("MC Dropout Samples:")
    print(MC_SAMPLES)

    # ========================================================
    # TEST INPUT VOLUME
    # ========================================================

    volume = np.random.randn(
        *PATCH_SIZE
    ).astype(np.float32)

    print()
    print("Input Volume Shape:")
    print(volume.shape)

    # ========================================================
    # VOLUME RECONSTRUCTOR
    # ========================================================

    # --------------------------------------------------------
    # The current VolumeReconstructor expects:
    #
    #     predictor.predict(patch_tensor)
    #
    # to return a dictionary containing:
    #
    #     reconstruction_samples
    #     log_variance_samples
    #
    # It returns:
    #
    #     reconstructed_volume,
    #     predictive_std_volume
    #
    # The second output is predictive STANDARD DEVIATION,
    # not predictive variance.
    # --------------------------------------------------------

    reconstructor = VolumeReconstructor(
        predictor=mc_dropout,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE
    )

    print()
    print("Patch Size:")
    print(PATCH_SIZE)

    print()
    print("Patch Stride:")
    print(PATCH_STRIDE)

    # ========================================================
    # RECONSTRUCT VOLUME
    # ========================================================

    reconstructed_volume, predictive_std_volume = (
        reconstructor.reconstruct(
            volume
        )
    )

    # ========================================================
    # VALIDATE OUTPUT TYPES
    # ========================================================

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTION OUTPUT")
    print("=" * 60)

    assert isinstance(
        reconstructed_volume,
        np.ndarray
    ), (
        "Reconstructed volume must be "
        "a NumPy array."
    )

    assert isinstance(
        predictive_std_volume,
        np.ndarray
    ), (
        "Predictive standard deviation must be "
        "a NumPy array."
    )

    print("Output type checks: PASSED")

    # ========================================================
    # VALIDATE OUTPUT SHAPES
    # ========================================================

    print()
    print(
        "Reconstruction Shape:",
        reconstructed_volume.shape
    )

    print(
        "Predictive Std Shape:",
        predictive_std_volume.shape
    )

    expected_shape = PATCH_SIZE

    assert (
        reconstructed_volume.shape
        == expected_shape
    ), (
        "Unexpected reconstructed volume shape: "
        f"{reconstructed_volume.shape}"
    )

    assert (
        predictive_std_volume.shape
        == expected_shape
    ), (
        "Unexpected predictive standard deviation "
        f"shape: {predictive_std_volume.shape}"
    )

    print()
    print("Output shape checks: PASSED")

    # ========================================================
    # FINITE-VALUE CHECKS
    # ========================================================

    assert np.isfinite(
        reconstructed_volume
    ).all(), (
        "Reconstructed volume contains "
        "non-finite values."
    )

    assert np.isfinite(
        predictive_std_volume
    ).all(), (
        "Predictive standard deviation contains "
        "non-finite values."
    )

    print("Finite-value checks: PASSED")

    # ========================================================
    # UNCERTAINTY CHECK
    # ========================================================

    # --------------------------------------------------------
    # Predictive standard deviation must be non-negative.
    # --------------------------------------------------------

    assert np.all(
        predictive_std_volume >= 0
    ), (
        "Predictive standard deviation contains "
        "negative values."
    )

    print("Predictive uncertainty check: PASSED")

    # ========================================================
    # OUTPUT STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("OUTPUT STATISTICS")
    print("=" * 60)

    print()
    print("Reconstructed Volume:")
    print("Minimum:", reconstructed_volume.min())
    print("Maximum:", reconstructed_volume.max())
    print("Mean:", reconstructed_volume.mean())

    print()
    print("Predictive Standard Deviation:")
    print("Minimum:", predictive_std_volume.min())
    print("Maximum:", predictive_std_volume.max())
    print("Mean:", predictive_std_volume.mean())

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTOR V3: PASSED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_volume_reconstructor_v3()