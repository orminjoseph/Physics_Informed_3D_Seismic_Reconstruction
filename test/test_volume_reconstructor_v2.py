"""
============================================================
Testing Volume Reconstructor V2
============================================================

Tests full-volume seismic reconstruction using:

    Network3D
        |
        v
    MCDropout3D
        |
        v
    VolumeReconstructor

The test validates:

    1. Model loading
    2. MC Dropout prediction
    3. Reconstruction output
    4. Predictive uncertainty output
    5. Output shapes
    6. Finite numerical values
    7. Non-negative predictive uncertainty

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
    SYNTHETIC_PATCH_SIZE,
    MC_DROPOUT_SAMPLES,
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    "outputs/"
    "experiment_20260729_201904/"
    "checkpoints/"
    "best_model.pth"
)

MC_SAMPLES = MC_DROPOUT_SAMPLES

# ------------------------------------------------------------
# Current synthetic configuration:
#
#     SYNTHETIC_PATCH_SIZE = (64, 128, 128)
#
# No separate synthetic stride is currently defined.
# Therefore, use non-overlapping patches.
# ------------------------------------------------------------

PATCH_SIZE = tuple(SYNTHETIC_PATCH_SIZE)
PATCH_STRIDE = PATCH_SIZE


# ============================================================
# TEST
# ============================================================

def test_volume_reconstructor_v2():

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTOR V2 TEST")
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

    volume = torch.randn(
        *PATCH_SIZE,
        dtype=torch.float32
    )

    print()
    print("Input Volume Shape:")
    print(volume.shape)

    # --------------------------------------------------------
    # Convert:
    #
    #     [D, H, W]
    #
    # to:
    #
    #     [B, C, D, H, W]
    #
    # Example:
    #
    #     [64, 128, 128]
    #             |
    #             v
    #     [1, 1, 64, 128, 128]
    # --------------------------------------------------------

    volume_tensor = (
        volume
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    print()
    print("Model Input Shape:")
    print(volume_tensor.shape)

    # ========================================================
    # RUN MC DROPOUT
    # ========================================================

    mc_results = mc_dropout.predict(
        volume_tensor
    )

    # --------------------------------------------------------
    # Extract authoritative MC Dropout outputs.
    # --------------------------------------------------------

    reconstruction_samples = (
        mc_results["reconstruction_samples"]
    )

    reconstruction_mean = (
        mc_results["reconstruction_mean"]
    )

    reconstruction_epistemic_variance = (
        mc_results["reconstruction_epistemic_variance"]
    )

    # ========================================================
    # MC OUTPUT VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("MC DROPOUT OUTPUT")
    print("=" * 60)

    print(
        "Reconstruction Samples:",
        reconstruction_samples.shape
    )

    print(
        "Reconstruction Mean:",
        reconstruction_mean.shape
    )

    print(
        "Epistemic Variance:",
        reconstruction_epistemic_variance.shape
    )

    expected_sample_shape = (
        MC_SAMPLES,
        1,
        1,
        *PATCH_SIZE
    )

    expected_output_shape = (
        1,
        1,
        *PATCH_SIZE
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert (
        reconstruction_samples.shape
        == expected_sample_shape
    ), (
        "Unexpected reconstruction sample shape: "
        f"{reconstruction_samples.shape}"
    )

    assert (
        reconstruction_mean.shape
        == expected_output_shape
    ), (
        "Unexpected reconstruction mean shape: "
        f"{reconstruction_mean.shape}"
    )

    assert (
        reconstruction_epistemic_variance.shape
        == expected_output_shape
    ), (
        "Unexpected epistemic variance shape: "
        f"{reconstruction_epistemic_variance.shape}"
    )

    print()
    print("MC output shape checks: PASSED")

    # --------------------------------------------------------
    # Finite-value checks
    # --------------------------------------------------------

    assert torch.isfinite(
        reconstruction_samples
    ).all(), (
        "Reconstruction samples contain "
        "non-finite values."
    )

    assert torch.isfinite(
        reconstruction_mean
    ).all(), (
        "Reconstruction mean contains "
        "non-finite values."
    )

    assert torch.isfinite(
        reconstruction_epistemic_variance
    ).all(), (
        "Epistemic variance contains "
        "non-finite values."
    )

    print("MC finite-value checks: PASSED")

    # --------------------------------------------------------
    # Variance must be non-negative.
    # --------------------------------------------------------

    assert torch.all(
        reconstruction_epistemic_variance >= 0
    ), (
        "Epistemic variance contains "
        "negative values."
    )

    print("MC variance check: PASSED")

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
            volume.numpy()
        )
    )

    # ========================================================
    # VALIDATE RECONSTRUCTION RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("VOLUME RECONSTRUCTION OUTPUT")
    print("=" * 60)

    print(
        "Reconstructed Volume Shape:",
        reconstructed_volume.shape
    )

    print(
        "Predictive Std Volume Shape:",
        predictive_std_volume.shape
    )

    expected_volume_shape = PATCH_SIZE

    # --------------------------------------------------------
    # Output shape checks
    # --------------------------------------------------------

    assert (
        reconstructed_volume.shape
        == expected_volume_shape
    ), (
        "Unexpected reconstructed volume shape: "
        f"{reconstructed_volume.shape}"
    )

    assert (
        predictive_std_volume.shape
        == expected_volume_shape
    ), (
        "Unexpected predictive standard deviation "
        f"shape: {predictive_std_volume.shape}"
    )

    print()
    print("Reconstruction shape checks: PASSED")

    # --------------------------------------------------------
    # Output type checks
    # --------------------------------------------------------

    assert isinstance(
        reconstructed_volume,
        np.ndarray
    ), (
        "Reconstructed volume must be a NumPy array."
    )

    assert isinstance(
        predictive_std_volume,
        np.ndarray
    ), (
        "Predictive standard deviation must be "
        "a NumPy array."
    )

    print("Reconstruction type checks: PASSED")

    # --------------------------------------------------------
    # Finite-value checks
    # --------------------------------------------------------

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

    print("Reconstruction finite-value checks: PASSED")

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
    print("VOLUME RECONSTRUCTOR V2 TEST: PASSED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_volume_reconstructor_v2()