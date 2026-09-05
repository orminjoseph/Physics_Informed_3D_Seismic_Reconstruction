"""
============================================================
F3 MONTE CARLO DROPOUT TEST
============================================================

Evaluates epistemic uncertainty on an F3 seismic patch
using the authoritative MCDropout3D implementation.

Uncertainty pathway:

    F3 seismic patch
          |
          v
       Network3D
          |
          v
      MCDropout3D
          |
          +---- MC reconstruction samples
          |
          +---- Mean reconstruction
          |
          +---- Epistemic reconstruction variance

Author: Ormin Joseph
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from models.mc_dropout import MCDropout3D
from utils.visualization import Visualizer


# ============================================================
# F3 DATASET PATH
# ============================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


# ============================================================
# MODEL CHECKPOINT
# ============================================================

CHECKPOINT = (
    "checkpoints/best_model.pth"
)


# ============================================================
# MC DROPOUT CONFIGURATION
# ============================================================

MC_SAMPLES = 20


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("F3 MONTE CARLO DROPOUT TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Select computation device.
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Load the F3 seismic dataset.
    # --------------------------------------------------------

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    # --------------------------------------------------------
    # Retrieve the first F3 patch.
    #
    # Dataset output:
    #
    # corrupted = incomplete seismic patch
    # target    = original seismic patch
    # mask      = observed/missing-data mask
    # velocity  = velocity information
    # --------------------------------------------------------

    corrupted, target, mask, velocity = dataset[0][:4]

    print()
    print("F3 Corrupted Patch Shape:")
    print(corrupted.shape)

    print("F3 Target Patch Shape:")
    print(target.shape)

    print("F3 Mask Shape:")
    print(mask.shape)

    print("F3 Velocity Shape:")
    print(velocity.shape)

    # --------------------------------------------------------
    # Create the 3D seismic reconstruction network.
    # --------------------------------------------------------

    model = Network3D()

    # --------------------------------------------------------
    # Load the trained checkpoint if it exists.
    #
    # The authoritative MCDropout3D class performs the
    # stochastic forward passes. We therefore do NOT use
    # --------------------------------------------------------

    checkpoint_data = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Create the authoritative MC Dropout engine.
    # --------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES
    )

    # --------------------------------------------------------
    # Run stochastic MC Dropout inference.
    #
    # The returned dictionary contains:
    #
    # reconstruction_samples
    # reconstruction_mean
    # reconstruction_epistemic_variance
    #
    # travel_time_samples
    # travel_time_mean
    # travel_time_epistemic_variance
    #
    # log_variance_samples
    # log_variance_mean
    # log_variance_epistemic_variance
    # --------------------------------------------------------

    results = mc_dropout.predict(
        corrupted.to(device)
    )

    # --------------------------------------------------------
    # Extract reconstruction uncertainty.
    # --------------------------------------------------------

    reconstruction_samples = results[
        "reconstruction_samples"
    ]

    mean_prediction = results[
        "reconstruction_mean"
    ]

    epistemic_variance = results[
        "reconstruction_epistemic_variance"
    ]

    # --------------------------------------------------------
    # Extract travel-time uncertainty.
    #
    # This is kept separate from reconstruction predictive
    # uncertainty.
    # --------------------------------------------------------

    travel_time_mean = results[
        "travel_time_mean"
    ]

    travel_time_epistemic_variance = results[
        "travel_time_epistemic_variance"
    ]

    # --------------------------------------------------------
    # Extract log-variance outputs.
    #
    # The epistemic variance of log-variance is diagnostic
    # only. It must NOT be added directly to reconstruction
    # epistemic variance.
    # --------------------------------------------------------

    log_variance_mean = results[
        "log_variance_mean"
    ]

    log_variance_epistemic_variance = results[
        "log_variance_epistemic_variance"
    ]

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("MC DROPOUT OUTPUT VALIDATION")
    print("=" * 60)

    print()
    print("MC Reconstruction Samples:")
    print(reconstruction_samples.shape)

    print()
    print("Mean Reconstruction:")
    print(mean_prediction.shape)

    print()
    print("Epistemic Reconstruction Variance:")
    print(epistemic_variance.shape)

    # --------------------------------------------------------
    # Check that all outputs contain finite values.
    # --------------------------------------------------------

    assert torch.isfinite(
        reconstruction_samples
    ).all(), (
        "MC reconstruction samples contain "
        "non-finite values."
    )

    assert torch.isfinite(
        mean_prediction
    ).all(), (
        "Mean reconstruction contains "
        "non-finite values."
    )

    assert torch.isfinite(
        epistemic_variance
    ).all(), (
        "Epistemic variance contains "
        "non-finite values."
    )

    # --------------------------------------------------------
    # Epistemic variance must be non-negative.
    # --------------------------------------------------------

    assert (
        epistemic_variance >= 0
    ).all(), (
        "Epistemic variance contains "
        "negative values."
    )

    print()
    print("Finite-value checks: PASSED")
    print("Non-negative variance check: PASSED")

    # ========================================================
    # STOCHASTICITY CHECK
    # ========================================================

    # --------------------------------------------------------
    # MC Dropout should produce different predictions when
    # dropout is active.
    # --------------------------------------------------------

    first_prediction = reconstruction_samples[0]
    second_prediction = reconstruction_samples[1]

    mean_difference = torch.mean(
        torch.abs(
            first_prediction - second_prediction
        )
    ).item()

    print()
    print(
        "Mean difference between first two "
        "MC reconstruction samples:"
    )
    print(mean_difference)

    assert mean_difference > 0.0, (
        "MC Dropout samples are identical. "
        "Dropout may not be active."
    )

    print()
    print("Stochasticity check: PASSED")

    # ========================================================
    # EPISTEMIC UNCERTAINTY SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("EPISTEMIC UNCERTAINTY")
    print("=" * 60)

    print()
    print(
        "Minimum:",
        epistemic_variance.min().item()
    )

    print(
        "Maximum:",
        epistemic_variance.max().item()
    )

    print(
        "Mean:",
        epistemic_variance.mean().item()
    )

    # ========================================================
    # TRAVEL-TIME UNCERTAINTY
    # ========================================================

    print()
    print("=" * 60)
    print("TRAVEL-TIME EPISTEMIC UNCERTAINTY")
    print("=" * 60)

    print(
        "Mean:",
        travel_time_epistemic_variance.mean().item()
    )

    # ========================================================
    # LOG-VARIANCE DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 60)
    print("LOG-VARIANCE EPISTEMIC DIAGNOSTIC")
    print("=" * 60)

    print(
        "Mean:",
        log_variance_epistemic_variance.mean().item()
    )

    print(
        "NOTE: Log-variance epistemic variance is "
        "diagnostic only."
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualizer = Visualizer()

    # --------------------------------------------------------
    # Save the MC mean reconstruction.
    # --------------------------------------------------------

    visualizer.save_slice(
        mean_prediction.detach().cpu().squeeze(0),
        "f3_mc_mean_prediction.png",
        "F3 MC Dropout Mean Prediction"
    )

    # --------------------------------------------------------
    # Save the epistemic variance.
    # --------------------------------------------------------

    visualizer.save_slice(
        epistemic_variance.detach().cpu().squeeze(0),
        "f3_mc_epistemic_variance.png",
        "F3 MC Dropout Epistemic Variance"
    )

    # ========================================================
    # TEST COMPLETION
    # ========================================================

    print()
    print("=" * 60)
    print("F3 MONTE CARLO DROPOUT TEST: PASSED")
    print("=" * 60)

    print()
    print("Images Saved Successfully")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()