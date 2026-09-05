"""
============================================================
MC DROPOUT CALIBRATION TEST
============================================================

Checks the relationship between MC Dropout epistemic
uncertainty and reconstruction error on an F3 seismic patch.

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
          |
          v
    Compare with reconstruction error

Important:
    This test evaluates the relationship between epistemic
    variance and reconstruction error.

    It does NOT calculate total predictive uncertainty.

    It does NOT establish scientific calibration from one
    untrained or randomly initialized model.

Author: Ormin Joseph
============================================================
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.f3_dataset import F3Dataset
from models.mc_dropout import MCDropout3D
from models.network import Network3D


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

CHECKPOINT = "checkpoints/best_model.pth"


# ============================================================
# MC DROPOUT CONFIGURATION
# ============================================================

MC_SAMPLES = 20


# ============================================================
# OUTPUT PATHS
# ============================================================

CSV_FILE = (
    "outputs/reports/"
    "mc_dropout_calibration.csv"
)

PLOT_FILE = (
    "outputs/figures/"
    "mc_dropout_calibration.png"
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MC DROPOUT CALIBRATION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Select computation device.
    # --------------------------------------------------------

    device = torch.device(
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
    # --------------------------------------------------------

    corrupted, target, mask, velocity = dataset[0][:4]

    # --------------------------------------------------------
    # Create the 3D reconstruction network.
    # --------------------------------------------------------

    model = Network3D()

    # --------------------------------------------------------
    # Load the trained checkpoint.
    #
    # The checkpoint is expected to contain:
    #
    #     "model_state_dict"
    #
    # This test uses the trained model rather than the
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
    # Run stochastic inference.
    #
    # The authoritative implementation returns:
    #
    #     reconstruction_samples
    #     reconstruction_mean
    #     reconstruction_epistemic_variance
    #
    # plus travel-time and log-variance outputs.
    # --------------------------------------------------------

    results = mc_dropout.predict(
        corrupted.to(device)
    )

    # --------------------------------------------------------
    # Extract the reconstruction outputs.
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

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("OUTPUT VALIDATION")
    print("=" * 60)

    print()
    print("MC Reconstruction Samples:")
    print(reconstruction_samples.shape)

    print()
    print("Mean Reconstruction:")
    print(mean_prediction.shape)

    print()
    print("Epistemic Variance:")
    print(epistemic_variance.shape)

    # --------------------------------------------------------
    # Check expected shapes.
    #
    # The F3 patch is expected to have shape:
    #
    #     [1, 1, 64, 64, 64]
    #
    # The MC sample tensor is expected to have shape:
    #
    #     [20, 1, 1, 64, 64, 64]
    # --------------------------------------------------------

    expected_output_shape = (
        1, 1, 64, 64, 64
    )

    expected_sample_shape = (
        MC_SAMPLES,
        1, 1, 64, 64, 64
    )

    assert tuple(
        reconstruction_samples.shape
    ) == expected_sample_shape, (
        "Unexpected reconstruction sample shape: "
        f"{reconstruction_samples.shape}"
    )

    assert tuple(
        mean_prediction.shape
    ) == expected_output_shape, (
        "Unexpected mean reconstruction shape: "
        f"{mean_prediction.shape}"
    )

    assert tuple(
        epistemic_variance.shape
    ) == expected_output_shape, (
        "Unexpected epistemic variance shape: "
        f"{epistemic_variance.shape}"
    )

    # --------------------------------------------------------
    # Check finite values.
    # --------------------------------------------------------

    assert torch.isfinite(
        reconstruction_samples
    ).all(), (
        "Reconstruction samples contain "
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
    print("Shape checks: PASSED")
    print("Finite-value checks: PASSED")
    print("Non-negative variance check: PASSED")

    # ========================================================
    # INDEPENDENT EPISTEMIC VARIANCE CHECK
    # ========================================================

    # --------------------------------------------------------
    # Independently calculate population variance:
    #
    #     Var(X) = mean((X - mean(X))²)
    #
    # This verifies that the authoritative MC Dropout engine
    # is calculating epistemic variance correctly.
    # --------------------------------------------------------

    independent_epistemic_variance = (
        (
            reconstruction_samples
            - mean_prediction.unsqueeze(0)
        ) ** 2
    ).mean(dim=0)

    maximum_variance_difference = torch.max(
        torch.abs(
            epistemic_variance
            - independent_epistemic_variance
        )
    ).item()

    print()
    print(
        "Maximum difference between implemented "
        "and independently calculated epistemic variance:"
    )
    print(maximum_variance_difference)

    assert torch.allclose(
        epistemic_variance,
        independent_epistemic_variance,
        rtol=1e-5,
        atol=1e-6
    ), (
        "Implemented epistemic variance does not "
        "match the independent calculation."
    )

    print()
    print("Independent epistemic variance check: PASSED")

    # ========================================================
    # RECONSTRUCTION ERROR
    # ========================================================

    # --------------------------------------------------------
    # Calculate absolute reconstruction error.
    #
    # Both tensors have shape:
    #
    #     [1, 1, 64, 64, 64]
    #
    # The result is an absolute-error map.
    # --------------------------------------------------------

    error_map = torch.abs(
        mean_prediction - target.to(device)
    )

    # --------------------------------------------------------
    # Flatten the error and uncertainty maps.
    #
    # This allows a voxel-by-voxel comparison.
    # --------------------------------------------------------

    error_flat = (
        error_map
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    uncertainty_flat = (
        epistemic_variance
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # --------------------------------------------------------
    # Check that the flattened arrays have equal length.
    # --------------------------------------------------------

    assert len(error_flat) == len(
        uncertainty_flat
    ), (
        "Error and uncertainty arrays "
        "have different lengths."
    )

    # ========================================================
    # CORRELATION
    # ========================================================

    # --------------------------------------------------------
    # Pearson correlation between:
    #
    #     absolute reconstruction error
    #
    # and:
    #
    #     epistemic reconstruction variance
    #
    # A positive correlation suggests that regions with
    # greater epistemic uncertainty tend to have greater
    # reconstruction error.
    #
    # A single correlation does not establish calibration.
    # --------------------------------------------------------

    error_std = np.std(error_flat)
    uncertainty_std = np.std(uncertainty_flat)

    if error_std == 0.0 or uncertainty_std == 0.0:

        correlation = float("nan")

        print()
        print(
            "Correlation cannot be calculated because "
            "one variable has zero variance."
        )

    else:

        correlation = np.corrcoef(
            error_flat,
            uncertainty_flat
        )[0, 1]

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("MC DROPOUT CALIBRATION")
    print("=" * 60)

    print()
    print(
        f"Correlation(Error, Epistemic Variance): "
        f"{correlation:.4f}"
    )

    print()
    print(
        "Epistemic Variance Minimum:",
        epistemic_variance.min().item()
    )

    print(
        "Epistemic Variance Maximum:",
        epistemic_variance.max().item()
    )

    print(
        "Epistemic Variance Mean:",
        epistemic_variance.mean().item()
    )

    print()
    print(
        "Mean Absolute Reconstruction Error:",
        error_flat.mean()
    )

    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Correlation_Error_Epistemic_Variance",
                "Mean_Absolute_Error",
                "Minimum_Epistemic_Variance",
                "Maximum_Epistemic_Variance",
                "Mean_Epistemic_Variance",
                "MC_Samples"
            ]
        )

        writer.writerow(
            [
                correlation,
                error_flat.mean(),
                epistemic_variance.min().item(),
                epistemic_variance.max().item(),
                epistemic_variance.mean().item(),
                MC_SAMPLES
            ]
        )

    # ========================================================
    # SCATTER PLOT
    # ========================================================

    # --------------------------------------------------------
    # Use a reproducible random sample of voxels.
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    sample_size = min(
        5000,
        len(error_flat)
    )

    sample = rng.choice(
        len(error_flat),
        sample_size,
        replace=False
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        uncertainty_flat[sample],
        error_flat[sample],
        alpha=0.3,
        s=5
    )

    plt.xlabel(
        "MC Dropout Epistemic Variance"
    )

    plt.ylabel(
        "Absolute Reconstruction Error"
    )

    plt.title(
        "MC Dropout Calibration\n"
        f"Correlation={correlation:.4f}"
    )

    plt.grid(True)

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # COMPLETION
    # ========================================================

    print()
    print("CSV saved to:")
    print(CSV_FILE)

    print()
    print("Plot saved to:")
    print(PLOT_FILE)

    print()
    print("=" * 60)
    print("MC DROPOUT CALIBRATION TEST: PASSED")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()