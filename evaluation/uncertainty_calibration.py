"""
=========================================================
Uncertainty Calibration
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluates whether higher predictive uncertainty is associated
with higher reconstruction error.

The script is DATASET-MODE AWARE and can operate with:

    DATASET_MODE = "synthetic"
    DATASET_MODE = "f3"

Predictive uncertainty is computed as:

    Predictive Variance
        = Aleatoric Variance
        + Epistemic Variance

and therefore:

    Predictive Std
        = sqrt(Aleatoric Std^2 + Epistemic Std^2)

Outputs
-------
1. uncertainty_calibration.csv
2. uncertainty_calibration.png

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import (
    DATASET_MODE,
    CHECKPOINT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    DEVICE,
)

from dataset.build_dataset import build_dataset
from inference.predictor import Predictor
from models.network import Network3D


# =========================================================
# Configuration
# =========================================================

NUM_PATCHES = 20
NUM_BINS = 10

CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "uncertainty_calibration"
)

CSV_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "uncertainty_calibration.csv"
)

FIGURE_FILE = os.path.join(
    FIGURE_DIR,
    "uncertainty_calibration",
    "uncertainty_calibration.png"
)


# =========================================================
# Utility Functions
# =========================================================

def validate_finite_array(
    array,
    name
):
    """
    Validate that an array contains finite values.
    """

    array = np.asarray(
        array,
        dtype=np.float64
    )

    if array.size == 0:

        raise ValueError(
            f"{name} is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


# =========================================================
# Tensor Preparation
# =========================================================

def prepare_input_tensor(
    corrupted
):
    """
    Convert dataset input into the 5D tensor format expected
    by the network:

        [Batch, Channel, Depth, Height, Width]
    """

    if corrupted.ndim == 4:

        corrupted = corrupted.unsqueeze(0)

    elif corrupted.ndim == 5:

        pass

    else:

        raise ValueError(
            "Unexpected corrupted input shape: "
            f"{tuple(corrupted.shape)}"
        )

    return corrupted


# =========================================================
# Dataset Information
# =========================================================

def describe_dataset(
    dataset
):
    """
    Print dataset information.
    """

    print()
    print(
        f"Dataset mode : {DATASET_MODE}"
    )

    print(
        f"Dataset size : {len(dataset)}"
    )

    print(
        f"Number of patches evaluated : "
        f"{min(NUM_PATCHES, len(dataset))}"
    )


# =========================================================
# Collect Error and Uncertainty
# =========================================================

def collect_calibration_data(
    dataset,
    predictor
):
    """
    Collect voxel-wise reconstruction error and
    predictive uncertainty.

    Returns
    -------
    errors : numpy.ndarray
        Absolute reconstruction errors.

    uncertainties : numpy.ndarray
        Predictive standard deviations.
    """

    all_errors = []
    all_uncertainties = []

    number_of_patches = min(
        NUM_PATCHES,
        len(dataset)
    )

    for patch_index in range(
        number_of_patches
    ):

        print(
            f"Processing patch "
            f"{patch_index + 1}/"
            f"{number_of_patches}"
        )

        # -------------------------------------------------
        # Dataset sample
        # -------------------------------------------------

        sample = dataset[
            patch_index
        ]

        if len(sample) < 4:

            raise ValueError(
                "Dataset sample must contain at least "
                "input, target, mask and velocity."
            )

        corrupted = sample[0]
        target = sample[1]

        # -------------------------------------------------
        # Prepare input
        # -------------------------------------------------

        corrupted = prepare_input_tensor(
            corrupted
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            aleatoric_std,
            epistemic_std
        ) = predictor.predict(
            corrupted
        )

        # -------------------------------------------------
        # Prepare target
        # -------------------------------------------------

        if target.ndim == 4:

            target = target.unsqueeze(0)

        elif target.ndim == 5:

            pass

        else:

            raise ValueError(
                "Unexpected target shape: "
                f"{tuple(target.shape)}"
            )

        # -------------------------------------------------
        # Shape validation
        # -------------------------------------------------

        if (
            reconstruction.shape
            != target.shape
        ):

            raise ValueError(
                "Reconstruction and target shapes "
                "do not match.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target.shape)}"
            )

        if (
            aleatoric_std.shape
            != reconstruction.shape
        ):

            raise ValueError(
                "Aleatoric uncertainty shape does not "
                "match reconstruction shape."
            )

        if (
            epistemic_std.shape
            != reconstruction.shape
        ):

            raise ValueError(
                "Epistemic uncertainty shape does not "
                "match reconstruction shape."
            )

        # -------------------------------------------------
        # Reconstruction error
        # -------------------------------------------------

        error = torch_abs(
            reconstruction
            -
            target.to(
                reconstruction.device
            )
        )

        # -------------------------------------------------
        # Predictive uncertainty
        # -------------------------------------------------
        #
        # Predictive variance:
        #
        #     sigma_pred^2
        #         =
        #     sigma_alea^2
        #         +
        #     sigma_epi^2
        #
        # Therefore:
        #
        #     sigma_pred
        #         =
        #     sqrt(
        #         sigma_alea^2
        #         +
        #         sigma_epi^2
        #     )

        predictive_std = (
            aleatoric_std.pow(2)
            +
            epistemic_std.pow(2)
        ).sqrt()

        # -------------------------------------------------
        # Convert to NumPy
        # -------------------------------------------------

        error = (
            error
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        predictive_std = (
            predictive_std
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        error = validate_finite_array(
            error,
            "Reconstruction error"
        )

        predictive_std = validate_finite_array(
            predictive_std,
            "Predictive uncertainty"
        )

        # -------------------------------------------------
        # Flatten
        # -------------------------------------------------

        all_errors.extend(
            error.reshape(-1)
        )

        all_uncertainties.extend(
            predictive_std.reshape(-1)
        )

    # -----------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------

    all_errors = np.asarray(
        all_errors,
        dtype=np.float64
    )

    all_uncertainties = np.asarray(
        all_uncertainties,
        dtype=np.float64
    )

    if len(all_errors) != len(
        all_uncertainties
    ):

        raise RuntimeError(
            "Error and uncertainty arrays "
            "have different lengths."
        )

    if len(all_errors) == 0:

        raise RuntimeError(
            "No calibration data were collected."
        )

    return (
        all_errors,
        all_uncertainties
    )


# =========================================================
# PyTorch-independent absolute value
# =========================================================

def torch_abs(
    tensor
):
    """
    Compute absolute value using PyTorch tensor operations.

    Imported locally so this utility remains isolated.
    """

    import torch

    return torch.abs(
        tensor
    )


# =========================================================
# Min-Max Normalization
# =========================================================

def normalize_to_unit_interval(
    values
):
    """
    Normalize values to [0, 1].

    This is used here to create a relative calibration
    diagnostic between uncertainty magnitude and error
    magnitude.

    IMPORTANT:
    This is not a probability calibration metric.
    """

    values = validate_finite_array(
        values,
        "Values to normalize"
    )

    minimum = values.min()
    maximum = values.max()

    if (
        maximum
        -
        minimum
        <
        1e-12
    ):

        return np.zeros_like(
            values
        )

    return (
        values
        -
        minimum
    ) / (
        maximum
        -
        minimum
    )


# =========================================================
# Calibration Binning
# =========================================================

def calculate_calibration(
    normalized_uncertainty,
    normalized_error,
    num_bins=10
):
    """
    Calculate bin-wise calibration statistics.

    Returns
    -------
    calibration : pandas.DataFrame
    ece : float
    """

    bins = np.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    records = []

    total_samples = len(
        normalized_uncertainty
    )

    for i in range(
        num_bins
    ):

        lower = bins[i]
        upper = bins[i + 1]

        if i == num_bins - 1:

            indices = (
                (normalized_uncertainty >= lower)
                &
                (normalized_uncertainty <= upper)
            )

        else:

            indices = (
                (normalized_uncertainty >= lower)
                &
                (normalized_uncertainty < upper)
            )

        count = int(
            indices.sum()
        )

        if count == 0:

            continue

        mean_uncertainty = float(
            normalized_uncertainty[
                indices
            ].mean()
        )

        mean_error = float(
            normalized_error[
                indices
            ].mean()
        )

        absolute_gap = abs(
            mean_uncertainty
            -
            mean_error
        )

        records.append({

            "Bin_Lower":
                lower,

            "Bin_Upper":
                upper,

            "Sample_Count":
                count,

            "Mean_Uncertainty":
                mean_uncertainty,

            "Mean_Error":
                mean_error,

            "Absolute_Gap":
                absolute_gap
        })

    calibration = pd.DataFrame(
        records
    )

    if calibration.empty:

        raise RuntimeError(
            "No populated calibration bins."
        )

    # -----------------------------------------------------
    # Proper sample-weighted ECE
    # -----------------------------------------------------

    ece = (
        (
            calibration["Sample_Count"]
            /
            total_samples
        )
        *
        calibration["Absolute_Gap"]
    ).sum()

    return (
        calibration,
        float(ece)
    )


# =========================================================
# Pearson Correlation
# =========================================================

def calculate_correlation(
    uncertainty,
    error
):
    """
    Calculate Pearson correlation between uncertainty
    and reconstruction error.
    """

    if (
        np.std(uncertainty) < 1e-12
        or
        np.std(error) < 1e-12
    ):

        return np.nan

    return float(
        np.corrcoef(
            uncertainty,
            error
        )[0, 1]
    )


# =========================================================
# Reliability Diagram
# =========================================================

def plot_calibration(
    calibration,
    ece,
    correlation
):
    """
    Generate and save the uncertainty calibration plot.
    """

    os.makedirs(
        os.path.dirname(
            FIGURE_FILE
        ),
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        calibration[
            "Mean_Uncertainty"
        ],
        calibration[
            "Mean_Error"
        ],
        marker="o",
        linewidth=2,
        label="Calibration Curve"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=2,
        label="Perfect Calibration"
    )

    plt.xlabel(
        "Normalized Predictive Uncertainty"
    )

    plt.ylabel(
        "Normalized Reconstruction Error"
    )

    plt.title(
        "Predictive Uncertainty Calibration\n"
        f"ECE = {ece:.4f}, "
        f"Pearson r = {correlation:.4f}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# Main
# =========================================================

def main():

    print()
    print(
        "=" * 60
    )
    print(
        "UNCERTAINTY CALIBRATION"
    )
    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # Display configuration
    # -----------------------------------------------------

    print()
    print(
        f"Data mode : {DATASET_MODE}"
    )

    print(
        f"Device    : {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    # -----------------------------------------------------
    # Validate checkpoint
    # -----------------------------------------------------

    if not os.path.exists(
        CHECKPOINT
    ):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # -----------------------------------------------------
    # Build dataset according to DATASET_MODE
    # -----------------------------------------------------

    dataset = build_dataset()

    describe_dataset(
        dataset
    )

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Collect data
    # -----------------------------------------------------

    (
        all_errors,
        all_uncertainties
    ) = collect_calibration_data(
        dataset,
        predictor
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    normalized_errors = (
        normalize_to_unit_interval(
            all_errors
        )
    )

    normalized_uncertainties = (
        normalize_to_unit_interval(
            all_uncertainties
        )
    )

    # -----------------------------------------------------
    # Calibration
    # -----------------------------------------------------

    (
        calibration,
        ece
    ) = calculate_calibration(
        normalized_uncertainties,
        normalized_errors,
        num_bins=NUM_BINS
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    correlation = calculate_correlation(
        all_uncertainties,
        all_errors
    )

    # -----------------------------------------------------
    # Save calibration table
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    calibration.to_csv(
        CSV_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plot_calibration(
        calibration,
        ece,
        correlation
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print(
        "-" * 60
    )

    print(
        "CALIBRATION RESULTS"
    )

    print(
        "-" * 60
    )

    print(
        f"Data mode                  : "
        f"{DATASET_MODE}"
    )

    print(
        f"Number of voxels           : "
        f"{len(all_errors):,}"
    )

    print(
        f"Mean absolute error        : "
        f"{all_errors.mean():.6f}"
    )

    print(
        f"Mean predictive uncertainty: "
        f"{all_uncertainties.mean():.6f}"
    )

    print(
        f"Pearson correlation        : "
        f"{correlation:.6f}"
    )

    print(
        f"Expected Calibration Error : "
        f"{ece:.6f}"
    )

    print()
    print(
        calibration
    )

    print()
    print(
        "Saved:"
    )

    print(
        CSV_FILE
    )

    print(
        FIGURE_FILE
    )

    print()
    print(
        "=" * 60
    )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()