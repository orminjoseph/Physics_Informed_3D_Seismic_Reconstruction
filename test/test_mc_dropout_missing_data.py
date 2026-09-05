"""
============================================================
MC DROPOUT VS MISSING DATA TEST
============================================================

Measures whether predictive uncertainty increases as
missing seismic data increases.

Uncertainty components:
    1. Aleatoric uncertainty:
       Mean of exp(log_variance) across MC samples.

    2. Epistemic uncertainty:
       Population variance of MC reconstruction samples.

    3. Predictive uncertainty:
       Aleatoric variance + Epistemic variance.

The test uses the authoritative MCDropout3D implementation
from models.mc_dropout.py.

Author: Ormin Joseph
============================================================
"""

import csv
import os

import matplotlib.pyplot as plt
import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)


# ============================================================
# CONFIGURATION
# ============================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"

PATCH_SIZE = (64, 64, 64)
STRIDE = (64, 64, 64)

NUM_MC_SAMPLES = 20

MISSING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]

OUTPUT_REPORT_DIR = "outputs/reports"
OUTPUT_FIGURE_DIR = "outputs/figures"

CSV_FILE = os.path.join(
    OUTPUT_REPORT_DIR,
    "mc_dropout_missing_data.csv"
)

PLOT_FILE = os.path.join(
    OUTPUT_FIGURE_DIR,
    "mc_dropout_missing_data.png"
)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load the trained model checkpoint.

    Supports checkpoints containing:
        model_state_dict

    or a direct model state dictionary.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("MC DROPOUT VS MISSING DATA TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Create the network
    # --------------------------------------------------------

    model = Network3D()

    model = load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT,
        device=device
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Create the authoritative MC Dropout engine
    # --------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=NUM_MC_SAMPLES
    )

    # --------------------------------------------------------
    # Create the predictive uncertainty estimator
    # --------------------------------------------------------

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
    )

    results = []

    print()

    print(
        "{:<15} {:<20} {:<20} {:<20}".format(
            "Missing %",
            "Aleatoric Var.",
            "Epistemic Var.",
            "Predictive Var."
        )
    )

    print("-" * 80)

    # ========================================================
    # TEST EACH MISSING-DATA LEVEL
    # ========================================================

    for missing_level in MISSING_LEVELS:

        print()
        print(
            f"Testing missing level: "
            f"{missing_level * 100:.0f}%"
        )

        # ----------------------------------------------------
        # Create dataset with the current missing probability
        # ----------------------------------------------------

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            missing_probability=missing_level
        )

        # ----------------------------------------------------
        # Obtain one corrupted seismic patch
        # ----------------------------------------------------

        sample = dataset[0]

        corrupted = sample[0]

        # ----------------------------------------------------
        # Ensure batch dimension exists
        # ----------------------------------------------------

        if corrupted.dim() == 4:
            corrupted = corrupted.unsqueeze(0)

        corrupted = corrupted.to(device)

        # ----------------------------------------------------
        # Run MC Dropout
        #
        # Returns:
        #   reconstruction_samples
        #   reconstruction_mean
        #   reconstruction_epistemic_variance
        #   travel_time_samples
        #   travel_time_mean
        #   travel_time_epistemic_variance
        #   log_variance_samples
        #   log_variance_mean
        #   log_variance_epistemic_variance
        # ----------------------------------------------------

        mc_results = mc_dropout.predict(
            corrupted
        )

        reconstruction_samples = (
            mc_results["reconstruction_samples"]
        )

        reconstruction_mean = (
            mc_results["reconstruction_mean"]
        )

        reconstruction_epistemic_variance = (
            mc_results[
                "reconstruction_epistemic_variance"
            ]
        )

        log_variance_samples = (
            mc_results["log_variance_samples"]
        )

        # ----------------------------------------------------
        # Calculate MC-integrated aleatoric variance
        #
        # CORRECT:
        #
        #   mean(exp(log_variance_samples))
        #
        # NOT:
        #
        #   exp(mean(log_variance_samples))
        # ----------------------------------------------------

        aleatoric_variance = (
            uncertainty_estimator.aleatoric_variance(
                log_variance_samples
            )
        )

        # ----------------------------------------------------
        # Calculate predictive variance
        #
        #   Predictive variance =
        #       Aleatoric variance
        #       +
        #       Epistemic variance
        # ----------------------------------------------------

        predictive_variance = (
            uncertainty_estimator.predictive_variance(
                aleatoric_variance,
                reconstruction_epistemic_variance
            )
        )

        # ----------------------------------------------------
        # Calculate mean uncertainty values
        # ----------------------------------------------------

        mean_aleatoric_variance = (
            aleatoric_variance.mean().item()
        )

        mean_epistemic_variance = (
            reconstruction_epistemic_variance.mean().item()
        )

        mean_predictive_variance = (
            predictive_variance.mean().item()
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            "{:<15} {:<20.6f} {:<20.6f} {:<20.6f}".format(
                f"{int(missing_level * 100)}%",
                mean_aleatoric_variance,
                mean_epistemic_variance,
                mean_predictive_variance
            )
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append([
            int(missing_level * 100),
            mean_aleatoric_variance,
            mean_epistemic_variance,
            mean_predictive_variance
        ])

    # ========================================================
    # SAVE CSV RESULTS
    # ========================================================

    os.makedirs(
        OUTPUT_REPORT_DIR,
        exist_ok=True
    )

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Missing_Percentage",
            "Mean_Aleatoric_Variance",
            "Mean_Epistemic_Variance",
            "Mean_Predictive_Variance"
        ])

        writer.writerows(results)

    # ========================================================
    # PREPARE PLOT DATA
    # ========================================================

    labels = [
        f"{result[0]}%"
        for result in results
    ]

    aleatoric_values = [
        result[1]
        for result in results
    ]

    epistemic_values = [
        result[2]
        for result in results
    ]

    predictive_values = [
        result[3]
        for result in results
    ]

    # ========================================================
    # CREATE UNCERTAINTY PLOT
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        labels,
        aleatoric_values,
        marker="o",
        label="Aleatoric Variance"
    )

    plt.plot(
        labels,
        epistemic_values,
        marker="s",
        label="Epistemic Variance"
    )

    plt.plot(
        labels,
        predictive_values,
        marker="^",
        label="Predictive Variance"
    )

    plt.title(
        "MC Dropout Uncertainty vs Missing Data"
    )

    plt.xlabel(
        "Missing Data (%)"
    )

    plt.ylabel(
        "Mean Variance"
    )

    plt.legend()

    plt.grid(True)

    # ========================================================
    # SAVE PLOT
    # ========================================================

    os.makedirs(
        OUTPUT_FIGURE_DIR,
        exist_ok=True
    )

    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("CSV saved to:")
    print(CSV_FILE)

    print()
    print("Plot saved to:")
    print(PLOT_FILE)

    print()
    print("=" * 60)
    print("MC DROPOUT VS MISSING DATA TEST COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()