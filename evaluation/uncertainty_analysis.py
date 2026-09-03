"""
=========================================================
Uncertainty Analysis
=========================================================

Computes:

1. Mean uncertainty
2. Standard deviation uncertainty
3. Minimum uncertainty
4. Maximum uncertainty

Generates:

- uncertainty_statistics.csv
- uncertainty_histogram.png

The output is stored inside the current experiment directory.

Example:

    outputs/

        f3_training/
            reports/
                uncertainty_statistics.csv
                uncertainty_histogram.png

=========================================================
"""

import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

from models.network import Network3D

from inference.predictor import Predictor

from dataset.build_dataset import build_dataset

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR
)


def analyze_uncertainty():

    print()
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 60)

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME
    )

    print(
        "Device     :",
        device
    )

    # =====================================================
    # BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print(
        "Dataset Length:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Uncertainty-analysis dataset is empty."
        )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print(
        "Checkpoint:",
        checkpoint
    )

    if not os.path.exists(checkpoint):

        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete training and make sure "
            "best_model.pth exists."
        )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model = Network3D()

    # =====================================================
    # PREDICTOR
    # =====================================================

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    # =====================================================
    # COLLECT UNCERTAINTY VALUES
    # =====================================================

    all_uncertainties = []

    for index in range(
        len(dataset)
    ):

        print(
            f"Analyzing sample "
            f"{index + 1}/{len(dataset)}"
        )

        # -------------------------------------------------
        # CURRENT DATASET OUTPUT
        # -------------------------------------------------

        (
            input_cube,
            target_cube,
            mask,
            velocity_model
        ) = dataset[index]

        # -------------------------------------------------
        # CURRENT PREDICTOR OUTPUT
        #
        # Predictor returns:
        #
        # reconstruction
        # travel_time
        # uncertainty
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            uncertainty
        ) = predictor.predict(
            input_cube
        )

        # -------------------------------------------------
        # STORE UNCERTAINTY VALUES
        # -------------------------------------------------

        all_uncertainties.extend(
            uncertainty
            .flatten()
            .numpy()
        )

    # =====================================================
    # CONVERT TO PANDAS SERIES
    # =====================================================

    all_uncertainties = pd.Series(
        all_uncertainties
    )

    # =====================================================
    # COMPUTE STATISTICS
    # =====================================================

    statistics = {

        "Mean_Uncertainty":
            all_uncertainties.mean(),

        "Std_Uncertainty":
            all_uncertainties.std(),

        "Min_Uncertainty":
            all_uncertainties.min(),

        "Max_Uncertainty":
            all_uncertainties.max()

    }

    # =====================================================
    # REPORT DIRECTORY
    # =====================================================

    report_dir = REPORT_DIR

    os.makedirs(
        report_dir,
        exist_ok=True
    )

    # =====================================================
    # SAVE STATISTICS
    # =====================================================

    statistics_file = os.path.join(
        report_dir,
        "uncertainty_statistics.csv"
    )

    pd.DataFrame(
        [statistics]
    ).to_csv(
        statistics_file,
        index=False
    )

    # =====================================================
    # UNCERTAINTY HISTOGRAM
    # =====================================================

    histogram_file = os.path.join(
        report_dir,
        "uncertainty_histogram.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        all_uncertainties,
        bins=50
    )

    plt.xlabel(
        "Uncertainty"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Predictive Uncertainty Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        histogram_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS COMPLETE")
    print("=" * 60)

    print()

    print(
        "Mean Uncertainty :",
        statistics["Mean_Uncertainty"]
    )

    print(
        "Std Uncertainty  :",
        statistics["Std_Uncertainty"]
    )

    print(
        "Min Uncertainty  :",
        statistics["Min_Uncertainty"]
    )

    print(
        "Max Uncertainty  :",
        statistics["Max_Uncertainty"]
    )

    print()
    print(
        "Statistics saved:",
        statistics_file
    )

    print(
        "Histogram saved:",
        histogram_file
    )

    return statistics


if __name__ == "__main__":

    analyze_uncertainty()