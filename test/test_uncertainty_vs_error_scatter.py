"""
============================================================
UNCERTAINTY VS ERROR SCATTER
============================================================

Checks whether uncertainty increases
with reconstruction error.

Author: Ormin Joseph
============================================================
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from inference.predictor import Predictor
from models.network import Network3D


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"


def main():

    print("=" * 60)
    print("UNCERTAINTY VS ERROR SCATTER")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    reconstruction = (
        reconstruction.squeeze()
        .cpu()
        .numpy()
    )

    uncertainty = (
        uncertainty.squeeze()
        .cpu()
        .numpy()
    )

    target = target.squeeze().numpy()

    error = np.abs(
        reconstruction - target
    )

    error_flat = error.flatten()

    uncertainty_flat = uncertainty.flatten()

    correlation = np.corrcoef(
        error_flat,
        uncertainty_flat
    )[0, 1]

    print()
    print(
        f"Correlation: {correlation:.4f}"
    )

    # -----------------------------------
    # Save CSV
    # -----------------------------------

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "uncertainty_vs_error_scatter.csv"
    )

    with open(
        csv_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Error", "Uncertainty"]
        )

        for e, u in zip(
                error_flat,
                uncertainty_flat
        ):
            writer.writerow([e, u])

    print()
    print("CSV saved to:")
    print(csv_file)

    # -----------------------------------
    # Scatter Plot
    # -----------------------------------

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plot_file = (
        "outputs/figures/"
        "uncertainty_vs_error_scatter.png"
    )

    # Sample points to avoid huge plot
    n_points = min(
        10000,
        len(error_flat)
    )

    indices = np.random.choice(
        len(error_flat),
        n_points,
        replace=False
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        error_flat[indices],
        uncertainty_flat[indices],
        alpha=0.3,
        s=5
    )

    plt.xlabel(
        "Absolute Reconstruction Error"
    )

    plt.ylabel(
        "Predictive Uncertainty"
    )

    plt.title(
        f"Error vs Uncertainty\n"
        f"Correlation = {correlation:.4f}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Plot saved to:")
    print(plot_file)


if __name__ == "__main__":
    main()