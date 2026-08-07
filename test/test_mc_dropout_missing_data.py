"""
============================================================
MC DROPOUT VS MISSING DATA TEST
============================================================

Measures whether epistemic uncertainty
increases as missing data increases.

Author: Ormin Joseph
============================================================
"""

import csv
import os
import torch
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from inference.mc_dropout_predictor import MCDropoutPredictor


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"

MISSING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]


def main():

    print("=" * 60)
    print("MC DROPOUT VS MISSING DATA TEST")
    print("=" * 60)

    results = []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = MCDropoutPredictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=device,
        num_samples=20
    )

    print()

    print(
        "{:<15} {:<20} {:<20}".format(
            "Missing %",
            "Mean Uncertainty",
            "Std Uncertainty"
        )
    )

    print("-" * 60)

    for missing_level in MISSING_LEVELS:

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=(64, 64, 64),
            stride=(64, 64, 64),
            missing_probability=missing_level
        )

        corrupted, target, mask, velocity = dataset[0]

        mean_prediction, uncertainty = predictor.predict(
            corrupted
        )

        mean_uncertainty = (
            uncertainty.mean()
            .item()
        )

        std_uncertainty = (
            uncertainty.std()
            .item()
        )

        print(
            "{:<15} {:<20.4f} {:<20.4f}".format(
                f"{int(missing_level*100)}%",
                mean_uncertainty,
                std_uncertainty
            )
        )

        results.append([
            int(missing_level * 100),
            mean_uncertainty,
            std_uncertainty
        ])

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "mc_dropout_missing_data.csv"
    )

    with open(
            csv_file,
            "w",
            newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Missing_Percentage",
            "Mean_Uncertainty",
            "Std_Uncertainty"
        ])

        writer.writerows(results)

    labels = [
        f"{r[0]}%"
        for r in results
    ]

    means = [
        r[1]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        labels,
        means,
        marker="o"
    )

    plt.title(
        "MC Dropout Uncertainty vs Missing Data"
    )

    plt.xlabel(
        "Missing Data (%)"
    )

    plt.ylabel(
        "Mean Uncertainty"
    )

    plt.grid(True)

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plot_file = (
        "outputs/figures/"
        "mc_dropout_missing_data.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("CSV saved to:")
    print(csv_file)

    print()
    print("Plot saved to:")
    print(plot_file)


if __name__ == "__main__":
    main()