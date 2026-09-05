import csv
import os
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

MISSING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]
def main():

    print("=" * 60)
    print("UNCERTAINTY VS MISSING DATA TEST")
    print("=" * 60)

    results = []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=device
    )

    print()

    print(
        "{:<15} {:<15} {:<15}".format(
            "Missing %",
            "Mean Uncertainty",
            "Std Uncertainty"
        )
    )

    print("-" * 50)

    for missing_level in MISSING_LEVELS:
        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=(64, 64, 64),
            stride=(64, 64, 64),
            missing_probability=missing_level
        )

        corrupted, target, mask, velocity = dataset[0][:4]

        reconstruction, uncertainty = predictor.predict(
            corrupted
        )

        mean_uncertainty = (
            uncertainty.mean().item()
        )

        std_uncertainty = (
            uncertainty.std().item()
        )
        print(
            "{:<15} {:<15.4f} {:<15.4f}".format(
                f"{int(missing_level * 100)}%",
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

        with open(
                "outputs/reports/uncertainty_missing_data.csv",
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

        labels = [f"{r[0]}%" for r in results]

        means = [r[1] for r in results]

        plt.figure(figsize=(8, 5))

        plt.plot(
            labels,
            means,
            marker="o"
        )

        plt.title(
            "Mean Uncertainty vs Missing Data"
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

        plt.savefig(
            "outputs/figures/uncertainty_missing_data.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print()
    print(
        "CSV saved to:"
    )
    print(
        "outputs/reports/uncertainty_missing_data.csv"
    )

    print()
    print(
        "Plot saved to:"
    )
    print(
        "outputs/figures/uncertainty_missing_data.png"
    )

if __name__ == "__main__":
    main()