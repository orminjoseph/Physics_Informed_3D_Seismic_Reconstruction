"""
=========================================================
Uncertainty Analysis
=========================================================

Computes:

1. Mean uncertainty
2. Standard deviation uncertainty
3. Maximum uncertainty
4. Minimum uncertainty

Generates:

- uncertainty_statistics.csv
- uncertainty_histogram.png

=========================================================
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from models.network import Network3D
from inference.predictor import Predictor
from dataset.build_dataset import build_dataset
from utils.config import DATASET_MODE


def analyze_uncertainty():

    print()
    print("=" * 60)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = build_dataset()

    checkpoint = os.path.join(
        "outputs",
        DATASET_MODE,
        "checkpoints",
        "best_model.pth"
    )

    model = Network3D()

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    all_uncertainties = []

    for index in range(len(dataset)):

        input_cube, target_cube, mask, velocity = dataset[index]

        reconstruction, uncertainty = predictor.predict(
            input_cube
        )

        all_uncertainties.extend(
            uncertainty.flatten().numpy()
        )

    all_uncertainties = pd.Series(
        all_uncertainties
    )

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

    report_dir = os.path.join(
        "outputs",
        DATASET_MODE,
        "reports"
    )

    os.makedirs(
        report_dir,
        exist_ok=True
    )

    pd.DataFrame(
        [statistics]
    ).to_csv(

        os.path.join(
            report_dir,
            "uncertainty_statistics.csv"
        ),

        index=False

    )

    plt.figure(figsize=(8,5))

    plt.hist(
        all_uncertainties,
        bins=50
    )

    plt.xlabel("Uncertainty")

    plt.ylabel("Frequency")

    plt.title(
        "Predictive Uncertainty Distribution"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            report_dir,
            "uncertainty_histogram.png"
        )

    )

    plt.close()

    print()
    print(statistics)

    return statistics


if __name__ == "__main__":
    analyze_uncertainty()