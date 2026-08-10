"""
Uncertainty Evaluation

Evaluates whether uncertainty correlates
with reconstruction quality.

Author: Ormin Joseph
"""

import csv
import os
import numpy as np
import torch

from dataset.f3_dataset import F3Dataset

from inference.predictor import Predictor

from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = (
    r"outputs"
    r"\current_experiment"
    r"\checkpoints"
    r"\best_model.pth"
)

NUM_TEST_PATCHES = 20
def evaluate_metrics(
    prediction,
    target
):

    return {

        "MAE": mae(
            prediction,
            target
        ).item(),

        "RMSE": rmse(
            prediction,
            target
        ).item(),

        "PSNR": psnr(
            prediction,
            target
        ).item(),

        "SNR": snr(
            prediction,
            target
        ).item(),

        "SSIM": ssim(
            prediction,
            target
        ).item()

    }
def main():

    print()
    print("=" * 60)
    print("UNCERTAINTY EVALUATION")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    results = []
    for patch_index in range(

            min(
                NUM_TEST_PATCHES,
                len(dataset)
            )
    ):
        print(
            f"Patch "
            f"{patch_index + 1}/"
            f"{NUM_TEST_PATCHES}"
        )

        corrupted, target, mask, velocity = dataset[
            patch_index
        ]

        target_batch = target.unsqueeze(0)

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        metrics = evaluate_metrics(
            reconstruction,
            target_batch
        )

        mean_uncertainty = (
            uncertainty.mean()
            .item()
        )

        max_uncertainty = (
            uncertainty.max()
            .item()
        )

        print(
            f"Mean Uncertainty = "
            f"{mean_uncertainty:.6f}"
        )

        row = {

            "Patch": patch_index,

            "Mean_Uncertainty":
                mean_uncertainty,

            "Max_Uncertainty":
                max_uncertainty,
            "Error_x_Uncertainty":
                mean_uncertainty * metrics["MAE"],

            **metrics
        }

        results.append(
            row
        )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "uncertainty_evaluation.csv"
    )

    with open(
            csv_file,
            "w",
            newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print("Saved:")
    print(csv_file)

    mean_unc = np.mean(
        [r["Mean_Uncertainty"]
         for r in results]
    )

    max_unc = np.max(
        [r["Mean_Uncertainty"]
         for r in results]
    )

    mean_mae = np.mean(
        [r["MAE"]
         for r in results]
    )

    mean_ssim = np.mean(
        [r["SSIM"]
         for r in results]
    )

    print(
        f"Total patches processed: "
        f"{len(results)}"
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Mean Uncertainty : "
        f"{mean_unc:.6f}"
    )

    print(
        f"Maximum Uncertainty : "
        f"{max_unc:.6f}"
    )

    print(
        f"Mean MAE : "
        f"{mean_mae:.6f}"
    )

    print(
        f"Mean SSIM : "
        f"{mean_ssim:.6f}"
    )

    print()

    correlation = np.corrcoef(
        [r["Mean_Uncertainty"] for r in results],
        [r["MAE"] for r in results]
    )[0, 1]

    print(
        f"Correlation (Uncertainty vs MAE): "
        f"{correlation:.4f}"
    )

if __name__ == "__main__":
    main()