
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def evaluate_metrics(
        prediction,
        target
):

    return {

        "MAE":
            mae(
                prediction,
                target
            ).item(),

        "RMSE":
            rmse(
                prediction,
                target
            ).item(),

        "PSNR":
            psnr(
                prediction,
                target
            ).item(),

        "SNR":
            snr(
                prediction,
                target
            ).item(),

        "SSIM":
            ssim(
                prediction,
                target
            ).item()
    }


def main():

    print()
    print("=" * 60)
    print("F3 MASK ROBUSTNESS")
    print("=" * 60)

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    mask_levels = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70
    ]

    results = []

    NUM_PATCHES = 20

    for missing_probability in mask_levels:

        print()
        print(
            f"Testing Missing Data = "
            f"{missing_probability:.0%}"
        )

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=(64,64,64),
            stride=(64,64,64),
            missing_probability=missing_probability
        )

        mae_values = []
        rmse_values = []
        psnr_values = []
        snr_values = []
        ssim_values = []

        for patch_index in range(
            min(
                NUM_PATCHES,
                len(dataset)
            )
        ):

            corrupted, target, mask, velocity = (
                dataset[patch_index]
            )

            reconstruction, uncertainty = (
                predictor.predict(
                    corrupted
                )
            )

            metrics = evaluate_metrics(
                reconstruction,
                target.unsqueeze(0)
            )

            mae_values.append(
                metrics["MAE"]
            )

            rmse_values.append(
                metrics["RMSE"]
            )

            psnr_values.append(
                metrics["PSNR"]
            )

            snr_values.append(
                metrics["SNR"]
            )

            ssim_values.append(
                metrics["SSIM"]
            )

        results.append({

            "Missing_Percentage":
                int(
                    missing_probability * 100
                ),

            "MAE":
                np.mean(
                    mae_values
                ),

            "RMSE":
                np.mean(
                    rmse_values
                ),

            "PSNR":
                np.mean(
                    psnr_values
                ),

            "SNR":
                np.mean(
                    snr_values
                ),

            "SSIM":
                np.mean(
                    ssim_values
                )
        })

    df = pd.DataFrame(
        results
    )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "f3_mask_robustness.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    print()
    print(df)

    plt.figure(
        figsize=(10,6)
    )

    plt.plot(
        df["Missing_Percentage"],
        df["MAE"],
        marker="o",
        label="MAE"
    )

    plt.plot(
        df["Missing_Percentage"],
        df["RMSE"],
        marker="o",
        label="RMSE"
    )

    plt.plot(
        df["Missing_Percentage"],
        df["SSIM"],
        marker="o",
        label="SSIM"
    )

    plt.xlabel(
        "Missing Data (%)"
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.title(
        "F3 Mask Robustness"
    )

    plt.legend()

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    figure_file = (
        "outputs/figures/"
        "f3_mask_robustness.png"
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(csv_file)
    print(figure_file)


if __name__ == "__main__":
    main()