"""
============================================================
MISSING DATA SENSITIVITY TEST
============================================================
"""

import csv
import os
import torch
import matplotlib.pyplot as plt
from dataset.f3_dataset import F3Dataset

from inference.predictor import Predictor
from models.network import Network3D

from evaluation.metrics import EvaluationMetrics


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
    print("MISSING DATA SENSITIVITY TEST")
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
        "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "Missing %",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        )
    )

    print("-" * 70)

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

        target = target.unsqueeze(0)

        mae = EvaluationMetrics.mae(
            reconstruction,
            target
        ).item()

        rmse = EvaluationMetrics.rmse(
            reconstruction,
            target
        ).item()

        psnr = EvaluationMetrics.psnr(
            reconstruction,
            target
        ).item()

        snr = EvaluationMetrics.snr(
            reconstruction,
            target
        ).item()

        ssim = EvaluationMetrics.ssim(
            reconstruction,
            target
        ).item()

        print(
            "{:<15} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.4f}".format(
                f"{int(missing_level * 100)}%",
                mae,
                rmse,
                psnr,
                snr,
                ssim
            )
        )
        results.append([
            int(missing_level * 100),
            mae,
            rmse,
            psnr,
            snr,
            ssim
        ])

    os.makedirs("outputs/reports", exist_ok=True)

    with open(
            "outputs/reports/missing_data_sensitivity.csv",
            "w",
            newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Missing_Percentage",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ])

        writer.writerows(results)

    print()
    print("CSV saved to:")
    print("outputs/reports/missing_data_sensitivity.csv")

    missing_labels = [f"{r[0]}%" for r in results]

    mae_values = [r[1] for r in results]

    rmse_values = [r[2] for r in results]

    psnr_values = [r[3] for r in results]

    snr_values = [r[4] for r in results]

    ssim_values = [r[5] for r in results]

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 12)
    )

    axes[0, 0].plot(
        missing_labels,
        mae_values,
        marker="o"
    )
    axes[0, 0].set_title("MAE vs Missing Data")
    axes[0, 0].set_ylabel("MAE")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        missing_labels,
        rmse_values,
        marker="s"
    )
    axes[0, 1].set_title("RMSE vs Missing Data")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].grid(True)

    axes[1, 0].plot(
        missing_labels,
        psnr_values,
        marker="^"
    )
    axes[1, 0].set_title("PSNR vs Missing Data")
    axes[1, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        missing_labels,
        ssim_values,
        marker="d"
    )

    axes[2, 0].plot(
        missing_labels,
        snr_values,
        marker="o"
    )

    axes[2, 0].set_title("SNR vs Missing Data")

    axes[2, 0].set_ylabel("SNR (dB)")

    axes[2, 0].grid(True)
    axes[2, 1].axis("off")

    axes[1, 1].set_title("SSIM vs Missing Data")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].grid(True)

    for ax in axes.flat:
        if ax != axes[2, 1]:
            ax.set_xlabel("Missing Data (%)")

    plt.tight_layout()

    plot_file = (
        "outputs/figures/"
        "missing_data_sensitivity.png"
    )

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