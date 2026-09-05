"""
============================================================
EPOCH SENSITIVITY TEST
============================================================
"""

import torch
import csv
import os
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

CHECKPOINTS = {

    "Epoch_1":
        "checkpoints/epoch_sensitivity/epoch_0001.pth",

    "Epoch_3":
        "checkpoints/epoch_sensitivity/epoch_0003.pth",

    "Epoch_5":
        "checkpoints/epoch_sensitivity/epoch_0005.pth",

    "Best_Model":
        "checkpoints/best_model.pth"

}


def evaluate_checkpoint(
        checkpoint_path,
        device
):

    predictor = Predictor(
        model=Network3D(),
        checkpoint=checkpoint_path,
        device=device
    )

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
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

    return mae, rmse, psnr, snr, ssim


def main():

    print("=" * 60)
    print("EPOCH SENSITIVITY TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "Checkpoint",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        )
    )

    print("-" * 70)

    results = []

    for name, path in CHECKPOINTS.items():

        mae, rmse, psnr, snr, ssim = evaluate_checkpoint(
            path,
            device
        )

        print(
            "{:<15} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.4f}".format(
                name,
                mae,
                rmse,
                psnr,
                snr,
                ssim
            )
        )
        results.append([
            name,
            mae,
            rmse,
            psnr,
            snr,
            ssim
        ])
    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = "outputs/reports/epoch_sensitivity.csv"

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Checkpoint",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ])

        writer.writerows(results)

    print()
    print(f"CSV saved to: {csv_file}")

    checkpoints = [row[0] for row in results]

    mae_values = [row[1] for row in results]
    rmse_values = [row[2] for row in results]
    psnr_values = [row[3] for row in results]
    ssim_values = [row[5] for row in results]
    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        checkpoints,
        mae_values,
        marker="o",
        label="MAE"
    )

    plt.plot(
        checkpoints,
        rmse_values,
        marker="s",
        label="RMSE"
    )

    plt.plot(
        checkpoints,
        psnr_values,
        marker="^",
        label="PSNR"
    )

    plt.plot(
        checkpoints,
        ssim_values,
        marker="d",
        label="SSIM"
    )

    plt.title(
        "Epoch Sensitivity Analysis"
    )

    plt.xlabel(
        "Checkpoint"
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.grid(True)

    plt.legend()

    plot_file = (
        "outputs/figures/"
        "epoch_sensitivity.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved to: {plot_file}"
    )

if __name__ == "__main__":
    main()