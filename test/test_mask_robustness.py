"""
============================================================
MASK ROBUSTNESS TEST
============================================================
"""

import torch
import csv
import os
import matplotlib.pyplot as plt
from dataset.f3_dataset import F3Dataset
from dataset.mask_generator import MaskGenerator

from inference.predictor import Predictor
from models.network import Network3D

from evaluation.metrics import (
    EvaluationMetrics
)


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"


MASK_TYPES = [

    "random_trace",

    "regular_trace",

    "inline_strip",

    "crossline_strip",

    "checkerboard"

]


def main():

    print("=" * 60)
    print("MASK ROBUSTNESS TEST")
    print("=" * 60)

    results = []

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    target = target.unsqueeze(0)

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
        "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "Mask",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        )
    )

    print("-" * 60)

    for mask_type in MASK_TYPES:

        generator = MaskGenerator(
            mask_type=mask_type,
            missing_probability=0.30
        )

        generated_mask = torch.tensor(
            generator.generate(
                target.squeeze().shape
            ),
            dtype=torch.float32
        )

        generated_mask = generated_mask.unsqueeze(0)

        corrupted_cube = (
            target * generated_mask
        )

        reconstruction, uncertainty = predictor.predict(
            corrupted_cube
        )

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
            "{:<20} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.4f}".format(
                mask_type,
                mae,
                rmse,
                psnr,
                snr,
                ssim
            )
        )
        results.append([
            mask_type,
            mae,
            rmse,
            psnr,
            snr,
            ssim
        ])

    os.makedirs("outputs/reports", exist_ok=True)

    with open(
            "outputs/reports/mask_robustness.csv",
            "w",
            newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Mask_Type",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ])

        writer.writerows(results)
    print()
    print("CSV saved to:")
    print("outputs/reports/mask_robustness.csv")

    mask_labels = [r[0] for r in results]

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

    axes[0, 0].plot(mask_labels, mae_values, marker="o")
    axes[0, 0].set_title("MAE vs Mask Type")
    axes[0, 0].grid(True)

    axes[0, 1].plot(mask_labels, rmse_values, marker="s")
    axes[0, 1].set_title("RMSE vs Mask Type")
    axes[0, 1].grid(True)

    axes[1, 0].plot(mask_labels, psnr_values, marker="^")
    axes[1, 0].set_title("PSNR vs Mask Type")
    axes[1, 0].grid(True)

    axes[1, 1].plot(mask_labels, ssim_values, marker="d")
    axes[1, 1].set_title("SSIM vs Mask Type")
    axes[1, 1].grid(True)

    axes[2, 0].plot(mask_labels, snr_values, marker="o")
    axes[2, 0].set_title("SNR vs Mask Type")
    axes[2, 0].grid(True)

    axes[2, 1].axis("off")

    for ax in axes.flat:
        ax.set_xlabel("Mask Type")

    plt.tight_layout()

    plot_file = (
        "outputs/figures/"
        "mask_robustness.png"
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