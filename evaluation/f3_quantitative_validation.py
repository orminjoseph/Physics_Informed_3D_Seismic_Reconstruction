import os
import numpy as np
import pandas as pd

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
    print("F3 QUANTITATIVE VALIDATION")
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

    mae_values = []
    rmse_values = []
    psnr_values = []
    snr_values = []
    ssim_values = []

    NUM_PATCHES = 20

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

        print(
            f"Patch {patch_index}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"SSIM={metrics['SSIM']:.4f}"
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

    # -----------------------------
    # AFTER LOOP
    # -----------------------------

    results = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ],

        "Mean_Value": [

            np.mean(mae_values),

            np.mean(rmse_values),

            np.mean(psnr_values),

            np.mean(snr_values),

            np.mean(ssim_values)
        ]
    })

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "f3_quantitative_validation.csv"
    )

    results.to_csv(
        csv_file,
        index=False
    )

    print()
    print(results)

    print()
    print("Saved:")
    print(csv_file)


if __name__ == "__main__":
    main()