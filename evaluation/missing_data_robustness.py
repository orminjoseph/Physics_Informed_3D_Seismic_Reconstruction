import os
import csv
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
    print("MISSING DATA ROBUSTNESS")
    print("=" * 60)

    missing_levels = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50
    ]

    NUM_TEST_PATCHES = 20

    results = []
    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )
    for missing_probability in missing_levels:
        print()
        print(
            f"Testing "
            f"{int(missing_probability * 100)}%"
            f" missing traces"
        )

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=(64, 64, 64),
            stride=(64, 64, 64),
            missing_probability=missing_probability
        )
        mae_values = []
        rmse_values = []
        psnr_values = []
        snr_values = []
        ssim_values = []

        for patch_index in range(

                min(
                    NUM_TEST_PATCHES,
                    len(dataset)
                )
        ):
            corrupted, target, mask, velocity = (
                dataset[patch_index]
            )

            target_batch = (
                target.unsqueeze(0)
            )
            reconstruction, uncertainty = (
                predictor.predict(
                    corrupted
                )
            )
            metrics = evaluate_metrics(
                reconstruction,
                target_batch
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

            "Missing_Probability":
                missing_probability,

            "MAE":
                np.mean(mae_values),

            "RMSE":
                np.mean(rmse_values),

            "PSNR":
                np.mean(psnr_values),

            "SNR":
                np.mean(snr_values),

            "SSIM":
                np.mean(ssim_values)

        })

        print(
            f"MAE={np.mean(mae_values):.4f}, "
            f"RMSE={np.mean(rmse_values):.4f}, "
            f"SSIM={np.mean(ssim_values):.4f}"
        )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "missing_data_robustness.csv"
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        csv_file,
        index=False
    )

    print()
    print("Saved:")
    print(csv_file)

    print()
    print(df)

if __name__ == "__main__":
    main()