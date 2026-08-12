"""
=========================================================
Model Evaluation
=========================================================

Loads best model checkpoint and computes:

MAE
RMSE
PSNR
SNR
SSIM

=========================================================
"""

import os
import torch
import pandas as pd

from models.network import Network3D

from inference.predictor import Predictor

from dataset.build_dataset import build_dataset

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import DATASET_MODE


def evaluate(
        model_override=None
):
    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = build_dataset()

    if model_override is None:

        model = Network3D()

    else:

        model = model_override

    checkpoint = os.path.join(
        "outputs",
        DATASET_MODE,
        "checkpoints",
        "best_model.pth"
    )

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    total_mae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_snr = 0.0
    total_ssim = 0.0

    num_samples = len(dataset)

    for i in range(num_samples):

        input_cube, target_cube, mask, velocity = dataset[i]

        reconstruction, uncertainty = predictor.predict(
            input_cube
        )

        total_mae += mae(
            reconstruction,
            target_cube.unsqueeze(0)
        ).item()

        total_rmse += rmse(
            reconstruction,
            target_cube.unsqueeze(0)
        ).item()

        total_psnr += psnr(
            reconstruction,
            target_cube.unsqueeze(0)
        ).item()

        total_snr += snr(
            reconstruction,
            target_cube.unsqueeze(0)
        ).item()

        total_ssim += ssim(
            reconstruction,
            target_cube.unsqueeze(0)
        ).item()

    results = {

        "MAE": total_mae / num_samples,
        "RMSE": total_rmse / num_samples,
        "PSNR": total_psnr / num_samples,
        "SNR": total_snr / num_samples,
        "SSIM": total_ssim / num_samples

    }

    print()
    print(results)

    output_dir = os.path.join(
        "outputs",
        DATASET_MODE,
        "reports"
    )

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(
        [results]
    ).to_csv(
        os.path.join(
            output_dir,
            "evaluation_metrics.csv"
        ),
        index=False
    )

    return results


if __name__ == "__main__":
    evaluate()