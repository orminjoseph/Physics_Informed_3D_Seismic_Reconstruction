"""
============================================================
EPOCH SENSITIVITY TEST
============================================================
"""

import torch

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
        "checkpoints/checkpoint_epoch_1.pth",

    "Epoch_2":
        "checkpoints/checkpoint_epoch_2.pth",

    "Epoch_3":
        "checkpoints/checkpoint_epoch_3.pth",

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

    corrupted, target, mask, velocity = dataset[0]

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

    ssim = EvaluationMetrics.ssim(
        reconstruction,
        target
    ).item()

    return mae, rmse, psnr, ssim


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
        "{:<15} {:<10} {:<10} {:<10} {:<10}".format(
            "Checkpoint",
            "MAE",
            "RMSE",
            "PSNR",
            "SSIM"
        )
    )

    print("-" * 70)

    for name, path in CHECKPOINTS.items():

        mae, rmse, psnr, ssim = evaluate_checkpoint(
            path,
            device
        )

        print(
            "{:<15} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.4f}".format(
                name,
                mae,
                rmse,
                psnr,
                ssim
            )
        )


if __name__ == "__main__":
    main()