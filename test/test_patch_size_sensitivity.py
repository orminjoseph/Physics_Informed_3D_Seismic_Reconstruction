"""
============================================================
PATCH SIZE SENSITIVITY TEST
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

CHECKPOINT = "checkpoints/best_model.pth"

PATCH_SIZES = [

    (32, 32, 32),

    (64, 64, 64),

    (96, 96, 96)

]


def main():

    print("=" * 60)
    print("PATCH SIZE SENSITIVITY TEST")
    print("=" * 60)

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
        "{:<15} {:<10} {:<10} {:<10} {:<10}".format(
            "Patch Size",
            "MAE",
            "RMSE",
            "PSNR",
            "SSIM"
        )
    )

    print("-" * 70)

    for patch_size in PATCH_SIZES:

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=patch_size,
            stride=patch_size,
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

        print(
            "{:<15} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.4f}".format(
                str(patch_size),
                mae,
                rmse,
                psnr,
                ssim
            )
        )


if __name__ == "__main__":
    main()