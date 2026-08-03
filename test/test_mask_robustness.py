"""
============================================================
MASK ROBUSTNESS TEST
============================================================
"""

import torch

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
        "{:<20} {:<10} {:<10} {:<10} {:<10}".format(
            "Mask",
            "MAE",
            "RMSE",
            "PSNR",
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

        ssim = EvaluationMetrics.ssim(
            reconstruction,
            target
        ).item()

        print(
            "{:<20} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.4f}".format(
                mask_type,
                mae,
                rmse,
                psnr,
                ssim
            )
        )


if __name__ == "__main__":
    main()