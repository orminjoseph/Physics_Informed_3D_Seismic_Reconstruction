"""
=========================================================
Test Monte Carlo Dropout Predictor
=========================================================
"""

import torch

from models.network import Network3D
from inference.mc_dropout_predictor import (
    MCDropoutPredictor
)
from utils.helpers import get_device


def main():

    checkpoint_path = (
        "outputs/"
        "experiment_20260729_201904/"
        "checkpoints/"
        "best_model.pth"
    )

    predictor = MCDropoutPredictor(

        model=Network3D(),

        checkpoint=checkpoint_path,

        device=get_device(),

        num_samples=20

    )

    sample = torch.randn(

        1,
        64,
        128,
        128

    )

    reconstruction, uncertainty = (

        predictor.predict(sample)

    )

    print()

    print("=" * 60)

    print("MC DROPOUT PREDICTOR TEST")

    print("=" * 60)

    print()

    print(
        "Reconstruction Shape :",
        reconstruction.shape
    )

    print(
        "Uncertainty Shape    :",
        uncertainty.shape
    )

    print()

    print(
        "Mean Uncertainty :",
        uncertainty.mean().item()
    )

    print(
        "Max Uncertainty  :",
        uncertainty.max().item()
    )


if __name__ == "__main__":

    main()