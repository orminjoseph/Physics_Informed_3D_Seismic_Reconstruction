"""
============================================================
F3 MONTE CARLO DROPOUT TEST
============================================================

Evaluates epistemic uncertainty on an F3 patch.

Author: Ormin Joseph
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset

from models.network import Network3D

from inference.mc_dropout_predictor import (
    MCDropoutPredictor
)

from utils.visualization import Visualizer


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = (
    "checkpoints/best_model.pth"
)


def main():

    print("=" * 60)
    print("F3 MONTE CARLO DROPOUT TEST")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = Network3D()

    predictor = MCDropoutPredictor(
        model=model,
        checkpoint=CHECKPOINT,
        device=device,
        num_samples=20
    )

    mean_prediction, epistemic_uncertainty = (
        predictor.predict(corrupted)
    )

    print()
    print("=" * 60)
    print("EPISTEMIC UNCERTAINTY")
    print("=" * 60)

    print(
        "Mean :",
        epistemic_uncertainty.mean().item()
    )

    print(
        "Max  :",
        epistemic_uncertainty.max().item()
    )

    visualizer = Visualizer()

    visualizer.save_slice(
        mean_prediction.squeeze(0),
        "f3_mc_mean_prediction.png",
        "MC Dropout Mean Prediction"
    )

    visualizer.save_slice(
        epistemic_uncertainty.squeeze(0),
        "f3_mc_epistemic_uncertainty.png",
        "MC Dropout Epistemic Uncertainty"
    )

    print()
    print("Images Saved Successfully")


if __name__ == "__main__":
    main()