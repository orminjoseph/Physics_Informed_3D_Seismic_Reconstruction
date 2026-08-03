"""
============================================================
TOTAL PREDICTIVE UNCERTAINTY TEST
============================================================

Aleatoric + Epistemic Uncertainty

Author: Ormin Joseph
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset

from models.network import Network3D

from inference.predictor import Predictor
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
    print("TOTAL UNCERTAINTY TEST")
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

    # ------------------------------------------
    # Aleatoric uncertainty
    # ------------------------------------------

    model_1 = Network3D()

    predictor = Predictor(
        model=model_1,
        checkpoint=CHECKPOINT,
        device=device
    )

    reconstruction, aleatoric = (
        predictor.predict(corrupted)
    )

    # ------------------------------------------
    # Epistemic uncertainty
    # ------------------------------------------

    model_2 = Network3D()

    mc_predictor = MCDropoutPredictor(
        model=model_2,
        checkpoint=CHECKPOINT,
        device=device,
        num_samples=20
    )

    _, epistemic = (
        mc_predictor.predict(corrupted)
    )

    # ------------------------------------------
    # Total uncertainty
    # ------------------------------------------

    total_uncertainty = (
        aleatoric + epistemic
    )

    print()
    print("=" * 60)
    print("UNCERTAINTY SUMMARY")
    print("=" * 60)

    print(
        "Mean Aleatoric :",
        aleatoric.mean().item()
    )

    print(
        "Mean Epistemic :",
        epistemic.mean().item()
    )

    print(
        "Mean Total     :",
        total_uncertainty.mean().item()
    )

    visualizer = Visualizer()

    visualizer.save_slice(
        aleatoric.squeeze(0),
        "f3_aleatoric_uncertainty.png",
        "Aleatoric Uncertainty"
    )

    visualizer.save_slice(
        epistemic.squeeze(0),
        "f3_epistemic_uncertainty.png",
        "Epistemic Uncertainty"
    )

    visualizer.save_slice(
        total_uncertainty.squeeze(0),
        "f3_total_uncertainty.png",
        "Total Predictive Uncertainty"
    )

    print()
    print("Images Saved Successfully")


if __name__ == "__main__":
    main()