"""
============================================================
F3 VISUALIZATION TEST
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from inference.predictor import Predictor
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
    r"outputs"
    r"\experiment_20260731_040411"
    r"\checkpoints"
    r"\best_model.pth"
)


def main():

    print()
    print("=" * 60)
    print("F3 VISUALIZATION TEST")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    device = torch.device("cpu")

    model = Network3D(
        in_channels=1,
        out_channels=1
    )

    predictor = Predictor(
        model=model,
        checkpoint=CHECKPOINT,
        device=device
    )

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    visualizer = Visualizer()

    visualizer.save_slice(
        corrupted,
        "f3_corrupted.png",
        "Corrupted F3 Patch"
    )

    visualizer.save_slice(
        target,
        "f3_target.png",
        "Ground Truth F3 Patch"
    )

    visualizer.save_slice(
        reconstruction.squeeze(0),
        "f3_reconstruction.png",
        "Reconstructed F3 Patch"
    )

    visualizer.save_slice(
        uncertainty.squeeze(0),
        "f3_uncertainty.png",
        "Predictive Uncertainty"
    )

    print()
    print("Images Saved Successfully")


if __name__ == "__main__":
    main()