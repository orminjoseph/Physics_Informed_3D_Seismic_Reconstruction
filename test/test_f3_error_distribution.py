"""
============================================================
F3 ERROR DISTRIBUTION TEST
============================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from inference.predictor import Predictor
from models.network import Network3D


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"


def main():

    print("=" * 60)
    print("F3 ERROR DISTRIBUTION TEST")
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

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=device
    )

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    error = torch.abs(
        reconstruction.squeeze() -
        target.squeeze()
    )

    errors = error.flatten().numpy()

    print()
    print("=" * 60)
    print("ERROR STATISTICS")
    print("=" * 60)

    print("Mean     :", np.mean(errors))
    print("Median   :", np.median(errors))
    print("Std      :", np.std(errors))
    print("95th %   :", np.percentile(errors, 95))
    print("Max      :", np.max(errors))

    plt.figure(figsize=(8, 5))

    plt.hist(
        errors,
        bins=100
    )

    plt.title(
        "F3 Error Distribution"
    )

    plt.xlabel(
        "Absolute Error"
    )

    plt.ylabel(
        "Voxel Count"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/comparisons/f3_error_distribution.png",
        dpi=300
    )

    plt.close()

    print()
    print(
        "Figure Saved:"
    )

    print(
        "outputs/comparisons/f3_error_distribution.png"
    )


if __name__ == "__main__":
    main()