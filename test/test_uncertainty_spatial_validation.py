# test/test_uncertainty_spatial_validation.py

"""
============================================================
UNCERTAINTY SPATIAL VALIDATION
============================================================

Visual comparison of:

1. Original
2. Corrupted
3. Reconstruction
4. Error Map
5. Uncertainty Map

Author: Ormin Joseph
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
    print("UNCERTAINTY SPATIAL VALIDATION")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    target = target.squeeze().numpy()

    corrupted = corrupted.squeeze().numpy()

    reconstruction = (
        reconstruction.squeeze()
        .cpu()
        .numpy()
    )

    uncertainty = (
        uncertainty.squeeze()
        .cpu()
        .numpy()
    )

    error_map = np.abs(
        target -
        reconstruction
    )

    # Middle depth slice

    z = target.shape[0] // 2

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes[0, 0].imshow(
        target[z],
        cmap="seismic",
        aspect="auto"
    )
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(
        corrupted[z],
        cmap="seismic",
        aspect="auto"
    )
    axes[0, 1].set_title("Corrupted")

    axes[0, 2].imshow(
        reconstruction[z],
        cmap="seismic",
        aspect="auto"
    )
    axes[0, 2].set_title("Reconstruction")

    axes[1, 0].imshow(
        error_map[z],
        cmap="hot",
        aspect="auto"
    )
    axes[1, 0].set_title("Error Map")

    axes[1, 1].imshow(
        uncertainty[z],
        cmap="viridis",
        aspect="auto"
    )
    axes[1, 1].set_title("Uncertainty Map")

    axes[1, 2].imshow(
        velocity.squeeze().numpy()[z],
        cmap="jet",
        aspect="auto"
    )
    axes[1, 2].set_title("Velocity Model")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/uncertainty_spatial_validation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Figure saved to:")
    print(
        "outputs/figures/"
        "uncertainty_spatial_validation.png"
    )


if __name__ == "__main__":
    main()