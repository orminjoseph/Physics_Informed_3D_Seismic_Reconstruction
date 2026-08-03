"""
============================================================
F3 SPATIAL ERROR MAP TEST
============================================================
"""

import os
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
    print("F3 SPATIAL ERROR MAP")
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

    target = target.squeeze().numpy()

    reconstruction = (
        reconstruction.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    error_map = np.abs(
        target - reconstruction
    )

    slice_index = target.shape[0] // 2

    target_slice = target[slice_index]
    reconstruction_slice = reconstruction[slice_index]
    error_slice = error_map[slice_index]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        target_slice,
        cmap="gray",
        aspect="auto"
    )
    axes[0].set_title(
        "Ground Truth"
    )

    axes[1].imshow(
        reconstruction_slice,
        cmap="gray",
        aspect="auto"
    )
    axes[1].set_title(
        "Reconstruction"
    )

    axes[2].imshow(
        error_slice,
        cmap="hot",
        aspect="auto"
    )
    axes[2].set_title(
        "Absolute Error"
    )

    plt.tight_layout()

    os.makedirs(
        "outputs/comparisons",
        exist_ok=True
    )

    save_path = (
        "outputs/comparisons/"
        "f3_spatial_error_map.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Figure Saved:")
    print(save_path)


if __name__ == "__main__":
    main()