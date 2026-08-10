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

CHECKPOINT = (
    r"outputs"
    r"\current_experiment"
    r"\checkpoints"
    r"\best_model.pth"
)

NUM_PATCHES = 20


def compute_mae(prediction, target):
    return torch.mean(
        torch.abs(
            prediction - target
        )
    ).item()


def save_visualization(
    corrupted,
    target,
    reconstruction,
    uncertainty,
    save_path,
    title
):

    slice_id = target.shape[1] // 2

    corrupted_slice = (
        corrupted[0, slice_id]
        .cpu()
        .numpy()
    )

    target_slice = (
        target[0, slice_id]
        .cpu()
        .numpy()
    )

    reconstruction_slice = (
        reconstruction[0, 0, slice_id]
        .cpu()
        .numpy()
    )

    error_slice = np.abs(
        target_slice -
        reconstruction_slice
    )

    uncertainty_slice = (
        uncertainty[0, 0, slice_id]
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    images = [
        corrupted_slice,
        target_slice,
        reconstruction_slice,
        error_slice,
        uncertainty_slice
    ]

    titles = [
        "Corrupted",
        "Ground Truth",
        "Reconstruction",
        "Absolute Error",
        "Uncertainty"
    ]

    for ax, image, name in zip(
            axes,
            images,
            titles
    ):
        ax.imshow(
            image,
            cmap="gray",
            aspect="auto"
        )

        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    print()
    print("=" * 60)
    print("RECONSTRUCTION REPORT")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    patch_results = []

    for patch_index in range(
            min(
                NUM_PATCHES,
                len(dataset)
            )
    ):

        print(
            f"Evaluating patch "
            f"{patch_index + 1}/"
            f"{NUM_PATCHES}"
        )

        corrupted, target, mask, velocity = (
            dataset[patch_index]
        )

        target_batch = (
            target.unsqueeze(0)
        )

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        mae_value = compute_mae(
            reconstruction,
            target_batch
        )

        mean_uncertainty = (
            uncertainty.mean()
            .item()
        )

        patch_results.append(
            {
                "index": patch_index,
                "mae": mae_value,
                "uncertainty":
                    mean_uncertainty,
                "corrupted":
                    corrupted,
                "target":
                    target,
                "reconstruction":
                    reconstruction,
                "uncertainty_map":
                    uncertainty
            }
        )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    best_patch = min(
        patch_results,
        key=lambda x: x["mae"]
    )

    worst_patch = max(
        patch_results,
        key=lambda x: x["mae"]
    )

    uncertainty_patch = max(
        patch_results,
        key=lambda x:
        x["uncertainty"]
    )

    median_patch = sorted(
        patch_results,
        key=lambda x: x["mae"]
    )[len(patch_results) // 2]

    save_visualization(
        best_patch["corrupted"],
        best_patch["target"],
        best_patch["reconstruction"],
        best_patch["uncertainty_map"],
        "outputs/reports/best_patch.png",
        "Best Patch"
    )

    save_visualization(
        median_patch["corrupted"],
        median_patch["target"],
        median_patch["reconstruction"],
        median_patch["uncertainty_map"],
        "outputs/reports/average_patch.png",
        "Average Patch"
    )

    save_visualization(
        worst_patch["corrupted"],
        worst_patch["target"],
        worst_patch["reconstruction"],
        worst_patch["uncertainty_map"],
        "outputs/reports/worst_patch.png",
        "Worst Patch"
    )

    save_visualization(
        uncertainty_patch["corrupted"],
        uncertainty_patch["target"],
        uncertainty_patch["reconstruction"],
        uncertainty_patch["uncertainty_map"],
        "outputs/reports/highest_uncertainty_patch.png",
        "Highest Uncertainty Patch"
    )

    print()
    print("Report saved:")
    print("outputs/reports/")
    print("best_patch.png")
    print("average_patch.png")
    print("worst_patch.png")
    print("highest_uncertainty_patch.png")


if __name__ == "__main__":
    main()