import os
import matplotlib.pyplot as plt
import torch

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

PATCH_INDICES = [
    0,
    5,
    10
]


def main():

    os.makedirs(
        "outputs/figures/f3_validation",
        exist_ok=True
    )

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

    print()
    print("=" * 60)
    print("F3 VALIDATION VISUALIZATION")
    print("=" * 60)

    for patch_index in PATCH_INDICES:

        print(
            f"Processing patch {patch_index}"
        )

        corrupted, target, mask, velocity = (
            dataset[patch_index]
        )

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        print(
            "Corrupted:",
            corrupted.shape
        )

        print(
            "Target:",
            target.shape
        )

        print(
            "Reconstruction:",
            reconstruction.shape
        )

        print(
            "Uncertainty:",
            uncertainty.shape
        )

        slice_index = 32

        input_slice = (
            corrupted[
                0,
                slice_index,
                :
            ]
            .cpu()
            .numpy()
        )

        target_slice = (
            target[
                0,
                slice_index,
                :
            ]
            .cpu()
            .numpy()
        )

        recon_slice = (
            reconstruction[
                0,
                0,
                slice_index,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        uncertainty_slice = (
            uncertainty[
                0,
                0,
                slice_index,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        error_slice = abs(
            recon_slice -
            target_slice
        )

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        axes[0].imshow(
            input_slice,
            cmap="seismic",
            aspect="auto"
        )
        axes[0].set_title(
            "Corrupted"
        )

        axes[1].imshow(
            target_slice,
            cmap="seismic",
            aspect="auto"
        )
        axes[1].set_title(
            "Ground Truth"
        )

        axes[2].imshow(
            recon_slice,
            cmap="seismic",
            aspect="auto"
        )
        axes[2].set_title(
            "Reconstruction"
        )

        axes[3].imshow(
            error_slice,
            cmap="hot",
            aspect="auto"
        )
        axes[3].set_title(
            "Absolute Error"
        )

        axes[4].imshow(
            uncertainty_slice,
            cmap="viridis",
            aspect="auto"
        )
        axes[4].set_title(
            "Uncertainty"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_file = (
            f"outputs/figures/f3_validation/"
            f"f3_patch_{patch_index}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )


if __name__ == "__main__":
    main()