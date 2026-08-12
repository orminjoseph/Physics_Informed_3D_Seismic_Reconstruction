"""
=========================================================
Reconstruction Gallery
=========================================================

Generate reconstruction figures for thesis.

=========================================================
"""

import os
import torch
import matplotlib.pyplot as plt

from models.network import Network3D
from inference.predictor import Predictor
from dataset.build_dataset import build_dataset

from utils.config import DATASET_MODE


def generate_gallery(
        number_of_samples=5
):

    print()
    print("=" * 60)
    print("RECONSTRUCTION GALLERY")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = build_dataset()

    model = Network3D()

    checkpoint = os.path.join(
        "outputs",
        DATASET_MODE,
        "checkpoints",
        "best_model.pth"
    )

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    output_dir = os.path.join(
        "outputs",
        DATASET_MODE,
        "reports",
        "gallery"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for index in range(

            min(
                number_of_samples,
                len(dataset)
            )

    ):

        input_cube, target_cube, mask, velocity = dataset[index]

        reconstruction, uncertainty = predictor.predict(
            input_cube
        )

        reconstruction = reconstruction.squeeze().numpy()
        uncertainty = uncertainty.squeeze().numpy()

        input_cube = input_cube.squeeze().numpy()
        target_cube = target_cube.squeeze().numpy()

        error = abs(
            target_cube -
            reconstruction
        )

        middle = target_cube.shape[0] // 2

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        axes[0].imshow(
            input_cube[middle],
            cmap="gray"
        )
        axes[0].set_title("Input")

        axes[1].imshow(
            target_cube[middle],
            cmap="gray"
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(
            reconstruction[middle],
            cmap="gray"
        )
        axes[2].set_title("Reconstruction")

        axes[3].imshow(
            error[middle],
            cmap="hot"
        )
        axes[3].set_title("Absolute Error")

        axes[4].imshow(
            uncertainty[middle],
            cmap="hot"
        )
        axes[4].set_title("Uncertainty")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                output_dir,
                f"sample_{index}.png"
            )

        )

        plt.close()

    print(
        f"Saved gallery to: {output_dir}"
    )


if __name__ == "__main__":
    generate_gallery()