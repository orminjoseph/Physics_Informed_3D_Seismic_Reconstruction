"""
=========================================================
Reconstruction Gallery
=========================================================

Generate reconstruction figures for thesis.

Figure layout:

    Input
    Ground Truth
    Reconstruction
    Absolute Error
    Aleatoric Uncertainty

The gallery uses the CURRENT EXPERIMENT_NAME so that
different experiments remain completely independent.

Example:

    outputs/

        f3_training/
            reports/
                gallery/

        synthetic_training/
            reports/
                gallery/

=========================================================
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.network import Network3D

from inference.predictor import Predictor

from dataset.build_dataset import build_dataset

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR
)


def generate_gallery(
        number_of_samples=5
):
    """
    Generate reconstruction figures for selected samples.

    Parameters
    ----------
    number_of_samples : int
        Maximum number of samples to visualize.
    """

    # =====================================================
    # VALIDATE NUMBER OF SAMPLES
    # =====================================================

    if not isinstance(number_of_samples, int):

        raise TypeError(
            "number_of_samples must be an integer."
        )

    if number_of_samples <= 0:

        raise ValueError(
            "number_of_samples must be greater than zero."
        )

    # =====================================================
    # HEADER
    # =====================================================

    print()
    print("=" * 60)
    print("RECONSTRUCTION GALLERY")
    print("=" * 60)

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME
    )

    print(
        "Device     :",
        device
    )

    # =====================================================
    # BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print()
    print(
        "Dataset Length:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Gallery dataset is empty."
        )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print(
        "Checkpoint:",
        checkpoint
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete training and make sure "
            "best_model.pth exists."
        )

    # =====================================================
    # PREDICTOR
    # =====================================================

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = os.path.join(
        REPORT_DIR,
        "gallery"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print(
        "Gallery output:",
        output_dir
    )

    # =====================================================
    # NUMBER OF SAMPLES
    # =====================================================

    samples_to_generate = min(
        number_of_samples,
        len(dataset)
    )

    print()
    print(
        "Samples to generate:",
        samples_to_generate
    )

    # =====================================================
    # GENERATE FIGURES
    # =====================================================

    for index in range(
        samples_to_generate
    ):

        print()
        print(
            f"Generating sample "
            f"{index + 1}/{samples_to_generate}"
        )

        # -------------------------------------------------
        # CURRENT DATASET CONVENTION
        # -------------------------------------------------

        (
            input_cube,
            target_cube,
            mask,
            velocity_model
        ) = dataset[index][:4]

        # -------------------------------------------------
        # CURRENT PREDICTOR CONVENTION
        #
        # Predictor returns:
        #
        # reconstruction
        # travel_time
        # log_variance
        # aleatoric_std
        #
        # Only reconstruction and aleatoric_std are
        # required for this gallery.
        # -------------------------------------------------

        (
            reconstruction,
            _,
            _,
            aleatoric_std
        ) = predictor.predict(
            input_cube
        )

        # -------------------------------------------------
        # CONVERT TENSORS TO NUMPY ARRAYS
        #
        # detach() removes the computation graph.
        # cpu() ensures compatibility with NumPy.
        # squeeze() removes batch/channel dimensions.
        # -------------------------------------------------

        reconstruction = (
            reconstruction
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        aleatoric_std = (
            aleatoric_std
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        input_cube = (
            input_cube
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        target_cube = (
            target_cube
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        # -------------------------------------------------
        # ABSOLUTE ERROR
        # -------------------------------------------------

        error = np.abs(
            target_cube -
            reconstruction
        )

        # -------------------------------------------------
        # MIDDLE DEPTH SLICE
        # -------------------------------------------------

        middle = (
            target_cube.shape[0] // 2
        )

        # =================================================
        # FIGURE
        # =================================================

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        # -------------------------------------------------
        # INPUT
        # -------------------------------------------------

        axes[0].imshow(
            input_cube[middle],
            cmap="gray"
        )

        axes[0].set_title(
            "Input"
        )

        # -------------------------------------------------
        # GROUND TRUTH
        # -------------------------------------------------

        axes[1].imshow(
            target_cube[middle],
            cmap="gray"
        )

        axes[1].set_title(
            "Ground Truth"
        )

        # -------------------------------------------------
        # RECONSTRUCTION
        # -------------------------------------------------

        axes[2].imshow(
            reconstruction[middle],
            cmap="gray"
        )

        axes[2].set_title(
            "Reconstruction"
        )

        # -------------------------------------------------
        # ABSOLUTE ERROR
        # -------------------------------------------------

        axes[3].imshow(
            error[middle],
            cmap="hot"
        )

        axes[3].set_title(
            "Absolute Error"
        )

        # -------------------------------------------------
        # ALEATORIC UNCERTAINTY
        # -------------------------------------------------

        axes[4].imshow(
            aleatoric_std[middle],
            cmap="hot"
        )

        axes[4].set_title(
            "Aleatoric Uncertainty"
        )

        # -------------------------------------------------
        # REMOVE AXES
        # -------------------------------------------------

        for ax in axes:

            ax.axis("off")

        # =================================================
        # SAVE FIGURE
        # =================================================

        output_file = os.path.join(
            output_dir,
            f"sample_{index:03d}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            "Saved:",
            output_file
        )

    # =====================================================
    # COMPLETE
    # =====================================================

    print()
    print("=" * 60)
    print("RECONSTRUCTION GALLERY COMPLETE")
    print("=" * 60)

    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Samples generated:",
        samples_to_generate
    )

    print(
        "Saved gallery to:",
        output_dir
    )


if __name__ == "__main__":

    generate_gallery()