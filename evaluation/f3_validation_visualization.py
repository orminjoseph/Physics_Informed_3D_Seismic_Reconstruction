"""
=========================================================
F3 Validation Visualization
=========================================================

Visualization of seismic reconstruction results on the
F3 dataset.

This evaluation is F3-specific.

For each selected F3 patch, the script visualizes:

    1. Corrupted seismic input
    2. Ground-truth seismic data
    3. Reconstructed seismic data
    4. Absolute reconstruction error
    5. Aleatoric uncertainty
    6. Epistemic uncertainty
    7. Predictive uncertainty

The current Predictor interface returns:

    reconstruction
    travel_time
    aleatoric_std
    epistemic_std

Predictive uncertainty is calculated as:

    predictive_variance =
        aleatoric_variance + epistemic_variance

and therefore:

    predictive_std =
        sqrt(
            aleatoric_std^2 +
            epistemic_std^2
        )

Author: Ormin Joseph
=========================================================
"""

import os

import matplotlib.pyplot as plt
import torch

from utils.config import (
    DATASET_MODE,
    F3_PATH,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    DEVICE,
)

from dataset.f3_dataset import F3Dataset

from inference.predictor import Predictor

from models.network import Network3D


# =========================================================
# F3-ONLY VALIDATION GUARD
# =========================================================

if DATASET_MODE.lower() != "f3":
    raise RuntimeError(
        "F3 validation visualization requires "
        "DATASET_MODE='f3'."
    )


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

OUTPUT_DIRECTORY = os.path.join(
    FIGURE_DIR,
    "f3_validation"
)

PATCH_SIZE = (
    64,
    64,
    64
)

STRIDE = (
    64,
    64,
    64
)

MISSING_PROBABILITY = 0.30

PATCH_INDICES = [
    0,
    5,
    10
]

# Central depth slice for visualization.
SLICE_INDEX = 32


# =========================================================
# TENSOR VALIDATION UTILITY
# =========================================================

def prepare_5d_tensor(
    tensor,
    name
):
    """
    Ensure a seismic tensor has shape:

        [B, C, D, H, W]

    Dataset patches normally have shape:

        [C, D, H, W]

    so a batch dimension is added when necessary.
    """

    if not torch.is_tensor(tensor):

        raise TypeError(
            f"{name} must be a torch.Tensor. "
            f"Received: {type(tensor)}"
        )

    if tensor.ndim == 4:

        return tensor.unsqueeze(0)

    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        f"Unexpected {name} shape: "
        f"{tuple(tensor.shape)}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("F3 VALIDATION VISUALIZATION")
    print("=" * 60)

    print()
    print(f"Dataset mode : {DATASET_MODE}")
    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"F3 dataset   : {F3_PATH}")

    # -----------------------------------------------------
    # Verify F3 dataset
    # -----------------------------------------------------

    if not os.path.isfile(F3_PATH):

        raise FileNotFoundError(
            f"F3 seismic dataset not found:\n"
            f"{F3_PATH}"
        )

    # -----------------------------------------------------
    # Verify checkpoint
    # -----------------------------------------------------

    if not os.path.isfile(CHECKPOINT):

        raise FileNotFoundError(
            f"Best model checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Create F3 dataset
    # -----------------------------------------------------

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        missing_probability=MISSING_PROBABILITY,
    )

    print()
    print(
        f"F3 Dataset Length : {len(dataset)}"
    )

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=DEVICE,
    )

    # =====================================================
    # PROCESS SELECTED PATCHES
    # =====================================================

    for patch_index in PATCH_INDICES:

        # -------------------------------------------------
        # Validate patch index
        # -------------------------------------------------

        if patch_index < 0 or patch_index >= len(dataset):

            print(
                f"Skipping patch {patch_index}: "
                f"index outside dataset range."
            )

            continue

        print()
        print(
            "-" * 60
        )
        print(
            f"Processing F3 patch {patch_index}"
        )

        # -------------------------------------------------
        # Obtain F3 patch
        # -------------------------------------------------

        corrupted, target, mask, velocity = (
            dataset[patch_index][:4]
        )

        # -------------------------------------------------
        # Prepare model input
        # -------------------------------------------------

        corrupted_input = prepare_5d_tensor(
            corrupted,
            "corrupted input"
        )

        target_input = prepare_5d_tensor(
            target,
            "target"
        )

        # -------------------------------------------------
        # Display original shapes
        # -------------------------------------------------

        print(
            "Corrupted:",
            tuple(corrupted_input.shape)
        )

        print(
            "Target:",
            tuple(target_input.shape)
        )

        # -------------------------------------------------
        # Run prediction
        #
        # Current Predictor API:
        #
        # reconstruction
        # travel_time
        # aleatoric_std
        # epistemic_std
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            aleatoric_std,
            epistemic_std,
        ) = predictor.predict(
            corrupted_input
        )

        # -------------------------------------------------
        # Ensure reconstruction has batch dimension
        # -------------------------------------------------

        reconstruction = prepare_5d_tensor(
            reconstruction,
            "reconstruction"
        )

        aleatoric_std = prepare_5d_tensor(
            aleatoric_std,
            "aleatoric uncertainty"
        )

        epistemic_std = prepare_5d_tensor(
            epistemic_std,
            "epistemic uncertainty"
        )

        # -------------------------------------------------
        # Check reconstruction and target compatibility
        # -------------------------------------------------

        if reconstruction.shape != target_input.shape:

            raise ValueError(
                "Reconstruction and target shapes "
                "do not match:\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target_input.shape)}"
            )

        # -------------------------------------------------
        # Display prediction shapes
        # -------------------------------------------------

        print(
            "Reconstruction:",
            tuple(reconstruction.shape)
        )

        print(
            "Aleatoric Std:",
            tuple(aleatoric_std.shape)
        )

        print(
            "Epistemic Std:",
            tuple(epistemic_std.shape)
        )

        # -------------------------------------------------
        # Calculate predictive standard deviation
        #
        # Predictive variance:
        #
        #   V_pred = V_alea + V_epi
        #
        # Since the predictor returns standard deviations:
        #
        #   V_alea = sigma_alea^2
        #   V_epi  = sigma_epi^2
        #
        # Therefore:
        #
        #   sigma_pred =
        #       sqrt(
        #           sigma_alea^2 +
        #           sigma_epi^2
        #       )
        # -------------------------------------------------

        predictive_std = torch.sqrt(
            torch.clamp(
                aleatoric_std.pow(2)
                +
                epistemic_std.pow(2),
                min=0.0
            )
        )

        # -------------------------------------------------
        # Select central depth slice
        # -------------------------------------------------

        depth_size = corrupted_input.shape[2]

        if SLICE_INDEX >= depth_size:

            slice_index = depth_size // 2

        else:

            slice_index = SLICE_INDEX

        # -------------------------------------------------
        # Extract 2D seismic slices
        # -------------------------------------------------

        input_slice = (
            corrupted_input[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        target_slice = (
            target_input[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        recon_slice = (
            reconstruction[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        aleatoric_slice = (
            aleatoric_std[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        epistemic_slice = (
            epistemic_std[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        predictive_slice = (
            predictive_std[
                0,
                0,
                slice_index,
                :,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------
        # Calculate absolute reconstruction error
        # -------------------------------------------------

        error_slice = (
            abs(
                recon_slice -
                target_slice
            )
        )

        # =================================================
        # CREATE FIGURE
        # =================================================

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(18, 9)
        )

        # -------------------------------------------------
        # Corrupted seismic input
        # -------------------------------------------------

        axes[0, 0].imshow(
            input_slice,
            cmap="seismic",
            aspect="auto"
        )

        axes[0, 0].set_title(
            "Corrupted Input"
        )

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        axes[0, 1].imshow(
            target_slice,
            cmap="seismic",
            aspect="auto"
        )

        axes[0, 1].set_title(
            "Ground Truth"
        )

        # -------------------------------------------------
        # Reconstruction
        # -------------------------------------------------

        axes[0, 2].imshow(
            recon_slice,
            cmap="seismic",
            aspect="auto"
        )

        axes[0, 2].set_title(
            "Reconstruction"
        )

        # -------------------------------------------------
        # Absolute error
        # -------------------------------------------------

        axes[0, 3].imshow(
            error_slice,
            cmap="hot",
            aspect="auto"
        )

        axes[0, 3].set_title(
            "Absolute Error"
        )

        # -------------------------------------------------
        # Aleatoric uncertainty
        # -------------------------------------------------

        axes[1, 0].imshow(
            aleatoric_slice,
            cmap="viridis",
            aspect="auto"
        )

        axes[1, 0].set_title(
            "Aleatoric Uncertainty"
        )

        # -------------------------------------------------
        # Epistemic uncertainty
        # -------------------------------------------------

        axes[1, 1].imshow(
            epistemic_slice,
            cmap="viridis",
            aspect="auto"
        )

        axes[1, 1].set_title(
            "Epistemic Uncertainty"
        )

        # -------------------------------------------------
        # Predictive uncertainty
        # -------------------------------------------------

        axes[1, 2].imshow(
            predictive_slice,
            cmap="viridis",
            aspect="auto"
        )

        axes[1, 2].set_title(
            "Predictive Uncertainty"
        )

        # -------------------------------------------------
        # Empty panel for clean layout
        # -------------------------------------------------

        axes[1, 3].axis("off")

        # -------------------------------------------------
        # Remove axes
        # -------------------------------------------------

        for row in axes:

            for ax in row:

                if ax.axison:

                    ax.axis("off")

        # -------------------------------------------------
        # Overall figure title
        # -------------------------------------------------

        fig.suptitle(
            f"F3 Patch {patch_index} "
            f"- Depth Slice {slice_index}",
            fontsize=14
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        # =================================================
        # SAVE FIGURE
        # =================================================

        output_file = os.path.join(
            OUTPUT_DIRECTORY,
            f"f3_patch_{patch_index}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print()
        print(
            f"Saved: {output_file}"
        )

    # =====================================================
    # COMPLETION
    # =====================================================

    print()
    print("=" * 60)
    print("F3 VALIDATION VISUALIZATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIRECTORY}"
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()