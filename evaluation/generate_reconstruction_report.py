"""
=========================================================
Reconstruction Report
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generate reconstruction visualizations for the currently
configured dataset mode.

The script is DATA-MODE AGNOSTIC.

It does NOT hard-code:
    - F3 dataset paths
    - F3Dataset
    - checkpoint directories
    - report directories
    - dataset-specific patch construction

Instead, it obtains these from the project configuration
and dataset factory.

Supported workflow
------------------
config.py
    |
    +--> DATASET_MODE
    +--> EXPERIMENT_NAME
    +--> CHECKPOINT_DIR
    +--> REPORT_DIR
            |
            v
     build_dataset()
            |
            v
       Predictor
            |
            v
    Reconstruction + Uncertainty
            |
            v
     Report visualizations

Outputs
-------
REPORT_DIR/
    reconstruction/
        best_patch.png
        median_patch.png
        worst_patch.png
        highest_uncertainty_patch.png

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataset.build_dataset import build_dataset
from inference.predictor import Predictor
from models.network import Network3D

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
)


# =========================================================
# Configuration
# =========================================================

# Number of dataset patches to inspect.
NUM_PATCHES = 20

# Device used for evaluation.
DEVICE = "cpu"

# Best trained model.
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

# Reconstruction-report output directory.
RECONSTRUCTION_REPORT_DIR = os.path.join(
    REPORT_DIR,
    "reconstruction"
)


# =========================================================
# Utility Functions
# =========================================================

def compute_mae(prediction, target):
    """
    Compute Mean Absolute Error between prediction and target.

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    Returns
    -------
    float
        MAE value.
    """

    # Ensure both tensors have compatible dimensions.
    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{prediction.shape} vs {target.shape}"
        )

    # Calculate MAE.
    mae = torch.mean(
        torch.abs(
            prediction - target
        )
    )

    # Return a Python float.
    return mae.item()


def prepare_batch(tensor):
    """
    Convert a single dataset sample into the batch format
    expected by the Predictor.

    Expected dataset sample:
        [C, D, H, W]

    Predictor input:
        [B, C, D, H, W]
    """

    # Validate tensor type.
    if not isinstance(tensor, torch.Tensor):

        raise TypeError(
            "Expected torch.Tensor, "
            f"received {type(tensor)}"
        )

    # Add batch dimension if necessary.
    if tensor.ndim == 4:

        return tensor.unsqueeze(0)

    # If already batched, keep it unchanged.
    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        "Expected tensor with 4 or 5 dimensions, "
        f"received shape {tuple(tensor.shape)}"
    )


def detach_cpu(tensor):
    """
    Safely detach a tensor and move it to CPU.
    """

    if not isinstance(tensor, torch.Tensor):

        raise TypeError(
            "Expected torch.Tensor."
        )

    return tensor.detach().cpu()


def get_central_depth_slice(tensor):
    """
    Extract the central depth slice from a seismic tensor.

    Handles:
        [B, C, D, H, W]
        [B, D, H, W]
        [C, D, H, W]
        [D, H, W]
    """

    tensor = detach_cpu(tensor)

    # [B, C, D, H, W]
    if tensor.ndim == 5:

        return tensor[0, 0, tensor.shape[2] // 2].numpy()

    # [B, D, H, W]
    if tensor.ndim == 4:

        return tensor[0, tensor.shape[1] // 2].numpy()

    # [C, D, H, W]
    if tensor.ndim == 4:

        return tensor[0, tensor.shape[1] // 2].numpy()

    # [D, H, W]
    if tensor.ndim == 3:

        return tensor[tensor.shape[0] // 2].numpy()

    raise ValueError(
        "Unsupported tensor shape for visualization: "
        f"{tuple(tensor.shape)}"
    )


def normalize_reconstruction_shape(
    reconstruction,
    target
):
    """
    Ensure reconstruction and target have the same shape.

    The current Predictor normally returns:

        reconstruction: [B, 1, D, H, W]

    while dataset targets are normally:

        target: [C, D, H, W]

    This function converts the target into the same
    batched/channel format as the reconstruction.
    """

    reconstruction = detach_cpu(reconstruction)
    target = detach_cpu(target)

    # Target [C, D, H, W] -> [B, C, D, H, W]
    if target.ndim == 4:

        target = target.unsqueeze(0)

    if reconstruction.ndim != 5:

        raise ValueError(
            "Expected reconstruction shape "
            "[B, C, D, H, W], received "
            f"{tuple(reconstruction.shape)}"
        )

    if target.ndim != 5:

        raise ValueError(
            "Expected target shape "
            "[B, C, D, H, W], received "
            f"{tuple(target.shape)}"
        )

    if reconstruction.shape != target.shape:

        raise ValueError(
            "Reconstruction and target shapes do not match: "
            f"{tuple(reconstruction.shape)} vs "
            f"{tuple(target.shape)}"
        )

    return reconstruction, target


# =========================================================
# Visualization
# =========================================================

def save_visualization(
    corrupted,
    target,
    reconstruction,
    uncertainty,
    save_path,
    title
):
    """
    Save a five-panel reconstruction visualization.

    Panels
    ------
    1. Corrupted input
    2. Ground truth
    3. Reconstruction
    4. Absolute error
    5. Predictive uncertainty

    Parameters
    ----------
    corrupted : torch.Tensor
        Corrupted/input seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    reconstruction : torch.Tensor
        Model reconstruction.

    uncertainty : torch.Tensor
        Predictive uncertainty standard deviation.

    save_path : str
        Output image path.

    title : str
        Figure title.
    """

    # Convert target and reconstruction to matching
    # [B, C, D, H, W] shapes.
    reconstruction, target = (
        normalize_reconstruction_shape(
            reconstruction,
            target
        )
    )

    # Convert tensors to CPU.
    corrupted = detach_cpu(corrupted)
    uncertainty = detach_cpu(uncertainty)

    # Extract central slices.
    target_slice = (
        target[
            0,
            0,
            target.shape[2] // 2
        ].numpy()
    )

    reconstruction_slice = (
        reconstruction[
            0,
            0,
            reconstruction.shape[2] // 2
        ].numpy()
    )

    # Handle corrupted input.
    if corrupted.ndim == 4:

        corrupted_slice = (
            corrupted[
                0,
                corrupted.shape[1] // 2
            ].numpy()
        )

    elif corrupted.ndim == 5:

        corrupted_slice = (
            corrupted[
                0,
                0,
                corrupted.shape[2] // 2
            ].numpy()
        )

    else:

        raise ValueError(
            "Unsupported corrupted tensor shape: "
            f"{tuple(corrupted.shape)}"
        )

    # Calculate absolute reconstruction error.
    error_slice = np.abs(
        target_slice -
        reconstruction_slice
    )

    # Extract uncertainty slice.
    if uncertainty.ndim == 5:

        uncertainty_slice = (
            uncertainty[
                0,
                0,
                uncertainty.shape[2] // 2
            ].numpy()
        )

    elif uncertainty.ndim == 4:

        uncertainty_slice = (
            uncertainty[
                0,
                uncertainty.shape[1] // 2
            ].numpy()
        )

    else:

        raise ValueError(
            "Unsupported uncertainty tensor shape: "
            f"{tuple(uncertainty.shape)}"
        )

    # Create output figure.
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # Images to display.
    images = [
        corrupted_slice,
        target_slice,
        reconstruction_slice,
        error_slice,
        uncertainty_slice,
    ]

    # Panel titles.
    titles = [
        "Corrupted",
        "Ground Truth",
        "Reconstruction",
        "Absolute Error",
        "Predictive Uncertainty",
    ]

    # Draw each panel.
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

    # Overall figure title.
    fig.suptitle(title)

    # Improve spacing.
    plt.tight_layout()

    # Ensure destination directory exists.
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    # Save figure.
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Close figure to prevent memory accumulation.
    plt.close(fig)


# =========================================================
# Model Construction
# =========================================================

def create_model():
    """
    Create Network3D using the current configuration.
    """

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    return model


# =========================================================
# Main Report Generation
# =========================================================

def main():

    print()
    print("=" * 60)
    print("RECONSTRUCTION REPORT")
    print("=" * 60)

    print(
        f"Experiment    : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode  : {DATASET_MODE}"
    )

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Checkpoint    : {CHECKPOINT_PATH}"
    )

    print(
        f"Output        : {RECONSTRUCTION_REPORT_DIR}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Validate checkpoint
    # -----------------------------------------------------

    if not os.path.isfile(CHECKPOINT_PATH):

        raise FileNotFoundError(
            "Best model checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # -----------------------------------------------------
    # Build dataset according to DATASET_MODE
    # -----------------------------------------------------

    dataset = build_dataset()

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "The configured dataset is empty."
        )

    print(
        f"Dataset Length: {len(dataset)}"
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = create_model()

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=CHECKPOINT_PATH,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Store patch results
    # -----------------------------------------------------

    patch_results = []

    number_to_evaluate = min(
        NUM_PATCHES,
        len(dataset)
    )

    # -----------------------------------------------------
    # Evaluate selected patches
    # -----------------------------------------------------

    with torch.no_grad():

        for patch_index in range(
            number_to_evaluate
        ):

            print(
                f"Evaluating patch "
                f"{patch_index + 1}/"
                f"{number_to_evaluate}"
            )

            # ---------------------------------------------
            # Obtain dataset sample
            # ---------------------------------------------

            sample = dataset[patch_index]

            if not isinstance(
                sample,
                (tuple, list)
            ):

                raise TypeError(
                    "Dataset must return a tuple/list."
                )

            if len(sample) < 2:

                raise ValueError(
                    "Dataset sample must contain at "
                    "least input and target."
                )

            # Current project convention:
            #
            # sample[0] = corrupted/input seismic
            # sample[1] = target seismic
            # sample[2] = mask
            # sample[3] = velocity model
            corrupted = sample[0]
            target = sample[1]

            # ---------------------------------------------
            # Prepare model input
            # ---------------------------------------------

            corrupted_batch = prepare_batch(
                corrupted
            )

            target_batch = prepare_batch(
                target
            )

            # ---------------------------------------------
            # Model prediction
            # ---------------------------------------------

            prediction_result = (
                predictor.predict(
                    corrupted_batch
                )
            )

            # ---------------------------------------------
            # Current Predictor API
            #
            # reconstruction,
            # travel_time,
            # aleatoric_std,
            # epistemic_std
            # ---------------------------------------------

            if not isinstance(
                prediction_result,
                (tuple, list)
            ):

                raise TypeError(
                    "Predictor.predict() must return "
                    "a tuple/list."
                )

            if len(prediction_result) != 4:

                raise ValueError(
                    "Expected Predictor.predict() "
                    "to return four values:\n"
                    "(reconstruction, travel_time, "
                    "aleatoric_std, epistemic_std)\n"
                    f"Received {len(prediction_result)} values."
                )

            (
                reconstruction,
                travel_time,
                aleatoric_std,
                epistemic_std,
            ) = prediction_result

            # ---------------------------------------------
            # Validate reconstruction
            # ---------------------------------------------

            reconstruction, target_batch = (
                normalize_reconstruction_shape(
                    reconstruction,
                    target_batch
                )
            )

            # ---------------------------------------------
            # Predictive uncertainty
            #
            # Predictive variance:
            #
            #   Var_predictive =
            #       Var_aleatoric +
            #       Var_epistemic
            #
            # Therefore predictive standard deviation:
            #
            #   Std_predictive =
            #       sqrt(
            #           aleatoric_std² +
            #           epistemic_std²
            #       )
            # ---------------------------------------------

            aleatoric_std = detach_cpu(
                aleatoric_std
            )

            epistemic_std = detach_cpu(
                epistemic_std
            )

            predictive_std = torch.sqrt(
                torch.clamp(
                    aleatoric_std ** 2 +
                    epistemic_std ** 2,
                    min=0.0
                )
            )

            # ---------------------------------------------
            # Compute reconstruction MAE
            # ---------------------------------------------

            mae_value = compute_mae(
                reconstruction,
                target_batch
            )

            # ---------------------------------------------
            # Compute mean predictive uncertainty
            # ---------------------------------------------

            mean_uncertainty = (
                predictive_std.mean().item()
            )

            # ---------------------------------------------
            # Store results
            # ---------------------------------------------

            patch_results.append(
                {
                    "index": patch_index,

                    "mae": mae_value,

                    "uncertainty":
                        mean_uncertainty,

                    "corrupted":
                        corrupted.detach().cpu(),

                    "target":
                        target.detach().cpu(),

                    "reconstruction":
                        reconstruction.detach().cpu(),

                    "uncertainty_map":
                        predictive_std.detach().cpu(),

                    "aleatoric_std":
                        aleatoric_std.detach().cpu(),

                    "epistemic_std":
                        epistemic_std.detach().cpu(),
                }
            )

    # -----------------------------------------------------
    # Ensure results exist
    # -----------------------------------------------------

    if not patch_results:

        raise RuntimeError(
            "No patch results were generated."
        )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    os.makedirs(
        RECONSTRUCTION_REPORT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Identify representative patches
    # -----------------------------------------------------

    # Lowest MAE = best reconstruction.
    best_patch = min(
        patch_results,
        key=lambda x: x["mae"]
    )

    # Highest MAE = worst reconstruction.
    worst_patch = max(
        patch_results,
        key=lambda x: x["mae"]
    )

    # Highest predictive uncertainty.
    uncertainty_patch = max(
        patch_results,
        key=lambda x: x["uncertainty"]
    )

    # Median MAE patch.
    sorted_results = sorted(
        patch_results,
        key=lambda x: x["mae"]
    )

    median_patch = sorted_results[
        len(sorted_results) // 2
    ]

    # -----------------------------------------------------
    # Save visualizations
    # -----------------------------------------------------

    save_visualization(
        best_patch["corrupted"],
        best_patch["target"],
        best_patch["reconstruction"],
        best_patch["uncertainty_map"],
        os.path.join(
            RECONSTRUCTION_REPORT_DIR,
            "best_patch.png"
        ),
        (
            f"Best Patch | "
            f"MAE = {best_patch['mae']:.6f}"
        )
    )

    save_visualization(
        median_patch["corrupted"],
        median_patch["target"],
        median_patch["reconstruction"],
        median_patch["uncertainty_map"],
        os.path.join(
            RECONSTRUCTION_REPORT_DIR,
            "median_patch.png"
        ),
        (
            f"Median Patch | "
            f"MAE = {median_patch['mae']:.6f}"
        )
    )

    save_visualization(
        worst_patch["corrupted"],
        worst_patch["target"],
        worst_patch["reconstruction"],
        worst_patch["uncertainty_map"],
        os.path.join(
            RECONSTRUCTION_REPORT_DIR,
            "worst_patch.png"
        ),
        (
            f"Worst Patch | "
            f"MAE = {worst_patch['mae']:.6f}"
        )
    )

    save_visualization(
        uncertainty_patch["corrupted"],
        uncertainty_patch["target"],
        uncertainty_patch["reconstruction"],
        uncertainty_patch["uncertainty_map"],
        os.path.join(
            RECONSTRUCTION_REPORT_DIR,
            "highest_uncertainty_patch.png"
        ),
        (
            f"Highest Predictive Uncertainty | "
            f"Mean Std = "
            f"{uncertainty_patch['uncertainty']:.6f}"
        )
    )

    # -----------------------------------------------------
    # Save patch summary
    # -----------------------------------------------------

    summary_rows = []

    for result in patch_results:

        summary_rows.append(
            {
                "Experiment":
                    EXPERIMENT_NAME,

                "Dataset_Mode":
                    DATASET_MODE,

                "Patch_Index":
                    result["index"],

                "MAE":
                    result["mae"],

                "Predictive_Uncertainty_Mean":
                    result["uncertainty"],

                "Aleatoric_STD_Mean":
                    result["aleatoric_std"].mean().item(),

                "Epistemic_STD_Mean":
                    result["epistemic_std"].mean().item(),

                "Predictive_STD_Mean":
                    result["uncertainty"],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        RECONSTRUCTION_REPORT_DIR,
        "reconstruction_patch_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # -----------------------------------------------------
    # Final report message
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("RECONSTRUCTION REPORT COMPLETED")
    print("=" * 60)

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset    : {DATASET_MODE}"
    )

    print(
        f"Patches    : {len(patch_results)}"
    )

    print()

    print(
        "Best Patch:"
    )

    print(
        f"    Index : {best_patch['index']}"
    )

    print(
        f"    MAE   : {best_patch['mae']:.6f}"
    )

    print()

    print(
        "Median Patch:"
    )

    print(
        f"    Index : {median_patch['index']}"
    )

    print(
        f"    MAE   : {median_patch['mae']:.6f}"
    )

    print()

    print(
        "Worst Patch:"
    )

    print(
        f"    Index : {worst_patch['index']}"
    )

    print(
        f"    MAE   : {worst_patch['mae']:.6f}"
    )

    print()

    print(
        "Highest Predictive Uncertainty:"
    )

    print(
        f"    Index : "
        f"{uncertainty_patch['index']}"
    )

    print(
        f"    Mean Std : "
        f"{uncertainty_patch['uncertainty']:.6f}"
    )

    print()

    print(
        "Report directory:"
    )

    print(
        RECONSTRUCTION_REPORT_DIR
    )

    print()

    print(
        "Summary file:"
    )

    print(
        summary_file
    )

    print("=" * 60)


# =========================================================
# Script Entry Point
# =========================================================

if __name__ == "__main__":

    main()