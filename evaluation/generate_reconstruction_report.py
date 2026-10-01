"""
=========================================================
RECONSTRUCTION REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

PURPOSE
-------
Generate representative reconstruction visualizations
for the currently configured dataset mode.

The script is DATA-MODE AGNOSTIC.

It obtains:

    Dataset
    Experiment name
    Device
    Checkpoint directory
    Report directory
    Network configuration

from the central project configuration and dataset factory.

IMPORTANT
---------
This script performs MODEL INFERENCE.

It does NOT:

    - train the model
    - modify model parameters
    - retrain the model
    - modify the checkpoint
    - recompute training losses
    - perform statistical significance testing
    - perform ablation training

It uses the already trained:

    best_model.pth

from the active experiment.

CURRENT PREDICTOR API
---------------------
The current deterministic Predictor returns:

    reconstruction
    travel_time
    log_variance
    aleatoric_std

The deterministic Predictor does NOT estimate epistemic
uncertainty.

Therefore this report uses:

    aleatoric_std

for the uncertainty visualization.

Predictive uncertainty requires the separate MC-Dropout
evaluation pipeline.

SUPPORTED WORKFLOW
------------------

utils/config.py
       |
       +--> DATASET_MODE
       +--> EXPERIMENT_NAME
       +--> DEVICE
       +--> CHECKPOINT_DIR
       +--> REPORT_DIR
       +--> USE_ATTENTION
       +--> USE_RESIDUAL
       +--> USE_UNCERTAINTY
       |
       v
build_dataset()
       |
       v
Network3D
       |
       v
Predictor
       |
       v
Reconstruction
+
Aleatoric Uncertainty
       |
       v
Representative Figures

OUTPUTS
-------
REPORT_DIR/
    reconstruction/
        best_patch.png
        median_patch.png
        worst_patch.png
        highest_aleatoric_uncertainty_patch.png

        reconstruction_patch_summary.csv

AUTHOR
------
Ormin Joseph
=========================================================
"""

# =========================================================
# STANDARD LIBRARY
# =========================================================

from pathlib import Path


# =========================================================
# THIRD-PARTY LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# =========================================================
# PROJECT IMPORTS
# =========================================================

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
    DEVICE as CONFIG_DEVICE,
)


# =========================================================
# CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Number of dataset patches to inspect.
# ---------------------------------------------------------
#
# This is intentionally kept local to the report script
# because it controls only how many representative patches
# are visualized.
#
# It does NOT affect training or model parameters.
# ---------------------------------------------------------

NUM_PATCHES = 20


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
#
# The device is obtained from utils.config.
#
# Supported configuration values in the project are:
#
#     "cpu"
#     "cuda"
#     "auto"
#
# Resolve "auto" here.
# ---------------------------------------------------------

if CONFIG_DEVICE == "auto":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

elif CONFIG_DEVICE == "cuda":

    if not torch.cuda.is_available():

        raise RuntimeError(
            "DEVICE='cuda' was requested in config.py, "
            "but CUDA is not available."
        )

    DEVICE = torch.device(
        "cuda"
    )

elif CONFIG_DEVICE == "cpu":

    DEVICE = torch.device(
        "cpu"
    )

else:

    raise ValueError(
        "Unsupported DEVICE configuration: "
        f"{CONFIG_DEVICE}"
    )


# ---------------------------------------------------------
# Best trained model checkpoint
# ---------------------------------------------------------

CHECKPOINT_PATH = (
    Path(CHECKPOINT_DIR)
    / "best_model.pth"
)


# ---------------------------------------------------------
# Reconstruction report directory
# ---------------------------------------------------------

RECONSTRUCTION_REPORT_DIR = (
    Path(REPORT_DIR)
    / "reconstruction"
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================


def compute_mae(
        prediction,
        target
):
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

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )

    if not torch.isfinite(
        prediction
    ).all():

        raise ValueError(
            "Prediction contains NaN or Inf values."
        )

    if not torch.isfinite(
        target
    ).all():

        raise ValueError(
            "Target contains NaN or Inf values."
        )

    mae = torch.mean(
        torch.abs(
            prediction - target
        )
    )

    return float(
        mae.item()
    )


# ---------------------------------------------------------
# Prepare batch
# ---------------------------------------------------------

def prepare_batch(
        tensor
):
    """
    Convert a dataset tensor into [B,C,D,H,W].

    Accepted input:
        [C,D,H,W]

    or:
        [B,C,D,H,W]

    Returns
    -------
    torch.Tensor
        Tensor with shape [B,C,D,H,W].
    """

    if not isinstance(
        tensor,
        torch.Tensor
    ):

        raise TypeError(
            "Expected torch.Tensor, "
            f"received {type(tensor)}."
        )

    if tensor.ndim == 4:

        tensor = tensor.unsqueeze(0)

    elif tensor.ndim != 5:

        raise ValueError(
            "Expected tensor with shape "
            "[C,D,H,W] or [B,C,D,H,W]. "
            f"Received {tuple(tensor.shape)}"
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            "Input tensor contains NaN or Inf values."
        )

    return tensor


# ---------------------------------------------------------
# Detach and move to CPU
# ---------------------------------------------------------

def detach_cpu(
        tensor
):
    """
    Detach tensor from computation graph and move to CPU.
    """

    if not isinstance(
        tensor,
        torch.Tensor
    ):

        raise TypeError(
            "Expected torch.Tensor."
        )

    return (
        tensor.detach()
        .cpu()
    )


# ---------------------------------------------------------
# Normalize reconstruction and target shapes
# ---------------------------------------------------------

def normalize_reconstruction_shape(
        reconstruction,
        target
):
    """
    Ensure reconstruction and target both have shape:

        [B,C,D,H,W]
    """

    reconstruction = detach_cpu(
        reconstruction
    )

    target = detach_cpu(
        target
    )

    # -----------------------------------------------------
    # Target [C,D,H,W] -> [B,C,D,H,W]
    # -----------------------------------------------------

    if target.ndim == 4:

        target = target.unsqueeze(0)

    # -----------------------------------------------------
    # Validate dimensions
    # -----------------------------------------------------

    if reconstruction.ndim != 5:

        raise ValueError(
            "Expected reconstruction with shape "
            "[B,C,D,H,W]. "
            f"Received {tuple(reconstruction.shape)}"
        )

    if target.ndim != 5:

        raise ValueError(
            "Expected target with shape "
            "[B,C,D,H,W]. "
            f"Received {tuple(target.shape)}"
        )

    # -----------------------------------------------------
    # Validate shape equality
    # -----------------------------------------------------

    if reconstruction.shape != target.shape:

        raise ValueError(
            "Reconstruction and target shapes do not match:\n"
            f"Reconstruction: {tuple(reconstruction.shape)}\n"
            f"Target        : {tuple(target.shape)}"
        )

    return (
        reconstruction,
        target
    )


# =========================================================
# VISUALIZATION
# =========================================================


def extract_central_slice(
        tensor
):
    """
    Extract the central depth slice from a seismic tensor.

    Expected final format:

        [B,C,D,H,W]

    Returns
    -------
    numpy.ndarray
        Central depth slice with shape [H,W].
    """

    tensor = detach_cpu(
        tensor
    )

    if tensor.ndim == 4:

        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 5:

        raise ValueError(
            "Expected tensor with shape "
            "[B,C,D,H,W]. "
            f"Received {tuple(tensor.shape)}"
        )

    depth_index = (
        tensor.shape[2] // 2
    )

    return (
        tensor[
            0,
            0,
            depth_index
        ]
        .numpy()
    )


# ---------------------------------------------------------
# Save reconstruction visualization
# ---------------------------------------------------------

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
    5. Aleatoric uncertainty

    IMPORTANT
    ---------
    The fifth panel is explicitly labelled ALEATORIC
    UNCERTAINTY because the deterministic Predictor does
    not estimate epistemic uncertainty.
    """

    # -----------------------------------------------------
    # Normalize reconstruction/target shapes
    # -----------------------------------------------------

    reconstruction, target = (
        normalize_reconstruction_shape(
            reconstruction,
            target
        )
    )

    # -----------------------------------------------------
    # Prepare corrupted input
    # -----------------------------------------------------

    corrupted = prepare_batch(
        corrupted
    )

    corrupted = detach_cpu(
        corrupted
    )

    # -----------------------------------------------------
    # Prepare uncertainty
    # -----------------------------------------------------

    uncertainty = prepare_batch(
        uncertainty
    )

    uncertainty = detach_cpu(
        uncertainty
    )

    # -----------------------------------------------------
    # Validate uncertainty shape
    # -----------------------------------------------------

    if uncertainty.shape != reconstruction.shape:

        raise ValueError(
            "Aleatoric uncertainty shape does not match "
            "reconstruction shape:\n"
            f"Uncertainty   : {tuple(uncertainty.shape)}\n"
            f"Reconstruction: {tuple(reconstruction.shape)}"
        )

    # -----------------------------------------------------
    # Extract central slices
    # -----------------------------------------------------

    corrupted_slice = extract_central_slice(
        corrupted
    )

    target_slice = extract_central_slice(
        target
    )

    reconstruction_slice = extract_central_slice(
        reconstruction
    )

    uncertainty_slice = extract_central_slice(
        uncertainty
    )

    # -----------------------------------------------------
    # Absolute reconstruction error
    # -----------------------------------------------------

    error_slice = np.abs(
        target_slice -
        reconstruction_slice
    )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # -----------------------------------------------------
    # Image data
    # -----------------------------------------------------

    images = [

        corrupted_slice,

        target_slice,

        reconstruction_slice,

        error_slice,

        uncertainty_slice,

    ]

    # -----------------------------------------------------
    # Panel titles
    # -----------------------------------------------------

    panel_titles = [

        "Corrupted",

        "Ground Truth",

        "Reconstruction",

        "Absolute Error",

        "Aleatoric Uncertainty",

    ]

    # -----------------------------------------------------
    # Draw panels
    # -----------------------------------------------------

    for axis, image, panel_title in zip(
        axes,
        images,
        panel_titles
    ):

        axis.imshow(
            image,
            cmap="gray",
            aspect="auto"
        )

        axis.set_title(
            panel_title
        )

        axis.axis(
            "off"
        )

    # -----------------------------------------------------
    # Overall title
    # -----------------------------------------------------

    fig.suptitle(
        title
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    plt.tight_layout()

    # -----------------------------------------------------
    # Ensure output directory exists
    # -----------------------------------------------------

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    # -----------------------------------------------------
    # Close figure
    # -----------------------------------------------------

    plt.close(
        fig
    )


# =========================================================
# MODEL CONSTRUCTION
# =========================================================


def create_model():
    """
    Create Network3D using the active project configuration.
    """

    model = Network3D(

        use_attention=
            USE_ATTENTION,

        use_residual=
            USE_RESIDUAL,

        use_uncertainty=
            USE_UNCERTAINTY,

    )

    return model


# =========================================================
# MAIN REPORT GENERATION
# =========================================================


def main():

    print()
    print("=" * 70)
    print("RECONSTRUCTION REPORT")
    print("=" * 70)

    print(
        f"Experiment       : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode     : {DATASET_MODE}"
    )

    print(
        f"Configured Device: {CONFIG_DEVICE}"
    )

    print(
        f"Resolved Device  : {DEVICE}"
    )

    print(
        f"Checkpoint       : {CHECKPOINT_PATH}"
    )

    print(
        f"Output Directory : "
        f"{RECONSTRUCTION_REPORT_DIR}"
    )

    print("=" * 70)

    # =====================================================
    # VALIDATE CHECKPOINT
    # =====================================================

    if not CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            "Best model checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    if CHECKPOINT_PATH.stat().st_size == 0:

        raise RuntimeError(
            "Best model checkpoint exists but is empty:\n"
            f"{CHECKPOINT_PATH}"
        )

    # =====================================================
    # BUILD DATASET
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 1: BUILD DATASET")
    print("-" * 70)

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
        f"Dataset length: {len(dataset)}"
    )

    number_to_evaluate = min(
        NUM_PATCHES,
        len(dataset)
    )

    print(
        f"Patches to inspect: "
        f"{number_to_evaluate}"
    )

    # =====================================================
    # CREATE MODEL
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 2: CREATE MODEL")
    print("-" * 70)

    model = create_model()

    print(
        "Network3D created successfully."
    )

    print(
        f"Attention    : {USE_ATTENTION}"
    )

    print(
        f"Residual     : {USE_RESIDUAL}"
    )

    print(
        f"Uncertainty  : {USE_UNCERTAINTY}"
    )

    # =====================================================
    # CREATE PREDICTOR
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 3: LOAD BEST MODEL")
    print("-" * 70)

    predictor = Predictor(

        model=model,

        checkpoint=str(
            CHECKPOINT_PATH
        ),

        device=DEVICE,

    )

    print(
        "Best model checkpoint loaded successfully."
    )

    # =====================================================
    # STORAGE
    # =====================================================

    patch_results = []

    # =====================================================
    # EVALUATE PATCHES
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 4: GENERATE RECONSTRUCTIONS")
    print("-" * 70)

    for patch_index in range(
        number_to_evaluate
    ):

        print(
            f"Evaluating patch "
            f"{patch_index + 1}/"
            f"{number_to_evaluate}"
        )

        # -------------------------------------------------
        # Obtain dataset sample
        # -------------------------------------------------

        sample = dataset[
            patch_index
        ]

        if not isinstance(
            sample,
            (tuple, list)
        ):

            raise TypeError(
                "Dataset sample must be a "
                "tuple or list."
            )

        if len(sample) < 2:

            raise ValueError(
                "Dataset sample must contain at least "
                "input and target."
            )

        # -------------------------------------------------
        # Current dataset convention
        #
        # sample[0] = input/corrupted cube
        # sample[1] = target cube
        # sample[2] = mask
        # sample[3] = velocity model
        # -------------------------------------------------

        corrupted = sample[0]

        target = sample[1]

        # -------------------------------------------------
        # Prepare model input
        # -------------------------------------------------

        corrupted_batch = prepare_batch(
            corrupted
        )

        target_batch = prepare_batch(
            target
        )

        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------
        #
        # Predictor.predict() returns:
        #
        #     reconstruction
        #     travel_time
        #     log_variance
        #     aleatoric_std
        #
        # It does NOT return epistemic uncertainty.
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            log_variance,
            aleatoric_std,
        ) = predictor.predict(
            corrupted_batch
        )

        # -------------------------------------------------
        # Normalize reconstruction and target
        # -------------------------------------------------

        reconstruction, target_batch = (
            normalize_reconstruction_shape(
                reconstruction,
                target_batch
            )
        )

        # -------------------------------------------------
        # Prepare aleatoric uncertainty
        # -------------------------------------------------

        aleatoric_std = prepare_batch(
            aleatoric_std
        )

        # -------------------------------------------------
        # Validate uncertainty dimensions
        # -------------------------------------------------

        if (
            aleatoric_std.shape
            != reconstruction.shape
        ):

            raise ValueError(
                "Aleatoric uncertainty shape does not "
                "match reconstruction shape:\n"
                f"Aleatoric: "
                f"{tuple(aleatoric_std.shape)}\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}"
            )

        # -------------------------------------------------
        # Move outputs to CPU
        # -------------------------------------------------

        reconstruction_cpu = detach_cpu(
            reconstruction
        )

        target_cpu = detach_cpu(
            target_batch
        )

        aleatoric_std_cpu = detach_cpu(
            aleatoric_std
        )

        corrupted_cpu = detach_cpu(
            corrupted_batch
        )

        # -------------------------------------------------
        # Compute MAE
        # -------------------------------------------------

        mae_value = compute_mae(
            reconstruction_cpu,
            target_cpu
        )

        # -------------------------------------------------
        # Mean aleatoric uncertainty
        # -------------------------------------------------

        mean_aleatoric_std = float(
            aleatoric_std_cpu.mean().item()
        )

        # -------------------------------------------------
        # Validate uncertainty
        # -------------------------------------------------

        if not torch.isfinite(
            aleatoric_std_cpu
        ).all():

            raise ValueError(
                "Aleatoric uncertainty contains "
                "NaN or Inf values."
            )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        patch_results.append(
            {

                "Patch_Index":
                    patch_index,

                "MAE":
                    mae_value,

                "Mean_Aleatoric_STD":
                    mean_aleatoric_std,

                "Corrupted":
                    corrupted_cpu,

                "Target":
                    target_cpu,

                "Reconstruction":
                    reconstruction_cpu,

                "Aleatoric_STD":
                    aleatoric_std_cpu,

            }
        )

    # =====================================================
    # VALIDATE RESULTS
    # =====================================================

    if not patch_results:

        raise RuntimeError(
            "No reconstruction results were generated."
        )

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    RECONSTRUCTION_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # IDENTIFY REPRESENTATIVE PATCHES
    # =====================================================

    # -----------------------------------------------------
    # Best reconstruction
    # -----------------------------------------------------

    best_patch = min(
        patch_results,
        key=lambda result:
        result["MAE"]
    )

    # -----------------------------------------------------
    # Worst reconstruction
    # -----------------------------------------------------

    worst_patch = max(
        patch_results,
        key=lambda result:
        result["MAE"]
    )

    # -----------------------------------------------------
    # Highest aleatoric uncertainty
    # -----------------------------------------------------

    highest_uncertainty_patch = max(
        patch_results,
        key=lambda result:
        result["Mean_Aleatoric_STD"]
    )

    # -----------------------------------------------------
    # Median MAE patch
    # -----------------------------------------------------

    sorted_results = sorted(
        patch_results,
        key=lambda result:
        result["MAE"]
    )

    median_patch = sorted_results[
        len(sorted_results) // 2
    ]

    # =====================================================
    # SAVE FIGURES
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 5: SAVE REPRESENTATIVE FIGURES")
    print("-" * 70)

    # -----------------------------------------------------
    # Best patch
    # -----------------------------------------------------

    save_visualization(

        best_patch["Corrupted"],

        best_patch["Target"],

        best_patch["Reconstruction"],

        best_patch["Aleatoric_STD"],

        (
            RECONSTRUCTION_REPORT_DIR
            / "best_patch.png"
        ),

        (
            "Best Reconstruction Patch | "
            f"MAE = {best_patch['MAE']:.6f}"
        ),

    )

    # -----------------------------------------------------
    # Median patch
    # -----------------------------------------------------

    save_visualization(

        median_patch["Corrupted"],

        median_patch["Target"],

        median_patch["Reconstruction"],

        median_patch["Aleatoric_STD"],

        (
            RECONSTRUCTION_REPORT_DIR
            / "median_patch.png"
        ),

        (
            "Median Reconstruction Patch | "
            f"MAE = {median_patch['MAE']:.6f}"
        ),

    )

    # -----------------------------------------------------
    # Worst patch
    # -----------------------------------------------------

    save_visualization(

        worst_patch["Corrupted"],

        worst_patch["Target"],

        worst_patch["Reconstruction"],

        worst_patch["Aleatoric_STD"],

        (
            RECONSTRUCTION_REPORT_DIR
            / "worst_patch.png"
        ),

        (
            "Worst Reconstruction Patch | "
            f"MAE = {worst_patch['MAE']:.6f}"
        ),

    )

    # -----------------------------------------------------
    # Highest uncertainty patch
    # -----------------------------------------------------

    save_visualization(

        highest_uncertainty_patch["Corrupted"],

        highest_uncertainty_patch["Target"],

        highest_uncertainty_patch["Reconstruction"],

        highest_uncertainty_patch["Aleatoric_STD"],

        (
            RECONSTRUCTION_REPORT_DIR
            / "highest_aleatoric_uncertainty_patch.png"
        ),

        (
            "Highest Aleatoric Uncertainty | "
            f"Mean Std = "
            f"{highest_uncertainty_patch['Mean_Aleatoric_STD']:.6f}"
        ),

    )

    # =====================================================
    # SAVE PATCH SUMMARY
    # =====================================================

    print()
    print("-" * 70)
    print("STEP 6: SAVE PATCH SUMMARY")
    print("-" * 70)

    summary_rows = []

    for result in patch_results:

        summary_rows.append(
            {

                "Experiment":
                    EXPERIMENT_NAME,

                "Dataset_Mode":
                    DATASET_MODE,

                "Patch_Index":
                    result["Patch_Index"],

                "MAE":
                    result["MAE"],

                "Mean_Aleatoric_STD":
                    result[
                        "Mean_Aleatoric_STD"
                    ],

            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        RECONSTRUCTION_REPORT_DIR
        / "reconstruction_patch_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("RECONSTRUCTION REPORT COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Experiment          : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode        : "
        f"{DATASET_MODE}"
    )

    print(
        f"Device              : "
        f"{DEVICE}"
    )

    print(
        f"Patches Evaluated   : "
        f"{len(patch_results)}"
    )

    print()

    print(
        "Best Reconstruction:"
    )

    print(
        f"    Patch Index : "
        f"{best_patch['Patch_Index']}"
    )

    print(
        f"    MAE         : "
        f"{best_patch['MAE']:.6f}"
    )

    print()

    print(
        "Median Reconstruction:"
    )

    print(
        f"    Patch Index : "
        f"{median_patch['Patch_Index']}"
    )

    print(
        f"    MAE         : "
        f"{median_patch['MAE']:.6f}"
    )

    print()

    print(
        "Worst Reconstruction:"
    )

    print(
        f"    Patch Index : "
        f"{worst_patch['Patch_Index']}"
    )

    print(
        f"    MAE         : "
        f"{worst_patch['MAE']:.6f}"
    )

    print()

    print(
        "Highest Aleatoric Uncertainty:"
    )

    print(
        f"    Patch Index : "
        f"{highest_uncertainty_patch['Patch_Index']}"
    )

    print(
        f"    Mean Std    : "
        f"{highest_uncertainty_patch['Mean_Aleatoric_STD']:.6f}"
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        RECONSTRUCTION_REPORT_DIR
    )

    print()

    print(
        "Patch summary:"
    )

    print(
        summary_file
    )

    print()
    print("=" * 70)


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()