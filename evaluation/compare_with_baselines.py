"""
=========================================================
Baseline Comparison
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Compare:

1. Nearest Neighbor
2. Linear Interpolation
3. Physics-Informed 3D Network

The active experiment and output directories are controlled
by utils.config.

Expected experiment structure:

    outputs/
        <EXPERIMENT_NAME>/
            checkpoints/
                best_model.pth
            reports/
                baseline_comparison.csv

Dataset output convention:

    corrupted/input
    target
    mask
    velocity_model

Predictor output convention:

    reconstruction
    travel_time
    predictive_uncertainty
    aleatoric_std

Author: Ormin Joseph
=========================================================
"""


# =========================================================
# IMPORTS
# =========================================================

import csv
import os

import numpy as np
import torch


# =========================================================
# MODEL AND INFERENCE
# =========================================================

from inference.predictor import Predictor
from models.network import Network3D


# =========================================================
# BASELINE METHODS
# =========================================================

from evaluation.baseline_nearest_neighbor import (
    nearest_neighbor_reconstruction
)

from evaluation.baseline_linear_interpolation import (
    linear_interpolation_reconstruction
)


# =========================================================
# RECONSTRUCTION METRICS
# =========================================================

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)


# =========================================================
# CONFIGURATION
# =========================================================

from utils.config import (
    EXPERIMENT_NAME,
    DATASET_MODE,
    CHECKPOINT_DIR,
    REPORT_DIR
)


# =========================================================
# DATASET BUILDER
# =========================================================
#
# IMPORTANT:
# build_dataset() is located in:
#
#     dataset/build_dataset.py
#
# It does not accept a "mode" argument.
#
# It automatically uses DATASET_MODE from utils.config.
# =========================================================

from dataset.build_dataset import build_dataset


# =========================================================
# EXPERIMENT CHECKPOINT
# =========================================================

CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def calculate_metrics(
    prediction,
    target
):
    """
    Compute reconstruction metrics.

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    Returns
    -------
    list
        [MAE, RMSE, PSNR, SNR, SSIM]
    """

    return [

        mae(
            prediction,
            target
        ).item(),

        rmse(
            prediction,
            target
        ).item(),

        psnr(
            prediction,
            target
        ).item(),

        snr(
            prediction,
            target
        ).item(),

        ssim(
            prediction,
            target
        ).item()

    ]


# =========================================================
# VALIDATE METRIC RESULT
# =========================================================

def validate_metrics(
    metrics,
    method_name
):
    """
    Validate that all calculated metrics are finite.

    Parameters
    ----------
    metrics : list
        Calculated metric values.

    method_name : str
        Name of reconstruction method.

    Returns
    -------
    list
        Validated metric values.
    """

    metrics = np.asarray(
        metrics,
        dtype=np.float64
    )

    if metrics.shape != (5,):

        raise RuntimeError(
            f"{method_name} returned an invalid "
            f"metric vector with shape {metrics.shape}."
        )

    if not np.all(
        np.isfinite(metrics)
    ):

        raise RuntimeError(
            f"{method_name} produced "
            "non-finite metric values."
        )

    return metrics.tolist()


# =========================================================
# VALIDATE DATASET SAMPLE
# =========================================================

def validate_sample(
    sample,
    patch_index
):
    """
    Validate the dataset sample structure.

    Expected convention:

        sample[0] -> corrupted/input
        sample[1] -> target
        sample[2] -> mask
        sample[3] -> velocity_model
    """

    if not isinstance(
        sample,
        (tuple, list)
    ):

        raise RuntimeError(
            f"Dataset sample {patch_index + 1} "
            "is not a tuple/list."
        )

    if len(sample) < 4:

        raise RuntimeError(
            f"Dataset sample {patch_index + 1} does not "
            "contain the expected four components:\n"
            "input, target, mask, velocity_model."
        )


# =========================================================
# CONVERT TO FLOAT TENSOR
# =========================================================

def ensure_tensor(
    value,
    name
):
    """
    Convert a dataset value to a float32 tensor.

    Parameters
    ----------
    value
        Input value.

    name : str
        Name used in error messages.

    Returns
    -------
    torch.Tensor
    """

    if not isinstance(
        value,
        torch.Tensor
    ):

        value = torch.as_tensor(
            value,
            dtype=torch.float32
        )

    else:

        value = value.float()

    if not torch.isfinite(
        value
    ).all():

        raise RuntimeError(
            f"{name} contains non-finite values."
        )

    return value


# =========================================================
# ADD BATCH DIMENSION
# =========================================================

def add_batch_dimension(
    tensor,
    name
):
    """
    Convert a seismic volume from:

        [C, D, H, W]

    to:

        [B, C, D, H, W]

    If the tensor is already 5D, it is returned unchanged.
    """

    if tensor.ndim == 4:

        tensor = tensor.unsqueeze(0)

    elif tensor.ndim == 5:

        pass

    else:

        raise RuntimeError(
            f"Unexpected {name} shape: "
            f"{tuple(tensor.shape)}. "
            "Expected [C,D,H,W] or [B,C,D,H,W]."
        )

    return tensor


# =========================================================
# MAIN BASELINE COMPARISON
# =========================================================

def main():

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

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
        "Dataset    :",
        DATASET_MODE
    )

    print(
        "Device     :",
        device
    )

    print(
        "Checkpoint :",
        CHECKPOINT
    )

    # =====================================================
    # CHECKPOINT VALIDATION
    # =====================================================

    if not os.path.isfile(
        CHECKPOINT
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT}\n\n"
            "Make sure the selected experiment has "
            "been trained and best_model.pth exists."
        )

    # =====================================================
    # BUILD ACTIVE DATASET
    # =====================================================

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # build_dataset() reads DATASET_MODE internally.
    #
    # Therefore:
    #
    #     build_dataset()
    #
    # NOT:
    #
    #     build_dataset(mode=DATASET_MODE)
    # -----------------------------------------------------

    dataset = build_dataset()

    print()
    print(
        "Dataset Length:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Dataset is empty."
        )

    # =====================================================
    # NUMBER OF TEST PATCHES
    # =====================================================

    NUM_TEST_PATCHES = min(
        20,
        len(dataset)
    )

    print()
    print(
        "Test Patches  :",
        NUM_TEST_PATCHES
    )

    # =====================================================
    # STORAGE FOR METRICS
    # =====================================================

    nn_all = []

    linear_all = []

    network_all = []

    # =====================================================
    # PHYSICS-INFORMED NETWORK
    # =====================================================

    print()
    print("=" * 70)
    print("LOADING PHYSICS-INFORMED NETWORK")
    print("=" * 70)

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    predictor = Predictor(

        model=model,

        checkpoint=CHECKPOINT,

        device=device

    )

    print(
        "Physics-Informed Network loaded successfully."
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    for patch_index in range(
        NUM_TEST_PATCHES
    ):

        print()
        print(
            f"Processing patch "
            f"{patch_index + 1}/"
            f"{NUM_TEST_PATCHES}"
        )

        # =================================================
        # LOAD DATASET SAMPLE
        # =================================================

        sample = dataset[
            patch_index
        ]

        validate_sample(
            sample,
            patch_index
        )

        (
            corrupted,
            target,
            mask,
            velocity
        ) = sample[:4]

        # =================================================
        # CONVERT DATA TO TENSORS
        # =================================================

        corrupted = ensure_tensor(
            corrupted,
            "Corrupted input"
        )

        target = ensure_tensor(
            target,
            "Target"
        )

        mask = ensure_tensor(
            mask,
            "Mask"
        )

        # =================================================
        # CHECK SHAPE CONSISTENCY
        # =================================================

        if corrupted.shape != target.shape:

            raise RuntimeError(
                "Corrupted input and target shapes do not "
                f"match for patch {patch_index + 1}:\n"
                f"Input : {tuple(corrupted.shape)}\n"
                f"Target: {tuple(target.shape)}"
            )

        if corrupted.shape != mask.shape:

            raise RuntimeError(
                "Corrupted input and mask shapes do not "
                f"match for patch {patch_index + 1}:\n"
                f"Input: {tuple(corrupted.shape)}\n"
                f"Mask : {tuple(mask.shape)}"
            )

        # =================================================
        # BATCH DIMENSION
        # =================================================

        corrupted_input = add_batch_dimension(
            corrupted,
            "corrupted input"
        )

        target_batch = add_batch_dimension(
            target,
            "target"
        )

        # =================================================
        # 1. NEAREST NEIGHBOR
        # =================================================

        print(
            "  -> Nearest Neighbor"
        )

        nn_prediction = (
            nearest_neighbor_reconstruction(
                corrupted,
                mask
            )
        )

        nn_prediction = ensure_tensor(
            nn_prediction,
            "Nearest Neighbor prediction"
        )

        nn_prediction = add_batch_dimension(
            nn_prediction,
            "Nearest Neighbor prediction"
        )

        nn_metrics = calculate_metrics(
            nn_prediction,
            target_batch
        )

        nn_metrics = validate_metrics(
            nn_metrics,
            "Nearest Neighbor"
        )

        nn_all.append(
            nn_metrics
        )

        # =================================================
        # 2. LINEAR INTERPOLATION
        # =================================================

        print(
            "  -> Linear Interpolation"
        )

        linear_prediction = (
            linear_interpolation_reconstruction(
                corrupted,
                mask
            )
        )

        linear_prediction = ensure_tensor(
            linear_prediction,
            "Linear Interpolation prediction"
        )

        linear_prediction = add_batch_dimension(
            linear_prediction,
            "Linear Interpolation prediction"
        )

        linear_metrics = calculate_metrics(
            linear_prediction,
            target_batch
        )

        linear_metrics = validate_metrics(
            linear_metrics,
            "Linear Interpolation"
        )

        linear_all.append(
            linear_metrics
        )

        # =================================================
        # 3. PHYSICS-INFORMED NETWORK
        # =================================================

        print(
            "  -> Physics-Informed Network"
        )

        (
            reconstruction,
            travel_time,
            predictive_uncertainty,
            aleatoric_std
        ) = predictor.predict(
            corrupted_input
        )

        reconstruction = ensure_tensor(
            reconstruction,
            "Physics-Informed reconstruction"
        )

        reconstruction = add_batch_dimension(
            reconstruction,
            "Physics-Informed reconstruction"
        )

        # -------------------------------------------------
        # CHECK NETWORK OUTPUT SHAPE
        # -------------------------------------------------

        if reconstruction.shape != target_batch.shape:

            raise RuntimeError(
                "Physics-Informed Network reconstruction "
                "shape does not match target shape for "
                f"patch {patch_index + 1}:\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target_batch.shape)}"
            )

        network_metrics = calculate_metrics(
            reconstruction,
            target_batch
        )

        network_metrics = validate_metrics(
            network_metrics,
            "Physics-Informed Network"
        )

        network_all.append(
            network_metrics
        )

    # =====================================================
    # CONVERT TO NUMPY ARRAYS
    # =====================================================

    nn_all = np.asarray(
        nn_all,
        dtype=np.float64
    )

    linear_all = np.asarray(
        linear_all,
        dtype=np.float64
    )

    network_all = np.asarray(
        network_all,
        dtype=np.float64
    )

    # =====================================================
    # COMPUTE AVERAGE METRICS
    # =====================================================

    nn_metrics = np.mean(
        nn_all,
        axis=0
    ).tolist()

    linear_metrics = np.mean(
        linear_all,
        axis=0
    ).tolist()

    network_metrics = np.mean(
        network_all,
        axis=0
    ).tolist()

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    csv_file = os.path.join(
        REPORT_DIR,
        "baseline_comparison.csv"
    )

    with open(
        csv_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        writer.writerow([
            "Method",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ])

        # -------------------------------------------------
        # NEAREST NEIGHBOR
        # -------------------------------------------------

        writer.writerow(
            [
                "Nearest_Neighbor"
            ]
            +
            nn_metrics
        )

        # -------------------------------------------------
        # LINEAR INTERPOLATION
        # -------------------------------------------------

        writer.writerow(
            [
                "Linear_Interpolation"
            ]
            +
            linear_metrics
        )

        # -------------------------------------------------
        # PHYSICS-INFORMED NETWORK
        # -------------------------------------------------

        writer.writerow(
            [
                "Physics_Informed_Network"
            ]
            +
            network_metrics
        )

        # -------------------------------------------------
        # NUMBER OF PATCHES
        # -----------------------------------------------------

        writer.writerow([
            "Num_Patches",
            NUM_TEST_PATCHES
        ])

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)

    print()
    print(
        f"Average over "
        f"{NUM_TEST_PATCHES} "
        f"{DATASET_MODE} patches"
    )

    # =====================================================
    # NEAREST NEIGHBOR
    # =====================================================

    print()
    print(
        "Nearest Neighbor:"
    )

    print(
        f"  MAE  : {nn_metrics[0]:.6f}"
    )

    print(
        f"  RMSE : {nn_metrics[1]:.6f}"
    )

    print(
        f"  PSNR : {nn_metrics[2]:.6f}"
    )

    print(
        f"  SNR  : {nn_metrics[3]:.6f}"
    )

    print(
        f"  SSIM : {nn_metrics[4]:.6f}"
    )

    # =====================================================
    # LINEAR INTERPOLATION
    # =====================================================

    print()
    print(
        "Linear Interpolation:"
    )

    print(
        f"  MAE  : {linear_metrics[0]:.6f}"
    )

    print(
        f"  RMSE : {linear_metrics[1]:.6f}"
    )

    print(
        f"  PSNR : {linear_metrics[2]:.6f}"
    )

    print(
        f"  SNR  : {linear_metrics[3]:.6f}"
    )

    print(
        f"  SSIM : {linear_metrics[4]:.6f}"
    )

    # =====================================================
    # PHYSICS-INFORMED NETWORK
    # =====================================================

    print()
    print(
        "Physics-Informed Network:"
    )

    print(
        f"  MAE  : {network_metrics[0]:.6f}"
    )

    print(
        f"  RMSE : {network_metrics[1]:.6f}"
    )

    print(
        f"  PSNR : {network_metrics[2]:.6f}"
    )

    print(
        f"  SNR  : {network_metrics[3]:.6f}"
    )

    print(
        f"  SSIM : {network_metrics[4]:.6f}"
    )

    # =====================================================
    # OUTPUT LOCATION
    # =====================================================

    print()
    print(
        "Results saved:"
    )

    print(
        csv_file
    )

    print()
    print("=" * 70)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 70)

    print()
    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Dataset:",
        DATASET_MODE
    )

    print(
        "Report directory:",
        REPORT_DIR
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()