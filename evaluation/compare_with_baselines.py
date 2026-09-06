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

    Both prediction and target must have identical shapes.
    """

    if prediction.shape != target.shape:

        raise RuntimeError(
            "Prediction and target shapes do not match:\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Target    : {tuple(target.shape)}"
        )

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

    The returned tensor is always CPU-resident at this stage.
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

        value = value.detach().float()

    if value.numel() == 0:

        raise RuntimeError(
            f"{name} is empty."
        )

    if not torch.isfinite(
        value
    ).all():

        raise RuntimeError(
            f"{name} contains non-finite values."
        )

    return value


# =========================================================
# VALIDATE MASK
# =========================================================

def validate_mask(
    mask,
    corrupted,
    patch_index
):
    """
    Validate the seismic observation mask.

    Convention:

        1.0 -> observed voxel
        0.0 -> missing voxel

    The mask must contain at least one observed voxel.
    """

    if mask.shape != corrupted.shape:

        raise RuntimeError(
            f"Mask shape does not match corrupted input "
            f"for patch {patch_index + 1}:\n"
            f"Input: {tuple(corrupted.shape)}\n"
            f"Mask : {tuple(mask.shape)}"
        )

    unique_values = torch.unique(mask)

    if unique_values.numel() == 0:

        raise RuntimeError(
            f"Mask for patch {patch_index + 1} is empty."
        )

    if not torch.all(
        (unique_values == 0.0) |
        (unique_values == 1.0)
    ):

        raise RuntimeError(
            f"Mask for patch {patch_index + 1} contains "
            "values other than 0.0 and 1.0:\n"
            f"{unique_values.tolist()}"
        )

    observed = torch.count_nonzero(
        mask > 0.5
    ).item()

    missing = torch.count_nonzero(
        mask <= 0.5
    ).item()

    total = mask.numel()

    if observed == 0:

        raise RuntimeError(
            f"Patch {patch_index + 1} contains no observed "
            "voxels. Nearest-neighbor and linear "
            "interpolation cannot be performed."
        )

    if missing == 0:

        print(
            "  Warning: patch contains no missing voxels."
        )

    print(
        f"  Mask voxels : total={total}, "
        f"observed={observed}, "
        f"missing={missing}"
    )


# =========================================================
# ADD BATCH DIMENSION
# =========================================================

def add_batch_dimension(
    tensor,
    name
):
    """
    Convert:

        [C, D, H, W]

    to:

        [B, C, D, H, W]

    If already 5D, leave unchanged.
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
# VALIDATE RECONSTRUCTION
# =========================================================

def validate_reconstruction(
    reconstruction,
    target,
    method_name
):
    """
    Validate reconstructed seismic volume.
    """

    reconstruction = ensure_tensor(
        reconstruction,
        f"{method_name} prediction"
    )

    reconstruction = add_batch_dimension(
        reconstruction,
        f"{method_name} prediction"
    )

    if reconstruction.shape != target.shape:

        raise RuntimeError(
            f"{method_name} reconstruction shape does not "
            "match target shape:\n"
            f"Reconstruction: {tuple(reconstruction.shape)}\n"
            f"Target        : {tuple(target.shape)}"
        )

    return reconstruction


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
    # EVALUATION MODE
    # =====================================================

    model.eval()

    # =====================================================
    # PATCH LOOP
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
        # SHAPE VALIDATION
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
        # MASK VALIDATION
        # =================================================

        validate_mask(
            mask,
            corrupted,
            patch_index
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

        try:

            nn_prediction = (
                nearest_neighbor_reconstruction(
                    corrupted,
                    mask
                )
            )

        except Exception as exc:

            raise RuntimeError(
                f"Nearest Neighbor reconstruction failed "
                f"on patch {patch_index + 1}.\n"
                f"Input shape: {tuple(corrupted.shape)}\n"
                f"Mask shape : {tuple(mask.shape)}\n"
                f"Original error: {exc}"
            ) from exc

        nn_prediction = validate_reconstruction(
            nn_prediction,
            target_batch,
            "Nearest Neighbor"
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

        try:

            linear_prediction = (
                linear_interpolation_reconstruction(
                    corrupted,
                    mask
                )
            )

        except Exception as exc:

            raise RuntimeError(
                f"Linear interpolation reconstruction "
                f"failed on patch {patch_index + 1}.\n"
                f"Input shape: {tuple(corrupted.shape)}\n"
                f"Mask shape : {tuple(mask.shape)}\n"
                f"Original error: {exc}"
            ) from exc

        linear_prediction = validate_reconstruction(
            linear_prediction,
            target_batch,
            "Linear Interpolation"
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

        try:

            with torch.no_grad():

                (
                    reconstruction,
                    travel_time,
                    predictive_uncertainty,
                    aleatoric_std
                ) = predictor.predict(
                    corrupted_input
                )

        except Exception as exc:

            raise RuntimeError(
                f"Physics-Informed Network inference "
                f"failed on patch {patch_index + 1}.\n"
                f"Input shape: {tuple(corrupted_input.shape)}\n"
                f"Device     : {device}\n"
                f"Original error: {exc}"
            ) from exc

        reconstruction = validate_reconstruction(
            reconstruction,
            target_batch,
            "Physics-Informed Network"
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

        writer.writerow([
            "Method",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM"
        ])

        writer.writerow(
            [
                "Nearest_Neighbor"
            ]
            +
            nn_metrics
        )

        writer.writerow(
            [
                "Linear_Interpolation"
            ]
            +
            linear_metrics
        )

        writer.writerow(
            [
                "Physics_Informed_Network"
            ]
            +
            network_metrics
        )

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

    print()
    print("Nearest Neighbor:")

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

    print()
    print("Linear Interpolation:")

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

    print()
    print("Physics-Informed Network:")

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

    print()
    print("Results saved:")

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