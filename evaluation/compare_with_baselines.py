"""
=========================================================
Baseline Comparison
=========================================================

Physics-Informed 3D Encoder-Decoder Framework

Compare:

1. Nearest Neighbor
2. Linear Interpolation
3. Physics-Informed Network

The experiment-specific checkpoint and report directories
are controlled by utils.config.

Expected experiment structure:

    outputs/
        <EXPERIMENT_NAME>/
            checkpoints/
                best_model.pth
            reports/
                baseline_comparison.csv

For example:

    outputs/
        f3_training/
            checkpoints/
                best_model.pth
            reports/
                baseline_comparison.csv

Dataset output convention:

    corrupted
    target
    mask
    velocity_model

Predictor output convention:

    reconstruction
    travel_time
    uncertainty

Author: Ormin Joseph
=========================================================
"""

import csv
import os

import numpy as np
import torch

from dataset.f3_dataset import F3Dataset

from evaluation.baseline_nearest_neighbor import (
    nearest_neighbor_reconstruction
)

from evaluation.baseline_linear_interpolation import (
    linear_interpolation_reconstruction
)

from inference.predictor import Predictor

from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR
)


# =========================================================
# F3 DATASET PATH
# =========================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


# =========================================================
# EXPERIMENT CHECKPOINT
# =========================================================
#
# IMPORTANT:
#
# Do NOT hard-code:
#
#     outputs\synthetic_training
#
# or:
#
#     outputs\f3_training
#
# The active experiment is controlled by config.py.
#
# Therefore, when:
#
#     EXPERIMENT_NAME = "f3_training"
#
# this automatically becomes:
#
#     outputs/f3_training/checkpoints/best_model.pth
#
# =========================================================

CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate(
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
# MAIN
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
    print("Experiment :", EXPERIMENT_NAME)
    print("Device     :", device)
    print("Checkpoint :", CHECKPOINT)

    # =====================================================
    # CHECKPOINT
    # =====================================================

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT}\n\n"
            "Make sure the selected experiment has been "
            "trained and that best_model.pth exists."
        )

    # =====================================================
    # CHECK F3 DATASET
    # =====================================================

    if not os.path.exists(F3_PATH):

        raise FileNotFoundError(
            "\nF3 seismic file not found:\n"
            f"{F3_PATH}\n\n"
            "Check the F3_PATH in baseline_comparison.py."
        )

    # =====================================================
    # BUILD F3 DATASET
    # =====================================================

    print()
    print("=" * 70)
    print("LOADING F3 DATASET")
    print("=" * 70)

    dataset = F3Dataset(

        segy_path=F3_PATH,

        patch_size=(
            64,
            64,
            64
        ),

        stride=(
            64,
            64,
            64
        ),

        missing_probability=0.30
    )

    print()
    print(
        "Dataset Length:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "F3 dataset is empty."
        )

    # =====================================================
    # NUMBER OF TEST PATCHES
    # =====================================================

    NUM_TEST_PATCHES = min(
        20,
        len(dataset)
    )

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

    predictor = Predictor(

        model=Network3D(),

        checkpoint=CHECKPOINT,

        device=device
    )

    # =====================================================
    # EVALUATE TEST PATCHES
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

        # -------------------------------------------------
        # LOAD F3 PATCH
        # -------------------------------------------------

        (
            corrupted,
            target,
            mask,
            velocity
        ) = dataset[patch_index]

        # -------------------------------------------------
        # TARGET BATCH DIMENSION
        # -------------------------------------------------

        target_batch = (
            target.unsqueeze(0)
        )

        # =================================================
        # 1. NEAREST NEIGHBOR
        # =================================================

        nn_prediction = (
            nearest_neighbor_reconstruction(
                corrupted,
                mask
            )
        )

        nn_prediction = (
            nn_prediction.unsqueeze(0)
        )

        nn_metrics = evaluate(
            nn_prediction,
            target_batch
        )

        nn_all.append(
            nn_metrics
        )

        # =================================================
        # 2. LINEAR INTERPOLATION
        # =================================================

        linear_prediction = (
            linear_interpolation_reconstruction(
                corrupted,
                mask
            )
        )

        linear_prediction = (
            linear_prediction.unsqueeze(0)
        )

        linear_metrics = evaluate(
            linear_prediction,
            target_batch
        )

        linear_all.append(
            linear_metrics
        )

        # =================================================
        # 3. PHYSICS-INFORMED NETWORK
        # =================================================

        (
            reconstruction,
            travel_time,
            uncertainty
        ) = predictor.predict(
            corrupted
        )

        network_metrics = evaluate(
            reconstruction,
            target_batch
        )

        network_all.append(
            network_metrics
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

        writer = csv.writer(file)

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
        # NUMBER OF TEST PATCHES
        # -------------------------------------------------

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
        f"{NUM_TEST_PATCHES} F3 patches"
    )

    # -----------------------------------------------------
    # NEAREST NEIGHBOR
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # LINEAR INTERPOLATION
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PHYSICS-INFORMED NETWORK
    # -----------------------------------------------------

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
        "Report directory:",
        REPORT_DIR
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()