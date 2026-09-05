"""
=========================================================
F3 Quantitative Validation
=========================================================

Quantitative validation of seismic reconstruction
performance on the F3 dataset.

This evaluation is F3-specific.

Metrics:
    - MAE
    - RMSE
    - PSNR
    - SNR
    - SSIM

The experiment evaluates multiple F3 seismic patches
using the trained best model checkpoint.

Author: Ormin Joseph
=========================================================
"""

import os
import numpy as np
import pandas as pd
import torch

from utils.config import (
    DATASET_MODE,
    F3_PATH,
    CHECKPOINT_DIR,
    REPORT_DIR,
    DEVICE,
)

from dataset.f3_dataset import F3Dataset
from inference.predictor import Predictor
from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =========================================================
# F3-ONLY VALIDATION GUARD
# =========================================================

if DATASET_MODE.lower() != "f3":
    raise RuntimeError(
        "F3 quantitative validation requires "
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
    REPORT_DIR,
    "f3_quantitative_validation"
)

CSV_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "f3_quantitative_validation.csv"
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

NUM_PATCHES = 20


# =========================================================
# METRIC EVALUATION
# =========================================================

def evaluate_metrics(
    prediction,
    target
):
    """
    Calculate reconstruction metrics.

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    Returns
    -------
    dict
        Dictionary containing reconstruction metrics.
    """

    return {

        "MAE": mae(
            prediction,
            target
        ).item(),

        "RMSE": rmse(
            prediction,
            target
        ).item(),

        "PSNR": psnr(
            prediction,
            target
        ).item(),

        "SNR": snr(
            prediction,
            target
        ).item(),

        "SSIM": ssim(
            prediction,
            target
        ).item(),
    }


# =========================================================
# MAIN VALIDATION FUNCTION
# =========================================================

def main():

    print()
    print("=" * 60)
    print("F3 QUANTITATIVE VALIDATION")
    print("=" * 60)

    print()
    print(f"Dataset mode : {DATASET_MODE}")
    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"F3 dataset   : {F3_PATH}")

    # -----------------------------------------------------
    # Verify checkpoint exists
    # -----------------------------------------------------

    if not os.path.isfile(CHECKPOINT):

        raise FileNotFoundError(
            f"Best model checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # -----------------------------------------------------
    # Verify F3 dataset exists
    # -----------------------------------------------------

    if not os.path.isfile(F3_PATH):

        raise FileNotFoundError(
            f"F3 seismic dataset not found:\n"
            f"{F3_PATH}"
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

    # -----------------------------------------------------
    # Storage for metric values
    # -----------------------------------------------------

    mae_values = []
    rmse_values = []
    psnr_values = []
    snr_values = []
    ssim_values = []

    # -----------------------------------------------------
    # Determine number of patches to evaluate
    # -----------------------------------------------------

    patches_to_evaluate = min(
        NUM_PATCHES,
        len(dataset)
    )

    print()
    print(
        f"Evaluating {patches_to_evaluate} "
        f"F3 patches..."
    )

    # =====================================================
    # PATCH EVALUATION LOOP
    # =====================================================

    for patch_index in range(
        patches_to_evaluate
    ):

        # -------------------------------------------------
        # Obtain F3 patch
        # -------------------------------------------------

        corrupted, target, mask, velocity = (
            dataset[patch_index][:4]
        )

        # -------------------------------------------------
        # Prepare corrupted input
        #
        # Dataset normally returns:
        # [C, D, H, W]
        #
        # Network expects:
        # [B, C, D, H, W]
        # -------------------------------------------------

        if corrupted.ndim == 4:

            corrupted_input = (
                corrupted.unsqueeze(0)
            )

        elif corrupted.ndim == 5:

            corrupted_input = corrupted

        else:

            raise ValueError(
                "Unexpected corrupted input shape: "
                f"{corrupted.shape}"
            )

        # -------------------------------------------------
        # Prepare target
        # -------------------------------------------------

        if target.ndim == 4:

            target_input = (
                target.unsqueeze(0)
            )

        elif target.ndim == 5:

            target_input = target

        else:

            raise ValueError(
                "Unexpected target shape: "
                f"{target.shape}"
            )

        # -------------------------------------------------
        # Check prediction/target compatibility
        # -------------------------------------------------

        if (
            corrupted_input.shape
            != target_input.shape
        ):

            raise ValueError(
                "Corrupted input and target "
                "shapes are incompatible:\n"
                f"Input : {corrupted_input.shape}\n"
                f"Target: {target_input.shape}"
            )

        # -------------------------------------------------
        # Model prediction
        #
        # Current Predictor API returns:
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
            epistemic_std
        ) = predictor.predict(
            corrupted_input
        )

        # -------------------------------------------------
        # Ensure reconstruction is batched
        # -------------------------------------------------

        if reconstruction.ndim == 4:

            reconstruction = (
                reconstruction.unsqueeze(0)
            )

        # -------------------------------------------------
        # Evaluate reconstruction metrics
        # -------------------------------------------------

        metrics = evaluate_metrics(
            reconstruction,
            target_input
        )

        # -------------------------------------------------
        # Store metrics
        # -------------------------------------------------

        mae_values.append(
            metrics["MAE"]
        )

        rmse_values.append(
            metrics["RMSE"]
        )

        psnr_values.append(
            metrics["PSNR"]
        )

        snr_values.append(
            metrics["SNR"]
        )

        ssim_values.append(
            metrics["SSIM"]
        )

        # -------------------------------------------------
        # Display patch results
        # -------------------------------------------------

        print(
            f"Patch {patch_index + 1:02d}/{patches_to_evaluate}: "
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"PSNR={metrics['PSNR']:.4f}, "
            f"SNR={metrics['SNR']:.4f}, "
            f"SSIM={metrics['SSIM']:.4f}"
        )

    # =====================================================
    # CHECK THAT RESULTS WERE GENERATED
    # =====================================================

    if not mae_values:

        raise RuntimeError(
            "No F3 patches were evaluated."
        )

    # =====================================================
    # CALCULATE MEAN METRICS
    # =====================================================

    results = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM",
        ],

        "Mean_Value": [

            np.mean(mae_values),

            np.mean(rmse_values),

            np.mean(psnr_values),

            np.mean(snr_values),

            np.mean(ssim_values),
        ]
    })

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    results.to_csv(
        CSV_FILE,
        index=False
    )

    # =====================================================
    # DISPLAY FINAL RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("F3 QUANTITATIVE VALIDATION RESULTS")
    print("=" * 60)

    print()
    print(results.to_string(index=False))

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print()
    print("Saved:")
    print(CSV_FILE)


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()