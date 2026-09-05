"""
=========================================================
F3 Mask Robustness Evaluation
=========================================================

Tests the robustness of the trained seismic reconstruction
model under different missing-data probabilities.

The experiment evaluates:

    Missing Data Probability
            |
            v
       F3 Dataset
            |
            v
     Corrupted Seismic
            |
            v
       Predictor
            |
            v
     Reconstruction
            |
            v
   Reconstruction Metrics
            |
            v
   MAE / RMSE / PSNR / SNR / SSIM

The experiment is specifically designed for the F3 dataset.

Author: Ormin Joseph
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
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
    DATASET_MODE,
    F3_PATH,
    CHECKPOINT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    DEVICE,
)


# =========================================================
# CONFIGURATION
# =========================================================

# Checkpoint to use for evaluation.
# IMPORTANT:
# Use the trained BEST model, not latest_checkpoint.pth.
CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

# F3 patch configuration.
PATCH_SIZE = (64, 64, 64)
STRIDE = (64, 64, 64)

# Number of patches to evaluate at each missing-data level.
NUM_PATCHES = 20

# Missing-data probabilities to test.
MASK_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

# Output locations.
ROBUSTNESS_REPORT_DIR = os.path.join(
    REPORT_DIR,
    "f3_mask_robustness"
)

ROBUSTNESS_FIGURE_DIR = os.path.join(
    FIGURE_DIR,
    "f3_mask_robustness"
)

CSV_FILE = os.path.join(
    ROBUSTNESS_REPORT_DIR,
    "f3_mask_robustness.csv"
)

FIGURE_FILE = os.path.join(
    ROBUSTNESS_FIGURE_DIR,
    "f3_mask_robustness.png"
)


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
        Reconstructed seismic cube.

    target : torch.Tensor
        Ground-truth seismic cube.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, PSNR, SNR and SSIM.
    """

    return {

        "MAE":
            mae(
                prediction,
                target
            ).item(),

        "RMSE":
            rmse(
                prediction,
                target
            ).item(),

        "PSNR":
            psnr(
                prediction,
                target
            ).item(),

        "SNR":
            snr(
                prediction,
                target
            ).item(),

        "SSIM":
            ssim(
                prediction,
                target
            ).item()
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("F3 MASK ROBUSTNESS EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Verify that the experiment is configured for F3.
    # -----------------------------------------------------

    if DATASET_MODE.lower() != "f3":

        raise RuntimeError(
            "F3 mask robustness evaluation requires "
            "DATASET_MODE='f3'. "
            f"Current DATASET_MODE='{DATASET_MODE}'."
        )

    # -----------------------------------------------------
    # Verify F3 seismic file exists.
    # -----------------------------------------------------

    if not os.path.exists(F3_PATH):

        raise FileNotFoundError(
            f"F3 seismic file not found:\n{F3_PATH}"
        )

    # -----------------------------------------------------
    # Verify trained checkpoint exists.
    # -----------------------------------------------------

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Best model checkpoint not found:\n{CHECKPOINT}"
        )

    print()
    print(f"Dataset Mode : {DATASET_MODE}")
    print(f"F3 Dataset   : {F3_PATH}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Device       : {DEVICE}")

    # -----------------------------------------------------
    # Create the reconstruction predictor.
    # -----------------------------------------------------

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Storage for all robustness results.
    # -----------------------------------------------------

    results = []

    # =====================================================
    # TEST EACH MISSING-DATA LEVEL
    # =====================================================

    for missing_probability in MASK_LEVELS:

        print()
        print("-" * 70)

        print(
            f"Testing Missing Data = "
            f"{missing_probability:.0%}"
        )

        print("-" * 70)

        # -------------------------------------------------
        # Build F3 dataset with the current missing-data
        # probability.
        # -------------------------------------------------

        dataset = F3Dataset(
            segy_path=F3_PATH,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            missing_probability=missing_probability
        )

        print(
            f"Available patches : {len(dataset)}"
        )

        # -------------------------------------------------
        # Metric storage for this missing-data level.
        # -------------------------------------------------

        mae_values = []
        rmse_values = []
        psnr_values = []
        snr_values = []
        ssim_values = []

        # -------------------------------------------------
        # Evaluate a fixed number of patches.
        # -------------------------------------------------

        num_test_patches = min(
            NUM_PATCHES,
            len(dataset)
        )

        for patch_index in range(
            num_test_patches
        ):

            # -------------------------------------------------
            # F3Dataset returns:
            #
            # corrupted
            # target
            # mask
            # velocity
            #
            # We only need the first four components here.
            # -------------------------------------------------

            (
                corrupted,
                target,
                mask,
                velocity
            ) = dataset[patch_index][:4]

            # -------------------------------------------------
            # Add batch dimension to the corrupted cube.
            #
            # Dataset cube:
            #     [C, D, H, W]
            #
            # Predictor input:
            #     [B, C, D, H, W]
            # -------------------------------------------------

            if corrupted.ndim == 4:

                corrupted_input = (
                    corrupted.unsqueeze(0)
                )

            elif corrupted.ndim == 5:

                corrupted_input = corrupted

            else:

                raise ValueError(
                    "Unexpected corrupted seismic shape: "
                    f"{corrupted.shape}"
                )

            # -------------------------------------------------
            # Prediction.
            #
            # Current Predictor.predict() returns:
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
            # Make target compatible with prediction shape.
            # -------------------------------------------------

            if target.ndim == 4:

                target_eval = (
                    target.unsqueeze(0)
                )

            elif target.ndim == 5:

                target_eval = target

            else:

                raise ValueError(
                    "Unexpected target seismic shape: "
                    f"{target.shape}"
                )

            # -------------------------------------------------
            # Confirm shape compatibility.
            # -------------------------------------------------

            if reconstruction.shape != target_eval.shape:

                raise ValueError(
                    "Prediction and target shapes do not match.\n"
                    f"Prediction: {reconstruction.shape}\n"
                    f"Target:     {target_eval.shape}"
                )

            # -------------------------------------------------
            # Calculate reconstruction metrics.
            # -------------------------------------------------

            metrics = evaluate_metrics(
                reconstruction,
                target_eval
            )

            # -------------------------------------------------
            # Store metrics.
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

        # =====================================================
        # AVERAGE RESULTS FOR THIS MASK LEVEL
        # =====================================================

        if len(mae_values) == 0:

            raise RuntimeError(
                "No patches were available for "
                f"missing probability "
                f"{missing_probability:.2f}."
            )

        results.append({

            "Missing_Percentage":
                int(
                    missing_probability * 100
                ),

            "MAE":
                float(
                    np.mean(
                        mae_values
                    )
                ),

            "RMSE":
                float(
                    np.mean(
                        rmse_values
                    )
                ),

            "PSNR":
                float(
                    np.mean(
                        psnr_values
                    )
                ),

            "SNR":
                float(
                    np.mean(
                        snr_values
                    )
                ),

            "SSIM":
                float(
                    np.mean(
                        ssim_values
                    )
                ),

            "Num_Patches":
                len(mae_values)
        })

        print()
        print(
            f"MAE  : "
            f"{np.mean(mae_values):.6f}"
        )

        print(
            f"RMSE : "
            f"{np.mean(rmse_values):.6f}"
        )

        print(
            f"PSNR : "
            f"{np.mean(psnr_values):.6f}"
        )

        print(
            f"SNR  : "
            f"{np.mean(snr_values):.6f}"
        )

        print(
            f"SSIM : "
            f"{np.mean(ssim_values):.6f}"
        )

    # =====================================================
    # CREATE RESULTS DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        results
    )

    # =====================================================
    # SAVE CSV RESULTS
    # =====================================================

    os.makedirs(
        ROBUSTNESS_REPORT_DIR,
        exist_ok=True
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )

    # =====================================================
    # PRINT FINAL RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print("F3 MASK ROBUSTNESS RESULTS")
    print("=" * 70)

    print()
    print(df.to_string(index=False))

    # =====================================================
    # CREATE ROBUSTNESS FIGURE
    # =====================================================

    os.makedirs(
        ROBUSTNESS_FIGURE_DIR,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        df["Missing_Percentage"],
        df["MAE"],
        marker="o",
        label="MAE"
    )

    ax.plot(
        df["Missing_Percentage"],
        df["RMSE"],
        marker="o",
        label="RMSE"
    )

    ax.plot(
        df["Missing_Percentage"],
        df["SSIM"],
        marker="o",
        label="SSIM"
    )

    ax.set_xlabel(
        "Missing Data (%)"
    )

    ax.set_ylabel(
        "Metric Value"
    )

    ax.set_title(
        "F3 Mask Robustness"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print()
    print("=" * 70)
    print("F3 MASK ROBUSTNESS COMPLETE")
    print("=" * 70)

    print()
    print("Saved CSV:")
    print(CSV_FILE)

    print()
    print("Saved Figure:")
    print(FIGURE_FILE)


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()