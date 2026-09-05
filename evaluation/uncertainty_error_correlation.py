"""
=========================================================
Uncertainty-Error Correlation Analysis
=========================================================

Computes the Pearson and Spearman correlations between
voxel-wise reconstruction error and predictive uncertainty.

Predictive uncertainty is decomposed into:

    Aleatoric uncertainty
    +
    Epistemic uncertainty

The predictive standard deviation is:

    sigma_predictive =
        sqrt(sigma_aleatoric^2 + sigma_epistemic^2)

The analysis is performed on the reconstruction output
only. Travel-time predictions are not used here.

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from dataset.build_dataset import build_dataset

from inference.predictor import Predictor

from models.network import Network3D

from utils.config import (
    DATASET_MODE,
    CHECKPOINT_DIR,
    REPORT_DIR,
    DEVICE,
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
    "uncertainty_error_correlation"
)

CSV_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "uncertainty_error_correlation.csv"
)

NUM_PATCHES = 20


# =========================================================
# VALIDATION
# =========================================================

def validate_array(
    array,
    name
):
    """
    Validate a NumPy array before correlation analysis.
    """

    array = np.asarray(
        array,
        dtype=np.float64
    )

    if array.size == 0:
        raise ValueError(
            f"{name} is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():

    print()
    print("=" * 60)
    print("UNCERTAINTY-ERROR CORRELATION")
    print("=" * 60)

    print()
    print(f"Dataset Mode : {DATASET_MODE}")
    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")

    # -----------------------------------------------------
    # Verify checkpoint
    # -----------------------------------------------------

    if not os.path.isfile(
        CHECKPOINT
    ):
        raise FileNotFoundError(
            "Best model checkpoint not found:\n"
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
    # Build dataset
    # -----------------------------------------------------

    dataset = build_dataset()

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contains no samples."
        )

    print(
        f"Dataset Length: {len(dataset)}"
    )

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Storage for voxel-wise values
    # -----------------------------------------------------

    all_errors = []
    all_uncertainties = []

    number_of_patches = min(
        NUM_PATCHES,
        len(dataset)
    )

    print()
    print(
        f"Analyzing {number_of_patches} patch(es)..."
    )

    # =====================================================
    # PROCESS DATASET
    # =====================================================

    for patch_index in range(
        number_of_patches
    ):

        print(
            f"Processing patch "
            f"{patch_index + 1}/{number_of_patches}"
        )

        # -------------------------------------------------
        # Dataset sample
        # -------------------------------------------------

        corrupted, target, mask, velocity = (
            dataset[patch_index][:4]
        )

        # -------------------------------------------------
        # Predictor expects a batch dimension.
        #
        # Dataset samples are normally:
        #
        #     [C, D, H, W]
        #
        # Predictor expects:
        #
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
                "Unexpected corrupted input shape: "
                f"{tuple(corrupted.shape)}"
            )

        # -------------------------------------------------
        # Current Predictor interface
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
        # Prepare target
        # -------------------------------------------------

        if target.ndim == 4:

            target_tensor = (
                target.unsqueeze(0)
            )

        elif target.ndim == 5:

            target_tensor = target

        else:

            raise ValueError(
                "Unexpected target shape: "
                f"{tuple(target.shape)}"
            )

        # -------------------------------------------------
        # Check reconstruction compatibility
        # -------------------------------------------------

        if reconstruction.shape != target_tensor.shape:

            raise ValueError(
                "Reconstruction and target shapes "
                "do not match.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target_tensor.shape)}"
            )

        # -------------------------------------------------
        # Predictive uncertainty
        #
        # sigma_pred =
        # sqrt(
        #     sigma_aleatoric^2
        #     +
        #     sigma_epistemic^2
        # )
        # -------------------------------------------------

        predictive_std = (
            aleatoric_std.pow(2)
            +
            epistemic_std.pow(2)
        ).sqrt()

        # -------------------------------------------------
        # Reconstruction error
        # -------------------------------------------------

        error = (
            reconstruction
            -
            target_tensor
        ).abs()

        # -------------------------------------------------
        # Convert to NumPy
        # -------------------------------------------------

        error_np = (
            error
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        uncertainty_np = (
            predictive_std
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        # -------------------------------------------------
        # Validate arrays
        # -------------------------------------------------

        error_np = validate_array(
            error_np,
            "Reconstruction error"
        )

        uncertainty_np = validate_array(
            uncertainty_np,
            "Predictive uncertainty"
        )

        # -------------------------------------------------
        # Verify matching voxel counts
        # -------------------------------------------------

        if error_np.size != uncertainty_np.size:

            raise ValueError(
                "Error and uncertainty contain "
                "different numbers of voxels.\n"
                f"Error voxels: "
                f"{error_np.size}\n"
                f"Uncertainty voxels: "
                f"{uncertainty_np.size}"
            )

        # -------------------------------------------------
        # Store voxel-wise values
        # -------------------------------------------------

        all_errors.extend(
            error_np.flatten()
        )

        all_uncertainties.extend(
            uncertainty_np.flatten()
        )

    # =====================================================
    # CONVERT TO ARRAYS
    # =====================================================

    all_errors = validate_array(
        all_errors,
        "All reconstruction errors"
    )

    all_uncertainties = validate_array(
        all_uncertainties,
        "All predictive uncertainties"
    )

    # -----------------------------------------------------
    # Minimum observations
    # -----------------------------------------------------

    if len(all_errors) < 2:

        raise RuntimeError(
            "At least two observations are required "
            "for correlation analysis."
        )

    # =====================================================
    # CORRELATION ANALYSIS
    # =====================================================

    # -----------------------------------------------------
    # Pearson correlation
    #
    # Measures linear association.
    # -----------------------------------------------------

    if (
        np.std(all_errors) == 0
        or
        np.std(all_uncertainties) == 0
    ):

        pearson_corr = np.nan
        pearson_p = np.nan

    else:

        pearson_corr, pearson_p = pearsonr(
            all_errors,
            all_uncertainties
        )

    # -----------------------------------------------------
    # Spearman correlation
    #
    # Measures monotonic association.
    # -----------------------------------------------------

    if (
        np.std(all_errors) == 0
        or
        np.std(all_uncertainties) == 0
    ):

        spearman_corr = np.nan
        spearman_p = np.nan

    else:

        spearman_corr, spearman_p = spearmanr(
            all_errors,
            all_uncertainties
        )

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    results = pd.DataFrame({

        "Metric": [
            "Pearson",
            "Spearman"
        ],

        "Correlation": [
            pearson_corr,
            spearman_corr
        ],

        "P_Value": [
            pearson_p,
            spearman_p
        ],

        "Number_of_Voxels": [
            len(all_errors),
            len(all_errors)
        ]
    })

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results.to_csv(
        CSV_FILE,
        index=False
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("CORRELATION RESULTS")
    print("=" * 60)

    print()

    print(results.to_string(
        index=False
    ))

    print()
    print(
        f"Total voxels analyzed: "
        f"{len(all_errors):,}"
    )

    print()
    print("Saved:")
    print(CSV_FILE)

    print()
    print("=" * 60)
    print("UNCERTAINTY-ERROR CORRELATION COMPLETE")
    print("=" * 60)


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()