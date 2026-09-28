"""
======================================================================
UNCERTAINTY EVALUATION
======================================================================

Evaluates the relationship between predictive uncertainty and
reconstruction error for the Physics-Informed 3D Encoder-Decoder
Framework with Predictive Uncertainty.

Purpose
-------
This module evaluates whether predictive uncertainty is informative
about reconstruction difficulty.

The analysis includes:

    1. Aleatoric uncertainty
    2. Epistemic uncertainty from MC-Dropout
    3. Predictive uncertainty
    4. Predictive standard deviation
    5. Global reconstruction error
    6. Missing-region reconstruction error
    7. Observed-region reconstruction error
    8. Pearson correlation between uncertainty and error
    9. Spearman correlation between uncertainty and error
   10. Patch-level uncertainty/error statistics
   11. Exact observed-data preservation
   12. Measured missing-data rate

Interpretation
--------------
A positive correlation between uncertainty and reconstruction error
indicates that the uncertainty estimate tends to increase in regions
where reconstruction is more difficult.

This module does NOT claim that correlation alone establishes
calibration. Formal calibration analysis is handled separately by:

    evaluation/uncertainty_calibration.py

The current production Evaluator is used so that uncertainty
calculation remains consistent across the evaluation framework.

Author: Ormin Joseph
======================================================================
"""

# ---------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------

import os
import csv

# ---------------------------------------------------------------------
# Scientific-computing imports
# ---------------------------------------------------------------------

import numpy as np
import torch

# ---------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------

from scipy.stats import (
    pearsonr,
    spearmanr
)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

from dataset.f3_dataset import F3Dataset

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

from models.network import Network3D

# ---------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------

from inference.predictor import Predictor

from evaluation.evaluator import Evaluator

# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------

from utils.config import (
    F3_PATH,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY,
    CHECKPOINT_DIR,
    REPORT_DIR,
    MC_DROPOUT_SAMPLES,
    DEVICE
)


# =====================================================================
# CONFIGURATION
# =====================================================================

# Number of F3 patches to evaluate.
#
# This is intentionally limited for a diagnostic evaluation run.
# The complete F3 evaluation should use the dedicated F3 validation
# modules when the full dataset is required.
NUM_TEST_PATCHES = 20


# Numerical tolerance used when checking observed-data preservation.
OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def safe_correlation(
    uncertainty_values,
    error_values
):
    """
    Calculate Pearson and Spearman correlations safely.

    Parameters
    ----------
    uncertainty_values : array-like
        Patch-level uncertainty values.

    error_values : array-like
        Corresponding patch-level reconstruction errors.

    Returns
    -------
    dict
        Pearson and Spearman correlation coefficients and p-values.

    Notes
    -----
    Correlation is undefined when either variable is constant.
    In that situation NaN is returned rather than manufacturing
    a numerical result.
    """

    uncertainty_values = np.asarray(
        uncertainty_values,
        dtype=np.float64
    )

    error_values = np.asarray(
        error_values,
        dtype=np.float64
    )

    # ---------------------------------------------------------------
    # Keep only finite paired observations.
    # ---------------------------------------------------------------

    valid = (
        np.isfinite(uncertainty_values)
        &
        np.isfinite(error_values)
    )

    uncertainty_values = uncertainty_values[
        valid
    ]

    error_values = error_values[
        valid
    ]

    # ---------------------------------------------------------------
    # Correlation requires at least two paired observations.
    # ---------------------------------------------------------------

    if len(uncertainty_values) < 2:

        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": len(uncertainty_values)
        }

    # ---------------------------------------------------------------
    # Correlation is undefined when either variable is constant.
    # ---------------------------------------------------------------

    if (
        np.std(uncertainty_values) == 0.0
        or
        np.std(error_values) == 0.0
    ):

        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": len(uncertainty_values)
        }

    # ---------------------------------------------------------------
    # Pearson correlation.
    # ---------------------------------------------------------------

    pearson_result = pearsonr(
        uncertainty_values,
        error_values
    )

    # ---------------------------------------------------------------
    # Spearman rank correlation.
    # ---------------------------------------------------------------

    spearman_result = spearmanr(
        uncertainty_values,
        error_values
    )

    return {
        "pearson_r": float(
            pearson_result.statistic
        ),

        "pearson_p": float(
            pearson_result.pvalue
        ),

        "spearman_rho": float(
            spearman_result.statistic
        ),

        "spearman_p": float(
            spearman_result.pvalue
        ),

        "n": len(uncertainty_values)
    }


def to_float(value):
    """
    Convert a scalar tensor or NumPy value to Python float.
    """

    if torch.is_tensor(value):

        return float(
            value.detach().cpu().item()
        )

    return float(value)


# =====================================================================
# MAIN EVALUATION
# =====================================================================

def main():

    print()
    print("=" * 78)
    print("UNCERTAINTY–RECONSTRUCTION ERROR EVALUATION")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Display configuration.
    # -----------------------------------------------------------------

    print()
    print("Configuration")
    print("-" * 78)

    print(
        f"Device                : {DEVICE}"
    )

    print(
        f"MC-Dropout samples    : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"F3 patch size         : "
        f"{F3_PATCH_SIZE}"
    )

    print(
        f"F3 stride             : "
        f"{F3_STRIDE}"
    )

    print(
        f"Missing probability   : "
        f"{F3_MISSING_PROBABILITY}"
    )

    print(
        f"Maximum test patches  : "
        f"{NUM_TEST_PATCHES}"
    )

    # -----------------------------------------------------------------
    # Construct checkpoint path from the project configuration.
    # -----------------------------------------------------------------

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print(
        f"Checkpoint            : "
        f"{checkpoint}"
    )

    # -----------------------------------------------------------------
    # Validate checkpoint.
    # -----------------------------------------------------------------

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(
            "Trained model checkpoint was not found:\n"
            f"{checkpoint}\n\n"
            "Train the model and ensure best_model.pth exists "
            "in the configured checkpoint directory before "
            "running uncertainty evaluation."
        )

    # -----------------------------------------------------------------
    # Build the F3 dataset.
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("BUILDING F3 DATASET")
    print("=" * 78)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=F3_PATCH_SIZE,
        stride=F3_STRIDE,
        missing_probability=F3_MISSING_PROBABILITY
    )

    number_of_patches = min(
        NUM_TEST_PATCHES,
        len(dataset)
    )

    if number_of_patches == 0:

        raise RuntimeError(
            "The F3 dataset contains no available patches."
        )

    print()
    print(
        f"Available F3 patches  : {len(dataset)}"
    )

    print(
        f"Patches to evaluate   : "
        f"{number_of_patches}"
    )

    # -----------------------------------------------------------------
    # Build the production model.
    # -----------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    # -----------------------------------------------------------------
    # Use Predictor only for robust checkpoint loading.
    #
    # Predictor loads the trained checkpoint into the model.
    # The production Evaluator then operates directly on the loaded
    # model so that MC-Dropout uncertainty is calculated correctly.
    # -----------------------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=DEVICE
    )

    trained_model = predictor.model

    # -----------------------------------------------------------------
    # Create the production Evaluator.
    # -----------------------------------------------------------------

    evaluator = Evaluator(
        model=trained_model,
        device=DEVICE,
        mc_samples=MC_DROPOUT_SAMPLES
    )

    # -----------------------------------------------------------------
    # Storage for patch-level results.
    # -----------------------------------------------------------------

    results = []

    # -----------------------------------------------------------------
    # Evaluate patches individually.
    #
    # Each patch is converted to the dictionary format expected by
    # the production Evaluator.
    # -----------------------------------------------------------------

    for patch_index in range(
        number_of_patches
    ):

        print()
        print(
            "-" * 78
        )

        print(
            f"Patch "
            f"{patch_index + 1}/"
            f"{number_of_patches}"
        )

        # -------------------------------------------------------------
        # Retrieve F3 patch.
        # -------------------------------------------------------------

        sample = dataset[
            patch_index
        ]

        # -------------------------------------------------------------
        # Current F3Dataset returns:
        #
        #     input
        #     target
        #     mask
        #     velocity
        #
        # We only require the first three for reconstruction and
        # uncertainty evaluation.
        # -------------------------------------------------------------

        corrupted = sample[0]
        target = sample[1]
        mask = sample[2]

        # -------------------------------------------------------------
        # Validate tensor dimensions.
        #
        # Individual dataset tensors should be:
        #
        #     [C, D, H, W]
        # -------------------------------------------------------------

        if not (
            torch.is_tensor(corrupted)
            and
            torch.is_tensor(target)
            and
            torch.is_tensor(mask)
        ):

            raise TypeError(
                "F3 dataset returned a non-tensor "
                "input, target, or mask."
            )

        if not (
            corrupted.ndim == 4
            and
            target.ndim == 4
            and
            mask.ndim == 4
        ):

            raise ValueError(
                "F3 tensors must have shape "
                "[C,D,H,W]. "
                f"Received input={tuple(corrupted.shape)}, "
                f"target={tuple(target.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        # -------------------------------------------------------------
        # Construct the dictionary expected by Evaluator.
        # -------------------------------------------------------------

        sample_dict = {
            "input": corrupted,
            "target": target,
            "mask": mask
        }

        # -------------------------------------------------------------
        # One-sample dataset.
        # -------------------------------------------------------------

        class SingleSampleDataset(
            torch.utils.data.Dataset
        ):

            def __len__(self):
                return 1

            def __getitem__(
                self,
                index
            ):
                return sample_dict

        # -------------------------------------------------------------
        # DataLoader creates:
        #
        #     [B,C,D,H,W]
        #
        # from:
        #
        #     [C,D,H,W]
        # -------------------------------------------------------------

        dataloader = torch.utils.data.DataLoader(
            SingleSampleDataset(),
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        # -------------------------------------------------------------
        # Run the production Evaluator.
        # -------------------------------------------------------------

        evaluation_result = evaluator.evaluate(
            dataloader
        )

        # -------------------------------------------------------------
        # Extract global reconstruction metrics.
        # -------------------------------------------------------------

        global_mae = to_float(
            evaluation_result["mae"]
        )

        global_rmse = to_float(
            evaluation_result["rmse"]
        )

        global_psnr = to_float(
            evaluation_result["psnr"]
        )

        global_snr = to_float(
            evaluation_result["snr"]
        )

        global_ssim = to_float(
            evaluation_result["ssim"]
        )

        # -------------------------------------------------------------
        # Extract missing-region metrics.
        # -------------------------------------------------------------

        missing_mae = to_float(
            evaluation_result["missing_mae"]
        )

        missing_rmse = to_float(
            evaluation_result["missing_rmse"]
        )

        # -------------------------------------------------------------
        # Extract observed-region metrics.
        # -------------------------------------------------------------

        observed_mae = to_float(
            evaluation_result["observed_mae"]
        )

        observed_rmse = to_float(
            evaluation_result["observed_rmse"]
        )

        # -------------------------------------------------------------
        # Extract uncertainty decomposition.
        # -------------------------------------------------------------

        aleatoric_variance = to_float(
            evaluation_result[
                "aleatoric_variance"
            ]
        )

        epistemic_variance = to_float(
            evaluation_result[
                "epistemic_variance"
            ]
        )

        predictive_variance = to_float(
            evaluation_result[
                "predictive_variance"
            ]
        )

        predictive_std = to_float(
            evaluation_result[
                "predictive_std"
            ]
        )

        # -------------------------------------------------------------
        # Extract quality-control statistics.
        # -------------------------------------------------------------

        observed_preservation_error = to_float(
            evaluation_result[
                "observed_preservation_error"
            ]
        )

        measured_missing_rate = to_float(
            evaluation_result[
                "measured_missing_rate"
            ]
        )

        # -------------------------------------------------------------
        # Store one row for this patch.
        # -------------------------------------------------------------

        row = {

            "Patch":
                patch_index,

            "MAE":
                global_mae,

            "RMSE":
                global_rmse,

            "PSNR":
                global_psnr,

            "SNR":
                global_snr,

            "SSIM":
                global_ssim,

            "Missing_MAE":
                missing_mae,

            "Missing_RMSE":
                missing_rmse,

            "Observed_MAE":
                observed_mae,

            "Observed_RMSE":
                observed_rmse,

            "Aleatoric_Variance":
                aleatoric_variance,

            "Epistemic_Variance":
                epistemic_variance,

            "Predictive_Variance":
                predictive_variance,

            "Predictive_Std":
                predictive_std,

            "Observed_Preservation_Error":
                observed_preservation_error,

            "Measured_Missing_Rate":
                measured_missing_rate,

            "MC_Samples":
                MC_DROPOUT_SAMPLES
        }

        results.append(
            row
        )

        # -------------------------------------------------------------
        # Display patch results.
        # -------------------------------------------------------------

        print(
            f"MAE                  : "
            f"{global_mae:.6f}"
        )

        print(
            f"Missing MAE          : "
            f"{missing_mae:.6f}"
        )

        print(
            f"Missing RMSE         : "
            f"{missing_rmse:.6f}"
        )

        print(
            f"Aleatoric variance   : "
            f"{aleatoric_variance:.6f}"
        )

        print(
            f"Epistemic variance   : "
            f"{epistemic_variance:.10e}"
        )

        print(
            f"Predictive variance  : "
            f"{predictive_variance:.6f}"
        )

        print(
            f"Predictive std       : "
            f"{predictive_std:.6f}"
        )

        print(
            f"Missing rate         : "
            f"{measured_missing_rate:.6f}"
        )

        print(
            f"Observed preservation: "
            f"{observed_preservation_error:.6e}"
        )

    # =================================================================
    # VALIDATE RESULTS
    # =================================================================

    if not results:

        raise RuntimeError(
            "No uncertainty evaluation results were generated."
        )

    # -----------------------------------------------------------------
    # Convert patch-level values to arrays.
    # -----------------------------------------------------------------

    predictive_uncertainty = np.array(
        [
            row["Predictive_Std"]
            for row in results
        ],
        dtype=np.float64
    )

    aleatoric_uncertainty = np.array(
        [
            np.sqrt(
                max(
                    row["Aleatoric_Variance"],
                    0.0
                )
            )
            for row in results
        ],
        dtype=np.float64
    )

    epistemic_uncertainty = np.array(
        [
            np.sqrt(
                max(
                    row["Epistemic_Variance"],
                    0.0
                )
            )
            for row in results
        ],
        dtype=np.float64
    )

    global_mae_values = np.array(
        [
            row["MAE"]
            for row in results
        ],
        dtype=np.float64
    )

    missing_mae_values = np.array(
        [
            row["Missing_MAE"]
            for row in results
        ],
        dtype=np.float64
    )

    global_rmse_values = np.array(
        [
            row["RMSE"]
            for row in results
        ],
        dtype=np.float64
    )

    missing_rmse_values = np.array(
        [
            row["Missing_RMSE"]
            for row in results
        ],
        dtype=np.float64
    )

    # =================================================================
    # CORRELATION ANALYSIS
    # =================================================================

    print()
    print("=" * 78)
    print("UNCERTAINTY–ERROR CORRELATION")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Predictive uncertainty vs global MAE.
    # -----------------------------------------------------------------

    global_correlation = safe_correlation(
        predictive_uncertainty,
        global_mae_values
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs missing-region MAE.
    # -----------------------------------------------------------------

    missing_correlation = safe_correlation(
        predictive_uncertainty,
        missing_mae_values
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs global RMSE.
    # -----------------------------------------------------------------

    global_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        global_rmse_values
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs missing-region RMSE.
    # -----------------------------------------------------------------

    missing_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        missing_rmse_values
    )

    print()
    print(
        "Predictive uncertainty vs Global MAE"
    )

    print(
        f"Pearson r       : "
        f"{global_correlation['pearson_r']:.6f}"
    )

    print(
        f"Pearson p-value : "
        f"{global_correlation['pearson_p']:.6e}"
    )

    print(
        f"Spearman rho    : "
        f"{global_correlation['spearman_rho']:.6f}"
    )

    print(
        f"Spearman p-value: "
        f"{global_correlation['spearman_p']:.6e}"
    )

    print()
    print(
        "Predictive uncertainty vs Missing-region MAE"
    )

    print(
        f"Pearson r       : "
        f"{missing_correlation['pearson_r']:.6f}"
    )

    print(
        f"Pearson p-value : "
        f"{missing_correlation['pearson_p']:.6e}"
    )

    print(
        f"Spearman rho    : "
        f"{missing_correlation['spearman_rho']:.6f}"
    )

    print(
        f"Spearman p-value: "
        f"{missing_correlation['spearman_p']:.6e}"
    )

    print()
    print(
        "Predictive uncertainty vs Global RMSE"
    )

    print(
        f"Pearson r       : "
        f"{global_rmse_correlation['pearson_r']:.6f}"
    )

    print(
        f"Spearman rho    : "
        f"{global_rmse_correlation['spearman_rho']:.6f}"
    )

    print()
    print(
        "Predictive uncertainty vs Missing-region RMSE"
    )

    print(
        f"Pearson r       : "
        f"{missing_rmse_correlation['pearson_r']:.6f}"
    )

    print(
        f"Spearman rho    : "
        f"{missing_rmse_correlation['spearman_rho']:.6f}"
    )

    # =================================================================
    # SAVE PATCH-LEVEL RESULTS
    # =================================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    csv_file = os.path.join(
        REPORT_DIR,
        "uncertainty_evaluation.csv"
    )

    # -----------------------------------------------------------------
    # Write patch-level CSV.
    # -----------------------------------------------------------------

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =================================================================
    # SAVE CORRELATION SUMMARY
    # =================================================================

    correlation_file = os.path.join(
        REPORT_DIR,
        "uncertainty_error_correlation.csv"
    )

    correlation_rows = [

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Global_MAE",

            "N":
                global_correlation["n"],

            "Pearson_r":
                global_correlation["pearson_r"],

            "Pearson_p":
                global_correlation["pearson_p"],

            "Spearman_rho":
                global_correlation["spearman_rho"],

            "Spearman_p":
                global_correlation["spearman_p"]
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Missing_MAE",

            "N":
                missing_correlation["n"],

            "Pearson_r":
                missing_correlation["pearson_r"],

            "Pearson_p":
                missing_correlation["pearson_p"],

            "Spearman_rho":
                missing_correlation["spearman_rho"],

            "Spearman_p":
                missing_correlation["spearman_p"]
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Global_RMSE",

            "N":
                global_rmse_correlation["n"],

            "Pearson_r":
                global_rmse_correlation["pearson_r"],

            "Pearson_p":
                global_rmse_correlation["pearson_p"],

            "Spearman_rho":
                global_rmse_correlation["spearman_rho"],

            "Spearman_p":
                global_rmse_correlation["spearman_p"]
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Missing_RMSE",

            "N":
                missing_rmse_correlation["n"],

            "Pearson_r":
                missing_rmse_correlation["pearson_r"],

            "Pearson_p":
                missing_rmse_correlation["pearson_p"],

            "Spearman_rho":
                missing_rmse_correlation["spearman_rho"],

            "Spearman_p":
                missing_rmse_correlation["spearman_p"]
        }
    ]

    with open(
        correlation_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=correlation_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            correlation_rows
        )

    # =================================================================
    # SUMMARY STATISTICS
    # =================================================================

    mean_predictive_uncertainty = np.mean(
        predictive_uncertainty
    )

    mean_aleatoric_uncertainty = np.mean(
        aleatoric_uncertainty
    )

    mean_epistemic_uncertainty = np.mean(
        epistemic_uncertainty
    )

    mean_global_mae = np.mean(
        global_mae_values
    )

    mean_missing_mae = np.mean(
        missing_mae_values
    )

    mean_global_rmse = np.mean(
        global_rmse_values
    )

    mean_missing_rmse = np.mean(
        missing_rmse_values
    )

    max_predictive_uncertainty = np.max(
        predictive_uncertainty
    )

    # =================================================================
    # FINAL QUALITY-CONTROL CHECKS
    # =================================================================

    observed_errors = np.array(
        [
            row[
                "Observed_Preservation_Error"
            ]
            for row in results
        ],
        dtype=np.float64
    )

    measured_missing_rates = np.array(
        [
            row[
                "Measured_Missing_Rate"
            ]
            for row in results
        ],
        dtype=np.float64
    )

    # -----------------------------------------------------------------
    # Check finite values.
    # -----------------------------------------------------------------

    all_numeric_values = np.concatenate(
        [
            predictive_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            global_mae_values,
            missing_mae_values,
            global_rmse_values,
            missing_rmse_values
        ]
    )

    if not np.all(
        np.isfinite(
            all_numeric_values
        )
    ):

        raise RuntimeError(
            "Non-finite values were detected in "
            "the uncertainty evaluation results."
        )

    # -----------------------------------------------------------------
    # Check non-negative uncertainty.
    # -----------------------------------------------------------------

    if np.any(
        predictive_uncertainty < 0
    ):

        raise RuntimeError(
            "Negative predictive uncertainty detected."
        )

    if np.any(
        aleatoric_uncertainty < 0
    ):

        raise RuntimeError(
            "Negative aleatoric uncertainty detected."
        )

    if np.any(
        epistemic_uncertainty < 0
    ):

        raise RuntimeError(
            "Negative epistemic uncertainty detected."
        )

    # -----------------------------------------------------------------
    # Check observed-data preservation.
    # -----------------------------------------------------------------

    maximum_observed_error = np.max(
        observed_errors
    )

    if (
        maximum_observed_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation tolerance "
            "was exceeded.\n"
            f"Maximum error = "
            f"{maximum_observed_error:.6e}\n"
            f"Tolerance = "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.6e}"
        )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print("=" * 78)
    print("UNCERTAINTY EVALUATION SUMMARY")
    print("=" * 78)

    print()
    print(
        f"Patches evaluated              : "
        f"{len(results)}"
    )

    print(
        f"MC-Dropout samples             : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Mean predictive std            : "
        f"{mean_predictive_uncertainty:.6f}"
    )

    print(
        f"Maximum predictive std         : "
        f"{max_predictive_uncertainty:.6f}"
    )

    print(
        f"Mean aleatoric std             : "
        f"{mean_aleatoric_uncertainty:.6f}"
    )

    print(
        f"Mean epistemic std             : "
        f"{mean_epistemic_uncertainty:.6e}"
    )

    print(
        f"Mean global MAE                : "
        f"{mean_global_mae:.6f}"
    )

    print(
        f"Mean missing-region MAE        : "
        f"{mean_missing_mae:.6f}"
    )

    print(
        f"Mean global RMSE               : "
        f"{mean_global_rmse:.6f}"
    )

    print(
        f"Mean missing-region RMSE       : "
        f"{mean_missing_rmse:.6f}"
    )

    print(
        f"Mean measured missing rate    : "
        f"{np.mean(measured_missing_rates):.6f}"
    )

    print(
        f"Maximum observed preservation : "
        f"{maximum_observed_error:.6e}"
    )

    print()
    print(
        "Predictive uncertainty vs "
        "missing-region MAE:"
    )

    print(
        f"  Pearson r  = "
        f"{missing_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Spearman ρ = "
        f"{missing_correlation['spearman_rho']:.6f}"
    )

    print(
        f"  N          = "
        f"{missing_correlation['n']}"
    )

    print()
    print("Saved:")
    print(
        f"  {csv_file}"
    )

    print(
        f"  {correlation_file}"
    )

    print()
    print("=" * 78)
    print(
        "UNCERTAINTY EVALUATION COMPLETE"
    )
    print("=" * 78)


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()