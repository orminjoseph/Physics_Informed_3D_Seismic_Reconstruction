"""
======================================================================
UNCERTAINTY EVALUATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate whether predictive uncertainty is informative about
reconstruction difficulty.

This module evaluates:

    1. Aleatoric uncertainty
    2. Epistemic uncertainty from MC-Dropout
    3. Predictive uncertainty
    4. Predictive standard deviation
    5. Global reconstruction error
    6. Missing-region reconstruction error
    7. Observed-region reconstruction error
    8. Patch-level uncertainty/error correlation
    9. Observed-data preservation
   10. Measured missing-data rate

Important
---------
This module performs PATCH-LEVEL uncertainty/error analysis.

It does not perform formal uncertainty calibration.

Formal calibration is handled separately by:

    evaluation/uncertainty_calibration.py

Detailed voxel-level uncertainty/error correlation is handled
separately by:

    evaluation/uncertainty_error_correlation.py

The production Evaluator is used as the authoritative source for
reconstruction metrics and uncertainty decomposition.

Evaluation modes
----------------
FULL_F3
    Evaluates all available F3 patches.

DIAGNOSTIC_SUBSET
    Evaluates the number of patches specified by
    F3_UNCERTAINTY_NUM_PATCHES.

Outputs
-------
The following files are saved to the configured REPORT_DIR:

    uncertainty_evaluation.csv
    uncertainty_error_correlation.csv
    uncertainty_evaluation_metadata.json

Author: Ormin Joseph
======================================================================
"""


# =====================================================================
# 1. STANDARD-LIBRARY IMPORTS
# =====================================================================

import csv
import json
import os
import random
from datetime import datetime, timezone


# =====================================================================
# 2. SCIENTIFIC-COMPUTING IMPORTS
# =====================================================================

import numpy as np
import torch


# =====================================================================
# 3. STATISTICAL ANALYSIS
# =====================================================================

from scipy.stats import pearsonr
from scipy.stats import spearmanr


# =====================================================================
# 4. DATASET
# =====================================================================

from dataset.f3_dataset import F3Dataset


# =====================================================================
# 5. MODEL
# =====================================================================

from models.network import Network3D


# =====================================================================
# 6. INFERENCE AND EVALUATION
# =====================================================================

from inference.predictor import Predictor
from evaluation.evaluator import Evaluator


# =====================================================================
# 7. CENTRAL PROJECT CONFIGURATION
# =====================================================================

from utils.config import (
    F3_PATH,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY,

    CHECKPOINT_DIR,
    REPORT_DIR,

    MC_DROPOUT_SAMPLES,
    DEVICE,

    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,

    OBSERVED_PRESERVATION_TOLERANCE,

    F3_UNCERTAINTY_NUM_PATCHES,
    UNCERTAINTY_EVALUATION_SEED,
)


# =====================================================================
# 8. REPRODUCIBILITY
# =====================================================================

def set_reproducibility_seed(seed):
    """
    Set random seeds for reproducible evaluation.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    # Python random generator.
    random.seed(seed)

    # NumPy random generator.
    np.random.seed(seed)

    # PyTorch CPU generator.
    torch.manual_seed(seed)

    # PyTorch CUDA generator.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 9. SAFE SCALAR CONVERSION
# =====================================================================

def to_float(value):
    """
    Convert a scalar tensor or numerical scalar to Python float.

    Parameters
    ----------
    value : scalar tensor or numerical value

    Returns
    -------
    float
    """

    if torch.is_tensor(value):

        return float(
            value.detach().cpu().item()
        )

    return float(value)


# =====================================================================
# 10. SAFE CORRELATION
# =====================================================================

def safe_correlation(
    uncertainty_values,
    error_values,
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
        Pearson and Spearman coefficients,
        p-values, and number of valid observations.

    Notes
    -----
    Correlation is undefined if either variable is constant.
    In that situation NaN is returned rather than creating
    an artificial numerical result.
    """

    # ---------------------------------------------------------------
    # Convert inputs to NumPy arrays.
    # ---------------------------------------------------------------

    uncertainty_values = np.asarray(
        uncertainty_values,
        dtype=np.float64,
    )

    error_values = np.asarray(
        error_values,
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # Check matching dimensions.
    # ---------------------------------------------------------------

    if uncertainty_values.shape != error_values.shape:

        raise ValueError(
            "Uncertainty and error arrays must have identical "
            "shapes. "
            f"Uncertainty shape = "
            f"{uncertainty_values.shape}; "
            f"Error shape = "
            f"{error_values.shape}."
        )

    # ---------------------------------------------------------------
    # Keep only finite paired observations.
    # ---------------------------------------------------------------

    valid = (
        np.isfinite(uncertainty_values)
        &
        np.isfinite(error_values)
    )

    uncertainty_values = uncertainty_values[valid]

    error_values = error_values[valid]

    # ---------------------------------------------------------------
    # At least two observations are required.
    # ---------------------------------------------------------------

    if len(uncertainty_values) < 2:

        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": len(uncertainty_values),
        }

    # ---------------------------------------------------------------
    # Correlation is undefined for constant variables.
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
            "n": len(uncertainty_values),
        }

    # ---------------------------------------------------------------
    # Pearson correlation.
    # ---------------------------------------------------------------

    pearson_result = pearsonr(
        uncertainty_values,
        error_values,
    )

    # ---------------------------------------------------------------
    # Spearman rank correlation.
    # ---------------------------------------------------------------

    spearman_result = spearmanr(
        uncertainty_values,
        error_values,
    )

    # ---------------------------------------------------------------
    # Return standardized results.
    # ---------------------------------------------------------------

    return {
        "pearson_r":
            float(pearson_result.statistic),

        "pearson_p":
            float(pearson_result.pvalue),

        "spearman_rho":
            float(spearman_result.statistic),

        "spearman_p":
            float(spearman_result.pvalue),

        "n":
            len(uncertainty_values),
    }


# =====================================================================
# 11. SINGLE-SAMPLE DATASET ADAPTER
# =====================================================================

class SingleSampleDataset(torch.utils.data.Dataset):
    """
    Wrap one F3 sample in the dictionary format expected by
    the production Evaluator.

    The production Evaluator expects:

        {
            "input":  [C,D,H,W],
            "target": [C,D,H,W],
            "mask":   [C,D,H,W]
        }

    DataLoader adds the batch dimension:

        [B,C,D,H,W]
    """

    def __init__(
        self,
        corrupted,
        target,
        mask,
    ):

        self.sample = {
            "input": corrupted,
            "target": target,
            "mask": mask,
        }

    def __len__(self):

        return 1

    def __getitem__(
        self,
        index,
    ):

        return self.sample


# =====================================================================
# 12. F3 SAMPLE VALIDATION
# =====================================================================

def validate_f3_sample(
    corrupted,
    target,
    mask,
    patch_index,
):
    """
    Validate one F3 dataset sample before evaluation.
    """

    # ---------------------------------------------------------------
    # Tensor-type validation.
    # ---------------------------------------------------------------

    if not torch.is_tensor(corrupted):

        raise TypeError(
            f"Corrupted input for patch {patch_index} "
            "is not a torch.Tensor."
        )

    if not torch.is_tensor(target):

        raise TypeError(
            f"Target for patch {patch_index} "
            "is not a torch.Tensor."
        )

    if not torch.is_tensor(mask):

        raise TypeError(
            f"Mask for patch {patch_index} "
            "is not a torch.Tensor."
        )

    # ---------------------------------------------------------------
    # Dimension validation.
    #
    # Individual dataset samples must be:
    #
    #     [C,D,H,W]
    # ---------------------------------------------------------------

    if corrupted.ndim != 4:

        raise ValueError(
            f"Corrupted input for patch {patch_index} "
            "must have shape [C,D,H,W]. "
            f"Received {tuple(corrupted.shape)}."
        )

    if target.ndim != 4:

        raise ValueError(
            f"Target for patch {patch_index} "
            "must have shape [C,D,H,W]. "
            f"Received {tuple(target.shape)}."
        )

    if mask.ndim != 4:

        raise ValueError(
            f"Mask for patch {patch_index} "
            "must have shape [C,D,H,W]. "
            f"Received {tuple(mask.shape)}."
        )

    # ---------------------------------------------------------------
    # Shape consistency.
    # ---------------------------------------------------------------

    if not (
        corrupted.shape
        ==
        target.shape
        ==
        mask.shape
    ):

        raise ValueError(
            f"Input, target, and mask shapes do not match "
            f"for patch {patch_index}. "
            f"Input={tuple(corrupted.shape)}, "
            f"Target={tuple(target.shape)}, "
            f"Mask={tuple(mask.shape)}."
        )

    # ---------------------------------------------------------------
    # Finite-value validation.
    # ---------------------------------------------------------------

    if not torch.isfinite(corrupted).all():

        raise ValueError(
            f"Non-finite values detected in corrupted "
            f"input for patch {patch_index}."
        )

    if not torch.isfinite(target).all():

        raise ValueError(
            f"Non-finite values detected in target "
            f"for patch {patch_index}."
        )

    if not torch.isfinite(mask).all():

        raise ValueError(
            f"Non-finite values detected in mask "
            f"for patch {patch_index}."
        )

    # ---------------------------------------------------------------
    # Binary-mask validation.
    # ---------------------------------------------------------------

    unique_values = torch.unique(mask)

    valid_mask = torch.all(
        (unique_values == 0)
        |
        (unique_values == 1)
    )

    if not valid_mask:

        raise ValueError(
            f"Mask for patch {patch_index} must contain only "
            f"0 and 1. Found {unique_values.tolist()}."
        )


# =====================================================================
# 13. MAIN EVALUATION
# =====================================================================

def main():

    # =================================================================
    # REPRODUCIBILITY
    # =================================================================

    set_reproducibility_seed(
        UNCERTAINTY_EVALUATION_SEED
    )

    # =================================================================
    # HEADER
    # =================================================================

    print()
    print("=" * 78)
    print(
        "UNCERTAINTY–RECONSTRUCTION ERROR EVALUATION"
    )
    print("=" * 78)

    # =================================================================
    # DETERMINE EVALUATION MODE
    # =================================================================

    if F3_UNCERTAINTY_NUM_PATCHES is None:

        evaluation_mode = "FULL_F3"

    else:

        if (
            not isinstance(
                F3_UNCERTAINTY_NUM_PATCHES,
                int,
            )
            or
            F3_UNCERTAINTY_NUM_PATCHES <= 0
        ):

            raise ValueError(
                "F3_UNCERTAINTY_NUM_PATCHES must be "
                "a positive integer or None."
            )

        evaluation_mode = "DIAGNOSTIC_SUBSET"

    # =================================================================
    # DISPLAY CONFIGURATION
    # =================================================================

    print()
    print("Configuration")
    print("-" * 78)

    print(
        f"Device                    : "
        f"{DEVICE}"
    )

    print(
        f"MC-Dropout samples        : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Attention                 : "
        f"{USE_ATTENTION}"
    )

    print(
        f"Residual                  : "
        f"{USE_RESIDUAL}"
    )

    print(
        f"Uncertainty               : "
        f"{USE_UNCERTAINTY}"
    )

    print(
        f"F3 patch size             : "
        f"{F3_PATCH_SIZE}"
    )

    print(
        f"F3 stride                 : "
        f"{F3_STRIDE}"
    )

    print(
        f"Missing probability       : "
        f"{F3_MISSING_PROBABILITY}"
    )

    print(
        f"Evaluation seed           : "
        f"{UNCERTAINTY_EVALUATION_SEED}"
    )

    print(
        f"Evaluation mode           : "
        f"{evaluation_mode}"
    )

    if F3_UNCERTAINTY_NUM_PATCHES is None:

        print(
            "Patch limit               : "
            "ALL AVAILABLE PATCHES"
        )

    else:

        print(
            f"Patch limit               : "
            f"{F3_UNCERTAINTY_NUM_PATCHES}"
        )

    # =================================================================
    # CHECKPOINT
    # =================================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth",
    )

    print()
    print(
        f"Checkpoint                : "
        f"{checkpoint}"
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            "The trained model checkpoint was not found:\n"
            f"{checkpoint}\n\n"
            "Ensure that best_model.pth exists in the "
            "configured checkpoint directory."
        )

    # =================================================================
    # BUILD F3 DATASET
    # =================================================================

    print()
    print("=" * 78)
    print("BUILDING F3 DATASET")
    print("=" * 78)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=F3_PATCH_SIZE,
        stride=F3_STRIDE,
        missing_probability=F3_MISSING_PROBABILITY,
    )

    # =================================================================
    # DETERMINE NUMBER OF PATCHES
    # =================================================================

    if len(dataset) == 0:

        raise RuntimeError(
            "The F3 dataset contains no available patches."
        )

    if F3_UNCERTAINTY_NUM_PATCHES is None:

        number_of_patches = len(dataset)

    else:

        number_of_patches = min(
            F3_UNCERTAINTY_NUM_PATCHES,
            len(dataset),
        )

    print()
    print(
        f"Available F3 patches       : "
        f"{len(dataset)}"
    )

    print(
        f"Patches to evaluate        : "
        f"{number_of_patches}"
    )

    # =================================================================
    # BUILD MODEL
    # =================================================================

    print()
    print("=" * 78)
    print("BUILDING PRODUCTION MODEL")
    print("=" * 78)

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    # =================================================================
    # LOAD BEST CHECKPOINT
    # =================================================================

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=DEVICE,
    )

    trained_model = predictor.model

    # =================================================================
    # CREATE PRODUCTION EVALUATOR
    # =================================================================

    evaluator = Evaluator(
        model=trained_model,
        device=DEVICE,
        mc_samples=MC_DROPOUT_SAMPLES,
    )

    # =================================================================
    # STORAGE
    # =================================================================

    results = []

    # =================================================================
    # PATCH-BY-PATCH EVALUATION
    # =================================================================

    for patch_index in range(
        number_of_patches
    ):

        print()
        print("-" * 78)

        print(
            f"Evaluating patch "
            f"{patch_index + 1}/"
            f"{number_of_patches}"
        )

        # -------------------------------------------------------------
        # Retrieve sample.
        # -------------------------------------------------------------

        sample = dataset[
            patch_index
        ]

        # -------------------------------------------------------------
        # Current F3Dataset structure:
        #
        #     input
        #     target
        #     mask
        #     velocity
        #
        # Only input, target, and mask are required by Evaluator.
        # -------------------------------------------------------------

        corrupted = sample[0]
        target = sample[1]
        mask = sample[2]

        # -------------------------------------------------------------
        # Validate sample.
        # -------------------------------------------------------------

        validate_f3_sample(
            corrupted=corrupted,
            target=target,
            mask=mask,
            patch_index=patch_index,
        )

        # -------------------------------------------------------------
        # Build Evaluator-compatible dataset.
        # -------------------------------------------------------------

        single_sample_dataset = SingleSampleDataset(
            corrupted=corrupted,
            target=target,
            mask=mask,
        )

        # -------------------------------------------------------------
        # Build DataLoader.
        # -------------------------------------------------------------

        dataloader = torch.utils.data.DataLoader(
            single_sample_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # -------------------------------------------------------------
        # Run production evaluation.
        # -------------------------------------------------------------

        evaluation_result = evaluator.evaluate(
            dataloader
        )

        # =============================================================
        # RECONSTRUCTION METRICS
        # =============================================================

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

        # =============================================================
        # MISSING-REGION METRICS
        # =============================================================

        missing_mae = to_float(
            evaluation_result["missing_mae"]
        )

        missing_rmse = to_float(
            evaluation_result["missing_rmse"]
        )

        # =============================================================
        # OBSERVED-REGION METRICS
        # =============================================================

        observed_mae = to_float(
            evaluation_result["observed_mae"]
        )

        observed_rmse = to_float(
            evaluation_result["observed_rmse"]
        )

        # =============================================================
        # UNCERTAINTY DECOMPOSITION
        # =============================================================

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

        # =============================================================
        # QUALITY CONTROL
        # =============================================================

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

        # =============================================================
        # STORE RESULT
        # =============================================================

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
                MC_DROPOUT_SAMPLES,
        }

        results.append(
            row
        )

        # =============================================================
        # DISPLAY PATCH RESULT
        # =============================================================

        print(
            f"MAE                       : "
            f"{global_mae:.6f}"
        )

        print(
            f"Missing MAE               : "
            f"{missing_mae:.6f}"
        )

        print(
            f"Missing RMSE              : "
            f"{missing_rmse:.6f}"
        )

        print(
            f"Aleatoric variance        : "
            f"{aleatoric_variance:.6f}"
        )

        print(
            f"Epistemic variance        : "
            f"{epistemic_variance:.10e}"
        )

        print(
            f"Predictive variance       : "
            f"{predictive_variance:.6f}"
        )

        print(
            f"Predictive standard dev.  : "
            f"{predictive_std:.6f}"
        )

        print(
            f"Measured missing rate     : "
            f"{measured_missing_rate:.6f}"
        )

        print(
            f"Observed preservation     : "
            f"{observed_preservation_error:.6e}"
        )

    # =================================================================
    # ENSURE RESULTS EXIST
    # =================================================================

    if not results:

        raise RuntimeError(
            "No uncertainty evaluation results were generated."
        )

    # =================================================================
    # CONVERT RESULTS TO ARRAYS
    # =================================================================

    predictive_uncertainty = np.array(
        [
            row["Predictive_Std"]
            for row in results
        ],
        dtype=np.float64,
    )

    aleatoric_uncertainty = np.sqrt(
        np.maximum(
            np.array(
                [
                    row["Aleatoric_Variance"]
                    for row in results
                ],
                dtype=np.float64,
            ),
            0.0,
        )
    )

    epistemic_uncertainty = np.sqrt(
        np.maximum(
            np.array(
                [
                    row["Epistemic_Variance"]
                    for row in results
                ],
                dtype=np.float64,
            ),
            0.0,
        )
    )

    global_mae_values = np.array(
        [
            row["MAE"]
            for row in results
        ],
        dtype=np.float64,
    )

    missing_mae_values = np.array(
        [
            row["Missing_MAE"]
            for row in results
        ],
        dtype=np.float64,
    )

    global_rmse_values = np.array(
        [
            row["RMSE"]
            for row in results
        ],
        dtype=np.float64,
    )

    missing_rmse_values = np.array(
        [
            row["Missing_RMSE"]
            for row in results
        ],
        dtype=np.float64,
    )

    observed_preservation_values = np.array(
        [
            row[
                "Observed_Preservation_Error"
            ]
            for row in results
        ],
        dtype=np.float64,
    )

    measured_missing_rates = np.array(
        [
            row[
                "Measured_Missing_Rate"
            ]
            for row in results
        ],
        dtype=np.float64,
    )

    # =================================================================
    # PATCH-LEVEL CORRELATION ANALYSIS
    # =================================================================

    print()
    print("=" * 78)
    print("PATCH-LEVEL UNCERTAINTY–ERROR CORRELATION")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Predictive uncertainty vs global MAE.
    # -----------------------------------------------------------------

    global_mae_correlation = safe_correlation(
        predictive_uncertainty,
        global_mae_values,
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs missing-region MAE.
    # -----------------------------------------------------------------

    missing_mae_correlation = safe_correlation(
        predictive_uncertainty,
        missing_mae_values,
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs global RMSE.
    # -----------------------------------------------------------------

    global_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        global_rmse_values,
    )

    # -----------------------------------------------------------------
    # Predictive uncertainty vs missing-region RMSE.
    # -----------------------------------------------------------------

    missing_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        missing_rmse_values,
    )

    # =================================================================
    # DISPLAY CORRELATION RESULTS
    # =================================================================

    print()
    print("Predictive uncertainty vs Global MAE")

    print(
        f"  Pearson r       = "
        f"{global_mae_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Pearson p-value = "
        f"{global_mae_correlation['pearson_p']:.6e}"
    )

    print(
        f"  Spearman rho    = "
        f"{global_mae_correlation['spearman_rho']:.6f}"
    )

    print(
        f"  Spearman p-value= "
        f"{global_mae_correlation['spearman_p']:.6e}"
    )

    print()
    print("Predictive uncertainty vs Missing-region MAE")

    print(
        f"  Pearson r       = "
        f"{missing_mae_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Pearson p-value = "
        f"{missing_mae_correlation['pearson_p']:.6e}"
    )

    print(
        f"  Spearman rho    = "
        f"{missing_mae_correlation['spearman_rho']:.6f}"
    )

    print(
        f"  Spearman p-value= "
        f"{missing_mae_correlation['spearman_p']:.6e}"
    )

    print()
    print("Predictive uncertainty vs Global RMSE")

    print(
        f"  Pearson r       = "
        f"{global_rmse_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Spearman rho    = "
        f"{global_rmse_correlation['spearman_rho']:.6f}"
    )

    print()
    print("Predictive uncertainty vs Missing-region RMSE")

    print(
        f"  Pearson r       = "
        f"{missing_rmse_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Spearman rho    = "
        f"{missing_rmse_correlation['spearman_rho']:.6f}"
    )

    # =================================================================
    # CREATE REPORT DIRECTORY
    # =================================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # =================================================================
    # SAVE PATCH-LEVEL RESULTS
    # =================================================================

    csv_file = os.path.join(
        REPORT_DIR,
        "uncertainty_evaluation.csv",
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =================================================================
    # SAVE CORRELATION RESULTS
    # =================================================================

    correlation_file = os.path.join(
        REPORT_DIR,
        "uncertainty_error_correlation.csv",
    )

    correlation_rows = [

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Global_MAE",

            "N":
                global_mae_correlation["n"],

            "Pearson_r":
                global_mae_correlation["pearson_r"],

            "Pearson_p":
                global_mae_correlation["pearson_p"],

            "Spearman_rho":
                global_mae_correlation["spearman_rho"],

            "Spearman_p":
                global_mae_correlation["spearman_p"],
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Missing_MAE",

            "N":
                missing_mae_correlation["n"],

            "Pearson_r":
                missing_mae_correlation["pearson_r"],

            "Pearson_p":
                missing_mae_correlation["pearson_p"],

            "Spearman_rho":
                missing_mae_correlation["spearman_rho"],

            "Spearman_p":
                missing_mae_correlation["spearman_p"],
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
                global_rmse_correlation["spearman_p"],
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
                missing_rmse_correlation["spearman_p"],
        },
    ]

    with open(
        correlation_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=correlation_rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            correlation_rows
        )

    # =================================================================
    # QUALITY CONTROL
    # =================================================================

    all_uncertainty_values = np.concatenate(
        [
            predictive_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
        ]
    )

    all_error_values = np.concatenate(
        [
            global_mae_values,
            missing_mae_values,
            global_rmse_values,
            missing_rmse_values,
        ]
    )

    # -----------------------------------------------------------------
    # Finite-value check.
    # -----------------------------------------------------------------

    if not np.all(
        np.isfinite(
            all_uncertainty_values
        )
    ):

        raise RuntimeError(
            "Non-finite uncertainty values detected."
        )

    if not np.all(
        np.isfinite(
            all_error_values
        )
    ):

        raise RuntimeError(
            "Non-finite reconstruction-error values detected."
        )

    # -----------------------------------------------------------------
    # Non-negative uncertainty check.
    # -----------------------------------------------------------------

    if np.any(
        predictive_uncertainty < 0.0
    ):

        raise RuntimeError(
            "Negative predictive standard deviation detected."
        )

    if np.any(
        aleatoric_uncertainty < 0.0
    ):

        raise RuntimeError(
            "Negative aleatoric standard deviation detected."
        )

    if np.any(
        epistemic_uncertainty < 0.0
    ):

        raise RuntimeError(
            "Negative epistemic standard deviation detected."
        )

    # -----------------------------------------------------------------
    # Observed-data preservation check.
    # -----------------------------------------------------------------

    maximum_observed_error = np.max(
        observed_preservation_values
    )

    if (
        maximum_observed_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation tolerance exceeded.\n"
            f"Maximum observed preservation error = "
            f"{maximum_observed_error:.6e}\n"
            f"Configured tolerance = "
            f"{OBSERVED_PRESERVATION_TOLERANCE:.6e}"
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

    maximum_predictive_uncertainty = np.max(
        predictive_uncertainty
    )

    mean_missing_rate = np.mean(
        measured_missing_rates
    )

    # =================================================================
    # SAVE REPRODUCIBILITY METADATA
    # =================================================================

    metadata_file = os.path.join(
        REPORT_DIR,
        "uncertainty_evaluation_metadata.json",
    )

    metadata = {

        "evaluation": {

            "name":
                "uncertainty_evaluation",

            "evaluation_mode":
                evaluation_mode,

            "patches_available":
                len(dataset),

            "patches_evaluated":
                number_of_patches,

            "seed":
                UNCERTAINTY_EVALUATION_SEED,
        },

        "dataset": {

            "name":
                "F3",

            "path":
                F3_PATH,

            "patch_size":
                list(F3_PATCH_SIZE),

            "stride":
                list(F3_STRIDE),

            "missing_probability":
                F3_MISSING_PROBABILITY,
        },

        "model": {

            "architecture":
                "Physics-Informed 3D Encoder-Decoder",

            "attention":
                USE_ATTENTION,

            "residual":
                USE_RESIDUAL,

            "uncertainty":
                USE_UNCERTAINTY,

            "checkpoint":
                checkpoint,
        },

        "uncertainty": {

            "method":
                "MC-Dropout",

            "mc_samples":
                MC_DROPOUT_SAMPLES,

            "components": [
                "aleatoric",
                "epistemic",
                "predictive",
            ],
        },

        "quality_control": {

            "observed_preservation_tolerance":
                OBSERVED_PRESERVATION_TOLERANCE,

            "maximum_observed_preservation_error":
                float(
                    maximum_observed_error
                ),

            "mean_measured_missing_rate":
                float(
                    mean_missing_rate
                ),
        },

        "summary": {

            "mean_predictive_std":
                float(
                    mean_predictive_uncertainty
                ),

            "maximum_predictive_std":
                float(
                    maximum_predictive_uncertainty
                ),

            "mean_aleatoric_std":
                float(
                    mean_aleatoric_uncertainty
                ),

            "mean_epistemic_std":
                float(
                    mean_epistemic_uncertainty
                ),

            "mean_global_mae":
                float(
                    mean_global_mae
                ),

            "mean_missing_mae":
                float(
                    mean_missing_mae
                ),

            "mean_global_rmse":
                float(
                    mean_global_rmse
                ),

            "mean_missing_rmse":
                float(
                    mean_missing_rmse
                ),
        },

        "software": {

            "device":
                str(DEVICE),

            "pytorch_version":
                torch.__version__,

            "numpy_version":
                np.__version__,
        },

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
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
        f"Evaluation mode                 : "
        f"{evaluation_mode}"
    )

    print(
        f"Patches evaluated               : "
        f"{number_of_patches}"
    )

    print(
        f"MC-Dropout samples              : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Mean predictive std             : "
        f"{mean_predictive_uncertainty:.6f}"
    )

    print(
        f"Maximum predictive std          : "
        f"{maximum_predictive_uncertainty:.6f}"
    )

    print(
        f"Mean aleatoric std              : "
        f"{mean_aleatoric_uncertainty:.6f}"
    )

    print(
        f"Mean epistemic std              : "
        f"{mean_epistemic_uncertainty:.6e}"
    )

    print(
        f"Mean global MAE                 : "
        f"{mean_global_mae:.6f}"
    )

    print(
        f"Mean missing-region MAE         : "
        f"{mean_missing_mae:.6f}"
    )

    print(
        f"Mean global RMSE                : "
        f"{mean_global_rmse:.6f}"
    )

    print(
        f"Mean missing-region RMSE        : "
        f"{mean_missing_rmse:.6f}"
    )

    print(
        f"Mean measured missing rate      : "
        f"{mean_missing_rate:.6f}"
    )

    print(
        f"Maximum observed preservation   : "
        f"{maximum_observed_error:.6e}"
    )

    print()
    print(
        "Predictive uncertainty vs "
        "missing-region MAE:"
    )

    print(
        f"  Pearson r  = "
        f"{missing_mae_correlation['pearson_r']:.6f}"
    )

    print(
        f"  Spearman rho = "
        f"{missing_mae_correlation['spearman_rho']:.6f}"
    )

    print(
        f"  N = "
        f"{missing_mae_correlation['n']}"
    )

    print()
    print("Saved files:")
    print(
        f"  {csv_file}"
    )

    print(
        f"  {correlation_file}"
    )

    print(
        f"  {metadata_file}"
    )

    print()
    print("=" * 78)
    print(
        "UNCERTAINTY EVALUATION COMPLETE"
    )
    print("=" * 78)


# =====================================================================
# 13. SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()