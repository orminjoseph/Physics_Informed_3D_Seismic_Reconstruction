"""
======================================================================
UNCERTAINTY EVALUATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluates the relationship between predictive uncertainty and
reconstruction error using the ACTIVE DATASET selected in
utils.config.py.

The analysis includes:

    1. Aleatoric uncertainty
    2. Epistemic uncertainty from MC-Dropout
    3. Predictive uncertainty
    4. Predictive standard deviation
    5. Global reconstruction error
    6. Missing-region reconstruction error
    7. Observed-region reconstruction error
    8. Pearson correlation
    9. Spearman correlation
   10. Patch/sample-level uncertainty/error statistics
   11. Exact observed-data preservation
   12. Measured missing-data rate

Important
---------
This module does not construct an F3-specific dataset directly.

The active dataset is obtained through:

    from dataset.build_dataset import build_dataset

Therefore DATASET_MODE in utils.config.py controls whether the
evaluation uses:

    synthetic
    F3

The production Evaluator is retained so that uncertainty calculation
remains consistent with the rest of the evaluation framework.

Author: Ormin Joseph
======================================================================
"""

# ======================================================================
# 1. STANDARD LIBRARY
# ======================================================================

import os
import csv

# ======================================================================
# 2. SCIENTIFIC COMPUTING
# ======================================================================

import numpy as np
import torch

# ======================================================================
# 3. STATISTICS
# ======================================================================

from scipy.stats import (
    pearsonr,
    spearmanr,
)

# ======================================================================
# 4. DATASET
# ======================================================================

# IMPORTANT:
# Import the FUNCTION from the module, not the module itself.
from dataset.build_dataset import build_dataset

# ======================================================================
# 5. MODEL
# ======================================================================

from models.network import Network3D

# ======================================================================
# 6. INFERENCE / EVALUATION
# ======================================================================

from inference.predictor import Predictor
from evaluation.evaluator import Evaluator

# ======================================================================
# 7. CENTRALIZED PROJECT CONFIGURATION
# ======================================================================

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    MC_DROPOUT_SAMPLES,
    DEVICE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
)

# ======================================================================
# 8. EVALUATION CONFIGURATION
# ======================================================================

# None means evaluate every available sample.
#
# For a quick diagnostic test this may temporarily be changed to:
#
#     UNCERTAINTY_EVALUATION_NUM_SAMPLES = 2
#
# For the final thesis evaluation, keep it as None unless the
# configuration file intentionally specifies another value.
try:
    from utils.config import UNCERTAINTY_EVALUATION_NUM_SAMPLES
except ImportError:
    UNCERTAINTY_EVALUATION_NUM_SAMPLES = None


# Seed used to make the evaluation reproducible.
try:
    from utils.config import UNCERTAINTY_EVALUATION_SEED
except ImportError:
    UNCERTAINTY_EVALUATION_SEED = 42


# Numerical tolerance used when checking observed-data preservation.
OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# ======================================================================
# 9. DEVICE RESOLUTION
# ======================================================================

def resolve_device(device_setting):
    """
    Convert the centralized DEVICE configuration into an actual
    torch.device object.

    Supported values
    ----------------
    "cpu"
        Force CPU.

    "cuda"
        Require CUDA.

    "auto"
        Use CUDA when available; otherwise use CPU.

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """

    # --------------------------------------------------------------
    # Normalize the configuration value.
    # --------------------------------------------------------------

    device_setting = str(
        device_setting
    ).strip().lower()

    # --------------------------------------------------------------
    # Explicit CPU.
    # --------------------------------------------------------------

    if device_setting == "cpu":

        return torch.device(
            "cpu"
        )

    # --------------------------------------------------------------
    # Explicit CUDA.
    # --------------------------------------------------------------

    if device_setting == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' was requested, "
                "but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    # --------------------------------------------------------------
    # Automatic device selection.
    # --------------------------------------------------------------

    if device_setting == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    # --------------------------------------------------------------
    # Invalid configuration.
    # --------------------------------------------------------------

    raise ValueError(
        "Invalid DEVICE configuration: "
        f"{device_setting!r}. "
        "Expected 'cpu', 'cuda', or 'auto'."
    )


# ======================================================================
# 10. CORRELATION HELPER
# ======================================================================

def safe_correlation(
    uncertainty_values,
    error_values,
):
    """
    Calculate Pearson and Spearman correlations safely.

    Correlation is undefined when fewer than two valid observations
    exist or when either variable is constant.
    """

    uncertainty_values = np.asarray(
        uncertainty_values,
        dtype=np.float64,
    )

    error_values = np.asarray(
        error_values,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Keep only finite paired observations.
    # --------------------------------------------------------------

    valid = (
        np.isfinite(
            uncertainty_values
        )
        &
        np.isfinite(
            error_values
        )
    )

    uncertainty_values = (
        uncertainty_values[valid]
    )

    error_values = (
        error_values[valid]
    )

    # --------------------------------------------------------------
    # Correlation requires at least two observations.
    # --------------------------------------------------------------

    if len(
        uncertainty_values
    ) < 2:

        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": len(
                uncertainty_values
            ),
        }

    # --------------------------------------------------------------
    # Correlation is undefined for constant variables.
    # --------------------------------------------------------------

    if (
        np.std(
            uncertainty_values
        ) == 0.0
        or
        np.std(
            error_values
        ) == 0.0
    ):

        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": len(
                uncertainty_values
            ),
        }

    # --------------------------------------------------------------
    # Pearson correlation.
    # --------------------------------------------------------------

    pearson_result = pearsonr(
        uncertainty_values,
        error_values,
    )

    # --------------------------------------------------------------
    # Spearman correlation.
    # --------------------------------------------------------------

    spearman_result = spearmanr(
        uncertainty_values,
        error_values,
    )

    return {
        "pearson_r":
            float(
                pearson_result.statistic
            ),

        "pearson_p":
            float(
                pearson_result.pvalue
            ),

        "spearman_rho":
            float(
                spearman_result.statistic
            ),

        "spearman_p":
            float(
                spearman_result.pvalue
            ),

        "n":
            len(
                uncertainty_values
            ),
    }


# ======================================================================
# 11. SCALAR CONVERSION
# ======================================================================

def to_float(value):
    """
    Convert a scalar tensor or NumPy value to Python float.
    """

    if torch.is_tensor(
        value
    ):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    return float(
        value
    )


# ======================================================================
# 12. MAIN EVALUATION
# ======================================================================

def main():

    # ==============================================================
    # Resolve the actual PyTorch device.
    #
    # This is CRITICAL.
    #
    # DEVICE may be "auto", but Predictor cannot receive the literal
    # string "auto".
    # ==============================================================

    device = resolve_device(
        DEVICE
    )

    # ==============================================================
    # Reproducibility.
    # ==============================================================

    torch.manual_seed(
        UNCERTAINTY_EVALUATION_SEED
    )

    np.random.seed(
        UNCERTAINTY_EVALUATION_SEED
    )

    # ==============================================================
    # HEADER
    # ==============================================================

    print()
    print("=" * 78)
    print(
        "UNCERTAINTY–RECONSTRUCTION ERROR EVALUATION"
    )
    print("=" * 78)

    # ==============================================================
    # CONFIGURATION
    # ==============================================================

    print()
    print("Configuration")
    print("-" * 78)

    print(
        f"Experiment                 : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode               : "
        f"{DATASET_MODE}"
    )

    print(
        f"Configured device          : "
        f"{DEVICE}"
    )

    print(
        f"Resolved PyTorch device    : "
        f"{device}"
    )

    print(
        f"MC-Dropout samples         : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Attention                  : "
        f"{USE_ATTENTION}"
    )

    print(
        f"Residual connections       : "
        f"{USE_RESIDUAL}"
    )

    print(
        f"Uncertainty head           : "
        f"{USE_UNCERTAINTY}"
    )

    print(
        f"Evaluation seed            : "
        f"{UNCERTAINTY_EVALUATION_SEED}"
    )

    if (
        UNCERTAINTY_EVALUATION_NUM_SAMPLES
        is None
    ):

        print(
            "Sample limit               : "
            "ALL AVAILABLE SAMPLES"
        )

    else:

        print(
            f"Sample limit               : "
            f"{UNCERTAINTY_EVALUATION_NUM_SAMPLES}"
        )

    # ==============================================================
    # VALIDATE MC-DROPOUT CONFIGURATION
    # ==============================================================

    if (
        not isinstance(
            MC_DROPOUT_SAMPLES,
            int,
        )
        or
        MC_DROPOUT_SAMPLES < 2
    ):

        raise ValueError(
            "MC_DROPOUT_SAMPLES must be an "
            "integer >= 2 for uncertainty evaluation."
        )

    # ==============================================================
    # CHECKPOINT
    # ==============================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth",
    )

    print()
    print(
        f"Checkpoint                 : "
        f"{checkpoint}"
    )

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(
            "Trained model checkpoint was not found:\n"
            f"{checkpoint}\n\n"
            "Expected checkpoint:\n"
            f"{os.path.abspath(checkpoint)}"
        )

    # ==============================================================
    # BUILD ACTIVE DATASET
    # ==============================================================

    print()
    print("=" * 78)
    print(
        "BUILDING ACTIVE EVALUATION DATASET"
    )
    print("=" * 78)

    print()
    print(
        f"Dataset mode               : "
        f"{DATASET_MODE}"
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT construct F3Dataset directly here.
    #
    # build_dataset() reads DATASET_MODE from the centralized
    # configuration and creates the appropriate dataset.
    # --------------------------------------------------------------

    dataset = build_dataset()

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    total_samples = len(
        dataset
    )

    if total_samples == 0:

        raise RuntimeError(
            "The active evaluation dataset contains "
            "zero samples."
        )

    # --------------------------------------------------------------
    # Determine number of samples.
    # --------------------------------------------------------------

    if (
        UNCERTAINTY_EVALUATION_NUM_SAMPLES
        is None
    ):

        number_of_samples = (
            total_samples
        )

    else:

        if (
            not isinstance(
                UNCERTAINTY_EVALUATION_NUM_SAMPLES,
                int,
            )
            or
            UNCERTAINTY_EVALUATION_NUM_SAMPLES
            <= 0
        ):

            raise ValueError(
                "UNCERTAINTY_EVALUATION_NUM_SAMPLES "
                "must be None or a positive integer."
            )

        number_of_samples = min(
            UNCERTAINTY_EVALUATION_NUM_SAMPLES,
            total_samples,
        )

    print()
    print(
        f"Available samples          : "
        f"{total_samples}"
    )

    print(
        f"Samples to evaluate        : "
        f"{number_of_samples}"
    )

    # ==============================================================
    # BUILD PRODUCTION MODEL
    # ==============================================================

    print()
    print("=" * 78)
    print(
        "BUILDING PRODUCTION MODEL"
    )
    print("=" * 78)

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    print()
    print(
        "Network3D created successfully."
    )

    # ==============================================================
    # LOAD TRAINED CHECKPOINT
    # ==============================================================

    print()
    print("=" * 78)
    print(
        "LOADING TRAINED MODEL"
    )
    print("=" * 78)

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Pass the RESOLVED torch.device to Predictor.
    #
    # Do NOT pass:
    #
    #     device=DEVICE
    #
    # because DEVICE may be the string "auto".
    # --------------------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    trained_model = (
        predictor.model
    )

    # --------------------------------------------------------------
    # Ensure model is on the resolved device.
    # --------------------------------------------------------------

    trained_model = (
        trained_model.to(
            device
        )
    )

    # ==============================================================
    # CREATE PRODUCTION EVALUATOR
    # ==============================================================

    evaluator = Evaluator(
        model=trained_model,
        device=device,
        mc_samples=MC_DROPOUT_SAMPLES,
    )

    # ==============================================================
    # STORAGE
    # ==============================================================

    results = []

    # ==============================================================
    # SAMPLE-BY-SAMPLE EVALUATION
    # ==============================================================

    for sample_index in range(
        number_of_samples
    ):

        print()
        print("-" * 78)

        print(
            f"Sample "
            f"{sample_index + 1}/"
            f"{number_of_samples}"
        )

        # ----------------------------------------------------------
        # Retrieve sample.
        # ----------------------------------------------------------

        sample = dataset[
            sample_index
        ]

        if not isinstance(
            sample,
            (tuple, list),
        ):

            raise TypeError(
                "The active dataset must return "
                "a tuple or list containing at least "
                "input, target, and mask."
            )

        if len(
            sample
        ) < 3:

            raise ValueError(
                "The active dataset sample must contain "
                "at least input, target, and mask."
            )

        # ----------------------------------------------------------
        # Extract the three required components.
        #
        # Expected:
        #
        #     sample[0] = input
        #     sample[1] = target
        #     sample[2] = mask
        # ----------------------------------------------------------

        corrupted = sample[0]
        target = sample[1]
        mask = sample[2]

        # ----------------------------------------------------------
        # Validate tensors.
        # ----------------------------------------------------------

        if not (
            torch.is_tensor(
                corrupted
            )
            and
            torch.is_tensor(
                target
            )
            and
            torch.is_tensor(
                mask
            )
        ):

            raise TypeError(
                "The active dataset returned a non-tensor "
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
                "Dataset tensors must have shape "
                "[C,D,H,W]. "
                f"Received input="
                f"{tuple(corrupted.shape)}, "
                f"target="
                f"{tuple(target.shape)}, "
                f"mask="
                f"{tuple(mask.shape)}."
            )

        # ----------------------------------------------------------
        # Build Evaluator-compatible dictionary.
        # ----------------------------------------------------------

        sample_dict = {
            "input": corrupted,
            "target": target,
            "mask": mask,
        }

        # ----------------------------------------------------------
        # Preserve optional dataset information when available.
        # ----------------------------------------------------------

        if len(
            sample
        ) > 3:

            sample_dict[
                "velocity"
            ] = sample[3]

        if len(
            sample
        ) > 4:

            sample_dict[
                "mask_type"
            ] = sample[4]

        if len(
            sample
        ) > 5:

            sample_dict[
                "geological_mode"
            ] = sample[5]

        # ----------------------------------------------------------
        # One-sample Dataset adapter.
        # ----------------------------------------------------------

        class SingleSampleDataset(
            torch.utils.data.Dataset
        ):

            def __len__(
                self
            ):

                return 1

            def __getitem__(
                self,
                index,
            ):

                if index != 0:

                    raise IndexError(
                        "SingleSampleDataset contains "
                        "only one sample."
                    )

                return sample_dict

        # ----------------------------------------------------------
        # DataLoader.
        #
        # Dataset tensor:
        #
        #     [C,D,H,W]
        #
        # DataLoader:
        #
        #     [B,C,D,H,W]
        # ----------------------------------------------------------

        dataloader = (
            torch.utils.data.DataLoader(
                SingleSampleDataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )

        # ----------------------------------------------------------
        # Production evaluation.
        # ----------------------------------------------------------

        evaluation_result = (
            evaluator.evaluate(
                dataloader
            )
        )

        if not isinstance(
            evaluation_result,
            dict,
        ):

            raise TypeError(
                "Evaluator.evaluate() must return "
                "a dictionary."
            )

        # ----------------------------------------------------------
        # Extract metrics.
        # ----------------------------------------------------------

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

        missing_mae = to_float(
            evaluation_result[
                "missing_mae"
            ]
        )

        missing_rmse = to_float(
            evaluation_result[
                "missing_rmse"
            ]
        )

        observed_mae = to_float(
            evaluation_result[
                "observed_mae"
            ]
        )

        observed_rmse = to_float(
            evaluation_result[
                "observed_rmse"
            ]
        )

        # ----------------------------------------------------------
        # Extract uncertainty decomposition.
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Quality-control statistics.
        # ----------------------------------------------------------

        observed_preservation_error = (
            to_float(
                evaluation_result[
                    "observed_preservation_error"
                ]
            )
        )

        measured_missing_rate = (
            to_float(
                evaluation_result[
                    "measured_missing_rate"
                ]
            )
        )

        # ----------------------------------------------------------
        # Store result.
        # ----------------------------------------------------------

        results.append(
            {
                "Sample":
                    sample_index,

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
        )

        # ----------------------------------------------------------
        # Display current result.
        # ----------------------------------------------------------

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

    # ==============================================================
    # VALIDATION
    # ==============================================================

    if not results:

        raise RuntimeError(
            "No uncertainty evaluation results were generated."
        )

    # --------------------------------------------------------------
    # Convert values to arrays.
    # --------------------------------------------------------------

    predictive_uncertainty = np.array(
        [
            row["Predictive_Std"]
            for row in results
        ],
        dtype=np.float64,
    )

    aleatoric_uncertainty = np.array(
        [
            np.sqrt(
                max(
                    row[
                        "Aleatoric_Variance"
                    ],
                    0.0,
                )
            )
            for row in results
        ],
        dtype=np.float64,
    )

    epistemic_uncertainty = np.array(
        [
            np.sqrt(
                max(
                    row[
                        "Epistemic_Variance"
                    ],
                    0.0,
                )
            )
            for row in results
        ],
        dtype=np.float64,
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

    # ==============================================================
    # CORRELATIONS
    # ==============================================================

    global_correlation = safe_correlation(
        predictive_uncertainty,
        global_mae_values,
    )

    missing_correlation = safe_correlation(
        predictive_uncertainty,
        missing_mae_values,
    )

    global_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        global_rmse_values,
    )

    missing_rmse_correlation = safe_correlation(
        predictive_uncertainty,
        missing_rmse_values,
    )

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

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

    # ==============================================================
    # SAVE CORRELATION RESULTS
    # ==============================================================

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
                global_correlation["n"],

            "Pearson_r":
                global_correlation[
                    "pearson_r"
                ],

            "Pearson_p":
                global_correlation[
                    "pearson_p"
                ],

            "Spearman_rho":
                global_correlation[
                    "spearman_rho"
                ],

            "Spearman_p":
                global_correlation[
                    "spearman_p"
                ],
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Missing_MAE",

            "N":
                missing_correlation["n"],

            "Pearson_r":
                missing_correlation[
                    "pearson_r"
                ],

            "Pearson_p":
                missing_correlation[
                    "pearson_p"
                ],

            "Spearman_rho":
                missing_correlation[
                    "spearman_rho"
                ],

            "Spearman_p":
                missing_correlation[
                    "spearman_p"
                ],
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Global_RMSE",

            "N":
                global_rmse_correlation[
                    "n"
                ],

            "Pearson_r":
                global_rmse_correlation[
                    "pearson_r"
                ],

            "Pearson_p":
                global_rmse_correlation[
                    "pearson_p"
                ],

            "Spearman_rho":
                global_rmse_correlation[
                    "spearman_rho"
                ],

            "Spearman_p":
                global_rmse_correlation[
                    "spearman_p"
                ],
        },

        {
            "Uncertainty_Type":
                "Predictive_Std",

            "Error_Type":
                "Missing_RMSE",

            "N":
                missing_rmse_correlation[
                    "n"
                ],

            "Pearson_r":
                missing_rmse_correlation[
                    "pearson_r"
                ],

            "Pearson_p":
                missing_rmse_correlation[
                    "pearson_p"
                ],

            "Spearman_rho":
                missing_rmse_correlation[
                    "spearman_rho"
                ],

            "Spearman_p":
                missing_rmse_correlation[
                    "spearman_p"
                ],
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
            fieldnames=(
                correlation_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            correlation_rows
        )

    # ==============================================================
    # SUMMARY
    # ==============================================================

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

    # ==============================================================
    # QUALITY CONTROL
    # ==============================================================

    observed_errors = np.array(
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

    all_numeric_values = np.concatenate(
        [
            predictive_uncertainty,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            global_mae_values,
            missing_mae_values,
            global_rmse_values,
            missing_rmse_values,
        ]
    )

    if not np.all(
        np.isfinite(
            all_numeric_values
        )
    ):

        raise RuntimeError(
            "Non-finite values were detected "
            "in the uncertainty evaluation results."
        )

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

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================

    print()
    print("=" * 78)
    print(
        "UNCERTAINTY EVALUATION SUMMARY"
    )
    print("=" * 78)

    print()
    print(
        f"Dataset mode                  : "
        f"{DATASET_MODE}"
    )

    print(
        f"Samples evaluated             : "
        f"{len(results)}"
    )

    print(
        f"MC-Dropout samples             : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    print(
        f"Resolved device                : "
        f"{device}"
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
        f"  Spearman rho = "
        f"{missing_correlation['spearman_rho']:.6f}"
    )

    print(
        f"  N = "
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


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()