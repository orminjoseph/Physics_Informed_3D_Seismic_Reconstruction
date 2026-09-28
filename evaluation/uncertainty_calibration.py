"""
======================================================================
UNCERTAINTY CALIBRATION AND ERROR ALIGNMENT
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
This module evaluates whether predictive uncertainty is aligned with
reconstruction error.

The analysis includes:

1. Patch-level predictive uncertainty
2. Patch-level reconstruction error
3. Missing-region reconstruction error
4. Pearson correlation
5. Spearman correlation
6. Quantile-based uncertainty bins
7. Uncertainty-Error Alignment Gap (UEAG)
8. Observed-data preservation verification
9. Summary CSV files
10. Calibration plot

Important
---------
This module performs uncertainty-error alignment analysis.

UEAG is NOT Expected Calibration Error (ECE).

Formal probabilistic interval coverage should be evaluated separately
if required.

Author: Ormin Joseph
======================================================================
"""

# ======================================================================
# 1. IMPORTS
# ======================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    REPORT_DIR,
    BATCH_SIZE,
    MC_DROPOUT_SAMPLES,
    DEVICE,
)

from dataset.build_dataset import build_dataset

from models.network import Network3D

from inference.predictor import Predictor

from evaluation.evaluator import Evaluator


# ======================================================================
# 2. USER SETTINGS
# ======================================================================

# ----------------------------------------------------------------------
# Number of samples to analyse.
#
# None = use the complete dataset.
#
# For a quick debugging test, you may temporarily use:
#
# NUM_PATCHES = 5
#
# For final thesis evaluation:
#
# NUM_PATCHES = None
# ----------------------------------------------------------------------

NUM_PATCHES = None


# ----------------------------------------------------------------------
# Number of uncertainty bins.
# ----------------------------------------------------------------------

NUM_BINS = 10


# ----------------------------------------------------------------------
# Numerical tolerance.
# ----------------------------------------------------------------------

EPSILON = 1.0e-8


# ----------------------------------------------------------------------
# Expected observed-data preservation tolerance.
# ----------------------------------------------------------------------

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# ======================================================================
# 3. OUTPUT PATHS
# ======================================================================

# Convert REPORT_DIR into a Path object.
REPORT_PATH = Path(REPORT_DIR)

# Create the directory if it does not already exist.
REPORT_PATH.mkdir(parents=True, exist_ok=True)


# Output files.
PATCH_RESULTS_FILE = (
    REPORT_PATH / "uncertainty_calibration.csv"
)

BIN_RESULTS_FILE = (
    REPORT_PATH / "uncertainty_calibration_bins.csv"
)

SUMMARY_FILE = (
    REPORT_PATH / "uncertainty_calibration_summary.csv"
)

PLOT_FILE = (
    REPORT_PATH / "uncertainty_calibration.png"
)


# ======================================================================
# 4. DATASET ADAPTER
# ======================================================================

class SingleSampleDataset(torch.utils.data.Dataset):
    """
    Converts one dataset sample into the dictionary format expected
    by evaluation.Evaluator.
    """

    def __init__(self, sample):
        """
        Parameters
        ----------
        sample : tuple
            Expected dataset output:

            (
                input_cube,
                target_cube,
                mask,
                velocity,
                mask_type,
                geological_mode
            )
        """

        # Store the sample.
        self.sample = sample

    def __len__(self):
        """
        Return the number of samples.

        This adapter contains exactly one sample.
        """

        return 1

    def __getitem__(self, index):
        """
        Return one sample in Evaluator-compatible dictionary format.
        """

        # Prevent invalid indexing.
        if index != 0:
            raise IndexError("SingleSampleDataset contains one sample.")

        # Unpack the original dataset tuple.
        (
            input_cube,
            target_cube,
            mask,
            velocity,
            mask_type,
            geological_mode,
        ) = self.sample

        # Return the dictionary expected by Evaluator.
        return {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
            "velocity": velocity,
            "mask_type": mask_type,
            "geological_mode": geological_mode,
        }


# ======================================================================
# 5. NUMERICAL VALIDATION
# ======================================================================

def validate_finite_dataframe(dataframe):
    """
    Check that all numeric columns contain finite values.
    """

    # Select numeric columns only.
    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns

    # Check every numeric column.
    for column in numeric_columns:

        # Convert the column to NumPy values.
        values = dataframe[column].to_numpy(
            dtype=float
        )

        # Check for NaN or infinite values.
        if not np.all(np.isfinite(values)):

            raise ValueError(
                f"Non-finite values detected in column: {column}"
            )


# ======================================================================
# 6. CORRELATION ANALYSIS
# ======================================================================

def calculate_correlations(
    uncertainty,
    reconstruction_error,
    missing_error,
):
    """
    Calculate Pearson and Spearman correlations.

    Parameters
    ----------
    uncertainty : array-like
        Patch-level predictive standard deviation.

    reconstruction_error : array-like
        Patch-level global MAE.

    missing_error : array-like
        Patch-level missing-region MAE.

    Returns
    -------
    dict
        Correlation statistics.
    """

    # Convert all inputs to NumPy arrays.
    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    reconstruction_error = np.asarray(
        reconstruction_error,
        dtype=float,
    )

    missing_error = np.asarray(
        missing_error,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Pearson correlation:
    # uncertainty versus global reconstruction error.
    # --------------------------------------------------------------

    if len(uncertainty) >= 2:

        pearson_global, pearson_global_p = pearsonr(
            uncertainty,
            reconstruction_error,
        )

    else:

        pearson_global = np.nan
        pearson_global_p = np.nan

    # --------------------------------------------------------------
    # Spearman correlation:
    # uncertainty versus global reconstruction error.
    # --------------------------------------------------------------

    if len(uncertainty) >= 2:

        spearman_global, spearman_global_p = spearmanr(
            uncertainty,
            reconstruction_error,
        )

    else:

        spearman_global = np.nan
        spearman_global_p = np.nan

    # --------------------------------------------------------------
    # Pearson correlation:
    # uncertainty versus missing-region error.
    # --------------------------------------------------------------

    if len(uncertainty) >= 2:

        pearson_missing, pearson_missing_p = pearsonr(
            uncertainty,
            missing_error,
        )

    else:

        pearson_missing = np.nan
        pearson_missing_p = np.nan

    # --------------------------------------------------------------
    # Spearman correlation:
    # uncertainty versus missing-region error.
    # --------------------------------------------------------------

    if len(uncertainty) >= 2:

        spearman_missing, spearman_missing_p = spearmanr(
            uncertainty,
            missing_error,
        )

    else:

        spearman_missing = np.nan
        spearman_missing_p = np.nan

    # Return all statistics.
    return {
        "pearson_global_r": pearson_global,
        "pearson_global_p": pearson_global_p,
        "spearman_global_r": spearman_global,
        "spearman_global_p": spearman_global_p,
        "pearson_missing_r": pearson_missing,
        "pearson_missing_p": pearson_missing_p,
        "spearman_missing_r": spearman_missing,
        "spearman_missing_p": spearman_missing_p,
    }


# ======================================================================
# 7. UNCERTAINTY BINNING
# ======================================================================

def calculate_uncertainty_bins(
    dataframe,
    number_of_bins=10,
):
    """
    Divide patches into uncertainty quantile bins.

    The bins are based on predictive standard deviation.

    Returns
    -------
    calibration_table : pandas.DataFrame
        Statistics for every uncertainty bin.

    global_gap : float
        Mean absolute difference between normalized uncertainty
        and normalized global MAE.

    missing_gap : float
        Mean absolute difference between normalized uncertainty
        and normalized missing-region MAE.

    Notes
    -----
    These gaps are descriptive alignment measures.

    They are NOT Expected Calibration Error (ECE).
    """

    # Work on a copy so the original dataframe is not modified.
    data = dataframe.copy()

    # --------------------------------------------------------------
    # Create quantile-based bins.
    # --------------------------------------------------------------

    data["Uncertainty_Bin"] = pd.qcut(
        data["Predictive_Std"],
        q=number_of_bins,
        labels=False,
        duplicates="drop",
    )

    # --------------------------------------------------------------
    # Calculate statistics for each bin.
    # --------------------------------------------------------------

    calibration_table = (
        data
        .groupby(
            "Uncertainty_Bin",
            observed=True,
        )
        .agg(
            Number_of_Patches=(
                "Predictive_Std",
                "count",
            ),
            Mean_Predictive_Std=(
                "Predictive_Std",
                "mean",
            ),
            Mean_MAE=(
                "MAE",
                "mean",
            ),
            Mean_Missing_MAE=(
                "Missing_MAE",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------------
    # Calculate normalized alignment gaps.
    # --------------------------------------------------------------

    uncertainty_values = (
        calibration_table["Mean_Predictive_Std"]
        .to_numpy(dtype=float)
    )

    mae_values = (
        calibration_table["Mean_MAE"]
        .to_numpy(dtype=float)
    )

    missing_mae_values = (
        calibration_table["Mean_Missing_MAE"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------------
    # Normalization helper.
    # --------------------------------------------------------------

    def normalize(values):
        """
        Min-max normalization with protection against
        zero range.
        """

        minimum = np.min(values)
        maximum = np.max(values)

        denominator = maximum - minimum

        if denominator < EPSILON:
            return np.zeros_like(values)

        return (
            values - minimum
        ) / denominator

    # Normalize uncertainty.
    uncertainty_normalized = normalize(
        uncertainty_values
    )

    # Normalize global MAE.
    mae_normalized = normalize(
        mae_values
    )

    # Normalize missing-region MAE.
    missing_mae_normalized = normalize(
        missing_mae_values
    )

    # --------------------------------------------------------------
    # Calculate alignment gaps.
    # --------------------------------------------------------------

    global_gap = float(
        np.mean(
            np.abs(
                uncertainty_normalized
                - mae_normalized
            )
        )
    )

    missing_gap = float(
        np.mean(
            np.abs(
                uncertainty_normalized
                - missing_mae_normalized
            )
        )
    )

    # --------------------------------------------------------------
    # Add gap columns to the table.
    # --------------------------------------------------------------

    calibration_table[
        "Global_Alignment_Gap"
    ] = np.abs(
        uncertainty_normalized
        - mae_normalized
    )

    calibration_table[
        "Missing_Alignment_Gap"
    ] = np.abs(
        uncertainty_normalized
        - missing_mae_normalized
    )

    # Return all three outputs.
    return (
        calibration_table,
        global_gap,
        missing_gap,
    )


# ======================================================================
# 8. PLOT GENERATION
# ======================================================================

def generate_calibration_plot(
    dataframe,
    calibration_table,
):
    """
    Generate uncertainty-error alignment plots.
    """

    # Create a figure.
    plt.figure(
        figsize=(12, 5)
    )

    # --------------------------------------------------------------
    # Panel 1: Predictive uncertainty versus error.
    # --------------------------------------------------------------

    ax1 = plt.subplot(
        1,
        2,
        1,
    )

    ax1.scatter(
        dataframe["Predictive_Std"],
        dataframe["MAE"],
        alpha=0.7,
        label="Global MAE",
    )

    ax1.scatter(
        dataframe["Predictive_Std"],
        dataframe["Missing_MAE"],
        alpha=0.7,
        label="Missing-region MAE",
    )

    ax1.set_xlabel(
        "Predictive Standard Deviation"
    )

    ax1.set_ylabel(
        "Reconstruction Error"
    )

    ax1.set_title(
        "Uncertainty versus Reconstruction Error"
    )

    ax1.legend()

    ax1.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------------
    # Panel 2: Mean uncertainty and error by bin.
    # --------------------------------------------------------------

    ax2 = plt.subplot(
        1,
        2,
        2,
    )

    x = np.arange(
        len(calibration_table)
    )

    ax2.plot(
        x,
        calibration_table[
            "Mean_Predictive_Std"
        ],
        marker="o",
        label="Predictive Std",
    )

    ax2.plot(
        x,
        calibration_table[
            "Mean_MAE"
        ],
        marker="s",
        label="MAE",
    )

    ax2.plot(
        x,
        calibration_table[
            "Mean_Missing_MAE"
        ],
        marker="^",
        label="Missing MAE",
    )

    ax2.set_xlabel(
        "Uncertainty Quantile Bin"
    )

    ax2.set_ylabel(
        "Mean Value"
    )

    ax2.set_title(
        "Uncertainty-Error Alignment"
    )

    ax2.set_xticks(x)

    ax2.legend()

    ax2.grid(
        True,
        alpha=0.3,
    )

    # Adjust spacing.
    plt.tight_layout()

    # Save the figure.
    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    # Close the figure.
    plt.close()


# ======================================================================
# 9. MAIN CALIBRATION FUNCTION
# ======================================================================

def run_uncertainty_calibration():
    """
    Execute the complete uncertainty calibration/error alignment
    analysis.
    """

    print()
    print("=" * 80)
    print("UNCERTAINTY CALIBRATION AND ERROR ALIGNMENT")
    print("=" * 80)

    print()
    print(f"Dataset mode       : {DATASET_MODE}")
    print(f"Experiment         : {EXPERIMENT_NAME}")
    print(f"Device             : {DEVICE}")
    print(f"MC samples         : {MC_DROPOUT_SAMPLES}")
    print(f"Number of bins     : {NUM_BINS}")

    # --------------------------------------------------------------
    # STEP 1: BUILD DATASET
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 1: BUILD DATASET")
    print("-" * 80)

    dataset = build_dataset()

    total_samples = len(dataset)

    print(
        f"Total dataset samples: {total_samples}"
    )

    # Determine how many samples to evaluate.
    if NUM_PATCHES is None:

        number_of_samples = total_samples

    else:

        number_of_samples = min(
            NUM_PATCHES,
            total_samples,
        )

    print(
        f"Samples to evaluate: {number_of_samples}"
    )

    # --------------------------------------------------------------
    # STEP 2: CREATE MODEL
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 2: CREATE MODEL")
    print("-" * 80)

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    )

    print(
        "Network3D created successfully."
    )

    # --------------------------------------------------------------
    # STEP 3: LOAD CHECKPOINT THROUGH PREDICTOR
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 3: LOAD MODEL CHECKPOINT")
    print("-" * 80)

    checkpoint_path = (
        Path.cwd()
        / "OUTPUTS"
        / EXPERIMENT_NAME
        / "checkpoints"
        / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Model checkpoint was not found:\n"
            f"{checkpoint_path}"
        )

    # Predictor handles checkpoint loading.
    predictor = Predictor(
        model=model,
        checkpoint_path=str(checkpoint_path),
        device=DEVICE,
    )

    # Use the loaded model.
    loaded_model = predictor.model

    print(
        f"Checkpoint loaded:\n{checkpoint_path}"
    )

    # --------------------------------------------------------------
    # STEP 4: CREATE EVALUATOR
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 4: CREATE EVALUATOR")
    print("-" * 80)

    evaluator = Evaluator(
        model=loaded_model,
        device=DEVICE,
        mc_samples=MC_DROPOUT_SAMPLES,
    )

    print(
        "Evaluator created successfully."
    )

    # --------------------------------------------------------------
    # STEP 5: EVALUATE PATCHES
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 5: EVALUATE PATCHES")
    print("-" * 80)

    results = []

    for sample_index in range(
        number_of_samples
    ):

        print(
            f"Evaluating sample "
            f"{sample_index + 1}/"
            f"{number_of_samples}"
        )

        # ----------------------------------------------------------
        # Obtain one dataset sample.
        # ----------------------------------------------------------

        sample = dataset[sample_index]

        # ----------------------------------------------------------
        # Convert sample to Evaluator format.
        # ----------------------------------------------------------

        single_dataset = SingleSampleDataset(
            sample
        )

        # ----------------------------------------------------------
        # Create DataLoader.
        # ----------------------------------------------------------

        dataloader = torch.utils.data.DataLoader(
            single_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # ----------------------------------------------------------
        # Run production evaluator.
        # ----------------------------------------------------------

        evaluation_result = evaluator.evaluate(
            dataloader
        )

        # ----------------------------------------------------------
        # Extract required values.
        # ----------------------------------------------------------

        row = {
            "Sample_ID": sample_index,
            "MAE": float(
                evaluation_result["mae"]
            ),
            "RMSE": float(
                evaluation_result["rmse"]
            ),
            "Missing_MAE": float(
                evaluation_result["missing_mae"]
            ),
            "Missing_RMSE": float(
                evaluation_result["missing_rmse"]
            ),
            "Predictive_Variance": float(
                evaluation_result[
                    "predictive_variance"
                ]
            ),
            "Predictive_Std": float(
                evaluation_result[
                    "predictive_std"
                ]
            ),
            "Aleatoric_Variance": float(
                evaluation_result[
                    "aleatoric_variance"
                ]
            ),
            "Epistemic_Variance": float(
                evaluation_result[
                    "epistemic_variance"
                ]
            ),
            "Observed_Preservation_Error": float(
                evaluation_result[
                    "observed_preservation_error"
                ]
            ),
            "Measured_Missing_Rate": float(
                evaluation_result[
                    "measured_missing_rate"
                ]
            ),
            "MC_Samples": int(
                evaluation_result.get(
                    "mc_samples",
                    MC_DROPOUT_SAMPLES,
                )
            ),
        }

        # ----------------------------------------------------------
        # Validate uncertainty decomposition.
        # ----------------------------------------------------------

        decomposition_difference = abs(
            row["Predictive_Variance"]
            - (
                row["Aleatoric_Variance"]
                + row["Epistemic_Variance"]
            )
        )

        if (
            decomposition_difference
            > 1.0e-5
        ):

            raise ValueError(
                "Predictive uncertainty decomposition "
                "failed for sample "
                f"{sample_index}."
            )

        # ----------------------------------------------------------
        # Validate observed-data preservation.
        # ----------------------------------------------------------

        if (
            row[
                "Observed_Preservation_Error"
            ]
            > OBSERVED_PRESERVATION_TOLERANCE
        ):

            raise ValueError(
                "Observed-data preservation failed "
                f"for sample {sample_index}."
            )

        # Add row to results.
        results.append(row)

    # --------------------------------------------------------------
    # STEP 6: CREATE DATAFRAME
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 6: CREATE RESULTS TABLE")
    print("-" * 80)

    results_dataframe = pd.DataFrame(
        results
    )

    # Validate numerical values.
    validate_finite_dataframe(
        results_dataframe
    )

    # Save patch-level results.
    results_dataframe.to_csv(
        PATCH_RESULTS_FILE,
        index=False,
    )

    print(
        f"Patch results saved:\n"
        f"{PATCH_RESULTS_FILE}"
    )

    # --------------------------------------------------------------
    # STEP 7: CORRELATION ANALYSIS
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 7: CORRELATION ANALYSIS")
    print("-" * 80)

    correlation_results = calculate_correlations(
        results_dataframe[
            "Predictive_Std"
        ].values,
        results_dataframe[
            "MAE"
        ].values,
        results_dataframe[
            "Missing_MAE"
        ].values,
    )

    # --------------------------------------------------------------
    # STEP 8: UNCERTAINTY BINNING
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 8: UNCERTAINTY BINNING")
    print("-" * 80)

    (
        calibration_table,
        global_gap,
        missing_gap,
    ) = calculate_uncertainty_bins(
        results_dataframe,
        number_of_bins=NUM_BINS,
    )

    # Save calibration bins.
    calibration_table.to_csv(
        BIN_RESULTS_FILE,
        index=False,
    )

    print(
        f"Calibration bins saved:\n"
        f"{BIN_RESULTS_FILE}"
    )

    # --------------------------------------------------------------
    # STEP 9: SUMMARY
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 9: CREATE SUMMARY")
    print("-" * 80)

    summary = {
        "Dataset_Mode": DATASET_MODE,
        "Experiment_Name": EXPERIMENT_NAME,
        "Number_of_Patches": len(
            results_dataframe
        ),
        "MC_Samples": MC_DROPOUT_SAMPLES,
        "Mean_MAE": results_dataframe[
            "MAE"
        ].mean(),
        "Mean_RMSE": results_dataframe[
            "RMSE"
        ].mean(),
        "Mean_Missing_MAE": results_dataframe[
            "Missing_MAE"
        ].mean(),
        "Mean_Missing_RMSE": results_dataframe[
            "Missing_RMSE"
        ].mean(),
        "Mean_Predictive_Variance": results_dataframe[
            "Predictive_Variance"
        ].mean(),
        "Mean_Predictive_Std": results_dataframe[
            "Predictive_Std"
        ].mean(),
        "Mean_Aleatoric_Variance": results_dataframe[
            "Aleatoric_Variance"
        ].mean(),
        "Mean_Epistemic_Variance": results_dataframe[
            "Epistemic_Variance"
        ].mean(),
        "Mean_Missing_Rate": results_dataframe[
            "Measured_Missing_Rate"
        ].mean(),
        "Mean_Observed_Preservation_Error": results_dataframe[
            "Observed_Preservation_Error"
        ].mean(),
        "Pearson_Global_r": correlation_results[
            "pearson_global_r"
        ],
        "Pearson_Global_p": correlation_results[
            "pearson_global_p"
        ],
        "Spearman_Global_r": correlation_results[
            "spearman_global_r"
        ],
        "Spearman_Global_p": correlation_results[
            "spearman_global_p"
        ],
        "Pearson_Missing_r": correlation_results[
            "pearson_missing_r"
        ],
        "Pearson_Missing_p": correlation_results[
            "pearson_missing_p"
        ],
        "Spearman_Missing_r": correlation_results[
            "spearman_missing_r"
        ],
        "Spearman_Missing_p": correlation_results[
            "spearman_missing_p"
        ],
        "Uncertainty_Error_Alignment_Gap_Global": global_gap,
        "Uncertainty_Error_Alignment_Gap_Missing": missing_gap,
    }

    summary_dataframe = pd.DataFrame(
        [summary]
    )

    # Validate summary.
    validate_finite_dataframe(
        summary_dataframe
    )

    # Save summary.
    summary_dataframe.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"Summary saved:\n"
        f"{SUMMARY_FILE}"
    )

    # --------------------------------------------------------------
    # STEP 10: GENERATE PLOT
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print("STEP 10: GENERATE CALIBRATION PLOT")
    print("-" * 80)

    generate_calibration_plot(
        results_dataframe,
        calibration_table,
    )

    print(
        f"Plot saved:\n"
        f"{PLOT_FILE}"
    )

    # --------------------------------------------------------------
    # STEP 11: DISPLAY RESULTS
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("UNCERTAINTY CALIBRATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Pearson correlation "
        "(Global MAE): "
        f"{correlation_results['pearson_global_r']:.6f}"
    )

    print(
        "Spearman correlation "
        "(Global MAE): "
        f"{correlation_results['spearman_global_r']:.6f}"
    )

    print(
        "Pearson correlation "
        "(Missing MAE): "
        f"{correlation_results['pearson_missing_r']:.6f}"
    )

    print(
        "Spearman correlation "
        "(Missing MAE): "
        f"{correlation_results['spearman_missing_r']:.6f}"
    )

    print()
    print(
        "Global uncertainty-error alignment gap: "
        f"{global_gap:.6f}"
    )

    print(
        "Missing-region uncertainty-error "
        f"alignment gap: {missing_gap:.6f}"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  1. {PATCH_RESULTS_FILE}"
    )

    print(
        f"  2. {BIN_RESULTS_FILE}"
    )

    print(
        f"  3. {SUMMARY_FILE}"
    )

    print(
        f"  4. {PLOT_FILE}"
    )

    print()
    print("=" * 80)


# ======================================================================
# 10. SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    run_uncertainty_calibration()