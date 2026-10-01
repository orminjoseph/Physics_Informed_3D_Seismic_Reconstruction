"""
======================================================================
UNCERTAINTY STATISTICS GENERATOR
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Create the finalized uncertainty_statistics.csv required by:

    1. evaluation/generate_thesis_tables.py
    2. evaluation/generate_final_report.py

IMPORTANT
---------
This module DOES NOT:

    - retrain the model
    - rerun model inference
    - rerun MC-Dropout
    - regenerate uncertainty maps
    - modify uncertainty_evaluation.csv

Instead, it reads the already-generated:

    REPORT_DIR/uncertainty_evaluation.csv

and calculates descriptive summary statistics from those
validated patch-level results.

Output
------
REPORT_DIR/
    uncertainty_statistics.csv

The output contains the principal uncertainty decomposition:

    aleatoric_variance
    epistemic_variance
    predictive_variance

together with associated reconstruction-error and
quality-control statistics.

Author:
Ormin Joseph
======================================================================
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os

import numpy as np
import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# ======================================================================
# INPUT AND OUTPUT FILES
# ======================================================================

INPUT_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_evaluation.csv",
)

OUTPUT_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_statistics.csv",
)


# ======================================================================
# REQUIRED INPUT COLUMNS
# ======================================================================

REQUIRED_COLUMNS = [
    "MAE",
    "RMSE",
    "Missing_MAE",
    "Missing_RMSE",
    "Aleatoric_Variance",
    "Epistemic_Variance",
    "Predictive_Variance",
    "Predictive_Std",
    "Observed_Preservation_Error",
    "Measured_Missing_Rate",
    "MC_Samples",
]


# ======================================================================
# VALIDATE FINITE NUMERICAL VALUES
# ======================================================================

def validate_finite_values(
    dataframe,
    columns,
):
    """
    Validate that all specified numerical columns contain
    finite values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.

    columns : list
        Numerical columns to validate.

    Raises
    ------
    ValueError
        If non-numeric, NaN, or infinite values are found.
    """

    for column in columns:

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if values.isna().any():

            raise ValueError(
                "\nNon-numeric or missing values detected "
                f"in column '{column}'."
            )

        if not np.isfinite(
            values.to_numpy(
                dtype=float
            )
        ).all():

            raise ValueError(
                "\nNon-finite values detected "
                f"in column '{column}'."
            )


# ======================================================================
# LOAD INPUT DATA
# ======================================================================

def load_uncertainty_evaluation():
    """
    Load and validate the existing patch-level
    uncertainty evaluation results.

    Returns
    -------
    pandas.DataFrame
        Validated uncertainty evaluation dataframe.
    """

    print()
    print("=" * 80)
    print("LOADING UNCERTAINTY EVALUATION RESULTS")
    print("=" * 80)

    print()
    print(
        "Input file:"
    )

    print(
        INPUT_FILE
    )

    # --------------------------------------------------------------
    # Check that the source file exists.
    # --------------------------------------------------------------

    if not os.path.isfile(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            "\nRequired uncertainty evaluation file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run the existing uncertainty evaluation module first."
        )

    # --------------------------------------------------------------
    # Read CSV.
    # --------------------------------------------------------------

    try:

        dataframe = pd.read_csv(
            INPUT_FILE
        )

    except Exception as error:

        raise RuntimeError(
            "\nUnable to read uncertainty evaluation CSV:\n"
            f"{INPUT_FILE}\n"
            f"Error: {error}"
        ) from error

    # --------------------------------------------------------------
    # Check that the table is not empty.
    # --------------------------------------------------------------

    if dataframe.empty:

        raise ValueError(
            "\nThe uncertainty evaluation file is empty:\n"
            f"{INPUT_FILE}"
        )

    print()
    print(
        f"Rows loaded    : {len(dataframe)}"
    )

    print(
        f"Columns loaded : {len(dataframe.columns)}"
    )

    # --------------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe uncertainty evaluation file is missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------------
    # Validate numerical fields.
    # --------------------------------------------------------------

    validate_finite_values(
        dataframe,
        REQUIRED_COLUMNS,
    )

    return dataframe


# ======================================================================
# VALIDATE UNCERTAINTY DECOMPOSITION
# ======================================================================

def validate_uncertainty_decomposition(
    dataframe,
):
    """
    Verify the uncertainty decomposition:

        Predictive Variance
            =
        Aleatoric Variance
            +
        Epistemic Variance

    A small numerical tolerance is allowed because of
    floating-point arithmetic.
    """

    print()
    print(
        "Validating uncertainty decomposition..."
    )

    difference = np.abs(
        dataframe[
            "Predictive_Variance"
        ].to_numpy(dtype=float)
        -
        (
            dataframe[
                "Aleatoric_Variance"
            ].to_numpy(dtype=float)
            +
            dataframe[
                "Epistemic_Variance"
            ].to_numpy(dtype=float)
        )
    )

    maximum_difference = float(
        np.max(difference)
    )

    print(
        "Maximum decomposition difference:"
        f" {maximum_difference:.6e}"
    )

    if maximum_difference > 1.0e-5:

        raise ValueError(
            "\nPredictive uncertainty decomposition "
            "validation failed.\n"
            f"Maximum difference = "
            f"{maximum_difference:.6e}\n"
            "Allowed tolerance = 1.0e-5"
        )

    print(
        "[VALID] Uncertainty decomposition"
    )


# ======================================================================
# VALIDATE NON-NEGATIVE UNCERTAINTY
# ======================================================================

def validate_non_negative_uncertainty(
    dataframe,
):
    """
    Ensure all uncertainty variance values are non-negative.
    """

    uncertainty_columns = [
        "Aleatoric_Variance",
        "Epistemic_Variance",
        "Predictive_Variance",
        "Predictive_Std",
    ]

    print()
    print(
        "Validating non-negative uncertainty values..."
    )

    for column in uncertainty_columns:

        if (
            dataframe[column]
            < 0
        ).any():

            raise ValueError(
                "\nNegative uncertainty values detected "
                f"in column '{column}'."
            )

    print(
        "[VALID] Non-negative uncertainty values"
    )


# ======================================================================
# VALIDATE OBSERVED-DATA PRESERVATION
# ======================================================================

def validate_observed_preservation(
    dataframe,
):
    """
    Verify that observed seismic samples remain unchanged
    within the established tolerance.
    """

    tolerance = 1.0e-6

    maximum_error = float(
        dataframe[
            "Observed_Preservation_Error"
        ].max()
    )

    print()
    print(
        "Validating observed-data preservation..."
    )

    print(
        "Maximum observed-data preservation error:"
        f" {maximum_error:.6e}"
    )

    if maximum_error > tolerance:

        raise ValueError(
            "\nObserved-data preservation tolerance "
            "was exceeded.\n"
            f"Maximum error = {maximum_error:.6e}\n"
            f"Tolerance = {tolerance:.6e}"
        )

    print(
        "[VALID] Observed-data preservation"
    )


# ======================================================================
# CREATE UNCERTAINTY STATISTICS
# ======================================================================

def create_uncertainty_statistics(
    dataframe,
):
    """
    Create the final uncertainty statistics table.

    Numerical values are descriptive aggregates of the
    existing patch-level uncertainty evaluation.

    No raw values are modified.
    """

    print()
    print("=" * 80)
    print("CREATING UNCERTAINTY STATISTICS")
    print("=" * 80)

    # --------------------------------------------------------------
    # Create one summary row.
    # --------------------------------------------------------------

    statistics = {

        # ----------------------------------------------------------
        # Experiment information
        # ----------------------------------------------------------

        "Experiment_Name":
            EXPERIMENT_NAME,

        "Number_of_Patches":
            int(len(dataframe)),

        "MC_Samples":
            int(
                dataframe[
                    "MC_Samples"
                ].iloc[0]
            ),

        # ----------------------------------------------------------
        # Reconstruction performance
        # ----------------------------------------------------------

        "Mean_MAE":
            float(
                dataframe[
                    "MAE"
                ].mean()
            ),

        "Mean_RMSE":
            float(
                dataframe[
                    "RMSE"
                ].mean()
            ),

        "Mean_Missing_MAE":
            float(
                dataframe[
                    "Missing_MAE"
                ].mean()
            ),

        "Mean_Missing_RMSE":
            float(
                dataframe[
                    "Missing_RMSE"
                ].mean()
            ),

        # ----------------------------------------------------------
        # Uncertainty decomposition
        #
        # These are the key fields required by the final
        # report generator and thesis-table generator.
        # ----------------------------------------------------------

        "aleatoric_variance":
            float(
                dataframe[
                    "Aleatoric_Variance"
                ].mean()
            ),

        "epistemic_variance":
            float(
                dataframe[
                    "Epistemic_Variance"
                ].mean()
            ),

        "predictive_variance":
            float(
                dataframe[
                    "Predictive_Variance"
                ].mean()
            ),

        "predictive_std":
            float(
                dataframe[
                    "Predictive_Std"
                ].mean()
            ),

        # ----------------------------------------------------------
        # Additional uncertainty statistics
        # ----------------------------------------------------------

        "Mean_Aleatoric_Variance":
            float(
                dataframe[
                    "Aleatoric_Variance"
                ].mean()
            ),

        "Mean_Epistemic_Variance":
            float(
                dataframe[
                    "Epistemic_Variance"
                ].mean()
            ),

        "Mean_Predictive_Variance":
            float(
                dataframe[
                    "Predictive_Variance"
                ].mean()
            ),

        "Mean_Predictive_Std":
            float(
                dataframe[
                    "Predictive_Std"
                ].mean()
            ),

        "Maximum_Predictive_Variance":
            float(
                dataframe[
                    "Predictive_Variance"
                ].max()
            ),

        "Maximum_Predictive_Std":
            float(
                dataframe[
                    "Predictive_Std"
                ].max()
            ),

        # ----------------------------------------------------------
        # Quality-control statistics
        # ----------------------------------------------------------

        "Mean_Observed_Preservation_Error":
            float(
                dataframe[
                    "Observed_Preservation_Error"
                ].mean()
            ),

        "Maximum_Observed_Preservation_Error":
            float(
                dataframe[
                    "Observed_Preservation_Error"
                ].max()
            ),

        "Mean_Measured_Missing_Rate":
            float(
                dataframe[
                    "Measured_Missing_Rate"
                ].mean()
            ),

        "Maximum_Measured_Missing_Rate":
            float(
                dataframe[
                    "Measured_Missing_Rate"
                ].max()
            ),
    }

    # --------------------------------------------------------------
    # Convert to DataFrame.
    # --------------------------------------------------------------

    statistics_dataframe = pd.DataFrame(
        [statistics]
    )

    # --------------------------------------------------------------
    # Validate every numerical value.
    # --------------------------------------------------------------

    numeric_columns = (
        statistics_dataframe
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    validate_finite_values(
        statistics_dataframe,
        list(numeric_columns),
    )

    return statistics_dataframe


# ======================================================================
# SAVE OUTPUT
# ======================================================================

def save_uncertainty_statistics(
    dataframe,
):
    """
    Save the finalized uncertainty statistics CSV.
    """

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "[CREATED] uncertainty_statistics.csv"
    )

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Rows:"
        f" {len(dataframe)}"
    )

    print(
        "Columns:"
        f" {len(dataframe.columns)}"
    )


# ======================================================================
# MAIN FUNCTION
# ======================================================================

def generate_uncertainty_statistics():
    """
    Generate uncertainty_statistics.csv from the existing
    uncertainty_evaluation.csv.
    """

    print()
    print("=" * 80)
    print("FINAL UNCERTAINTY STATISTICS GENERATION")
    print("=" * 80)

    # --------------------------------------------------------------
    # Load existing patch-level results.
    # --------------------------------------------------------------

    dataframe = load_uncertainty_evaluation()

    # --------------------------------------------------------------
    # Validate uncertainty decomposition.
    # --------------------------------------------------------------

    validate_uncertainty_decomposition(
        dataframe
    )

    # --------------------------------------------------------------
    # Validate uncertainty values.
    # --------------------------------------------------------------

    validate_non_negative_uncertainty(
        dataframe
    )

    # --------------------------------------------------------------
    # Validate observed-data preservation.
    # --------------------------------------------------------------

    validate_observed_preservation(
        dataframe
    )

    # --------------------------------------------------------------
    # Create statistics.
    # --------------------------------------------------------------

    statistics_dataframe = (
        create_uncertainty_statistics(
            dataframe
        )
    )

    # --------------------------------------------------------------
    # Save final CSV.
    # --------------------------------------------------------------

    save_uncertainty_statistics(
        statistics_dataframe
    )

    # --------------------------------------------------------------
    # Display the resulting table.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("UNCERTAINTY STATISTICS")
    print("=" * 80)

    print()

    print(
        statistics_dataframe.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "UNCERTAINTY STATISTICS GENERATION COMPLETE"
    )
    print("=" * 80)

    return OUTPUT_FILE


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    generate_uncertainty_statistics()