"""
=====================================================================
Predictive Uncertainty vs Reconstruction Error
=====================================================================

Purpose
-------
Visualises the relationship between predictive uncertainty and
reconstruction error across evaluated seismic patches.

Primary analysis
----------------
The plot uses:

    X = Missing-region predictive standard deviation

    Y = Missing-region reconstruction MAE

This choice is consistent with the reconstruction problem because
observed seismic samples are explicitly preserved by the
data-consistency operation. Consequently, the missing region is the
appropriate primary region for analysing the relationship between
uncertainty and reconstruction error.

Statistical analysis
--------------------
The following quantities are reported:

    Pearson correlation coefficient (r)
    Pearson p-value
    R-squared (R²)
    Number of valid observations

The least-squares regression line is also plotted.

Important interpretation
------------------------
This figure evaluates the association between predictive uncertainty
and reconstruction error. It is NOT, by itself, a formal uncertainty
calibration assessment.

Formal calibration is handled by the dedicated uncertainty
calibration module.

Supported data modes
--------------------
    synthetic
    f3

Configuration
-------------
Dataset mode and report paths are obtained from utils/config.py.

Input
-----
    REPORT_DIR/uncertainty/uncertainty_evaluation.csv

Preferred input columns
-----------------------
    missing_predictive_std_mean
    missing_mae

Backward-compatible alternatives
---------------------------------
    predictive_std_mean
    mae

    Mean_Uncertainty
    MAE

The preferred missing-region fields are used whenever available.

Output
------
    REPORT_DIR/uncertainty/
        uncertainty_vs_error.png

Author
------
    Ormin Joseph
=====================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from utils.config import (
    DATASET_MODE,
    REPORT_DIR,
)


# =====================================================================
# 1. PATH CONFIGURATION
# =====================================================================

# Convert the configured report directory to a Path object.
REPORT_ROOT = Path(
    REPORT_DIR
)

# Directory containing uncertainty evaluation outputs.
UNCERTAINTY_DIRECTORY = (
    REPORT_ROOT / "uncertainty"
)

# Input uncertainty evaluation report.
CSV_FILE = (
    UNCERTAINTY_DIRECTORY
    / "uncertainty_evaluation.csv"
)

# Output figure.
OUTPUT_FILE = (
    UNCERTAINTY_DIRECTORY
    / "uncertainty_vs_error.png"
)


# =====================================================================
# 2. ANALYSIS CONFIGURATION
# =====================================================================

# Minimum number of valid paired observations required for
# correlation and regression analysis.
MIN_OBSERVATIONS = 3

# Numerical threshold used to detect effectively constant variables.
VARIANCE_EPSILON = 1.0e-12

# Resolution of the output figure.
FIGURE_DPI = 300


# =====================================================================
# 3. COLUMN DEFINITIONS
# =====================================================================

# Preferred final-evaluation columns.
#
# These represent the missing-region quantities and are therefore
# preferred for the primary thesis analysis.
PREFERRED_UNCERTAINTY_COLUMNS = [
    "missing_predictive_std_mean",
    "predictive_std_mean",
    "Mean_Uncertainty",
]

PREFERRED_ERROR_COLUMNS = [
    "missing_mae",
    "mae",
    "MAE",
]


# =====================================================================
# 4. LOAD UNCERTAINTY EVALUATION DATA
# =====================================================================

def load_uncertainty_data():
    """
    Load the uncertainty evaluation report.

    Returns
    -------
    pandas.DataFrame
        Loaded uncertainty evaluation data.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.
    ValueError
        If the CSV is empty.
    """

    # Verify that the input report exists.
    if not CSV_FILE.is_file():

        raise FileNotFoundError(
            "The uncertainty evaluation CSV was not found.\n\n"
            f"Expected file:\n{CSV_FILE}\n\n"
            "Run the uncertainty evaluation stage first."
        )

    # Load the CSV file.
    df = pd.read_csv(
        CSV_FILE
    )

    # Reject an empty report.
    if df.empty:

        raise ValueError(
            "The uncertainty evaluation CSV is empty.\n"
            f"File: {CSV_FILE}"
        )

    return df


# =====================================================================
# 5. SELECT ANALYSIS COLUMNS
# =====================================================================

def select_analysis_columns(
    df,
):
    """
    Select the appropriate uncertainty and reconstruction-error
    columns from the uncertainty evaluation report.

    The missing-region quantities are preferred because the observed
    seismic samples are preserved by data consistency.

    Returns
    -------
    tuple
        uncertainty column name,
        error column name.
    """

    # ---------------------------------------------------------------
    # Find the first available uncertainty column.
    # ---------------------------------------------------------------

    uncertainty_column = None

    for column in PREFERRED_UNCERTAINTY_COLUMNS:

        if column in df.columns:

            uncertainty_column = column

            break

    # ---------------------------------------------------------------
    # Find the first available error column.
    # ---------------------------------------------------------------

    error_column = None

    for column in PREFERRED_ERROR_COLUMNS:

        if column in df.columns:

            error_column = column

            break

    # ---------------------------------------------------------------
    # Validate uncertainty column.
    # ---------------------------------------------------------------

    if uncertainty_column is None:

        raise KeyError(
            "No suitable predictive-uncertainty column was found.\n\n"
            "Expected one of:\n"
            f"{PREFERRED_UNCERTAINTY_COLUMNS}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # ---------------------------------------------------------------
    # Validate reconstruction-error column.
    # ---------------------------------------------------------------

    if error_column is None:

        raise KeyError(
            "No suitable reconstruction-error column was found.\n\n"
            "Expected one of:\n"
            f"{PREFERRED_ERROR_COLUMNS}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return (
        uncertainty_column,
        error_column,
    )


# =====================================================================
# 6. PREPARE DATA
# =====================================================================

def prepare_data(
    df,
    uncertainty_column,
    error_column,
):
    """
    Convert uncertainty and MAE values to numeric values and remove
    invalid observations.

    Returns
    -------
    pandas.DataFrame
        Clean paired uncertainty-error observations.
    """

    # Select only the two variables required for the plot.
    data = df[
        [
            uncertainty_column,
            error_column,
        ]
    ].copy()

    # Convert uncertainty values to numeric.
    data[
        uncertainty_column
    ] = pd.to_numeric(
        data[
            uncertainty_column
        ],
        errors="coerce",
    )

    # Convert MAE values to numeric.
    data[
        error_column
    ] = pd.to_numeric(
        data[
            error_column
        ],
        errors="coerce",
    )

    # Convert infinite values to NaN.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows containing invalid paired observations.
    data = data.dropna(
        subset=[
            uncertainty_column,
            error_column,
        ]
    )

    # Verify that sufficient observations remain.
    if len(data) < MIN_OBSERVATIONS:

        raise ValueError(
            "Insufficient valid observations for correlation analysis.\n"
            f"Valid observations: {len(data)}\n"
            f"Required minimum: {MIN_OBSERVATIONS}"
        )

    return data


# =====================================================================
# 7. CALCULATE PEARSON CORRELATION
# =====================================================================

def calculate_correlation(
    x,
    y,
):
    """
    Calculate Pearson correlation and p-value.

    Parameters
    ----------
    x : numpy.ndarray
        Predictive uncertainty.

    y : numpy.ndarray
        Reconstruction MAE.

    Returns
    -------
    tuple
        Pearson correlation coefficient and p-value.
    """

    # Check for effectively constant uncertainty.
    if np.std(x) < VARIANCE_EPSILON:

        raise ValueError(
            "Predictive uncertainty has effectively zero variance. "
            "Pearson correlation cannot be calculated."
        )

    # Check for effectively constant reconstruction error.
    if np.std(y) < VARIANCE_EPSILON:

        raise ValueError(
            "Reconstruction MAE has effectively zero variance. "
            "Pearson correlation cannot be calculated."
        )

    # Calculate Pearson correlation.
    result = pearsonr(
        x,
        y,
    )

    correlation = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    return (
        correlation,
        p_value,
    )


# =====================================================================
# 8. CALCULATE REGRESSION
# =====================================================================

def calculate_regression(
    x,
    y,
):
    """
    Calculate the least-squares linear regression coefficients.

    Returns
    -------
    tuple
        slope and intercept.
    """

    # Fit:
    #
    #     y = m*x + b
    #
    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    return (
        float(slope),
        float(intercept),
    )


# =====================================================================
# 9. MAIN
# =====================================================================

def main():

    print()
    print("=" * 72)
    print(
        "PREDICTIVE UNCERTAINTY VS RECONSTRUCTION ERROR"
    )
    print("=" * 72)

    print()
    print(
        f"Dataset mode : {DATASET_MODE}"
    )

    print(
        f"Input CSV    : {CSV_FILE}"
    )

    print(
        f"Output figure: {OUTPUT_FILE}"
    )

    # =================================================================
    # LOAD DATA
    # =================================================================

    df = load_uncertainty_data()

    print()
    print(
        f"Rows loaded  : {len(df)}"
    )

    # =================================================================
    # SELECT COLUMNS
    # =================================================================

    (
        uncertainty_column,
        error_column,
    ) = select_analysis_columns(
        df
    )

    print()
    print(
        f"Uncertainty  : {uncertainty_column}"
    )

    print(
        f"Error metric : {error_column}"
    )

    # =================================================================
    # PREPARE DATA
    # =================================================================

    data = prepare_data(
        df,
        uncertainty_column,
        error_column,
    )

    # =================================================================
    # EXTRACT VARIABLES
    # =================================================================

    x = data[
        uncertainty_column
    ].to_numpy(
        dtype=np.float64
    )

    y = data[
        error_column
    ].to_numpy(
        dtype=np.float64
    )

    # =================================================================
    # VALIDATE ARRAYS
    # =================================================================

    if not np.all(
        np.isfinite(x)
    ):

        raise ValueError(
            "Predictive uncertainty contains invalid values."
        )

    if not np.all(
        np.isfinite(y)
    ):

        raise ValueError(
            "Reconstruction MAE contains invalid values."
        )

    # =================================================================
    # CALCULATE PEARSON CORRELATION
    # =================================================================

    (
        correlation,
        p_value,
    ) = calculate_correlation(
        x,
        y,
    )

    # =================================================================
    # CALCULATE REGRESSION
    # =================================================================

    (
        slope,
        intercept,
    ) = calculate_regression(
        x,
        y,
    )

    # =================================================================
    # CALCULATE R-SQUARED
    # =================================================================

    # For simple least-squares linear regression with an intercept,
    # R² is the square of the Pearson correlation coefficient.
    r_squared = (
        correlation ** 2
    )

    # =================================================================
    # GENERATE REGRESSION LINE
    # =================================================================

    x_line = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    y_line = (
        slope * x_line
        +
        intercept
    )

    # =================================================================
    # CREATE FIGURE
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    # ---------------------------------------------------------------
    # Scatter observations.
    # ---------------------------------------------------------------

    ax.scatter(
        x,
        y,
        alpha=0.75,
        label="Evaluation patches",
    )

    # ---------------------------------------------------------------
    # Regression line.
    # ---------------------------------------------------------------

    ax.plot(
        x_line,
        y_line,
        linewidth=2.0,
        label="Least-squares regression",
    )

    # ---------------------------------------------------------------
    # Axis labels.
    # ---------------------------------------------------------------

    ax.set_xlabel(
        "Mean Predictive Uncertainty"
    )

    ax.set_ylabel(
        "Missing-Region Reconstruction MAE"
    )

    # ---------------------------------------------------------------
    # Figure title.
    # ---------------------------------------------------------------

    ax.set_title(
        "Predictive Uncertainty vs Reconstruction Error"
    )

    # ---------------------------------------------------------------
    # Statistical annotation.
    # ---------------------------------------------------------------

    annotation = (
        f"Pearson r = {correlation:.4f}\n"
        f"p = {p_value:.4e}\n"
        f"R² = {r_squared:.4f}\n"
        f"n = {len(data)}"
    )

    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.85,
        },
    )

    # ---------------------------------------------------------------
    # Grid.
    # ---------------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3,
    )

    # ---------------------------------------------------------------
    # Legend.
    # ---------------------------------------------------------------

    ax.legend()

    # ---------------------------------------------------------------
    # Layout.
    # ---------------------------------------------------------------

    fig.tight_layout()

    # =================================================================
    # CREATE OUTPUT DIRECTORY
    # =================================================================

    UNCERTAINTY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # SAVE FIGURE
    # =================================================================

    fig.savefig(
        OUTPUT_FILE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # Close the figure to release memory.
    plt.close(
        fig
    )

    # =================================================================
    # DISPLAY RESULTS
    # =================================================================

    print()
    print("-" * 72)
    print(
        "CORRELATION AND REGRESSION RESULTS"
    )
    print("-" * 72)

    print()
    print(
        f"Valid observations : {len(data)}"
    )

    print(
        f"Pearson r          : {correlation:.6f}"
    )

    print(
        f"Pearson p-value    : {p_value:.6e}"
    )

    print(
        f"R-squared          : {r_squared:.6f}"
    )

    print()
    print(
        "Regression equation:"
    )

    print(
        "MAE = "
        f"{slope:.6f} × "
        f"{uncertainty_column} "
        f"+ {intercept:.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 72)
    print(
        "UNCERTAINTY VS RECONSTRUCTION ERROR PLOT COMPLETE"
    )
    print("=" * 72)
    print()


# =====================================================================
# 10. SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()