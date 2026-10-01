"""
=====================================================================
Uncertainty–Error Correlation Summary
=====================================================================

Purpose
-------
Summarises the relationship between predictive uncertainty and
reconstruction-quality metrics produced by the uncertainty evaluation
stage.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
This module measures statistical association between predictive
uncertainty and reconstruction metrics. Pearson correlation is NOT,
by itself, a formal uncertainty-calibration test.

Formal uncertainty calibration should be performed by the dedicated
uncertainty calibration module using appropriate coverage/reliability
measures.

Supported data modes
--------------------
    synthetic
    f3

The script is DATASET-MODE aware through utils/config.py and therefore
does not contain hard-coded dataset paths.

Input
-----
    REPORT_DIR/uncertainty/uncertainty_evaluation.csv

Required column
---------------
    Mean_Uncertainty

Expected reconstruction-quality columns
----------------------------------------
    MAE
    RMSE
    PSNR
    SNR
    SSIM

Output
------
    REPORT_DIR/uncertainty/
        uncertainty_calibration_summary.csv

Output columns
--------------
    Metric
    Correlation
    Absolute_Correlation
    Sample_Count

Interpretation
--------------
For MAE and RMSE:
    Positive correlation means uncertainty tends to increase as
    reconstruction error increases.

For PSNR, SNR and SSIM:
    Negative correlation means uncertainty tends to increase as
    reconstruction quality decreases.

The correlation coefficient does not establish causation and does
not constitute formal calibration.

Author
------
    Ormin Joseph
=====================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import (
    DATASET_MODE,
    REPORT_DIR,
)


# =====================================================================
# 1. PATH CONFIGURATION
# =====================================================================

# Convert the configured report directory to a Path object.
REPORT_ROOT = Path(REPORT_DIR)

# Directory containing all uncertainty-related evaluation reports.
UNCERTAINTY_DIRECTORY = (
    REPORT_ROOT / "uncertainty"
)

# Input produced by the uncertainty evaluation stage.
REPORT_FILE = (
    UNCERTAINTY_DIRECTORY
    / "uncertainty_evaluation.csv"
)

# Output produced by this module.
OUTPUT_FILE = (
    UNCERTAINTY_DIRECTORY
    / "uncertainty_calibration_summary.csv"
)


# =====================================================================
# 2. REQUIRED COLUMNS
# =====================================================================

# Column containing the predictive uncertainty measure.
UNCERTAINTY_COLUMN = "Mean_Uncertainty"

# Reconstruction-quality metrics to investigate.
METRICS = [
    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM",
]


# =====================================================================
# 3. NUMERICAL CONFIGURATION
# =====================================================================

# Minimum number of valid paired observations required before a
# correlation coefficient can be calculated.
MIN_CORRELATION_SAMPLES = 3

# Numerical threshold used to identify effectively constant variables.
VARIANCE_EPSILON = 1.0e-12


# =====================================================================
# 4. LOAD UNCERTAINTY REPORT
# =====================================================================

def load_report():
    """
    Load the uncertainty evaluation report.

    Returns
    -------
    pandas.DataFrame
        Loaded uncertainty evaluation data.

    Raises
    ------
    FileNotFoundError
        If the expected uncertainty evaluation report does not exist.

    ValueError
        If the report exists but contains no rows.
    """

    # Check whether the expected input report exists.
    if not REPORT_FILE.exists():

        raise FileNotFoundError(
            "The uncertainty evaluation report was not found.\n\n"
            f"Expected file:\n{REPORT_FILE}\n\n"
            "Run the uncertainty evaluation stage first."
        )

    # Read the CSV report.
    df = pd.read_csv(
        REPORT_FILE
    )

    # Reject an empty report.
    if df.empty:

        raise ValueError(
            "The uncertainty evaluation report is empty.\n"
            f"File: {REPORT_FILE}"
        )

    return df


# =====================================================================
# 5. VALIDATE REQUIRED COLUMNS
# =====================================================================

def validate_columns(df):
    """
    Validate the uncertainty column and identify the reconstruction
    metrics available in the report.

    Parameters
    ----------
    df : pandas.DataFrame
        Uncertainty evaluation report.

    Returns
    -------
    list
        Available reconstruction metrics.

    Raises
    ------
    KeyError
        If the predictive uncertainty column is missing or none of
        the expected reconstruction metrics are available.
    """

    # The uncertainty variable is mandatory.
    if UNCERTAINTY_COLUMN not in df.columns:

        raise KeyError(
            f"Required uncertainty column "
            f"'{UNCERTAINTY_COLUMN}' is missing.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # Identify which expected metrics are present.
    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    # At least one reconstruction metric is required.
    if not available_metrics:

        raise KeyError(
            "None of the expected reconstruction-quality metrics "
            "were found in the uncertainty evaluation report.\n\n"
            f"Expected metrics:\n{METRICS}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return available_metrics


# =====================================================================
# 6. PREPARE NUMERIC DATA
# =====================================================================

def prepare_data(
    df,
    metrics,
):
    """
    Convert the uncertainty and reconstruction metrics to numeric
    values.

    Invalid, infinite and non-numeric values are converted to NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Original uncertainty evaluation report.

    metrics : list
        Available reconstruction metrics.

    Returns
    -------
    pandas.DataFrame
        Numeric analysis data.
    """

    # Select only the variables required for correlation analysis.
    columns = [
        UNCERTAINTY_COLUMN,
        *metrics,
    ]

    data = df[
        columns
    ].copy()

    # Convert every selected column to numeric.
    for column in columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Replace positive and negative infinity with NaN.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


# =====================================================================
# 7. CALCULATE PEARSON CORRELATION
# =====================================================================

def calculate_correlation(
    uncertainty,
    metric,
):
    """
    Calculate the Pearson correlation coefficient.

    The calculation is performed only on finite paired observations.

    Parameters
    ----------
    uncertainty : numpy.ndarray
        Predictive uncertainty values.

    metric : numpy.ndarray
        Reconstruction-quality metric values.

    Returns
    -------
    float
        Pearson correlation coefficient.

        NaN is returned when:
        - insufficient observations are available;
        - uncertainty has effectively zero variance; or
        - the reconstruction metric has effectively zero variance.
    """

    # Identify observations where both variables are finite.
    valid = (
        np.isfinite(uncertainty)
        &
        np.isfinite(metric)
    )

    # Retain only valid paired observations.
    uncertainty = uncertainty[
        valid
    ]

    metric = metric[
        valid
    ]

    # Check whether enough paired observations exist.
    if len(uncertainty) < MIN_CORRELATION_SAMPLES:

        return np.nan

    # Pearson correlation is undefined when either variable is
    # effectively constant.
    if (
        np.std(uncertainty) < VARIANCE_EPSILON
        or
        np.std(metric) < VARIANCE_EPSILON
    ):

        return np.nan

    # Calculate the Pearson correlation coefficient.
    correlation_matrix = np.corrcoef(
        uncertainty,
        metric,
    )

    correlation = correlation_matrix[
        0,
        1,
    ]

    # Protect against unexpected numerical results.
    if not np.isfinite(correlation):

        return np.nan

    return float(
        correlation
    )


# =====================================================================
# 8. INTERPRET CORRELATION DIRECTION
# =====================================================================

def interpret_correlation(
    metric,
    correlation,
):
    """
    Provide a scientifically descriptive interpretation of the
    correlation direction.

    This function does not assign a quality ranking or score.

    Parameters
    ----------
    metric : str
        Reconstruction-quality metric.

    correlation : float
        Pearson correlation coefficient.

    Returns
    -------
    str
        Directional interpretation.
    """

    if not np.isfinite(correlation):

        return "Undefined"

    if abs(correlation) < VARIANCE_EPSILON:

        return "Approximately zero association"

    if metric in {
        "MAE",
        "RMSE",
    }:

        if correlation > 0:

            return (
                "Positive association: higher uncertainty "
                "corresponds to higher reconstruction error"
            )

        return (
            "Negative association: higher uncertainty "
            "corresponds to lower reconstruction error"
        )

    if metric in {
        "PSNR",
        "SNR",
        "SSIM",
    }:

        if correlation < 0:

            return (
                "Negative association: higher uncertainty "
                "corresponds to lower reconstruction quality"
            )

        return (
            "Positive association: higher uncertainty "
            "corresponds to higher reconstruction quality"
        )

    # Generic fallback for future metrics.
    if correlation > 0:

        return "Positive association"

    return "Negative association"


# =====================================================================
# 9. CALCULATE CORRELATION SUMMARY
# =====================================================================

def calculate_summary(
    df,
    metrics,
):
    """
    Calculate uncertainty correlations for all available
    reconstruction-quality metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared numeric uncertainty data.

    metrics : list
        Reconstruction metrics to analyse.

    Returns
    -------
    pandas.DataFrame
        Correlation summary.
    """

    results = []

    # Process each reconstruction-quality metric independently.
    for metric in metrics:

        # Select paired observations for this metric.
        subset = df[
            [
                UNCERTAINTY_COLUMN,
                metric,
            ]
        ].dropna()

        # Convert uncertainty values to NumPy arrays.
        uncertainty = (
            subset[
                UNCERTAINTY_COLUMN
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        # Convert metric values to NumPy arrays.
        metric_values = (
            subset[
                metric
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        # Calculate Pearson correlation.
        correlation = calculate_correlation(
            uncertainty,
            metric_values,
        )

        # Calculate absolute correlation only when defined.
        if np.isfinite(correlation):

            absolute_correlation = abs(
                correlation
            )

        else:

            absolute_correlation = np.nan

        # Store the result.
        results.append(
            {
                "Metric": metric,
                "Correlation": correlation,
                "Absolute_Correlation": (
                    absolute_correlation
                ),
                "Sample_Count": len(subset),
                "Interpretation": interpret_correlation(
                    metric,
                    correlation,
                ),
            }
        )

    # Convert results into a DataFrame.
    return pd.DataFrame(
        results
    )


# =====================================================================
# 10. QUALITY-CONTROL VALIDATION
# =====================================================================

def validate_results(
    result_df,
):
    """
    Perform final quality-control checks on the correlation summary.

    Raises
    ------
    ValueError
        If the output contains an invalid finite correlation.
    """

    # Check that the expected result columns exist.
    required_result_columns = {
        "Metric",
        "Correlation",
        "Absolute_Correlation",
        "Sample_Count",
        "Interpretation",
    }

    missing_columns = (
        required_result_columns
        - set(result_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Correlation summary is missing expected output "
            f"columns: {sorted(missing_columns)}"
        )

    # Validate every defined correlation.
    for correlation in result_df[
        "Correlation"
    ]:

        if np.isfinite(correlation):

            if not (
                -1.0
                <= correlation
                <= 1.0
            ):

                raise ValueError(
                    "An invalid Pearson correlation coefficient "
                    f"was produced: {correlation}"
                )


# =====================================================================
# 11. MAIN
# =====================================================================

def main():

    print()
    print("=" * 70)
    print(
        "UNCERTAINTY–ERROR CORRELATION SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        f"Data mode : {DATASET_MODE}"
    )

    print(
        f"Input     : {REPORT_FILE}"
    )

    print(
        f"Output    : {OUTPUT_FILE}"
    )

    # ---------------------------------------------------------------
    # Load the uncertainty evaluation report.
    # ---------------------------------------------------------------

    df = load_report()

    print()
    print(
        f"Rows loaded: {len(df)}"
    )

    # ---------------------------------------------------------------
    # Validate required columns.
    # ---------------------------------------------------------------

    available_metrics = validate_columns(
        df
    )

    print(
        f"Metrics available: "
        f"{', '.join(available_metrics)}"
    )

    # ---------------------------------------------------------------
    # Prepare numeric analysis data.
    # ---------------------------------------------------------------

    data = prepare_data(
        df,
        available_metrics,
    )

    # ---------------------------------------------------------------
    # Calculate correlation summary.
    # ---------------------------------------------------------------

    result_df = calculate_summary(
        data,
        available_metrics,
    )

    # ---------------------------------------------------------------
    # Validate calculated results.
    # ---------------------------------------------------------------

    validate_results(
        result_df
    )

    # ---------------------------------------------------------------
    # Create output directory.
    # ---------------------------------------------------------------

    UNCERTAINTY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save the correlation summary.
    # ---------------------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Display results.
    # ---------------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "CORRELATION RESULTS"
    )
    print("-" * 70)

    for _, row in result_df.iterrows():

        correlation = row[
            "Correlation"
        ]

        sample_count = int(
            row[
                "Sample_Count"
            ]
        )

        metric = row[
            "Metric"
        ]

        if np.isfinite(
            correlation
        ):

            print(
                f"{metric:>5s} : "
                f"{correlation:+.4f} "
                f"(n={sample_count})"
            )

        else:

            print(
                f"{metric:>5s} : "
                f"undefined "
                f"(n={sample_count})"
            )

    # ---------------------------------------------------------------
    # Final output information.
    # ---------------------------------------------------------------

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print(
        "UNCERTAINTY–ERROR CORRELATION SUMMARY COMPLETED"
    )
    print("=" * 70)
    print()


# =====================================================================
# 12. ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()