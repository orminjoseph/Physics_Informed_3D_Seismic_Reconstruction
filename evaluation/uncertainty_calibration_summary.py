"""
=========================================================
Uncertainty Calibration Summary
=========================================================

Computes the Pearson correlation between predictive
uncertainty and reconstruction-quality metrics.

Supported data modes
--------------------
    synthetic
    f3

The script is DATASET-MODE AWARE through config.py and
therefore does not contain hard-coded F3 paths.

Expected input file
-------------------
    REPORT_DIR/uncertainty/uncertainty_evaluation.csv

Expected columns
----------------
    Mean_Uncertainty
    MAE
    RMSE
    PSNR
    SNR
    SSIM

Output
------
    REPORT_DIR/uncertainty/
        uncertainty_calibration_summary.csv

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd

from utils.config import (
    DATASET_MODE,
    REPORT_DIR,
)


# =========================================================
# Configuration
# =========================================================

UNCERTAINTY_DIRECTORY = os.path.join(
    REPORT_DIR,
    "uncertainty"
)

REPORT_FILE = os.path.join(
    UNCERTAINTY_DIRECTORY,
    "uncertainty_evaluation.csv"
)

OUTPUT_FILE = os.path.join(
    UNCERTAINTY_DIRECTORY,
    "uncertainty_calibration_summary.csv"
)


# =========================================================
# Required Columns
# =========================================================

UNCERTAINTY_COLUMN = (
    "Mean_Uncertainty"
)

METRICS = [
    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM",
]


# =========================================================
# Load Report
# =========================================================

def load_report():
    """
    Load the uncertainty evaluation report.

    Returns
    -------
    pandas.DataFrame
        Loaded uncertainty evaluation data.
    """

    if not os.path.exists(
        REPORT_FILE
    ):

        raise FileNotFoundError(
            "Uncertainty evaluation report "
            "was not found:\n"
            f"{REPORT_FILE}\n\n"
            "Run uncertainty_analysis.py first."
        )

    df = pd.read_csv(
        REPORT_FILE
    )

    if df.empty:

        raise ValueError(
            "The uncertainty evaluation report "
            "is empty."
        )

    return df


# =========================================================
# Validate Columns
# =========================================================

def validate_columns(
    df
):
    """
    Check that the uncertainty column exists and identify
    which reconstruction metrics are available.
    """

    if UNCERTAINTY_COLUMN not in df.columns:

        raise KeyError(
            f"Required column "
            f"'{UNCERTAINTY_COLUMN}' "
            f"is missing from the report."
        )

    available_metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    if not available_metrics:

        raise KeyError(
            "None of the expected reconstruction "
            "metrics were found.\n"
            f"Expected at least one of: {METRICS}"
        )

    return available_metrics


# =========================================================
# Prepare Numeric Data
# =========================================================

def prepare_data(
    df,
    metrics
):
    """
    Convert required columns to numeric values and remove
    invalid observations.

    Only rows containing valid uncertainty and metric
    values are used for each correlation.
    """

    columns = [
        UNCERTAINTY_COLUMN
    ] + metrics

    data = df[
        columns
    ].copy()

    for column in columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


# =========================================================
# Pearson Correlation
# =========================================================

def calculate_correlation(
    uncertainty,
    metric
):
    """
    Calculate Pearson correlation.

    Returns NaN when there are insufficient observations
    or when either variable has zero variance.
    """

    valid = (
        np.isfinite(uncertainty)
        &
        np.isfinite(metric)
    )

    uncertainty = uncertainty[
        valid
    ]

    metric = metric[
        valid
    ]

    if len(uncertainty) < 2:

        return np.nan

    if (
        np.std(uncertainty) < 1e-12
        or
        np.std(metric) < 1e-12
    ):

        return np.nan

    return float(
        np.corrcoef(
            uncertainty,
            metric
        )[0, 1]
    )


# =========================================================
# Calculate Summary
# =========================================================

def calculate_summary(
    df,
    metrics
):
    """
    Calculate uncertainty correlations with all available
    reconstruction metrics.
    """

    results = []

    for metric in metrics:

        subset = df[
            [
                UNCERTAINTY_COLUMN,
                metric
            ]
        ].dropna()

        uncertainty = (
            subset[
                UNCERTAINTY_COLUMN
            ].to_numpy(
                dtype=np.float64
            )
        )

        metric_values = (
            subset[
                metric
            ].to_numpy(
                dtype=np.float64
            )
        )

        correlation = calculate_correlation(
            uncertainty,
            metric_values
        )

        results.append({

            "Metric":
                metric,

            "Correlation":
                correlation,

            "Absolute_Correlation":
                (
                    abs(correlation)
                    if np.isfinite(correlation)
                    else np.nan
                ),

            "Sample_Count":
                len(subset)
        })

    return pd.DataFrame(
        results
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "UNCERTAINTY CALIBRATION SUMMARY"
    )

    print(
        "=" * 60
    )

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

    # -----------------------------------------------------
    # Load report
    # -----------------------------------------------------

    df = load_report()

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    available_metrics = validate_columns(
        df
    )

    # -----------------------------------------------------
    # Prepare numeric data
    # -----------------------------------------------------

    data = prepare_data(
        df,
        available_metrics
    )

    # -----------------------------------------------------
    # Calculate correlations
    # -----------------------------------------------------

    result_df = calculate_summary(
        data,
        available_metrics
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    os.makedirs(
        UNCERTAINTY_DIRECTORY,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print(
        "-" * 60
    )

    print(
        "CORRELATION RESULTS"
    )

    print(
        "-" * 60
    )

    for _, row in result_df.iterrows():

        correlation = row[
            "Correlation"
        ]

        if np.isfinite(
            correlation
        ):

            print(
                f"{row['Metric']:5s} : "
                f"{correlation:.4f} "
                f"(n={int(row['Sample_Count'])})"
            )

        else:

            print(
                f"{row['Metric']:5s} : "
                f"undefined "
                f"(n={int(row['Sample_Count'])})"
            )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "=" * 60
    )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()