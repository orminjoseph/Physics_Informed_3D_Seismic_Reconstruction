"""
=========================================================
Uncertainty-Error Correlation Analysis
=========================================================

Analyzes the relationship between predictive uncertainty
and reconstruction error.

The analysis uses:

    X = Mean Predictive Uncertainty
    Y = Reconstruction MAE

The Pearson correlation coefficient is calculated and a
least-squares regression line is plotted.

This script is mode-aware and can therefore be used with
either:

    - Synthetic seismic data
    - F3 seismic data

The input CSV must contain the columns:

    Mean_Uncertainty
    MAE

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import (
    DATASET_MODE,
    REPORT_DIR,
)


# =========================================================
# CONFIGURATION
# =========================================================

UNCERTAINTY_DIRECTORY = os.path.join(
    REPORT_DIR,
    "uncertainty"
)

CSV_FILE = os.path.join(
    UNCERTAINTY_DIRECTORY,
    "uncertainty_evaluation.csv"
)

OUTPUT_FILE = os.path.join(
    UNCERTAINTY_DIRECTORY,
    "uncertainty_vs_error.png"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_uncertainty_data():
    """
    Load uncertainty evaluation results from CSV.
    """

    if not os.path.isfile(CSV_FILE):

        raise FileNotFoundError(
            "Uncertainty evaluation CSV was not found:\n"
            f"{CSV_FILE}\n\n"
            "Run the uncertainty evaluation script first."
        )

    df = pd.read_csv(
        CSV_FILE
    )

    required_columns = [
        "Mean_Uncertainty",
        "MAE"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The uncertainty evaluation CSV is missing "
            f"required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


# =========================================================
# VALIDATE DATA
# =========================================================

def validate_data(df):
    """
    Validate uncertainty and error values before
    calculating correlation.
    """

    data = df[
        [
            "Mean_Uncertainty",
            "MAE"
        ]
    ].copy()

    # -----------------------------------------------------
    # Convert values to numeric
    # -----------------------------------------------------

    data["Mean_Uncertainty"] = pd.to_numeric(
        data["Mean_Uncertainty"],
        errors="coerce"
    )

    data["MAE"] = pd.to_numeric(
        data["MAE"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Remove invalid values
    # -----------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    # -----------------------------------------------------
    # Check number of observations
    # -----------------------------------------------------

    if len(data) < 2:

        raise ValueError(
            "At least two valid observations are required "
            "to calculate a correlation."
        )

    return data


# =========================================================
# CALCULATE CORRELATION
# =========================================================

def calculate_correlation(
    x,
    y
):
    """
    Calculate Pearson correlation coefficient.
    """

    if np.std(x) == 0:

        raise ValueError(
            "Mean uncertainty has zero variance. "
            "Pearson correlation cannot be calculated."
        )

    if np.std(y) == 0:

        raise ValueError(
            "MAE has zero variance. "
            "Pearson correlation cannot be calculated."
        )

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    return correlation


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("UNCERTAINTY VS RECONSTRUCTION ERROR")
    print("=" * 60)

    print()
    print(
        f"Dataset mode : {DATASET_MODE}"
    )

    print(
        f"Input CSV    : {CSV_FILE}"
    )

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    df = load_uncertainty_data()

    # -----------------------------------------------------
    # Validate data
    # -----------------------------------------------------

    data = validate_data(
        df
    )

    # -----------------------------------------------------
    # Extract variables
    # -----------------------------------------------------

    x = data[
        "Mean_Uncertainty"
    ].to_numpy()

    y = data[
        "MAE"
    ].to_numpy()

    # -----------------------------------------------------
    # Calculate Pearson correlation
    # -----------------------------------------------------

    correlation = calculate_correlation(
        x,
        y
    )

    # -----------------------------------------------------
    # Calculate least-squares regression line
    # -----------------------------------------------------

    m, b = np.polyfit(
        x,
        y,
        1
    )

    # -----------------------------------------------------
    # Generate regression line
    # -----------------------------------------------------

    x_line = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_line = (
        m * x_line
        + b
    )

    # -----------------------------------------------------
    # Calculate R-squared
    # -----------------------------------------------------

    r_squared = (
        correlation ** 2
    )

    # =====================================================
    # CREATE FIGURE
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # -----------------------------------------------------
    # Scatter plot
    # -----------------------------------------------------

    ax.scatter(
        x,
        y,
        alpha=0.8
    )

    # -----------------------------------------------------
    # Regression line
    # -----------------------------------------------------

    ax.plot(
        x_line,
        y_line,
        linewidth=2
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    ax.set_xlabel(
        "Mean Predictive Uncertainty"
    )

    ax.set_ylabel(
        "Reconstruction MAE"
    )

    ax.set_title(
        "Predictive Uncertainty vs Reconstruction Error\n"
        f"Pearson r = {correlation:.4f}, "
        f"R² = {r_squared:.4f}"
    )

    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------

    ax.grid(
        True
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.tight_layout()

    # =====================================================
    # SAVE FIGURE
    # =====================================================

    os.makedirs(
        UNCERTAINTY_DIRECTORY,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print(
        f"Valid observations : {len(data)}"
    )

    print(
        f"Pearson correlation: "
        f"{correlation:.4f}"
    )

    print(
        f"R-squared          : "
        f"{r_squared:.4f}"
    )

    print()
    print(
        "Regression equation:"
    )

    print(
        f"MAE = "
        f"{m:.6f} × Mean_Uncertainty "
        f"+ {b:.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()