"""
=========================================================
Baseline Comparison Plot
=========================================================

Creates publication-quality comparison charts for the
six classical baseline methods and the proposed
Physics-Informed 3D Encoder–Decoder model.

Input:
    baseline_comparison.csv

Expected methods:

    1. Nearest Neighbor
    2. Linear Interpolation
    3. f-x Prediction
    4. Compressive Sensing
    5. Curvelet POCS
    6. Dictionary Learning
    7. Proposed Model

Metrics:

    Error metrics:
        MAE
        RMSE

    Quality metrics:
        PSNR
        SNR
        SSIM

The input/output locations are obtained from config.py
through REPORT_DIR.

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import REPORT_DIR


# =========================================================
# CONFIGURATION
# =========================================================

CSV_FILE = os.path.join(
    REPORT_DIR,
    "baseline_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    REPORT_DIR,
    "baseline_metrics_comparison.png"
)


# =========================================================
# EXPECTED METHODS
# =========================================================

EXPECTED_METHODS = [
    "Nearest Neighbor",
    "Linear Interpolation",
    "f-x Prediction",
    "Compressive Sensing",
    "Curvelet POCS",
    "Dictionary Learning",
    "Proposed Model"
]


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "Method",
    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM"
]


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_baseline_results(csv_file):
    """
    Load and validate the baseline comparison CSV file.

    Validation includes:

        1. File existence
        2. Non-empty dataframe
        3. Required columns
        4. Valid method names
        5. Exactly one row per method
        6. Numerical metric validation
        7. Finite metric values
    """

    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not os.path.exists(csv_file):

        raise FileNotFoundError(
            f"\nBaseline comparison file not found:\n"
            f"{csv_file}\n\n"
            "Run evaluation.baselines."
            "compare_with_baselines.py first."
        )

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    df = pd.read_csv(
        csv_file
    )

    # -----------------------------------------------------
    # Check empty dataframe
    # -----------------------------------------------------

    if df.empty:

        raise ValueError(
            f"\nBaseline comparison file is empty:\n"
            f"{csv_file}"
        )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe baseline comparison file is missing "
            f"required columns:\n{missing_columns}"
        )

    # -----------------------------------------------------
    # Validate Method column
    # -----------------------------------------------------

    if df["Method"].isna().any():

        raise ValueError(
            "\nThe 'Method' column contains missing values."
        )

    df["Method"] = (
        df["Method"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Check duplicate methods
    # -----------------------------------------------------

    duplicate_methods = (
        df["Method"]
        [df["Method"].duplicated(keep=False)]
        .unique()
        .tolist()
    )

    if duplicate_methods:

        raise ValueError(
            "\nDuplicate method entries detected:\n"
            f"{duplicate_methods}\n\n"
            "The baseline comparison plot requires "
            "exactly one result row per method."
        )

    # -----------------------------------------------------
    # Check expected methods
    # -----------------------------------------------------

    found_methods = set(
        df["Method"].tolist()
    )

    expected_methods = set(
        EXPECTED_METHODS
    )

    missing_methods = sorted(
        expected_methods - found_methods
    )

    unexpected_methods = sorted(
        found_methods - expected_methods
    )

    if missing_methods:

        raise ValueError(
            "\nExpected baseline method(s) are missing:\n"
            f"{missing_methods}\n\n"
            "The direct baseline comparison should contain "
            "six classical baselines plus the Proposed Model."
        )

    if unexpected_methods:

        raise ValueError(
            "\nUnexpected method name(s) detected:\n"
            f"{unexpected_methods}\n\n"
            "Check the method names in "
            "compare_with_baselines.py."
        )

    # -----------------------------------------------------
    # Validate numerical metrics
    # -----------------------------------------------------

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"
    ]

    for metric in metric_columns:

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

        if df[metric].isna().any():

            raise ValueError(
                f"\nMetric '{metric}' contains "
                "missing or non-numeric values."
            )

        values = df[metric].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():

            raise ValueError(
                f"\nMetric '{metric}' contains "
                "non-finite values."
            )

    # -----------------------------------------------------
    # Reorder rows according to the defined experimental
    # method order.
    # -----------------------------------------------------

    method_order = {
        method: index
        for index, method in enumerate(
            EXPECTED_METHODS
        )
    }

    df["_method_order"] = (
        df["Method"].map(method_order)
    )

    df = (
        df.sort_values("_method_order")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    return df


# =========================================================
# CREATE PLOT
# =========================================================

def create_baseline_plot(
        df,
        output_file
):
    """
    Create and save the publication-quality baseline
    comparison figure.
    """

    methods = (
        df["Method"]
        .tolist()
    )

    # -----------------------------------------------------
    # Metric groups
    # -----------------------------------------------------

    error_metrics = [
        "MAE",
        "RMSE"
    ]

    quality_metrics = [
        "PSNR",
        "SNR",
        "SSIM"
    ]

    # -----------------------------------------------------
    # X-axis positions
    # -----------------------------------------------------

    error_x = np.arange(
        len(error_metrics)
    )

    quality_x = np.arange(
        len(quality_metrics)
    )

    # -----------------------------------------------------
    # Number of methods
    # -----------------------------------------------------

    num_methods = len(
        methods
    )

    # -----------------------------------------------------
    # Bar width
    # -----------------------------------------------------

    width = min(
        0.8 / max(num_methods, 1),
        0.16
    )

    # =====================================================
    # FIGURE
    # =====================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    ax_error = axes[0]
    ax_quality = axes[1]

    # =====================================================
    # ERROR METRICS
    # =====================================================

    for i, method in enumerate(
        methods
    ):

        method_row = df[
            df["Method"] == method
        ].iloc[0]

        values = [
            method_row[metric]
            for metric in error_metrics
        ]

        offset = (
            i
            - (num_methods - 1) / 2
        ) * width

        ax_error.bar(
            error_x + offset,
            values,
            width=width,
            label=method
        )

    ax_error.set_xticks(
        error_x
    )

    ax_error.set_xticklabels(
        error_metrics,
        fontsize=10
    )

    ax_error.set_ylabel(
        "Metric Value"
    )

    ax_error.set_title(
        "Reconstruction Error Metrics",
        fontweight="bold"
    )

    ax_error.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax_error.text(
        0.5,
        -0.15,
        "Lower values indicate lower reconstruction error",
        transform=ax_error.transAxes,
        ha="center",
        fontsize=9
    )

    # =====================================================
    # QUALITY METRICS
    # =====================================================

    for i, method in enumerate(
        methods
    ):

        method_row = df[
            df["Method"] == method
        ].iloc[0]

        values = [
            method_row[metric]
            for metric in quality_metrics
        ]

        offset = (
            i
            - (num_methods - 1) / 2
        ) * width

        ax_quality.bar(
            quality_x + offset,
            values,
            width=width,
            label=method
        )

    ax_quality.set_xticks(
        quality_x
    )

    ax_quality.set_xticklabels(
        quality_metrics,
        fontsize=10
    )

    ax_quality.set_ylabel(
        "Metric Value"
    )

    ax_quality.set_title(
        "Reconstruction Quality Metrics",
        fontweight="bold"
    )

    ax_quality.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax_quality.text(
        0.5,
        -0.15,
        "Higher values generally indicate better reconstruction quality",
        transform=ax_quality.transAxes,
        ha="center",
        fontsize=9
    )

    # =====================================================
    # FIGURE TITLE
    # =====================================================

    fig.suptitle(
        "Comparison of Seismic Reconstruction Methods",
        fontsize=15,
        fontweight="bold"
    )

    # =====================================================
    # SHARED LEGEND
    # =====================================================

    handles, labels = (
        ax_error.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=9
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.82
        ]
    )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        os.path.dirname(
            output_file
        ),
        exist_ok=True
    )

    # =====================================================
    # SAVE FIGURE
    # =====================================================

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("BASELINE COMPARISON PLOT")
    print("=" * 70)

    print()
    print("Input:")
    print(CSV_FILE)

    print()
    print("Loading and validating baseline results...")

    df = load_baseline_results(
        CSV_FILE
    )

    print()
    print("Validated methods:")

    for number, method in enumerate(
        df["Method"],
        start=1
    ):

        print(
            f"  {number}. {method}"
        )

    print()
    print(
        "Number of methods:",
        len(df)
    )

    print()
    print(
        "Creating comparison figure..."
    )

    create_baseline_plot(
        df,
        OUTPUT_FILE
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print(
        "BASELINE COMPARISON PLOT COMPLETE"
    )
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()