"""
=========================================================
Baseline Comparison Plot
=========================================================

Creates publication-quality comparison charts for the
reconstruction model and baseline methods.

The script reads the baseline comparison results generated
by:

    evaluation/compare_with_baselines.py

The input/output locations are obtained from config.py
through REPORT_DIR rather than using hard-coded paths.

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
    """

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Baseline comparison file not found:\n{csv_file}\n\n"
            "Run evaluation/compare_with_baselines.py first."
        )

    df = pd.read_csv(csv_file)

    if df.empty:
        raise ValueError(
            f"Baseline comparison file is empty:\n{csv_file}"
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The baseline comparison file is missing "
            f"required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Validate method names
    # -----------------------------------------------------

    if df["Method"].isna().any():
        raise ValueError(
            "The 'Method' column contains missing values."
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
                f"Metric '{metric}' contains "
                "missing or non-numeric values."
            )

        if not np.isfinite(df[metric].to_numpy()).all():
            raise ValueError(
                f"Metric '{metric}' contains "
                "non-finite values."
            )

    return df


# =========================================================
# CREATE PLOT
# =========================================================

def create_baseline_plot(df, output_file):
    """
    Create and save the publication-quality baseline
    comparison figure.
    """

    methods = df["Method"].astype(str).tolist()

    # -----------------------------------------------------
    # Metric groups
    #
    # Error metrics:
    #   Lower is better
    #
    # Quality metrics:
    #   Higher is generally better
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

    error_x = np.arange(len(error_metrics))
    quality_x = np.arange(len(quality_metrics))

    # Number of methods
    num_methods = len(methods)

    # Bar width adapts to the number of methods
    width = min(
        0.8 / max(num_methods, 1),
        0.25
    )

    # -----------------------------------------------------
    # Create figure with two panels
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax_error = axes[0]
    ax_quality = axes[1]

    # =====================================================
    # ERROR METRICS
    # =====================================================

    for i, method in enumerate(methods):

        values = [
            df.loc[i, metric]
            for metric in error_metrics
        ]

        offset = (
            i - (num_methods - 1) / 2
        ) * width

        ax_error.bar(
            error_x + offset,
            values,
            width=width,
            label=method
        )

    ax_error.set_xticks(error_x)

    ax_error.set_xticklabels(
        error_metrics
    )

    ax_error.set_ylabel(
        "Metric Value"
    )

    ax_error.set_title(
        "Reconstruction Error Metrics"
    )

    ax_error.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # Lower error is better
    ax_error.text(
        0.5,
        -0.14,
        "Lower values indicate better reconstruction",
        transform=ax_error.transAxes,
        ha="center",
        fontsize=9
    )

    # =====================================================
    # QUALITY METRICS
    # =====================================================

    for i, method in enumerate(methods):

        values = [
            df.loc[i, metric]
            for metric in quality_metrics
        ]

        offset = (
            i - (num_methods - 1) / 2
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
        quality_metrics
    )

    ax_quality.set_ylabel(
        "Metric Value"
    )

    ax_quality.set_title(
        "Reconstruction Quality Metrics"
    )

    ax_quality.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # Higher quality is generally better
    ax_quality.text(
        0.5,
        -0.14,
        "Higher values generally indicate better reconstruction",
        transform=ax_quality.transAxes,
        ha="center",
        fontsize=9
    )

    # =====================================================
    # FIGURE TITLE
    # =====================================================

    fig.suptitle(
        "Baseline Comparison of Seismic Reconstruction Methods",
        fontsize=14,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # Shared legend
    # -----------------------------------------------------

    handles, labels = ax_error.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(num_methods, 4),
        frameon=False
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    plt.tight_layout(
        rect=[0, 0.05, 1, 0.86]
    )

    # -----------------------------------------------------
    # Ensure output directory exists
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save high-resolution figure
    # -----------------------------------------------------

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("BASELINE COMPARISON PLOT")
    print("=" * 60)

    print()
    print("Input:")
    print(CSV_FILE)

    print()
    print("Loading baseline results...")

    df = load_baseline_results(
        CSV_FILE
    )

    print()
    print("Methods:")
    for method in df["Method"]:
        print(f"  - {method}")

    print()
    print("Creating comparison figure...")

    create_baseline_plot(
        df,
        OUTPUT_FILE
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=" * 60)
    print("BASELINE COMPARISON PLOT COMPLETE")
    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()