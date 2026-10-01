"""
=====================================================================
FINAL PhD BASELINE COMPARISON PLOT
=====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

Purpose
-------
Create a publication-quality figure comparing the six classical
baseline reconstruction methods with the proposed Physics-Informed
3D Encoder–Decoder framework.

Input
-----

    baseline_comparison.csv

Expected methods
----------------

    1. Nearest Neighbor
    2. Linear Interpolation
    3. f-x Prediction
    4. Compressive Sensing
    5. Curvelet POCS
    6. Dictionary Learning
    7. Proposed Physics-Informed 3D Encoder–Decoder

Metrics
-------

    MAE
    RMSE
    PSNR
    SNR
    SSIM

Experimental principle
----------------------

The input CSV must originate from the common seven-method
reconstruction comparison in:

    evaluation/baselines/compare_with_baselines.py

All methods must therefore have been evaluated using the same:

    * target seismic cube;
    * corrupted seismic cube;
    * missing-data mask;
    * geological configuration;
    * random seed;
    * evaluation metrics.

This plotting module performs NO re-evaluation.

It only validates, organizes, and visualizes the already generated
baseline-comparison results.

Output
------

    <REPORT_DIR>/
        baseline_metrics_comparison.png

The figure contains five panels:

    (a) MAE
    (b) RMSE
    (c) PSNR
    (d) SNR
    (e) SSIM

Metric interpretation
---------------------

MAE:
    Lower values indicate lower reconstruction error.

RMSE:
    Lower values indicate lower reconstruction error.

PSNR:
    Higher values indicate greater reconstruction fidelity.

SNR:
    Higher values indicate greater signal preservation relative
    to reconstruction error.

SSIM:
    Higher values indicate greater structural similarity.

Important
---------

This figure is descriptive.

It does not perform statistical significance testing and does not
assign an overall ranking or winner to the reconstruction methods.

Formal statistical analysis belongs to the dedicated statistical
significance evaluation module.

=====================================================================
"""

# =====================================================================
# STANDARD LIBRARY
# =====================================================================

from pathlib import Path


# =====================================================================
# THIRD-PARTY LIBRARIES
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# PROJECT CONFIGURATION
# =====================================================================

from utils.config import REPORT_DIR


# =====================================================================
# 1. PATH CONFIGURATION
# =====================================================================

"""
The baseline comparison CSV is produced by:

    evaluation/baselines/compare_with_baselines.py

Both input and output paths are derived from REPORT_DIR so that
changing the experiment configuration automatically redirects the
plot to the corresponding experiment directory.
"""

REPORT_PATH = Path(REPORT_DIR)

CSV_FILE = (
    REPORT_PATH
    / "baseline_comparison.csv"
)

OUTPUT_FILE = (
    REPORT_PATH
    / "baseline_metrics_comparison.png"
)


# =====================================================================
# 2. EXPECTED METHOD ORDER
# =====================================================================

"""
This order must agree with the seven-method comparison experiment.

The proposed method name is deliberately written exactly as expected
from the current baseline comparison implementation.
"""

EXPECTED_METHODS = [
    "Nearest Neighbor",
    "Linear Interpolation",
    "f-x Prediction",
    "Compressive Sensing",
    "Curvelet POCS",
    "Dictionary Learning",
    "Proposed Physics-Informed 3D Encoder–Decoder",
]


# =====================================================================
# 3. REQUIRED CSV COLUMNS
# =====================================================================

REQUIRED_COLUMNS = [
    "Method",
    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM",
]


# =====================================================================
# 4. METRIC DEFINITIONS
# =====================================================================

METRICS = [
    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM",
]


# =====================================================================
# 5. METRIC INTERPRETATION
# =====================================================================

METRIC_DIRECTION = {
    "MAE": "Lower is better",
    "RMSE": "Lower is better",
    "PSNR": "Higher is better",
    "SNR": "Higher is better",
    "SSIM": "Higher is better",
}


# =====================================================================
# 6. LOAD AND VALIDATE BASELINE RESULTS
# =====================================================================

def load_baseline_results(csv_file):
    """
    Load and rigorously validate the baseline comparison results.

    Validation includes:

        1. Input-file existence.
        2. Non-empty CSV.
        3. Required columns.
        4. Missing method names.
        5. Unexpected method names.
        6. Duplicate methods.
        7. Numeric metric values.
        8. Finite metric values.
        9. Exact seven-method experimental structure.
       10. Reproducible method ordering.

    Parameters
    ----------
    csv_file : Path
        Baseline comparison CSV.

    Returns
    -------
    pandas.DataFrame
        Validated and ordered results.
    """

    # -----------------------------------------------------------------
    # Check input file
    # -----------------------------------------------------------------

    if not csv_file.is_file():

        raise FileNotFoundError(
            "\nBaseline comparison file was not found:\n"
            f"{csv_file}\n\n"
            "Run:\n"
            "evaluation/baselines/"
            "compare_with_baselines.py\n"
            "before running this plotting module."
        )

    # -----------------------------------------------------------------
    # Load CSV
    # -----------------------------------------------------------------

    dataframe = pd.read_csv(
        csv_file
    )

    # -----------------------------------------------------------------
    # Validate non-empty dataframe
    # -----------------------------------------------------------------

    if dataframe.empty:

        raise ValueError(
            "\nThe baseline comparison CSV is empty:\n"
            f"{csv_file}"
        )

    # -----------------------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe baseline comparison CSV is missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    # -----------------------------------------------------------------
    # Validate Method column
    # -----------------------------------------------------------------

    if dataframe["Method"].isna().any():

        raise ValueError(
            "\nThe 'Method' column contains missing values."
        )

    dataframe["Method"] = (
        dataframe["Method"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------------------
    # Validate duplicate methods
    # -----------------------------------------------------------------

    duplicate_methods = (
        dataframe.loc[
            dataframe["Method"].duplicated(
                keep=False
            ),
            "Method",
        ]
        .unique()
        .tolist()
    )

    if duplicate_methods:

        raise ValueError(
            "\nDuplicate method entries detected:\n"
            f"{duplicate_methods}\n\n"
            "The common seven-method comparison requires "
            "exactly one result row per method."
        )

    # -----------------------------------------------------------------
    # Identify methods present in the CSV
    # -----------------------------------------------------------------

    found_methods = set(
        dataframe["Method"].tolist()
    )

    expected_methods = set(
        EXPECTED_METHODS
    )

    # -----------------------------------------------------------------
    # Missing methods
    # -----------------------------------------------------------------

    missing_methods = sorted(
        expected_methods - found_methods
    )

    if missing_methods:

        raise ValueError(
            "\nExpected reconstruction method(s) are missing:\n"
            f"{missing_methods}\n\n"
            "Check baseline_comparison.csv and ensure that "
            "all seven reconstruction methods were completed."
        )

    # -----------------------------------------------------------------
    # Unexpected methods
    # -----------------------------------------------------------------

    unexpected_methods = sorted(
        found_methods - expected_methods
    )

    if unexpected_methods:

        raise ValueError(
            "\nUnexpected reconstruction method name(s) detected:\n"
            f"{unexpected_methods}\n\n"
            "The plotting module expects exactly the seven methods "
            "defined by the common baseline comparison."
        )

    # -----------------------------------------------------------------
    # Validate exact number of methods
    # -----------------------------------------------------------------

    if len(dataframe) != len(EXPECTED_METHODS):

        raise ValueError(
            "\nUnexpected number of baseline rows.\n"
            f"Expected: {len(EXPECTED_METHODS)}\n"
            f"Received: {len(dataframe)}"
        )

    # -----------------------------------------------------------------
    # Validate numerical metrics
    # -----------------------------------------------------------------

    for metric in METRICS:

        dataframe[metric] = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        # -------------------------------------------------------------
        # Missing/non-numeric values
        # -------------------------------------------------------------

        if dataframe[metric].isna().any():

            invalid_rows = dataframe.loc[
                dataframe[metric].isna(),
                ["Method", metric],
            ]

            raise ValueError(
                f"\nMetric '{metric}' contains missing or "
                f"non-numeric values:\n"
                f"{invalid_rows}"
            )

        # -------------------------------------------------------------
        # Finite-value validation
        # -------------------------------------------------------------

        values = dataframe[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():

            raise ValueError(
                f"\nMetric '{metric}' contains NaN or "
                "infinite values."
            )

    # -----------------------------------------------------------------
    # Create reproducible method ordering
    # -----------------------------------------------------------------

    method_order = {
        method: index
        for index, method in enumerate(
            EXPECTED_METHODS
        )
    }

    dataframe["_method_order"] = (
        dataframe["Method"].map(
            method_order
        )
    )

    dataframe = (
        dataframe
        .sort_values(
            "_method_order"
        )
        .drop(
            columns="_method_order"
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------------------
    # Final method-order verification
    # -----------------------------------------------------------------

    if (
        dataframe["Method"].tolist()
        != EXPECTED_METHODS
    ):

        raise RuntimeError(
            "\nMethod ordering validation failed."
        )

    return dataframe


# =====================================================================
# 7. CREATE PUBLICATION FIGURE
# =====================================================================

def create_baseline_plot(
    dataframe,
    output_file,
):
    """
    Create a five-panel publication-quality comparison figure.

    Each metric receives an independent panel because MAE/RMSE,
    PSNR/SNR and SSIM have different numerical scales.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Validated baseline results.

    output_file : Path
        Destination PNG file.
    """

    # -----------------------------------------------------------------
    # Extract methods
    # -----------------------------------------------------------------

    methods = dataframe[
        "Method"
    ].tolist()

    # -----------------------------------------------------------------
    # X positions
    # -----------------------------------------------------------------

    x_positions = np.arange(
        len(methods)
    )

    # -----------------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(13, 20),
        sharex=True,
    )

    # -----------------------------------------------------------------
    # Handle matplotlib axis container
    # -----------------------------------------------------------------

    axes = np.asarray(
        axes
    )

    # -----------------------------------------------------------------
    # Plot every metric independently
    # -----------------------------------------------------------------

    for axis, metric in zip(
        axes,
        METRICS,
    ):

        values = dataframe[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        # -------------------------------------------------------------
        # Bars
        # -------------------------------------------------------------

        bars = axis.bar(
            x_positions,
            values,
            width=0.68,
            edgecolor="black",
            linewidth=0.7,
        )

        # -------------------------------------------------------------
        # Axis label
        # -------------------------------------------------------------

        axis.set_ylabel(
            metric,
            fontsize=11,
            fontweight="bold",
        )

        # -------------------------------------------------------------
        # Panel title
        # -------------------------------------------------------------

        axis.set_title(
            f"{metric} — "
            f"{METRIC_DIRECTION[metric]}",
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=8,
        )

        # -------------------------------------------------------------
        # Grid
        # -------------------------------------------------------------

        axis.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.30,
        )

        # -------------------------------------------------------------
        # Keep grid behind bars
        # -------------------------------------------------------------

        axis.set_axisbelow(
            True
        )

        # -------------------------------------------------------------
        # Remove unnecessary top/right borders
        # -------------------------------------------------------------

        axis.spines[
            "top"
        ].set_visible(False)

        axis.spines[
            "right"
        ].set_visible(False)

        # -------------------------------------------------------------
        # Value labels
        # -------------------------------------------------------------

        for bar, value in zip(
            bars,
            values,
        ):

            axis.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                clip_on=False,
            )

    # =================================================================
    # X-AXIS
    # =================================================================

    axes[-1].set_xticks(
        x_positions
    )

    axes[-1].set_xticklabels(
        methods,
        rotation=25,
        ha="right",
        fontsize=10,
    )

    axes[-1].set_xlabel(
        "Reconstruction Method",
        fontsize=11,
        fontweight="bold",
    )

    # =================================================================
    # OVERALL TITLE
    # =================================================================

    fig.suptitle(
        "Comparison of Seismic Reconstruction Methods",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    # =================================================================
    # FIGURE DESCRIPTION
    # =================================================================

    fig.text(
        0.5,
        0.978,
        "Common seven-method reconstruction comparison",
        ha="center",
        va="top",
        fontsize=10,
    )

    # =================================================================
    # LAYOUT
    # =================================================================

    fig.tight_layout(
        rect=[
            0.03,
            0.03,
            0.99,
            0.965,
        ]
    )

    # =================================================================
    # OUTPUT DIRECTORY
    # =================================================================

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # SAVE HIGH-RESOLUTION FIGURE
    # =================================================================

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    # =================================================================
    # CLOSE FIGURE
    # =================================================================

    plt.close(
        fig
    )


# =====================================================================
# 8. PRINT VALIDATED RESULTS
# =====================================================================

def print_validation_summary(
    dataframe,
):
    """
    Print a concise validation summary before plotting.
    """

    print()
    print(
        "Validated reconstruction methods:"
    )

    for number, method in enumerate(
        dataframe["Method"],
        start=1,
    ):

        print(
            f"  {number}. {method}"
        )

    print()
    print(
        "Validated metrics:"
    )

    for metric in METRICS:

        print(
            f"  {metric}: "
            f"{dataframe[metric].min():.6f} "
            f"to "
            f"{dataframe[metric].max():.6f}"
        )

    print()
    print(
        "Method count:",
        len(dataframe),
    )


# =====================================================================
# 9. MAIN
# =====================================================================

def main():
    """
    Execute the complete baseline comparison plotting workflow.
    """

    print()
    print("=" * 70)
    print("BASELINE COMPARISON PLOT")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Display paths
    # -----------------------------------------------------------------

    print()
    print(
        "Input file:"
    )

    print(
        CSV_FILE
    )

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )

    # -----------------------------------------------------------------
    # Load and validate
    # -----------------------------------------------------------------

    print()
    print(
        "Loading and validating baseline results..."
    )

    dataframe = load_baseline_results(
        CSV_FILE
    )

    # -----------------------------------------------------------------
    # Validation summary
    # -----------------------------------------------------------------

    print_validation_summary(
        dataframe
    )

    # -----------------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------------

    print()
    print(
        "Creating publication-quality figure..."
    )

    create_baseline_plot(
        dataframe,
        OUTPUT_FILE,
    )

    # -----------------------------------------------------------------
    # Confirm output
    # -----------------------------------------------------------------

    if not OUTPUT_FILE.is_file():

        raise RuntimeError(
            "\nThe baseline comparison figure was not "
            "created successfully."
        )

    print()
    print(
        "Figure saved successfully:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print(
        "BASELINE COMPARISON PLOT COMPLETE"
    )
    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()