"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Compile existing evaluation outputs into a final
evaluation report.

IMPORTANT
---------

This module DOES NOT:

    - retrain the model
    - rerun model inference
    - rerun uncertainty analysis
    - rerun baseline reconstruction
    - rerun ablation experiments
    - rerun statistical significance testing
    - regenerate thesis tables from raw data

Those operations are handled by the appropriate
evaluation modules.

This module ONLY:

    1. Checks existing evaluation outputs.
    2. Loads existing CSV files.
    3. Validates required columns.
    4. Reports available figures.
    5. Reports available thesis tables.
    6. Records the best-model checkpoint.
    7. Records experiment configuration.
    8. Generates final_report.txt.

Required CSV outputs
--------------------

    evaluation_metrics.csv
    uncertainty_statistics.csv
    baseline_comparison.csv
    statistical_significance.csv
    ablation_study.csv
    ablation_summary.csv

Optional output directories
---------------------------

    gallery/
    uncertainty/
    reconstruction/
    thesis_tables/

Optional figures may include:

    baseline_error_metrics.png
    baseline_quality_metrics.png
    uncertainty_analysis.png
    reconstruction figures
    uncertainty correlation figures
    calibration figures
    robustness figures
    ablation figures

The report is configuration-driven and uses REPORT_DIR,
CHECKPOINT_DIR and EXPERIMENT_NAME from utils/config.py.

Author:
Ormin Joseph
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import os
from datetime import datetime

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
    CHECKPOINT_DIR,
    DATASET_MODE,
)


# =========================================================
# REPORT PATHS
# =========================================================

# Final text report.
FINAL_REPORT_FILE = os.path.join(
    REPORT_DIR,
    "final_report.txt",
)

# Best trained-model checkpoint.
CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth",
)


# =========================================================
# OPTIONAL OUTPUT DIRECTORIES
# =========================================================

# Reconstruction gallery.
GALLERY_DIR = os.path.join(
    REPORT_DIR,
    "gallery",
)

# Uncertainty outputs.
UNCERTAINTY_DIR = os.path.join(
    REPORT_DIR,
    "uncertainty",
)

# Reconstruction-analysis outputs.
RECONSTRUCTION_DIR = os.path.join(
    REPORT_DIR,
    "reconstruction",
)

# Thesis tables.
THESIS_TABLES_DIR = os.path.join(
    REPORT_DIR,
    "thesis_tables",
)


# =========================================================
# OPTIONAL FIGURE FILES
# =========================================================

BASELINE_ERROR_FIGURE = os.path.join(
    REPORT_DIR,
    "baseline_error_metrics.png",
)

BASELINE_QUALITY_FIGURE = os.path.join(
    REPORT_DIR,
    "baseline_quality_metrics.png",
)

BASELINE_COMPARISON_FIGURE = os.path.join(
    REPORT_DIR,
    "baseline_metrics_comparison.png",
)


# =========================================================
# REQUIRED CSV FILES
# =========================================================

REQUIRED_CSV_FILES = {

    "Evaluation Metrics":
        os.path.join(
            REPORT_DIR,
            "evaluation_metrics.csv",
        ),

    "Uncertainty Statistics":
        os.path.join(
            REPORT_DIR,
            "uncertainty_statistics.csv",
        ),

    "Baseline Comparison":
        os.path.join(
            REPORT_DIR,
            "baseline_comparison.csv",
        ),

    "Statistical Significance":
        os.path.join(
            REPORT_DIR,
            "statistical_significance.csv",
        ),

    "Ablation Study":
        os.path.join(
            REPORT_DIR,
            "ablation_study.csv",
        ),

    "Ablation Summary":
        os.path.join(
            REPORT_DIR,
            "ablation_summary.csv",
        ),
}


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = {

    # -----------------------------------------------------
    # Main model evaluation
    # -----------------------------------------------------

    "Evaluation Metrics": [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

    # -----------------------------------------------------
    # Uncertainty statistics
    #
    # The uncertainty evaluator may contain additional
    # fields. These are the minimum fields required for
    # this final report.
    # -----------------------------------------------------

    "Uncertainty Statistics": [
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
    ],

    # -----------------------------------------------------
    # Seven-method baseline comparison
    #
    # IMPORTANT:
    # The current baseline producer uses "Method",
    # NOT "Model".
    # -----------------------------------------------------

    "Baseline Comparison": [
        "Method",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

    # -----------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------

    "Statistical Significance": [
        "Comparison",
        "N_Pairs",
        "Raw_P_Value",
        "Holm_Adjusted_P_Value",
    ],

    # -----------------------------------------------------
    # Per-sample ablation results
    # -----------------------------------------------------

    "Ablation Study": [
        "Model",
        "Sample_ID",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

    # -----------------------------------------------------
    # Ablation summary
    # -----------------------------------------------------

    "Ablation Summary": [
        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],
}


# =========================================================
# EXPECTED BASELINE METHODS
# =========================================================

EXPECTED_BASELINE_METHODS = [
    "Nearest Neighbor",
    "Linear Interpolation",
    "f-x Prediction",
    "Compressive Sensing",
    "Curvelet POCS",
    "Dictionary Learning",
    "Proposed Physics-Informed 3D Encoder–Decoder",
]


# =========================================================
# FILE VALIDATION
# =========================================================

def validate_file(file_path):
    """
    Check whether a file exists and is non-empty.

    Parameters
    ----------
    file_path : str
        File to validate.

    Returns
    -------
    bool
        True if the file exists and is non-empty.
    """

    if not os.path.isfile(
        file_path
    ):

        return False

    if os.path.getsize(
        file_path
    ) == 0:

        return False

    return True


# =========================================================
# CSV LOADING
# =========================================================

def load_csv(file_path):
    """
    Safely load a CSV file.

    Parameters
    ----------
    file_path : str
        CSV file path.

    Returns
    -------
    pandas.DataFrame or None
        Loaded dataframe or None if loading fails.
    """

    try:

        dataframe = pd.read_csv(
            file_path
        )

        if dataframe.empty:

            return None

        return dataframe

    except Exception as error:

        print(
            "[WARNING] Could not read CSV:\n"
            f"{file_path}\n"
            f"Reason: {error}"
        )

        return None


# =========================================================
# COLUMN VALIDATION
# =========================================================

def validate_columns(
    dataframe,
    required_columns,
    table_name,
):
    """
    Validate that all required columns exist.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe being validated.

    required_columns : list
        Required column names.

    table_name : str
        Human-readable table name.

    Returns
    -------
    bool
        True when all required columns exist.
    """

    if dataframe is None:

        return False

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        print(
            f"[WARNING] {table_name} is missing "
            f"columns: {missing_columns}"
        )

        return False

    return True


# =========================================================
# BASELINE VALIDATION
# =========================================================

def validate_baseline_methods(
    dataframe,
):
    """
    Validate the seven-method baseline comparison.

    The method order is a controlled experimental order,
    not a performance ranking.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Baseline comparison dataframe.

    Returns
    -------
    bool
        True when the expected seven methods are present.
    """

    if dataframe is None:

        return False

    if "Method" not in dataframe.columns:

        print(
            "[WARNING] Baseline comparison does not "
            "contain the required 'Method' column."
        )

        return False

    methods = (
        dataframe["Method"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # -----------------------------------------------------
    # Check duplicates
    # -----------------------------------------------------

    if len(methods) != len(
        set(methods)
    ):

        duplicates = [
            method
            for method in set(methods)
            if methods.count(method) > 1
        ]

        print(
            "[WARNING] Duplicate baseline methods found:\n"
            f"{duplicates}"
        )

        return False

    # -----------------------------------------------------
    # Check missing expected methods
    # -----------------------------------------------------

    missing_methods = [
        method
        for method in EXPECTED_BASELINE_METHODS
        if method not in methods
    ]

    if missing_methods:

        print(
            "[WARNING] Missing expected baseline "
            f"method(s): {missing_methods}"
        )

        return False

    # -----------------------------------------------------
    # Check unexpected methods
    # -----------------------------------------------------

    unexpected_methods = [
        method
        for method in methods
        if method not in EXPECTED_BASELINE_METHODS
    ]

    if unexpected_methods:

        print(
            "[WARNING] Unexpected baseline "
            f"method(s): {unexpected_methods}"
        )

        return False

    # -----------------------------------------------------
    # Exactly seven methods
    # -----------------------------------------------------

    if len(methods) != len(
        EXPECTED_BASELINE_METHODS
    ):

        print(
            "[WARNING] Baseline comparison contains "
            f"{len(methods)} methods; expected "
            f"{len(EXPECTED_BASELINE_METHODS)}."
        )

        return False

    return True


# =========================================================
# PNG FILE DISCOVERY
# =========================================================

def get_png_files(directory):
    """
    Return PNG files from a directory.

    Parameters
    ----------
    directory : str
        Directory to inspect.

    Returns
    -------
    list
        Sorted PNG filenames.
    """

    if not os.path.isdir(
        directory
    ):

        return []

    return sorted(
        file
        for file in os.listdir(
            directory
        )
        if file.lower().endswith(
            ".png"
        )
    )


# =========================================================
# THESIS TABLE DISCOVERY
# =========================================================

def get_thesis_tables():
    """
    Return CSV files generated for thesis tables.

    Returns
    -------
    list
        Sorted thesis-table filenames.
    """

    if not os.path.isdir(
        THESIS_TABLES_DIR
    ):

        return []

    return sorted(
        file
        for file in os.listdir(
            THESIS_TABLES_DIR
        )
        if file.lower().endswith(
            ".csv"
        )
    )


# =========================================================
# SECTION WRITER
# =========================================================

def write_section(
    report,
    title,
):
    """
    Write a formatted section heading.
    """

    report.write(
        "\n"
    )

    report.write(
        "=" * 80
    )

    report.write(
        "\n"
    )

    report.write(
        title
    )

    report.write(
        "\n"
    )

    report.write(
        "=" * 80
    )

    report.write(
        "\n"
    )


# =========================================================
# DATAFRAME WRITER
# =========================================================

def write_dataframe(
    report,
    dataframe,
):
    """
    Write a DataFrame into the text report.
    """

    if dataframe is None:

        report.write(
            "No data available.\n"
        )

        return

    report.write(
        dataframe.to_string(
            index=False
        )
    )

    report.write(
        "\n"
    )


# =========================================================
# PATH WRITER
# =========================================================

def write_file_list(
    report,
    directory,
    files,
    empty_message,
):
    """
    Write discovered files to the report.
    """

    if files:

        for file in files:

            report.write(
                "  - "
                f"{os.path.join(directory, file)}"
                "\n"
            )

    else:

        report.write(
            f"  {empty_message}\n"
        )


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================

def generate_final_report():
    """
    Compile existing evaluation outputs into final_report.txt.

    No model computation is performed.
    """

    # =====================================================
    # HEADER
    # =====================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL REPORT GENERATOR"
    )

    print(
        "=" * 80
    )

    # -----------------------------------------------------
    # Ensure report directory exists
    # -----------------------------------------------------

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Store validation results
    # -----------------------------------------------------

    results = {}

    # =====================================================
    # CHECK REQUIRED CSV FILES
    # =====================================================

    print()
    print(
        "Checking required evaluation outputs..."
    )

    for name, file_path in (
        REQUIRED_CSV_FILES.items()
    ):

        # -------------------------------------------------
        # Check physical file
        # -------------------------------------------------

        if not validate_file(
            file_path
        ):

            results[name] = {
                "path": file_path,
                "dataframe": None,
                "valid": False,
            }

            print(
                f"[MISSING] {name}"
            )

            continue

        # -------------------------------------------------
        # Load CSV
        # -------------------------------------------------

        dataframe = load_csv(
            file_path
        )

        if dataframe is None:

            results[name] = {
                "path": file_path,
                "dataframe": None,
                "valid": False,
            }

            print(
                f"[INVALID] {name}"
            )

            continue

        # -------------------------------------------------
        # Validate required columns
        # -------------------------------------------------

        valid = validate_columns(
            dataframe,
            REQUIRED_COLUMNS[name],
            name,
        )

        # -------------------------------------------------
        # Additional baseline validation
        # -------------------------------------------------

        if (
            valid
            and name == "Baseline Comparison"
        ):

            valid = validate_baseline_methods(
                dataframe
            )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results[name] = {
            "path": file_path,
            "dataframe": dataframe,
            "valid": valid,
        }

        if valid:

            print(
                f"[AVAILABLE] {name}"
            )

        else:

            print(
                f"[INVALID] {name}"
            )

    # =====================================================
    # DISCOVER OPTIONAL ARTIFACTS
    # =====================================================

    gallery_files = get_png_files(
        GALLERY_DIR
    )

    uncertainty_files = get_png_files(
        UNCERTAINTY_DIR
    )

    reconstruction_files = get_png_files(
        RECONSTRUCTION_DIR
    )

    thesis_tables = get_thesis_tables()

    # -----------------------------------------------------
    # Individual baseline figures
    # -----------------------------------------------------

    baseline_error_available = (
        validate_file(
            BASELINE_ERROR_FIGURE
        )
    )

    baseline_quality_available = (
        validate_file(
            BASELINE_QUALITY_FIGURE
        )
    )

    baseline_comparison_available = (
        validate_file(
            BASELINE_COMPARISON_FIGURE
        )
    )

    # =====================================================
    # WRITE FINAL REPORT
    # =====================================================

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        # =================================================
        # REPORT HEADER
        # =================================================

        report.write(
            "PHYSICS-INFORMED 3D SEISMIC "
            "RECONSTRUCTION\n"
        )

        report.write(
            "FINAL EVALUATION REPORT\n"
        )

        report.write(
            "=" * 80
        )

        report.write(
            "\n\n"
        )

        report.write(
            f"Experiment: "
            f"{EXPERIMENT_NAME}\n"
        )

        report.write(
            f"Dataset Mode: "
            f"{DATASET_MODE}\n"
        )

        report.write(
            f"Report Directory: "
            f"{REPORT_DIR}\n"
        )

        report.write(
            f"Best Checkpoint Directory: "
            f"{CHECKPOINT_DIR}\n"
        )

        report.write(
            f"Report Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # =================================================
        # 1. MODEL EVALUATION
        # =================================================

        write_section(
            report,
            "1. MODEL EVALUATION",
        )

        evaluation_result = results.get(
            "Evaluation Metrics"
        )

        if (
            evaluation_result
            and evaluation_result["valid"]
        ):

            write_dataframe(
                report,
                evaluation_result[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Evaluation metrics are "
                "unavailable or invalid.\n"
            )

        # =================================================
        # 2. PREDICTIVE UNCERTAINTY
        # =================================================

        write_section(
            report,
            "2. PREDICTIVE UNCERTAINTY",
        )

        uncertainty_result = results.get(
            "Uncertainty Statistics"
        )

        if (
            uncertainty_result
            and uncertainty_result["valid"]
        ):

            write_dataframe(
                report,
                uncertainty_result[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Uncertainty statistics are "
                "unavailable or invalid.\n"
            )

        report.write(
            "\nUncertainty figures:\n"
        )

        write_file_list(
            report,
            UNCERTAINTY_DIR,
            uncertainty_files,
            "None found.",
        )

        # =================================================
        # 3. BASELINE COMPARISON
        # =================================================

        write_section(
            report,
            "3. BASELINE COMPARISON",
        )

        baseline_result = results.get(
            "Baseline Comparison"
        )

        if (
            baseline_result
            and baseline_result["valid"]
        ):

            report.write(
                "Validated seven-method "
                "baseline comparison:\n\n"
            )

            write_dataframe(
                report,
                baseline_result[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Baseline comparison results "
                "are unavailable or invalid.\n"
            )

        report.write(
            "\nBaseline figures:\n"
        )

        if baseline_error_available:

            report.write(
                f"  - "
                f"{BASELINE_ERROR_FIGURE}\n"
            )

        if baseline_quality_available:

            report.write(
                f"  - "
                f"{BASELINE_QUALITY_FIGURE}\n"
            )

        if baseline_comparison_available:

            report.write(
                f"  - "
                f"{BASELINE_COMPARISON_FIGURE}\n"
            )

        if not any(
            [
                baseline_error_available,
                baseline_quality_available,
                baseline_comparison_available,
            ]
        ):

            report.write(
                "  None found.\n"
            )

        # =================================================
        # 4. STATISTICAL SIGNIFICANCE
        # =================================================

        write_section(
            report,
            "4. STATISTICAL SIGNIFICANCE",
        )

        significance_result = results.get(
            "Statistical Significance"
        )

        if (
            significance_result
            and significance_result["valid"]
        ):

            write_dataframe(
                report,
                significance_result[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Statistical significance results "
                "are unavailable or invalid.\n"
            )

        # =================================================
        # 5. ABLATION STUDY
        # =================================================

        write_section(
            report,
            "5. ABLATION STUDY",
        )

        # -------------------------------------------------
        # Ablation summary
        # -------------------------------------------------

        ablation_summary = results.get(
            "Ablation Summary"
        )

        if (
            ablation_summary
            and ablation_summary["valid"]
        ):

            report.write(
                "Ablation Summary:\n\n"
            )

            write_dataframe(
                report,
                ablation_summary[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Ablation summary is "
                "unavailable or invalid.\n"
            )

        report.write(
            "\n"
        )

        # -------------------------------------------------
        # Per-sample ablation
        # -------------------------------------------------

        ablation_study = results.get(
            "Ablation Study"
        )

        if (
            ablation_study
            and ablation_study["valid"]
        ):

            report.write(
                "Per-Sample Ablation Results:\n\n"
            )

            write_dataframe(
                report,
                ablation_study[
                    "dataframe"
                ],
            )

        else:

            report.write(
                "Per-sample ablation results "
                "are unavailable or invalid.\n"
            )

        # =================================================
        # 6. RECONSTRUCTION ANALYSIS
        # =================================================

        write_section(
            report,
            "6. RECONSTRUCTION ANALYSIS",
        )

        report.write(
            "Available reconstruction-analysis "
            "figures:\n"
        )

        write_file_list(
            report,
            RECONSTRUCTION_DIR,
            reconstruction_files,
            "None found.",
        )

        # =================================================
        # 7. THESIS TABLES
        # =================================================

        write_section(
            report,
            "7. THESIS TABLES",
        )

        if thesis_tables:

            report.write(
                "Generated thesis table files:\n\n"
            )

            for table in thesis_tables:

                report.write(
                    f"  - "
                    f"{os.path.join(THESIS_TABLES_DIR, table)}"
                    "\n"
                )

        else:

            report.write(
                "No thesis table files found.\n"
            )

        # =================================================
        # 8. RECONSTRUCTION GALLERY
        # =================================================

        write_section(
            report,
            "8. RECONSTRUCTION GALLERY",
        )

        report.write(
            "Generated gallery figures:\n"
        )

        write_file_list(
            report,
            GALLERY_DIR,
            gallery_files,
            "None found.",
        )

        # =================================================
        # 9. BEST MODEL CHECKPOINT
        # =================================================

        write_section(
            report,
            "9. BEST MODEL CHECKPOINT",
        )

        if validate_file(
            CHECKPOINT_FILE
        ):

            checkpoint_size = (
                os.path.getsize(
                    CHECKPOINT_FILE
                )
            )

            report.write(
                "Best checkpoint:\n"
            )

            report.write(
                f"{CHECKPOINT_FILE}\n"
            )

            report.write(
                f"Checkpoint size: "
                f"{checkpoint_size:,} bytes\n"
            )

        else:

            report.write(
                "Best model checkpoint was "
                "not found.\n"
            )

        # =================================================
        # 10. OUTPUT STATUS
        # =================================================

        write_section(
            report,
            "10. OUTPUT STATUS",
        )

        valid_csv_outputs = sum(
            1
            for result in results.values()
            if result["valid"]
        )

        total_required_csv = len(
            results
        )

        missing_csv_outputs = [
            name
            for name, result in results.items()
            if not result["valid"]
        ]

        report.write(
            f"Valid required CSV outputs: "
            f"{valid_csv_outputs}/"
            f"{total_required_csv}\n"
        )

        report.write(
            f"Gallery figures: "
            f"{len(gallery_files)}\n"
        )

        report.write(
            f"Uncertainty figures: "
            f"{len(uncertainty_files)}\n"
        )

        report.write(
            f"Reconstruction figures: "
            f"{len(reconstruction_files)}\n"
        )

        report.write(
            f"Thesis tables: "
            f"{len(thesis_tables)}\n"
        )

        report.write(
            f"Baseline error figure: "
            f"{'Available' if baseline_error_available else 'Not found'}\n"
        )

        report.write(
            f"Baseline quality figure: "
            f"{'Available' if baseline_quality_available else 'Not found'}\n"
        )

        report.write(
            f"Baseline comparison figure: "
            f"{'Available' if baseline_comparison_available else 'Not found'}\n"
        )

        report.write(
            f"Best checkpoint: "
            f"{'Available' if validate_file(CHECKPOINT_FILE) else 'Not found'}\n"
        )

        if missing_csv_outputs:

            report.write(
                "\nMissing or invalid required CSV outputs:\n"
            )

            for name in missing_csv_outputs:

                report.write(
                    f"  - {name}\n"
                )

            report.write(
                "\nSTATUS: SOME REQUIRED "
                "EVALUATION OUTPUTS ARE "
                "MISSING OR INVALID.\n"
            )

        else:

            report.write(
                "\nSTATUS: ALL REQUIRED CSV "
                "EVALUATION OUTPUTS ARE "
                "AVAILABLE AND VALID.\n"
            )

        # =================================================
        # 11. SCOPE OF THIS REPORT
        # =================================================

        write_section(
            report,
            "11. SCOPE OF THIS REPORT",
        )

        report.write(
            "This document is a compilation of "
            "existing evaluation outputs.\n\n"
        )

        report.write(
            "The final report generator does not "
            "perform model training, inference, "
            "uncertainty estimation, baseline "
            "reconstruction, ablation experiments, "
            "robustness experiments, or statistical "
            "hypothesis testing.\n\n"
        )

        report.write(
            "All numerical results reported here "
            "are read from previously generated "
            "evaluation files.\n"
        )

        # =================================================
        # 12. EVALUATION MODULE SEPARATION
        # =================================================

        write_section(
            report,
            "12. EVALUATION MODULE SEPARATION",
        )

        report.write(
            "The evaluation framework separates "
            "the following analyses:\n\n"
        )

        report.write(
            "1. Main reconstruction evaluation\n"
        )

        report.write(
            "2. Predictive uncertainty analysis\n"
        )

        report.write(
            "3. Uncertainty-error association\n"
        )

        report.write(
            "4. Formal uncertainty calibration\n"
        )

        report.write(
            "5. Baseline reconstruction comparison\n"
        )

        report.write(
            "6. Ablation analysis\n"
        )

        report.write(
            "7. Noise robustness\n"
        )

        report.write(
            "8. Missing-data robustness\n"
        )

        report.write(
            "9. Geological-complexity robustness\n"
        )

        report.write(
            "10. Statistical significance analysis\n"
        )

        report.write(
            "11. Thesis-table generation\n"
        )

        report.write(
            "12. Final report compilation\n"
        )

        # =================================================
        # 13. FINAL NOTE
        # =================================================

        write_section(
            report,
            "13. FINAL NOTE",
        )

        report.write(
            "This report should be interpreted together "
            "with the underlying CSV files, figures, "
            "checkpoint, and individual evaluation "
            "module outputs.\n"
        )

    # =====================================================
    # CONSOLE SUMMARY
    # =====================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL REPORT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Report saved to:"
    )

    print(
        FINAL_REPORT_FILE
    )

    print()
    print(
        f"Valid required CSV outputs: "
        f"{valid_csv_outputs}/"
        f"{total_required_csv}"
    )

    print(
        f"Gallery figures: "
        f"{len(gallery_files)}"
    )

    print(
        f"Uncertainty figures: "
        f"{len(uncertainty_files)}"
    )

    print(
        f"Reconstruction figures: "
        f"{len(reconstruction_files)}"
    )

    print(
        f"Thesis tables: "
        f"{len(thesis_tables)}"
    )

    print(
        f"Best checkpoint: "
        f"{'Available' if validate_file(CHECKPOINT_FILE) else 'Not found'}"
    )

    print()

    if missing_csv_outputs:

        print(
            "STATUS: SOME REQUIRED "
            "EVALUATION OUTPUTS ARE "
            "MISSING OR INVALID."
        )

        print()
        print(
            "Missing/invalid outputs:"
        )

        for name in missing_csv_outputs:

            print(
                f"  - {name}"
            )

    else:

        print(
            "STATUS: ALL REQUIRED CSV "
            "EVALUATION OUTPUTS ARE "
            "AVAILABLE AND VALID."
        )

    print()

    print(
        "=" * 80
    )

    return FINAL_REPORT_FILE


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    generate_final_report()