"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

PURPOSE
-------
This script ONLY compiles previously generated experimental
outputs into a final thesis-oriented report.

This script DOES NOT:

    - rerun training
    - rerun model inference
    - rerun evaluation
    - retrain ablation models
    - recompute uncertainty
    - recompute baseline results
    - recompute statistical tests

All experimental results must already exist in the active
experiment directory defined by:

    utils.config.EXPERIMENT_NAME
    utils.config.REPORT_DIR

EXPECTED EXPERIMENT STRUCTURE
-----------------------------

outputs/
    <EXPERIMENT_NAME>/

        checkpoints/
            best_model.pth
            latest_checkpoint.pth
            ...

        reports/
            evaluation_metrics.csv
            uncertainty_statistics.csv
            baseline_comparison.csv
            statistical_significance.csv

            ablation_study.csv
            ablation_summary.csv

            gallery/
                *.png

            uncertainty/
                *.png

            thesis_tables/
                *.csv

            final_report.txt
            final_report_metadata.json


FINAL REPORT ROLE
-----------------

The generated report is an archival compilation of
previously produced experimental outputs.

It is therefore intentionally separated from:

    Training
    Inference
    Evaluation
    Uncertainty Analysis
    Baseline Comparison
    Ablation Experiments
    Statistical Testing

=========================================================
"""

# =========================================================
# STANDARD LIBRARY
# =========================================================

import json
from datetime import datetime
from pathlib import Path


# =========================================================
# THIRD-PARTY LIBRARIES
# =========================================================

import numpy as np
import pandas as pd


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# =========================================================
# REPORT PATHS
# =========================================================

REPORT_PATH = Path(REPORT_DIR)

FINAL_REPORT_FILE = (
    REPORT_PATH / "final_report.txt"
)

FINAL_REPORT_METADATA_FILE = (
    REPORT_PATH / "final_report_metadata.json"
)


# =========================================================
# CHECKPOINT PATH
# =========================================================
#
# The checkpoint directory is located beside the reports
# directory under the current experiment.
#
# Example:
#
# outputs/
#     synthetic_training/
#         checkpoints/
#         reports/
#
# Therefore:
#
# REPORT_DIR / ".." / "checkpoints"
#
# resolves to the correct experiment checkpoint directory.
# =========================================================

BEST_CHECKPOINT_FILE = (
    REPORT_PATH.parent
    / "checkpoints"
    / "best_model.pth"
)


# =========================================================
# REQUIRED RESULT FILES
# =========================================================
#
# These files are necessary for the analytical sections
# of the final report.
# =========================================================

REQUIRED_FILES = {

    "Evaluation Metrics":
        "evaluation_metrics.csv",

    "Uncertainty Statistics":
        "uncertainty_statistics.csv",

    "Baseline Comparison":
        "baseline_comparison.csv",

    "Statistical Significance":
        "statistical_significance.csv",

    "Ablation Study (Per Sample)":
        "ablation_study.csv",

    "Ablation Summary":
        "ablation_summary.csv",

}


# =========================================================
# OPTIONAL RESULT DIRECTORIES
# =========================================================
#
# Their absence does NOT prevent final report generation.
# =========================================================

OPTIONAL_DIRECTORIES = {

    "Reconstruction Gallery":
        "gallery",

    "Uncertainty Figures":
        "uncertainty",

    "Thesis Tables":
        "thesis_tables",

}


# =========================================================
# REQUIRED CSV STRUCTURES
# =========================================================

REQUIRED_COLUMNS = {

    "Evaluation Metrics": (

        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",

    ),

    "Baseline Comparison": (

        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",

    ),

    "Statistical Significance": (

        "Comparison",
        "N_Pairs",
        "Raw_P_Value",
        "Holm_Adjusted_P_Value",

    ),

    "Ablation Study (Per Sample)": (

        "Model",
        "Sample_ID",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",

    ),

    "Ablation Summary": (

        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",

    ),

}


# =========================================================
# NUMERIC METRIC COLUMNS
# =========================================================

NUMERIC_METRIC_COLUMNS = (

    "MAE",
    "RMSE",
    "PSNR",
    "SNR",
    "SSIM",

)


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_file_path(filename):
    """
    Construct a path inside the active experiment's
    report directory.

    Parameters
    ----------
    filename : str
        Filename inside REPORT_DIR.

    Returns
    -------
    pathlib.Path
        Complete report path.
    """

    return REPORT_PATH / filename


# ---------------------------------------------------------
# Validate file
# ---------------------------------------------------------

def validate_file(filepath):
    """
    Check whether a file exists and is non-empty.

    Parameters
    ----------
    filepath : pathlib.Path
        File to validate.

    Returns
    -------
    bool
        True if the file exists and contains data.
    """

    if not filepath.exists():

        return False

    if not filepath.is_file():

        return False

    if filepath.stat().st_size == 0:

        return False

    return True


# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------

def load_csv(filename):
    """
    Load a CSV file from REPORT_DIR.

    Parameters
    ----------
    filename : str
        CSV filename.

    Returns
    -------
    pandas.DataFrame or None
        Loaded dataframe, or None if loading fails.
    """

    filepath = get_file_path(
        filename
    )

    if not validate_file(filepath):

        print(
            f"[WARNING] Missing or empty file: {filepath}"
        )

        return None

    try:

        dataframe = pd.read_csv(
            filepath
        )

    except Exception as error:

        print(
            f"[WARNING] Could not read: {filepath}"
        )

        print(
            f"          Reason: {error}"
        )

        return None

    if dataframe.empty:

        print(
            f"[WARNING] CSV contains no rows: {filepath}"
        )

        return None

    return dataframe


# ---------------------------------------------------------
# Validate dataframe columns
# ---------------------------------------------------------

def validate_columns(
        dataframe,
        required_columns,
        dataset_name
):
    """
    Validate required dataframe columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe being checked.

    required_columns : tuple
        Required column names.

    dataset_name : str
        Human-readable dataset name.

    Returns
    -------
    bool
        True if all required columns exist.
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
            f"[WARNING] {dataset_name} is missing columns:"
        )

        for column in missing_columns:

            print(
                f"           - {column}"
            )

        return False

    return True


# ---------------------------------------------------------
# Validate numeric metrics
# ---------------------------------------------------------

def validate_numeric_metrics(
        dataframe,
        dataset_name
):
    """
    Validate that standard reconstruction metrics are
    numeric and finite.

    Invalid numerical values are treated as structural
    problems rather than silently removed.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe being checked.

    dataset_name : str
        Human-readable dataset name.

    Returns
    -------
    bool
        True if all available required metrics are valid.
    """

    if dataframe is None:

        return False

    validation_passed = True

    for column in NUMERIC_METRIC_COLUMNS:

        if column not in dataframe.columns:

            continue

        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        if numeric_values.isna().any():

            print(
                f"[WARNING] {dataset_name}: "
                f"non-numeric or missing values in {column}."
            )

            validation_passed = False

            continue

        values = numeric_values.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():

            print(
                f"[WARNING] {dataset_name}: "
                f"non-finite values in {column}."
            )

            validation_passed = False

    return validation_passed


# ---------------------------------------------------------
# Report section writer
# ---------------------------------------------------------

def write_section(
        file,
        title
):
    """
    Write a formatted section heading.
    """

    file.write("\n")
    file.write("=" * 80)
    file.write("\n")
    file.write(title)
    file.write("\n")
    file.write("=" * 80)
    file.write("\n")


# ---------------------------------------------------------
# Write dataframe
# ---------------------------------------------------------

def write_dataframe(
        file,
        dataframe
):
    """
    Write a dataframe in plain-text form.
    """

    if dataframe is None:

        file.write(
            "No valid data available.\n"
        )

        return

    file.write(
        dataframe.to_string(
            index=False
        )
    )

    file.write("\n")


# ---------------------------------------------------------
# Get PNG files
# ---------------------------------------------------------

def get_png_files(directory):
    """
    Return sorted PNG files from a directory.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing figures.

    Returns
    -------
    list
        Sorted PNG filenames.
    """

    if not directory.is_dir():

        return []

    return sorted(

        path.name

        for path in directory.iterdir()

        if path.is_file()
        and path.suffix.lower() == ".png"

    )


# ---------------------------------------------------------
# Get thesis tables
# ---------------------------------------------------------

def get_thesis_tables(directory):
    """
    Return sorted CSV files from the thesis_tables directory.

    Parameters
    ----------
    directory : pathlib.Path
        Thesis table directory.

    Returns
    -------
    list
        Sorted CSV filenames.
    """

    if not directory.is_dir():

        return []

    return sorted(

        path.name

        for path in directory.iterdir()

        if path.is_file()
        and path.suffix.lower() == ".csv"

    )


# ---------------------------------------------------------
# Write metadata JSON
# ---------------------------------------------------------

def write_metadata(
        generation_time,
        loaded_results,
        optional_status,
        checkpoint_available
):
    """
    Write machine-readable metadata describing the final
    report compilation.

    This does not contain new experimental measurements.
    It records which previously generated outputs were
    compiled.
    """

    metadata = {

        "report_type":
            "Final Thesis Experimental Report",

        "framework":
            "Physics-Informed 3D Encoder-Decoder Framework "
            "with Predictive Uncertainty for Seismic Data "
            "Reconstruction",

        "experiment_name":
            EXPERIMENT_NAME,

        "report_directory":
            str(REPORT_PATH),

        "final_report_file":
            str(FINAL_REPORT_FILE),

        "generation_timestamp":
            generation_time,

        "generation_scope":
            "Compilation of previously generated "
            "experimental outputs only",

        "training_rerun":
            False,

        "inference_rerun":
            False,

        "evaluation_rerun":
            False,

        "uncertainty_recomputed":
            False,

        "baseline_evaluation_rerun":
            False,

        "ablation_rerun":
            False,

        "statistical_testing_rerun":
            False,

        "required_outputs": {

            name: {

                "filename":
                    filename,

                "rows":
                    len(
                        loaded_results[name]
                    ),

                "available":
                    True,

            }

            for name, filename
            in REQUIRED_FILES.items()

        },

        "optional_outputs":
            optional_status,

        "best_model_checkpoint":
            {

                "available":
                    checkpoint_available,

                "path":
                    str(BEST_CHECKPOINT_FILE),

            },

    }

    with open(
        FINAL_REPORT_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4
        )


# =========================================================
# GENERATE FINAL REPORT
# =========================================================

def generate_final_report():
    """
    Compile all validated experimental outputs into the
    final thesis-oriented report.

    Returns
    -------
    pathlib.Path or None
        Path to the final report if successful.
    """

    print()
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    # =====================================================
    # EXPERIMENT INFORMATION
    # =====================================================

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME
    )

    print(
        "Report Dir :",
        REPORT_PATH
    )

    print(
        "Report File:",
        FINAL_REPORT_FILE
    )

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    REPORT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # CHECK REQUIRED RESULT FILES
    # =====================================================

    print()
    print("=" * 70)
    print("CHECKING REQUIRED RESULTS")
    print("=" * 70)

    loaded_results = {}

    missing_files = []

    invalid_files = []

    for name, filename in REQUIRED_FILES.items():

        filepath = get_file_path(
            filename
        )

        if not validate_file(filepath):

            print(
                f"[MISSING/EMPTY] "
                f"{name:<30} "
                f"{filepath}"
            )

            missing_files.append(
                name
            )

            continue

        dataframe = load_csv(
            filename
        )

        if dataframe is None:

            print(
                f"[INVALID] "
                f"{name:<30} "
                f"{filepath}"
            )

            invalid_files.append(
                name
            )

            continue

        print(
            f"[FOUND] "
            f"{name:<30} "
            f"{filepath} "
            f"({len(dataframe)} rows)"
        )

        loaded_results[name] = dataframe

    # =====================================================
    # STOP IF REQUIRED RESULTS ARE MISSING
    # =====================================================

    if missing_files or invalid_files:

        print()
        print("=" * 70)
        print("FINAL REPORT NOT GENERATED")
        print("=" * 70)

        if missing_files:

            print()
            print(
                "Missing or empty required files:"
            )

            for name in missing_files:

                print(
                    f"  - {name}"
                )

        if invalid_files:

            print()
            print(
                "Invalid required files:"
            )

            for name in invalid_files:

                print(
                    f"  - {name}"
                )

        print()
        print(
            "Repair or generate the corresponding "
            "experimental outputs before compiling "
            "the final report."
        )

        return None

    # =====================================================
    # VALIDATE RESULT STRUCTURES
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATING RESULT STRUCTURES")
    print("=" * 70)

    validation_failed = False

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    for dataset_name, required_columns in (
        REQUIRED_COLUMNS.items()
    ):

        if not validate_columns(
            loaded_results[dataset_name],
            required_columns,
            dataset_name
        ):

            validation_failed = True

    # -----------------------------------------------------
    # Numeric metrics
    # -----------------------------------------------------

    numeric_validation_datasets = (

        "Evaluation Metrics",
        "Baseline Comparison",
        "Ablation Study (Per Sample)",
        "Ablation Summary",

    )

    for dataset_name in numeric_validation_datasets:

        if not validate_numeric_metrics(
            loaded_results[dataset_name],
            dataset_name
        ):

            validation_failed = True

    # -----------------------------------------------------
    # Uncertainty dataframe
    # -----------------------------------------------------

    uncertainty = loaded_results[
        "Uncertainty Statistics"
    ]

    if uncertainty.empty:

        print(
            "[WARNING] Uncertainty Statistics is empty."
        )

        validation_failed = True

    # =====================================================
    # STOP IF STRUCTURAL VALIDATION FAILED
    # =====================================================

    if validation_failed:

        print()
        print("=" * 70)
        print("FINAL REPORT NOT GENERATED")
        print("=" * 70)

        print()
        print(
            "One or more required result files failed "
            "structural or numerical validation."
        )

        print(
            "No report was generated in order to prevent "
            "an incomplete or misleading thesis report."
        )

        return None

    # =====================================================
    # CHECK OPTIONAL OUTPUTS
    # =====================================================

    print()
    print("=" * 70)
    print("CHECKING OPTIONAL OUTPUTS")
    print("=" * 70)

    optional_status = {}

    for name, directory_name in OPTIONAL_DIRECTORIES.items():

        directory = (
            REPORT_PATH / directory_name
        )

        exists = directory.is_dir()

        optional_status[name] = exists

        if exists:

            print(
                f"[FOUND]    "
                f"{name:<30} "
                f"{directory}"
            )

        else:

            print(
                f"[OPTIONAL] "
                f"{name:<29} "
                f"not found"
            )

    # =====================================================
    # CHECK BEST MODEL CHECKPOINT
    # =====================================================

    checkpoint_available = (
        BEST_CHECKPOINT_FILE.is_file()
        and BEST_CHECKPOINT_FILE.stat().st_size > 0
    )

    print()

    if checkpoint_available:

        print(
            "[FOUND] Best model checkpoint:",
            BEST_CHECKPOINT_FILE
        )

    else:

        print(
            "[WARNING] Best model checkpoint not found:",
            BEST_CHECKPOINT_FILE
        )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    evaluation = loaded_results[
        "Evaluation Metrics"
    ]

    uncertainty = loaded_results[
        "Uncertainty Statistics"
    ]

    baseline = loaded_results[
        "Baseline Comparison"
    ]

    significance = loaded_results[
        "Statistical Significance"
    ]

    ablation = loaded_results[
        "Ablation Study (Per Sample)"
    ]

    ablation_summary = loaded_results[
        "Ablation Summary"
    ]

    # =====================================================
    # OPTIONAL OUTPUT DETAILS
    # =====================================================

    thesis_directory = (
        REPORT_PATH / "thesis_tables"
    )

    thesis_tables = get_thesis_tables(
        thesis_directory
    )

    gallery_directory = (
        REPORT_PATH / "gallery"
    )

    gallery_files = get_png_files(
        gallery_directory
    )

    uncertainty_directory = (
        REPORT_PATH / "uncertainty"
    )

    uncertainty_figures = get_png_files(
        uncertainty_directory
    )

    # =====================================================
    # GENERATION TIMESTAMP
    # =====================================================

    generation_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # =====================================================
    # GENERATE TEXT REPORT
    # =====================================================

    print()
    print("=" * 70)
    print("COMPILING FINAL REPORT")
    print("=" * 70)

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        # =================================================
        # TITLE
        # =================================================

        file.write(
            "PHYSICS-INFORMED 3D ENCODER-DECODER FRAMEWORK\n"
        )

        file.write(
            "WITH PREDICTIVE UNCERTAINTY FOR SEISMIC DATA "
            "RECONSTRUCTION\n"
        )

        file.write(
            "FINAL EXPERIMENT REPORT\n"
        )

        file.write("\n")

        file.write(
            f"Experiment: {EXPERIMENT_NAME}\n"
        )

        file.write(
            f"Report Directory: {REPORT_PATH}\n"
        )

        file.write(
            f"Report Generated: {generation_time}\n"
        )

        file.write("\n")

        file.write(
            "IMPORTANT: This report generator only compiles "
            "previously generated experimental outputs. "
            "It does not perform training, model inference, "
            "uncertainty estimation, baseline evaluation, "
            "ablation training, or statistical testing.\n"
        )

        # =================================================
        # 1. MODEL EVALUATION
        # =================================================

        write_section(
            file,
            "1. MODEL EVALUATION"
        )

        file.write(
            f"Rows: {len(evaluation)}\n\n"
        )

        write_dataframe(
            file,
            evaluation
        )

        # =================================================
        # 2. PREDICTIVE UNCERTAINTY
        # =================================================

        write_section(
            file,
            "2. PREDICTIVE UNCERTAINTY ANALYSIS"
        )

        file.write(
            f"Rows: {len(uncertainty)}\n\n"
        )

        write_dataframe(
            file,
            uncertainty
        )

        # =================================================
        # 3. BASELINE COMPARISON
        # =================================================

        write_section(
            file,
            "3. BASELINE COMPARISON"
        )

        file.write(
            f"Rows: {len(baseline)}\n\n"
        )

        write_dataframe(
            file,
            baseline
        )

        # =================================================
        # 4. STATISTICAL SIGNIFICANCE
        # =================================================

        write_section(
            file,
            "4. STATISTICAL SIGNIFICANCE"
        )

        file.write(
            f"Rows: {len(significance)}\n\n"
        )

        write_dataframe(
            file,
            significance
        )

        # =================================================
        # 5. ABLATION STUDY
        # =================================================

        write_section(
            file,
            "5. ABLATION STUDY"
        )

        file.write(
            "Per-sample ablation results:\n\n"
        )

        file.write(
            f"Rows: {len(ablation)}\n\n"
        )

        write_dataframe(
            file,
            ablation
        )

        file.write("\n")

        file.write(
            "Ablation model-level summary:\n\n"
        )

        file.write(
            f"Rows: {len(ablation_summary)}\n\n"
        )

        write_dataframe(
            file,
            ablation_summary
        )

        # =================================================
        # 6. THESIS TABLES
        # =================================================

        write_section(
            file,
            "6. THESIS TABLES"
        )

        if thesis_tables:

            file.write(
                f"Thesis table directory: "
                f"{thesis_directory}\n"
            )

            file.write(
                f"Number of thesis tables: "
                f"{len(thesis_tables)}\n\n"
            )

            for filename in thesis_tables:

                file.write(
                    f"  - {filename}\n"
                )

        else:

            file.write(
                "Thesis table directory not found or "
                "contains no CSV tables.\n"
            )

        # =================================================
        # 7. RECONSTRUCTION GALLERY
        # =================================================

        write_section(
            file,
            "7. RECONSTRUCTION GALLERY"
        )

        if gallery_files:

            file.write(
                f"Gallery directory: "
                f"{gallery_directory}\n"
            )

            file.write(
                f"Number of reconstruction figures: "
                f"{len(gallery_files)}\n\n"
            )

            for filename in gallery_files:

                file.write(
                    f"  - {filename}\n"
                )

        else:

            file.write(
                "Reconstruction gallery not found or "
                "contains no PNG figures.\n"
            )

        # =================================================
        # 8. UNCERTAINTY FIGURES
        # =================================================

        write_section(
            file,
            "8. UNCERTAINTY FIGURES"
        )

        if uncertainty_figures:

            file.write(
                f"Uncertainty figure directory: "
                f"{uncertainty_directory}\n"
            )

            file.write(
                f"Number of uncertainty figures: "
                f"{len(uncertainty_figures)}\n\n"
            )

            for filename in uncertainty_figures:

                file.write(
                    f"  - {filename}\n"
                )

        else:

            file.write(
                "Uncertainty figure directory not found "
                "or contains no PNG figures.\n"
            )

        # =================================================
        # 9. MODEL CHECKPOINT
        # =================================================

        write_section(
            file,
            "9. MODEL CHECKPOINT"
        )

        if checkpoint_available:

            file.write(
                "Best model checkpoint: AVAILABLE\n"
            )

            file.write(
                f"Checkpoint path: "
                f"{BEST_CHECKPOINT_FILE}\n"
            )

        else:

            file.write(
                "Best model checkpoint: NOT FOUND\n"
            )

        # =================================================
        # 10. EXPERIMENT REPORT STATUS
        # =================================================

        write_section(
            file,
            "10. EXPERIMENT REPORT STATUS"
        )

        file.write(
            "Evaluation metrics: AVAILABLE\n"
        )

        file.write(
            "Predictive uncertainty statistics: AVAILABLE\n"
        )

        file.write(
            "Baseline comparison: AVAILABLE\n"
        )

        file.write(
            "Statistical significance: AVAILABLE\n"
        )

        file.write(
            "Ablation per-sample results: AVAILABLE\n"
        )

        file.write(
            "Ablation summary: AVAILABLE\n"
        )

        file.write(
            "Best model checkpoint: "
            f"{'AVAILABLE' if checkpoint_available else 'NOT FOUND'}\n"
        )

        file.write(
            "Reconstruction gallery: "
            f"{'AVAILABLE' if gallery_files else 'NOT FOUND'}\n"
        )

        file.write(
            "Uncertainty figures: "
            f"{'AVAILABLE' if uncertainty_figures else 'NOT FOUND'}\n"
        )

        file.write(
            "Thesis tables: "
            f"{'AVAILABLE' if thesis_tables else 'NOT FOUND'}\n"
        )

        # =================================================
        # 11. EXPERIMENT OUTPUT SUMMARY
        # =================================================

        write_section(
            file,
            "11. EXPERIMENT OUTPUT SUMMARY"
        )

        file.write(
            f"Experiment Name: {EXPERIMENT_NAME}\n"
        )

        file.write(
            f"Report Directory: {REPORT_PATH}\n"
        )

        file.write(
            f"Evaluation rows: "
            f"{len(evaluation)}\n"
        )

        file.write(
            f"Uncertainty rows: "
            f"{len(uncertainty)}\n"
        )

        file.write(
            f"Baseline comparison rows: "
            f"{len(baseline)}\n"
        )

        file.write(
            f"Statistical significance rows: "
            f"{len(significance)}\n"
        )

        file.write(
            f"Ablation per-sample rows: "
            f"{len(ablation)}\n"
        )

        file.write(
            f"Ablation summary rows: "
            f"{len(ablation_summary)}\n"
        )

        file.write(
            f"Gallery figures: "
            f"{len(gallery_files)}\n"
        )

        file.write(
            f"Uncertainty figures: "
            f"{len(uncertainty_figures)}\n"
        )

        file.write(
            f"Thesis tables: "
            f"{len(thesis_tables)}\n"
        )

        # =================================================
        # 12. REPORT GENERATION NOTE
        # =================================================

        write_section(
            file,
            "12. REPORT GENERATION NOTE"
        )

        file.write(
            "This report is a compilation of previously "
            "generated experimental outputs.\n"
        )

        file.write(
            "No training was performed by this script.\n"
        )

        file.write(
            "No model inference was performed by this script.\n"
        )

        file.write(
            "No uncertainty estimation was performed by "
            "this script.\n"
        )

        file.write(
            "No baseline evaluation was performed by "
            "this script.\n"
        )

        file.write(
            "No ablation experiment was performed by "
            "this script.\n"
        )

        file.write(
            "No statistical significance test was performed "
            "by this script.\n"
        )

        file.write(
            "The report therefore represents an archival "
            "compilation of the experimental outputs "
            "available at the time of report generation.\n"
        )

    # =====================================================
    # GENERATE MACHINE-READABLE METADATA
    # =====================================================

    write_metadata(
        generation_time=generation_time,
        loaded_results=loaded_results,
        optional_status=optional_status,
        checkpoint_available=checkpoint_available
    )

    # =====================================================
    # COMPLETION MESSAGE
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL REPORT COMPLETE")
    print("=" * 70)

    print()
    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Final Report:"
    )

    print(
        FINAL_REPORT_FILE
    )

    print()
    print(
        "Metadata:"
    )

    print(
        FINAL_REPORT_METADATA_FILE
    )

    print()
    print(
        "Existing experimental outputs were compiled."
    )

    print(
        "No training or evaluation was rerun."
    )

    return FINAL_REPORT_FILE


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    generate_final_report()