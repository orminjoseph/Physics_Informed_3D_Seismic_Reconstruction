"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This script DOES NOT:
    - rerun training
    - rerun evaluation
    - retrain ablation models
    - recompute uncertainty
    - recompute statistical tests

It ONLY collects previously generated experimental outputs
from the CURRENT EXPERIMENT and compiles them into a
final thesis-oriented report.

Expected experiment structure:

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
                    uncertainty_analysis.png

                thesis_tables/
                    *.csv

                final_report.txt

The active experiment is controlled by:

    utils.config.EXPERIMENT_NAME
    utils.config.REPORT_DIR

=========================================================
"""

import os
from datetime import datetime

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR
)


# =========================================================
# REPORT FILE
# =========================================================

FINAL_REPORT_FILE = os.path.join(
    REPORT_DIR,
    "final_report.txt"
)


# =========================================================
# REQUIRED RESULT FILES
# =========================================================
#
# These files are required for the final analytical report.
#
# NOTE:
# ablation_study.csv is retained because it contains the
# per-sample ablation results required by the statistical
# significance analysis.
#
# ablation_summary.csv is also required because it provides
# the model-level summary for the thesis report.
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
        "ablation_summary.csv"

}


# =========================================================
# OPTIONAL RESULT DIRECTORIES
# =========================================================

OPTIONAL_DIRECTORIES = {

    "Reconstruction Gallery":
        "gallery",

    "Uncertainty Figures":
        "uncertainty",

    "Thesis Tables":
        "thesis_tables"

}


# =========================================================
# OPTIONAL CHECKPOINT
# =========================================================

BEST_CHECKPOINT_FILE = os.path.join(
    REPORT_DIR,
    "..",
    "checkpoints",
    "best_model.pth"
)

BEST_CHECKPOINT_FILE = os.path.normpath(
    BEST_CHECKPOINT_FILE
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_file_path(filename):
    """
    Construct the absolute/relative path of a report file.

    Parameters
    ----------
    filename : str
        Filename located inside REPORT_DIR.

    Returns
    -------
    str
        Full report path.
    """

    return os.path.join(
        REPORT_DIR,
        filename
    )


# ---------------------------------------------------------
# Check whether a file exists and is non-empty
# ---------------------------------------------------------

def validate_file(filepath):
    """
    Check whether a file exists and contains data.

    Parameters
    ----------
    filepath : str
        File path.

    Returns
    -------
    bool
        True if the file exists and is non-empty.
    """

    if not os.path.exists(filepath):

        return False

    if os.path.getsize(filepath) == 0:

        return False

    return True


# ---------------------------------------------------------
# Load CSV safely
# ---------------------------------------------------------

def load_csv(filename):
    """
    Load a CSV file from the current experiment's
    report directory.

    Parameters
    ----------
    filename : str
        CSV filename.

    Returns
    -------
    pandas.DataFrame or None
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

        if dataframe.empty:

            print(
                f"[WARNING] CSV contains no rows: {filepath}"
            )

            return None

        return dataframe

    except Exception as error:

        print(
            f"[WARNING] Could not read: {filepath}"
        )

        print(
            f"Reason: {error}"
        )

        return None


# ---------------------------------------------------------
# Validate dataframe columns
# ---------------------------------------------------------

def validate_columns(
        dataframe,
        required_columns,
        dataset_name
):
    """
    Validate that required columns exist.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to validate.

    required_columns : list
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
# Report section writer
# ---------------------------------------------------------

def write_section(
        file,
        title
):
    """
    Write a formatted report section.
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
    Write a dataframe into the text report.
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
# Count PNG files
# ---------------------------------------------------------

def get_png_files(directory):
    """
    Return sorted PNG files from a directory.
    """

    if not os.path.isdir(directory):

        return []

    return sorted(

        filename

        for filename in os.listdir(directory)

        if filename.lower().endswith(".png")

    )


# ---------------------------------------------------------
# Count thesis tables
# ---------------------------------------------------------

def get_thesis_tables(directory):
    """
    Return sorted CSV files from the thesis_tables directory.
    """

    if not os.path.isdir(directory):

        return []

    return sorted(

        filename

        for filename in os.listdir(directory)

        if filename.lower().endswith(".csv")

    )


# =========================================================
# GENERATE FINAL REPORT
# =========================================================

def generate_final_report():

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
        REPORT_DIR
    )

    print(
        "Report File:",
        FINAL_REPORT_FILE
    )

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
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
                f"[MISSING/EMPTY] {name:<30} {filepath}"
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
                f"[INVALID]       {name:<30} {filepath}"
            )

            invalid_files.append(
                name
            )

            continue

        print(
            f"[FOUND]         {name:<30} "
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
            "Run or repair the corresponding evaluation "
            "pipeline before generating the final report."
        )

        return None

    # =====================================================
    # VALIDATE REQUIRED CSV STRUCTURES
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATING RESULT STRUCTURES")
    print("=" * 70)

    validation_failed = False

    # -----------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------

    evaluation_required_columns = [

        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"

    ]

    if not validate_columns(

        loaded_results["Evaluation Metrics"],

        evaluation_required_columns,

        "Evaluation Metrics"

    ):

        validation_failed = True

    # -----------------------------------------------------
    # Baseline comparison
    # -----------------------------------------------------

    baseline_required_columns = [

        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"

    ]

    if not validate_columns(

        loaded_results["Baseline Comparison"],

        baseline_required_columns,

        "Baseline Comparison"

    ):

        validation_failed = True

    # -----------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------

    significance_required_columns = [

        "Comparison",
        "N_Pairs",
        "Raw_P_Value",
        "Holm_Adjusted_P_Value"

    ]

    if not validate_columns(

        loaded_results["Statistical Significance"],

        significance_required_columns,

        "Statistical Significance"

    ):

        validation_failed = True

    # -----------------------------------------------------
    # Ablation per-sample data
    # -----------------------------------------------------

    ablation_required_columns = [

        "Model",
        "Sample_ID",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"

    ]

    if not validate_columns(

        loaded_results["Ablation Study (Per Sample)"],

        ablation_required_columns,

        "Ablation Study"

    ):

        validation_failed = True

    # -----------------------------------------------------
    # Ablation summary
    # -----------------------------------------------------

    ablation_summary_required_columns = [

        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"

    ]

    if not validate_columns(

        loaded_results["Ablation Summary"],

        ablation_summary_required_columns,

        "Ablation Summary"

    ):

        validation_failed = True

    # -----------------------------------------------------
    # Uncertainty statistics
    # -----------------------------------------------------

    if loaded_results["Uncertainty Statistics"].empty:

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

        print(
            "One or more result files failed structural validation."
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

        directory = os.path.join(
            REPORT_DIR,
            directory_name
        )

        exists = os.path.isdir(
            directory
        )

        optional_status[name] = exists

        if exists:

            print(
                f"[FOUND]   {name:<30} {directory}"
            )

        else:

            print(
                f"[OPTIONAL] {name:<29} not found"
            )

    # =====================================================
    # CHECK BEST CHECKPOINT
    # =====================================================

    checkpoint_available = os.path.exists(
        BEST_CHECKPOINT_FILE
    )

    print()

    if checkpoint_available:

        print(
            "[FOUND]   Best model checkpoint:",
            BEST_CHECKPOINT_FILE
        )

    else:

        print(
            "[WARNING] Best model checkpoint not found:",
            BEST_CHECKPOINT_FILE
        )

    # =====================================================
    # GENERATE REPORT
    # =====================================================

    print()
    print("=" * 70)
    print("COMPILING FINAL REPORT")
    print("=" * 70)

    generation_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

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
            f"Report Directory: {REPORT_DIR}\n"
        )

        file.write(
            f"Report Generated: {generation_time}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "IMPORTANT: This report generator only compiles "
            "previously generated outputs. It does not "
            "perform training, model inference, uncertainty "
            "estimation, baseline evaluation, ablation "
            "training, or statistical testing.\n"
        )

        # =================================================
        # 1. MODEL EVALUATION
        # =================================================

        write_section(
            file,
            "1. MODEL EVALUATION"
        )

        evaluation = loaded_results[
            "Evaluation Metrics"
        ]

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

        uncertainty = loaded_results[
            "Uncertainty Statistics"
        ]

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

        baseline = loaded_results[
            "Baseline Comparison"
        ]

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

        significance = loaded_results[
            "Statistical Significance"
        ]

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

        ablation = loaded_results[
            "Ablation Study (Per Sample)"
        ]

        ablation_summary = loaded_results[
            "Ablation Summary"
        ]

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

        file.write(
            "\n"
        )

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

        thesis_directory = os.path.join(
            REPORT_DIR,
            "thesis_tables"
        )

        thesis_tables = get_thesis_tables(
            thesis_directory
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

        gallery_directory = os.path.join(
            REPORT_DIR,
            "gallery"
        )

        gallery_files = get_png_files(
            gallery_directory
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

        uncertainty_directory = os.path.join(
            REPORT_DIR,
            "uncertainty"
        )

        uncertainty_figures = get_png_files(
            uncertainty_directory
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
        # 9. CHECKPOINT
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
            f"{'AVAILABLE' if optional_status['Reconstruction Gallery'] else 'NOT FOUND'}\n"
        )

        file.write(
            "Uncertainty figures: "
            f"{'AVAILABLE' if optional_status['Uncertainty Figures'] else 'NOT FOUND'}\n"
        )

        file.write(
            "Thesis tables: "
            f"{'AVAILABLE' if optional_status['Thesis Tables'] else 'NOT FOUND'}\n"
        )

        # =================================================
        # 11. DATASET / EXPERIMENT SUMMARY
        # =================================================

        write_section(
            file,
            "11. EXPERIMENT OUTPUT SUMMARY"
        )

        file.write(
            f"Experiment Name: {EXPERIMENT_NAME}\n"
        )

        file.write(
            f"Report Directory: {REPORT_DIR}\n"
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
        # FINAL NOTE
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