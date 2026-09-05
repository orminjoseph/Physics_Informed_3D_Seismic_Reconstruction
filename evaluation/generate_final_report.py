"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Compiles existing evaluation outputs into a final report.

IMPORTANT:
This module does NOT rerun model evaluation, uncertainty
analysis, baseline comparison, statistical testing, ablation
studies, or thesis-table generation.

Those operations are handled by:

    evaluation/run_full_evaluation.py

This module only:
    1. Checks existing evaluation outputs.
    2. Loads existing CSV files.
    3. Reports available figures and thesis tables.
    4. Records the best-model checkpoint.
    5. Generates final_report.txt.

Author: Ormin Joseph
=========================================================
"""

import os
import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# =========================================================
# REPORT PATHS
# =========================================================

FINAL_REPORT_FILE = os.path.join(
    REPORT_DIR,
    "final_report.txt"
)

CHECKPOINT_FILE = os.path.join(
    os.path.dirname(REPORT_DIR),
    "checkpoints",
    "best_model.pth"
)

GALLERY_DIR = os.path.join(
    REPORT_DIR,
    "gallery"
)

UNCERTAINTY_DIR = os.path.join(
    REPORT_DIR,
    "uncertainty"
)

THESIS_TABLES_DIR = os.path.join(
    REPORT_DIR,
    "thesis_tables"
)


# =========================================================
# EXPECTED CSV FILES
# =========================================================

CSV_FILES = {
    "Evaluation Metrics":
        os.path.join(
            REPORT_DIR,
            "evaluation_metrics.csv"
        ),

    "Uncertainty Statistics":
        os.path.join(
            REPORT_DIR,
            "uncertainty_statistics.csv"
        ),

    "Baseline Comparison":
        os.path.join(
            REPORT_DIR,
            "baseline_comparison.csv"
        ),

    "Statistical Significance":
        os.path.join(
            REPORT_DIR,
            "statistical_significance.csv"
        ),

    "Ablation Study":
        os.path.join(
            REPORT_DIR,
            "ablation_study.csv"
        ),

    "Ablation Summary":
        os.path.join(
            REPORT_DIR,
            "ablation_summary.csv"
        ),
}


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = {

    "Evaluation Metrics": [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

    "Baseline Comparison": [
        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

    "Statistical Significance": [
        "Comparison",
        "N_Pairs",
        "Raw_P_Value",
        "Holm_Adjusted_P_Value",
    ],

    "Ablation Study": [
        "Model",
        "Sample_ID",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ],

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
# HELPER FUNCTIONS
# =========================================================

def validate_file(file_path):
    """
    Check whether a file exists and is non-empty.
    """

    if not os.path.isfile(file_path):
        return False

    if os.path.getsize(file_path) == 0:
        return False

    return True


def load_csv(file_path):
    """
    Safely load a CSV file.
    """

    try:

        dataframe = pd.read_csv(file_path)

        if dataframe.empty:
            return None

        return dataframe

    except Exception as error:

        print(
            f"[WARNING] Could not read CSV:\n"
            f"{file_path}\n"
            f"Reason: {error}"
        )

        return None


def validate_columns(
    dataframe,
    required_columns,
    table_name
):
    """
    Validate that required columns exist.
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


def get_png_files(directory):
    """
    Return PNG files from a directory.
    """

    if not os.path.isdir(directory):
        return []

    return sorted(
        file
        for file in os.listdir(directory)
        if file.lower().endswith(".png")
    )


def get_thesis_tables():
    """
    Return CSV files generated for thesis tables.
    """

    if not os.path.isdir(THESIS_TABLES_DIR):
        return []

    return sorted(
        file
        for file in os.listdir(THESIS_TABLES_DIR)
        if file.lower().endswith(".csv")
    )


def write_section(
    report,
    title
):
    """
    Write a formatted section heading.
    """

    report.write("\n")
    report.write("=" * 80)
    report.write("\n")
    report.write(title)
    report.write("\n")
    report.write("=" * 80)
    report.write("\n")


def write_dataframe(
    report,
    dataframe
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
        dataframe.to_string(index=False)
    )

    report.write("\n")


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================

def generate_final_report():

    print()
    print("=" * 80)
    print("FINAL REPORT GENERATOR")
    print("=" * 80)

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    results = {}

    # -----------------------------------------------------
    # CHECK CSV FILES
    # -----------------------------------------------------

    print()
    print("Checking evaluation outputs...")

    for name, file_path in CSV_FILES.items():

        if validate_file(file_path):

            dataframe = load_csv(file_path)

            if dataframe is not None:

                if name in REQUIRED_COLUMNS:

                    valid = validate_columns(
                        dataframe,
                        REQUIRED_COLUMNS[name],
                        name
                    )

                else:

                    valid = True

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

            else:

                results[name] = {
                    "path": file_path,
                    "dataframe": None,
                    "valid": False,
                }

                print(
                    f"[INVALID] {name}"
                )

        else:

            results[name] = {
                "path": file_path,
                "dataframe": None,
                "valid": False,
            }

            print(
                f"[MISSING] {name}"
            )

    # -----------------------------------------------------
    # FIGURES
    # -----------------------------------------------------

    gallery_files = get_png_files(
        GALLERY_DIR
    )

    uncertainty_files = get_png_files(
        UNCERTAINTY_DIR
    )

    thesis_tables = get_thesis_tables()

    # -----------------------------------------------------
    # WRITE FINAL REPORT
    # -----------------------------------------------------

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8"
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

        report.write("\n\n")

        report.write(
            f"Experiment: {EXPERIMENT_NAME}\n"
        )

        report.write(
            f"Report Directory: {REPORT_DIR}\n"
        )

        # =================================================
        # 1. MODEL EVALUATION
        # =================================================

        write_section(
            report,
            "1. MODEL EVALUATION"
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
                evaluation_result["dataframe"]
            )

        else:

            report.write(
                "Evaluation metrics are unavailable.\n"
            )

        # =================================================
        # 2. PREDICTIVE UNCERTAINTY
        # =================================================

        write_section(
            report,
            "2. PREDICTIVE UNCERTAINTY"
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
                uncertainty_result["dataframe"]
            )

        else:

            report.write(
                "Uncertainty statistics are unavailable.\n"
            )

        report.write("\n")

        report.write(
            "Uncertainty figures:\n"
        )

        if uncertainty_files:

            for file in uncertainty_files:

                report.write(
                    f"  - {os.path.join(UNCERTAINTY_DIR, file)}\n"
                )

        else:

            report.write(
                "  None found.\n"
            )

        # =================================================
        # 3. BASELINE COMPARISON
        # =================================================

        write_section(
            report,
            "3. BASELINE COMPARISON"
        )

        baseline_result = results.get(
            "Baseline Comparison"
        )

        if (
            baseline_result
            and baseline_result["valid"]
        ):

            write_dataframe(
                report,
                baseline_result["dataframe"]
            )

        else:

            report.write(
                "Baseline comparison results are unavailable.\n"
            )

        # =================================================
        # 4. STATISTICAL SIGNIFICANCE
        # =================================================

        write_section(
            report,
            "4. STATISTICAL SIGNIFICANCE"
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
                significance_result["dataframe"]
            )

        else:

            report.write(
                "Statistical significance results are unavailable.\n"
            )

        # =================================================
        # 5. ABLATION STUDY
        # =================================================

        write_section(
            report,
            "5. ABLATION STUDY"
        )

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
                ablation_summary["dataframe"]
            )

        else:

            report.write(
                "Ablation summary is unavailable.\n"
            )

        report.write("\n")

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
                ablation_study["dataframe"]
            )

        else:

            report.write(
                "Per-sample ablation results are unavailable.\n"
            )

        # =================================================
        # 6. THESIS TABLES
        # =================================================

        write_section(
            report,
            "6. THESIS TABLES"
        )

        if thesis_tables:

            report.write(
                "Generated thesis table files:\n\n"
            )

            for table in thesis_tables:

                report.write(
                    f"  - "
                    f"{os.path.join(THESIS_TABLES_DIR, table)}\n"
                )

        else:

            report.write(
                "No thesis table files found.\n"
            )

        # =================================================
        # 7. RECONSTRUCTION GALLERY
        # =================================================

        write_section(
            report,
            "7. RECONSTRUCTION GALLERY"
        )

        if gallery_files:

            report.write(
                "Generated gallery figures:\n\n"
            )

            for file in gallery_files:

                report.write(
                    f"  - {os.path.join(GALLERY_DIR, file)}\n"
                )

        else:

            report.write(
                "No reconstruction gallery figures found.\n"
            )

        # =================================================
        # 8. CHECKPOINT
        # =================================================

        write_section(
            report,
            "8. BEST MODEL CHECKPOINT"
        )

        if validate_file(CHECKPOINT_FILE):

            report.write(
                f"Best checkpoint:\n"
                f"{CHECKPOINT_FILE}\n"
            )

        else:

            report.write(
                "Best model checkpoint was not found.\n"
            )

        # =================================================
        # 9. OUTPUT STATUS
        # =================================================

        write_section(
            report,
            "9. OUTPUT STATUS"
        )

        successful_outputs = sum(
            1
            for result in results.values()
            if result["valid"]
        )

        total_outputs = len(results)

        report.write(
            f"Valid CSV outputs: "
            f"{successful_outputs}/{total_outputs}\n"
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
            f"Thesis tables: "
            f"{len(thesis_tables)}\n"
        )

        report.write("\n")

        if successful_outputs == total_outputs:

            report.write(
                "STATUS: ALL EXPECTED CSV OUTPUTS "
                "ARE AVAILABLE.\n"
            )

        else:

            report.write(
                "STATUS: SOME EXPECTED OUTPUTS "
                "ARE MISSING OR INVALID.\n"
            )

        # =================================================
        # 10. REPORT GENERATION NOTE
        # =================================================

        write_section(
            report,
            "10. REPORT GENERATION NOTE"
        )

        report.write(
            "This report was compiled from existing "
            "evaluation outputs.\n"
        )

        report.write(
            "No model training or evaluation computation "
            "was performed by the final report generator.\n"
        )

    # =====================================================
    # CONSOLE SUMMARY
    # =====================================================

    print()
    print("=" * 80)
    print("FINAL REPORT COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Report saved to:\n"
        f"{FINAL_REPORT_FILE}"
    )

    print()
    print(
        f"Valid CSV outputs: "
        f"{successful_outputs}/{total_outputs}"
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
        f"Thesis tables: "
        f"{len(thesis_tables)}"
    )

    print()

    return FINAL_REPORT_FILE


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    generate_final_report()