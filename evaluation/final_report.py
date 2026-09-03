"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework

This script DOES NOT rerun training or evaluation.

It collects previously generated results from the
CURRENT EXPERIMENT and compiles them into a final
thesis-oriented report.

Expected experiment structure:

    outputs/
        <EXPERIMENT_NAME>/
            checkpoints/
            reports/
                evaluation_metrics.csv
                uncertainty_statistics.csv
                baseline_comparison.csv
                statistical_significance.csv
                ablation_study.csv
                thesis_tables/        # if generated

The active experiment is controlled by:

    utils.config.EXPERIMENT_NAME
    utils.config.REPORT_DIR

=========================================================
"""

import os
import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR
)


# =========================================================
# REQUIRED RESULT FILES
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

    "Ablation Study":
        "ablation_study.csv"

}


# =========================================================
# OPTIONAL RESULT FILES
# =========================================================

OPTIONAL_FILES = {

    "Thesis Tables":
        "thesis_tables.csv"

}


# =========================================================
# REPORT FILE
# =========================================================

FINAL_REPORT_FILE = os.path.join(
    REPORT_DIR,
    "final_report.txt"
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def load_csv(
        filename
):
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

    filepath = os.path.join(
        REPORT_DIR,
        filename
    )

    if not os.path.exists(filepath):

        return None

    try:

        return pd.read_csv(
            filepath
        )

    except Exception as error:

        print(
            f"[WARNING] Could not read {filepath}"
        )

        print(
            f"Reason: {error}"
        )

        return None


# =========================================================
# REPORT SECTION WRITER
# =========================================================

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

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # =====================================================
    # CHECK REQUIRED FILES
    # =====================================================

    print()
    print("=" * 70)
    print("CHECKING EXISTING RESULTS")
    print("=" * 70)

    loaded_results = {}

    missing_files = []

    for name, filename in REQUIRED_FILES.items():

        filepath = os.path.join(
            REPORT_DIR,
            filename
        )

        if os.path.exists(filepath):

            print(
                f"[FOUND]   {name:<30} {filepath}"
            )

            loaded_results[name] = load_csv(
                filename
            )

        else:

            print(
                f"[MISSING] {name:<30} {filepath}"
            )

            missing_files.append(
                name
            )

    # =====================================================
    # CHECK OPTIONAL FILES
    # =====================================================

    for name, filename in OPTIONAL_FILES.items():

        filepath = os.path.join(
            REPORT_DIR,
            filename
        )

        if os.path.exists(filepath):

            print(
                f"[FOUND]   {name:<30} {filepath}"
            )

            loaded_results[name] = load_csv(
                filename
            )

        else:

            print(
                f"[OPTIONAL] {name:<28} not found"
            )

    # =====================================================
    # STOP IF REQUIRED RESULTS ARE MISSING
    # =====================================================

    if missing_files:

        print()
        print("=" * 70)
        print("FINAL REPORT NOT GENERATED")
        print("=" * 70)

        print()
        print(
            "The following required result files are missing:"
        )

        for name in missing_files:

            print(
                f"  - {name}"
            )

        print()
        print(
            "Run the corresponding evaluation step first."
        )

        return None

    # =====================================================
    # GENERATE REPORT
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
            "FINAL EXPERIMENT REPORT\n"
        )

        file.write(
            "\n"
        )

        file.write(
            f"Experiment: {EXPERIMENT_NAME}\n"
        )

        file.write(
            f"Report Directory: {REPORT_DIR}\n"
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
            evaluation.to_string(
                index=False
            )
        )

        file.write(
            "\n"
        )

        # =================================================
        # 2. UNCERTAINTY ANALYSIS
        # =================================================

        write_section(
            file,
            "2. PREDICTIVE UNCERTAINTY ANALYSIS"
        )

        uncertainty = loaded_results[
            "Uncertainty Statistics"
        ]

        file.write(
            uncertainty.to_string(
                index=False
            )
        )

        file.write(
            "\n"
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
            baseline.to_string(
                index=False
            )
        )

        file.write(
            "\n"
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
            significance.to_string(
                index=False
            )
        )

        file.write(
            "\n"
        )

        # =================================================
        # 5. ABLATION STUDY
        # =================================================

        write_section(
            file,
            "5. ABLATION STUDY"
        )

        ablation = loaded_results[
            "Ablation Study"
        ]

        file.write(
            ablation.to_string(
                index=False
            )
        )

        file.write(
            "\n"
        )

        # =================================================
        # 6. THESIS TABLES
        # =================================================

        if "Thesis Tables" in loaded_results:

            write_section(
                file,
                "6. THESIS TABLES"
            )

            thesis_tables = loaded_results[
                "Thesis Tables"
            ]

            file.write(
                thesis_tables.to_string(
                    index=False
                )
            )

            file.write(
                "\n"
            )

        # =================================================
        # 7. RECONSTRUCTION GALLERY
        # =================================================

        write_section(
            file,
            "7. RECONSTRUCTION GALLERY"
        )

        gallery_dir = os.path.join(
            REPORT_DIR,
            "gallery"
        )

        if os.path.exists(
            gallery_dir
        ):

            gallery_files = sorted(

                filename
                for filename in os.listdir(
                    gallery_dir
                )

                if filename.lower().endswith(
                    ".png"
                )
            )

            file.write(
                f"Gallery directory: {gallery_dir}\n"
            )

            file.write(
                f"Number of reconstruction figures: "
                f"{len(gallery_files)}\n"
            )

            for filename in gallery_files:

                file.write(
                    f"  - {filename}\n"
                )

        else:

            file.write(
                "Reconstruction gallery not found.\n"
            )

        # =================================================
        # 8. UNCERTAINTY FIGURE
        # =================================================

        write_section(
            file,
            "8. UNCERTAINTY FIGURES"
        )

        uncertainty_histogram = os.path.join(
            REPORT_DIR,
            "uncertainty_histogram.png"
        )

        if os.path.exists(
            uncertainty_histogram
        ):

            file.write(
                "Uncertainty histogram:\n"
            )

            file.write(
                f"  {uncertainty_histogram}\n"
            )

        else:

            file.write(
                "Uncertainty histogram not found.\n"
            )

        # =================================================
        # 9. REPORT STATUS
        # =================================================

        write_section(
            file,
            "9. EXPERIMENT REPORT STATUS"
        )

        file.write(
            "Training results: AVAILABLE\n"
        )

        file.write(
            "Model evaluation: AVAILABLE\n"
        )

        file.write(
            "Reconstruction gallery: AVAILABLE\n"
        )

        file.write(
            "Predictive uncertainty analysis: AVAILABLE\n"
        )

        file.write(
            "Baseline comparison: AVAILABLE\n"
        )

        file.write(
            "Statistical significance: AVAILABLE\n"
        )

        file.write(
            "Ablation study: AVAILABLE\n"
        )

        file.write(
            "Thesis tables: AVAILABLE/OPTIONAL\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "This report compiles previously generated "
            "experimental outputs. No training or model "
            "evaluation was performed by the final report "
            "generator.\n"
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
        "Existing evaluation outputs were compiled."
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