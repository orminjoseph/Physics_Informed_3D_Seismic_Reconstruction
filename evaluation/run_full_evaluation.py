"""
====================================================================
FULL EVALUATION PIPELINE
====================================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Master orchestration script for the complete PhD evaluation pipeline.

The runner supports:

    1. Optional training
    2. Model evaluation
    3. Reconstruction gallery
    4. Uncertainty analysis
    5. Uncertainty evaluation
    6. Uncertainty statistics
    7. Baseline comparison
    8. Ablation study
    9. Statistical significance
   10. Thesis tables
   11. Final report

Resume behaviour
----------------
When RESUME_EVALUATION = True:

    - Valid completed stages are skipped.
    - The first incomplete stage is executed.
    - Later stages are executed when their dependencies are ready.
    - Existing valid results are preserved.

When FORCE_RERUN_EVALUATION = True:

    - All evaluation stages are executed again.
    - FORCE_RERUN_EVALUATION takes priority over RESUME_EVALUATION.

Current experiment
------------------
DATASET_MODE = synthetic
EXPERIMENT_NAME = synthetic_training

The runner derives all output paths from utils.config.py.

Author: Ormin Joseph
====================================================================
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    OUTPUT_ROOT,
    CHECKPOINT_DIR,
    REPORT_DIR,
    RUN_TRAINING,
    RESUME_EVALUATION,
    FORCE_RERUN_EVALUATION,
    GALLERY_NUMBER_OF_SAMPLES,
)


# ====================================================================
# PATHS
# ====================================================================

REPORT_PATH = Path(REPORT_DIR)
CHECKPOINT_PATH = Path(CHECKPOINT_DIR)

BEST_MODEL_PATH = CHECKPOINT_PATH / "best_model.pth"


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def print_header(title: str) -> None:
    """
    Print a consistent section header.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_stage(stage_number: int, title: str) -> None:
    """
    Print the current pipeline stage.
    """

    print()
    print("-" * 78)
    print(f"STAGE {stage_number}: {title}")
    print("-" * 78)


def run_module(module_name: str, description: str) -> None:
    """
    Execute a Python module using the current project interpreter.

    Using sys.executable guarantees that the same Python interpreter
    used to launch this master script is also used for every stage.
    """

    print()
    print(f"Running: {description}")
    print(f"Module : python -m {module_name}")
    print()

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Evaluation stage failed: {module_name}\n"
            f"Return code: {result.returncode}"
        )


def file_is_valid(path: Path, minimum_size: int = 1) -> bool:
    """
    Check whether a required output file exists and is non-empty.
    """

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size >= minimum_size
    )


def csv_has_required_columns(
    path: Path,
    required_columns: list[str],
) -> bool:
    """
    Validate that a CSV exists and contains the required columns.

    This prevents the resume system from treating an incomplete or
    incorrectly generated CSV as a completed evaluation stage.
    """

    if not file_is_valid(path):
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                return False

            available_columns = set(reader.fieldnames)

            return set(required_columns).issubset(
                available_columns
            )

    except Exception:
        return False


def json_is_valid(path: Path) -> bool:
    """
    Validate a JSON metadata file.
    """

    if not file_is_valid(path):
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            json.load(file)

        return True

    except Exception:
        return False


def figure_is_valid(path: Path) -> bool:
    """
    Validate a generated figure.
    """

    return file_is_valid(path, minimum_size=100)


# ====================================================================
# STAGE VALIDATION
# ====================================================================

def model_evaluation_complete() -> bool:
    """
    Check whether the main model evaluation is complete.
    """

    path = REPORT_PATH / "evaluation_metrics.csv"

    return csv_has_required_columns(
        path,
        [
            "MAE",
            "RMSE",
            "PSNR",
            "SSIM",
        ],
    )


def reconstruction_gallery_complete() -> bool:
    """
    Check whether the reconstruction gallery contains the configured
    number of sample figures.
    """

    gallery_dir = REPORT_PATH / "gallery"

    if not gallery_dir.exists():
        return False

    number_of_samples = int(GALLERY_NUMBER_OF_SAMPLES)

    for index in range(number_of_samples):

        figure_path = gallery_dir / f"sample_{index:03d}.png"

        if not figure_is_valid(figure_path):
            return False

    return True


def uncertainty_analysis_complete() -> bool:
    """
    Check whether the uncertainty analysis figure exists.
    """

    path = (
        REPORT_PATH
        / "uncertainty"
        / "uncertainty_analysis.png"
    )

    return figure_is_valid(path)


def uncertainty_evaluation_complete() -> bool:
    """
    Check whether uncertainty evaluation has completed.

    Three outputs are required:

        uncertainty_evaluation.csv
        uncertainty_error_correlation.csv
        uncertainty_evaluation_metadata.json
    """

    evaluation_csv = (
        REPORT_PATH
        / "uncertainty_evaluation.csv"
    )

    correlation_csv = (
        REPORT_PATH
        / "uncertainty_error_correlation.csv"
    )

    metadata_json = (
        REPORT_PATH
        / "uncertainty_evaluation_metadata.json"
    )

    evaluation_valid = csv_has_required_columns(
        evaluation_csv,
        [
            "MAE",
            "RMSE",
            "Missing_MAE",
            "Missing_RMSE",
            "aleatoric_variance",
            "epistemic_variance",
            "predictive_variance",
            "predictive_std",
        ],
    )

    correlation_valid = file_is_valid(
        correlation_csv
    )

    metadata_valid = json_is_valid(
        metadata_json
    )

    return (
        evaluation_valid
        and correlation_valid
        and metadata_valid
    )


def uncertainty_statistics_complete() -> bool:
    """
    Check whether final uncertainty statistics have been generated.

    This stage is now considered COMPLETE for the current experiment
    because uncertainty_statistics.csv has already been generated and
    validated.
    """

    path = (
        REPORT_PATH
        / "uncertainty_statistics.csv"
    )

    return csv_has_required_columns(
        path,
        [
            "Experiment_Name",
            "Number_of_Patches",
            "MC_Samples",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_Missing_MAE",
            "Mean_Missing_RMSE",
            "Mean_Aleatoric_Variance",
            "Mean_Epistemic_Variance",
            "Mean_Predictive_Variance",
            "Mean_Predictive_Std",
            "Maximum_Predictive_Variance",
            "Maximum_Predictive_Std",
            "Mean_Observed_Preservation_Error",
            "Maximum_Observed_Preservation_Error",
            "Mean_Measured_Missing_Rate",
            "Maximum_Measured_Missing_Rate",
        ],
    )


def baseline_comparison_complete() -> bool:
    """
    Check whether the common seven-method baseline comparison
    has completed.
    """

    path = (
        REPORT_PATH
        / "baseline_comparison.csv"
    )

    summary_path = (
        REPORT_PATH
        / "baseline_comparison_summary.csv"
    )

    return (
        csv_has_required_columns(
            path,
            [
                "Method",
                "MAE",
                "RMSE",
                "PSNR",
                "SNR",
                "SSIM",
            ],
        )
        and file_is_valid(summary_path)
    )


def ablation_complete() -> bool:
    """
    Check whether the ablation study has completed.
    """

    path = (
        REPORT_PATH
        / "ablation_study.csv"
    )

    summary_path = (
        REPORT_PATH
        / "ablation_summary.csv"
    )

    metadata_path = (
        REPORT_PATH
        / "ablation_metadata.json"
    )

    return (
        csv_has_required_columns(
            path,
            [
                "Model",
                "MAE",
                "RMSE",
                "PSNR",
                "SSIM",
            ],
        )
        and file_is_valid(summary_path)
        and json_is_valid(metadata_path)
    )


def statistical_significance_complete() -> bool:
    """
    Check whether statistical significance analysis has completed.
    """

    path = (
        REPORT_PATH
        / "statistical_significance.csv"
    )

    metadata_path = (
        REPORT_PATH
        / "statistical_significance_metadata.json"
    )

    return (
        csv_has_required_columns(
            path,
            [
                "Comparison",
                "Metric",
                "p_value",
            ],
        )
        and json_is_valid(metadata_path)
    )


def thesis_tables_complete() -> bool:
    """
    Check whether all required thesis tables have been generated.
    """

    required_tables = [
        "Table_4_1_Main_Performance.csv",
        "Table_4_2_Ablation_Study.csv",
        "Table_4_3_Uncertainty_Statistics.csv",
        "Table_4_4_Statistical_Significance.csv",
    ]

    tables_dir = REPORT_PATH / "thesis_tables"

    if not tables_dir.exists():
        return False

    for table_name in required_tables:

        table_path = tables_dir / table_name

        if not file_is_valid(table_path):
            return False

    return True


def final_report_complete() -> bool:
    """
    Check whether the final report exists.
    """

    path = REPORT_PATH / "final_report.txt"

    return file_is_valid(path)


# ====================================================================
# STATUS REPORT
# ====================================================================

def display_pipeline_status() -> None:
    """
    Display the current state of every evaluation stage.
    """

    print_header("CURRENT EVALUATION PIPELINE STATUS")

    stages = [
        (
            "Model Evaluation",
            model_evaluation_complete(),
        ),
        (
            "Reconstruction Gallery",
            reconstruction_gallery_complete(),
        ),
        (
            "Uncertainty Analysis",
            uncertainty_analysis_complete(),
        ),
        (
            "Uncertainty Evaluation",
            uncertainty_evaluation_complete(),
        ),
        (
            "Uncertainty Statistics",
            uncertainty_statistics_complete(),
        ),
        (
            "Baseline Comparison",
            baseline_comparison_complete(),
        ),
        (
            "Ablation Study",
            ablation_complete(),
        ),
        (
            "Statistical Significance",
            statistical_significance_complete(),
        ),
        (
            "Thesis Tables",
            thesis_tables_complete(),
        ),
        (
            "Final Report",
            final_report_complete(),
        ),
    ]

    for number, (name, complete) in enumerate(
        stages,
        start=1,
    ):

        status = "[COMPLETE]" if complete else "[PENDING]"

        print(
            f"{number:02d}. {status:<12} {name}"
        )


# ====================================================================
# TRAINING
# ====================================================================

def run_training_if_required() -> None:
    """
    Train the model when RUN_TRAINING is enabled.

    Existing best_model.pth is preserved unless the training module
    itself updates it.
    """

    if not RUN_TRAINING:

        print(
            "RUN_TRAINING = False"
        )

        print(
            "Training skipped."
        )

        return

    print_header("TRAINING")

    print(
        "RUN_TRAINING = True"
    )

    print(
        "Training will be executed using the configured training module."
    )

    run_module(
        "train.train",
        "Training pipeline",
    )


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main() -> None:
    """
    Execute the complete resumable evaluation pipeline.
    """

    print_header(
        "PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION"
    )

    print(
        "RESUME-ENABLED FULL EVALUATION PIPELINE"
    )

    print()
    print(
        f"Experiment       : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode     : {DATASET_MODE}"
    )

    print(
        f"Output root      : {OUTPUT_ROOT}"
    )

    print(
        f"Report directory : {REPORT_PATH}"
    )

    print(
        f"Checkpoint       : {BEST_MODEL_PATH}"
    )

    print()
    print(
        f"RUN_TRAINING             : {RUN_TRAINING}"
    )

    print(
        f"RESUME_EVALUATION        : {RESUME_EVALUATION}"
    )

    print(
        f"FORCE_RERUN_EVALUATION  : {FORCE_RERUN_EVALUATION}"
    )

    print(
        f"GALLERY_NUMBER_OF_SAMPLES: {GALLERY_NUMBER_OF_SAMPLES}"
    )

    # ---------------------------------------------------------------
    # Create output directory if necessary.
    # ---------------------------------------------------------------

    REPORT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    run_training_if_required()

    # ---------------------------------------------------------------
    # Verify checkpoint before evaluation.
    # ---------------------------------------------------------------

    print_header("CHECKPOINT VALIDATION")

    if not file_is_valid(
        BEST_MODEL_PATH,
        minimum_size=1_000_000,
    ):

        raise FileNotFoundError(
            "\nRequired checkpoint was not found or is unexpectedly small:\n"
            f"{BEST_MODEL_PATH}\n\n"
            "The evaluation pipeline cannot continue without "
            "best_model.pth."
        )

    print(
        "[VALID] best_model.pth"
    )

    # ---------------------------------------------------------------
    # Show current state.
    # ---------------------------------------------------------------

    display_pipeline_status()

    # ---------------------------------------------------------------
    # Determine whether previous results may be reused.
    # ---------------------------------------------------------------

    resume = (
        RESUME_EVALUATION
        and not FORCE_RERUN_EVALUATION
    )

    if FORCE_RERUN_EVALUATION:

        print()
        print(
            "FORCE_RERUN_EVALUATION = True"
        )

        print(
            "All evaluation stages will be executed again."
        )

    elif RESUME_EVALUATION:

        print()
        print(
            "RESUME_EVALUATION = True"
        )

        print(
            "Valid completed stages will be skipped."
        )

    else:

        print()
        print(
            "RESUME_EVALUATION = False"
        )

        print(
            "Evaluation stages will be executed."
        )

    # =================================================================
    # STAGE 1 — MODEL EVALUATION
    # =================================================================

    print_stage(
        1,
        "MODEL EVALUATION",
    )

    if resume and model_evaluation_complete():

        print(
            "[SKIPPED] Model evaluation already completed."
        )

    else:

        run_module(
            "evaluation.evaluate",
            "Model evaluation",
        )

    # =================================================================
    # STAGE 2 — RECONSTRUCTION GALLERY
    # =================================================================

    print_stage(
        2,
        "RECONSTRUCTION GALLERY",
    )

    if resume and reconstruction_gallery_complete():

        print(
            "[SKIPPED] Reconstruction gallery already completed."
        )

    else:

        run_module(
            "evaluation.reconstruction_gallery",
            "Reconstruction gallery",
        )

    # =================================================================
    # STAGE 3 — UNCERTAINTY ANALYSIS
    # =================================================================

    print_stage(
        3,
        "UNCERTAINTY ANALYSIS",
    )

    if resume and uncertainty_analysis_complete():

        print(
            "[SKIPPED] Uncertainty analysis already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_analysis",
            "Uncertainty analysis",
        )

    # =================================================================
    # STAGE 4 — UNCERTAINTY EVALUATION
    # =================================================================

    print_stage(
        4,
        "UNCERTAINTY EVALUATION",
    )

    if resume and uncertainty_evaluation_complete():

        print(
            "[SKIPPED] Uncertainty evaluation already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_evaluation",
            "Uncertainty–reconstruction error evaluation",
        )

    # =================================================================
    # STAGE 5 — UNCERTAINTY STATISTICS
    # =================================================================

    print_stage(
        5,
        "UNCERTAINTY STATISTICS",
    )

    if resume and uncertainty_statistics_complete():

        print(
            "[SKIPPED] Uncertainty statistics already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_statistics",
            "Final uncertainty statistics generation",
        )

    # =================================================================
    # STAGE 6 — BASELINE COMPARISON
    # =================================================================

    print_stage(
        6,
        "COMMON SEVEN-METHOD BASELINE COMPARISON",
    )

    if resume and baseline_comparison_complete():

        print(
            "[SKIPPED] Baseline comparison already completed."
        )

    else:

        run_module(
            "evaluation.compare_with_baselines",
            "Seven-method reconstruction comparison",
        )

    # =================================================================
    # STAGE 7 — ABLATION STUDY
    # =================================================================

    print_stage(
        7,
        "ABLATION STUDY",
    )

    if resume and ablation_complete():

        print(
            "[SKIPPED] Ablation study already completed."
        )

    else:

        run_module(
            "evaluation.ablation_study",
            "Ablation study",
        )

    # =================================================================
    # STAGE 8 — STATISTICAL SIGNIFICANCE
    # =================================================================

    print_stage(
        8,
        "STATISTICAL SIGNIFICANCE",
    )

    if resume and statistical_significance_complete():

        print(
            "[SKIPPED] Statistical significance already completed."
        )

    else:

        run_module(
            "evaluation.statistical_significance",
            "Statistical significance analysis",
        )

    # =================================================================
    # STAGE 9 — THESIS TABLES
    # =================================================================

    print_stage(
        9,
        "THESIS TABLES",
    )

    if resume and thesis_tables_complete():

        print(
            "[SKIPPED] Thesis tables already completed."
        )

    else:

        run_module(
            "evaluation.thesis_tables",
            "Thesis table generation",
        )

    # =================================================================
    # STAGE 10 — FINAL REPORT
    # =================================================================

    print_stage(
        10,
        "FINAL REPORT",
    )

    if resume and final_report_complete():

        print(
            "[SKIPPED] Final report already completed."
        )

    else:

        run_module(
            "evaluation.final_report",
            "Final report generation",
        )

    # =================================================================
    # FINAL STATUS
    # =================================================================

    display_pipeline_status()

    print_header(
        "FULL EVALUATION PIPELINE COMPLETE"
    )

    print(
        "All configured evaluation stages have completed successfully."
    )

    print()
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset    : {DATASET_MODE}"
    )

    print(
        f"Reports    : {REPORT_PATH}"
    )


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()