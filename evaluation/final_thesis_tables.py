"""
=========================================================
FINAL THESIS TABLES
=========================================================

Creates thesis-ready CSV tables from finalized evaluation
outputs.

SOURCE FILES
------------

1. evaluation_metrics.csv
2. ablation_summary.csv
3. uncertainty_statistics.csv
4. statistical_significance.csv

IMPORTANT DESIGN PRINCIPLES
----------------------------

    - All paths are derived from REPORT_DIR.
    - DATASET_MODE is NOT used to construct the experiment
      directory.
    - EXPERIMENT_NAME is used only for reporting/metadata.
    - ablation_study.csv contains per-sample results and is
      retained for statistical analysis.
    - ablation_summary.csv contains model-level aggregate
      results and is therefore used for the thesis ablation
      table.
    - Source numerical values are copied without modification.
    - Source files are validated before thesis tables are
      written.
    - Missing source files are reported explicitly.
    - Invalid source files cause the corresponding table to
      be rejected rather than silently copied.
    - Original evaluation files are never modified.
    - A metadata JSON records table provenance and generation
      status.

OUTPUT
------

outputs/
    <EXPERIMENT_NAME>/
        reports/
            thesis_tables/
                Table_4_1_Main_Performance.csv
                Table_4_2_Ablation_Study.csv
                Table_4_3_Uncertainty_Statistics.csv
                Table_4_4_Statistical_Significance.csv
                thesis_tables_metadata.json

=========================================================
"""

import json
import os
from datetime import datetime

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# =========================================================
# SOURCE FILES
# =========================================================

MAIN_METRICS_FILE = os.path.join(
    REPORT_DIR,
    "evaluation_metrics.csv",
)

ABLATION_SUMMARY_FILE = os.path.join(
    REPORT_DIR,
    "ablation_summary.csv",
)

UNCERTAINTY_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_statistics.csv",
)

SIGNIFICANCE_FILE = os.path.join(
    REPORT_DIR,
    "statistical_significance.csv",
)


# =========================================================
# THESIS TABLE DIRECTORY
# =========================================================

THESIS_DIR = os.path.join(
    REPORT_DIR,
    "thesis_tables",
)


# =========================================================
# THESIS TABLE OUTPUT FILES
# =========================================================

TABLE_4_1_FILE = os.path.join(
    THESIS_DIR,
    "Table_4_1_Main_Performance.csv",
)

TABLE_4_2_FILE = os.path.join(
    THESIS_DIR,
    "Table_4_2_Ablation_Study.csv",
)

TABLE_4_3_FILE = os.path.join(
    THESIS_DIR,
    "Table_4_3_Uncertainty_Statistics.csv",
)

TABLE_4_4_FILE = os.path.join(
    THESIS_DIR,
    "Table_4_4_Statistical_Significance.csv",
)

METADATA_FILE = os.path.join(
    THESIS_DIR,
    "thesis_tables_metadata.json",
)


# =========================================================
# COMMON VALIDATION UTILITIES
# =========================================================

def validate_required_columns(
        dataframe,
        required_columns,
        table_name,
):
    """
    Validate that all required columns are present.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Table being validated.

    required_columns : list
        Required column names.

    table_name : str
        Human-readable table name.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"\n{table_name} is missing required columns:\n"
            f"{missing}"
        )


def validate_numeric_columns(
        dataframe,
        numeric_columns,
        table_name,
):
    """
    Validate that specified columns are numeric and contain
    only finite values.

    This prevents strings, NaN values, and infinite values
    from entering thesis tables unnoticed.
    """

    for column in numeric_columns:

        if column not in dataframe.columns:
            continue

        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if numeric_values.isna().any():

            raise ValueError(
                f"\n{table_name} contains non-numeric or "
                f"missing values in column '{column}'."
            )

        if not numeric_values.map(
            lambda value: pd.notna(value)
            and abs(float(value)) != float("inf")
        ).all():

            raise ValueError(
                f"\n{table_name} contains non-finite values "
                f"in column '{column}'."
            )


def load_source_table(
        source_file,
        table_name,
):
    """
    Load a source CSV without modifying it.

    Returns
    -------
    pandas.DataFrame or None
        Loaded dataframe, or None when the source file is
        missing.

    Raises
    ------
    ValueError
        If the source file exists but cannot be read or is
        empty.
    """

    print()
    print("-" * 70)
    print(f"PROCESSING: {table_name}")
    print("-" * 70)

    print(
        "Source:",
        source_file,
    )

    if not os.path.isfile(source_file):

        print(
            f"WARNING: Source file not found for "
            f"{table_name}."
        )

        return None

    try:

        dataframe = pd.read_csv(
            source_file,
        )

    except Exception as error:

        raise ValueError(
            f"\nUnable to read source file for "
            f"{table_name}:\n"
            f"{source_file}\n"
            f"Error: {error}"
        ) from error

    if dataframe.empty:

        raise ValueError(
            f"\nSource file for {table_name} is empty:\n"
            f"{source_file}"
        )

    print(
        "Rows   :",
        len(dataframe),
    )

    print(
        "Columns:",
        len(dataframe.columns),
    )

    return dataframe


# =========================================================
# VALIDATE MAIN PERFORMANCE TABLE
# =========================================================

def validate_main_performance(
        dataframe,
):
    """
    Validate the main model evaluation table.

    Expected metrics:

        MAE
        RMSE
        PSNR
        SNR
        SSIM
    """

    table_name = "Table 4.1 - Main Performance"

    required_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        table_name,
    )

    validate_numeric_columns(
        dataframe,
        required_columns,
        table_name,
    )

    # -----------------------------------------------------
    # Validate that at least one evaluation row exists.
    # -----------------------------------------------------

    if len(dataframe) == 0:

        raise ValueError(
            f"\n{table_name} contains no evaluation rows."
        )

    print(
        f"[VALID] {table_name}"
    )


# =========================================================
# VALIDATE ABLATION SUMMARY
# =========================================================

def validate_ablation_summary(
        dataframe,
):
    """
    Validate the aggregate ablation summary.

    The thesis ablation table should contain one row per
    model.

    Expected models include:

        Full_Model
        No_Attention
        No_Residual
        No_Uncertainty
        Plain_UNet

    The validation does not force the exact model list,
    because additional controlled ablations may be added
    later. It does, however, require unique model rows.
    """

    table_name = "Table 4.2 - Ablation Study"

    required_columns = [
        "Model",
        "Attention",
        "Residual",
        "Uncertainty",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        table_name,
    )

    # -----------------------------------------------------
    # Model names must be present.
    # -----------------------------------------------------

    if dataframe["Model"].isna().any():

        raise ValueError(
            f"\n{table_name} contains missing model names."
        )

    # -----------------------------------------------------
    # Exactly one row per model.
    # -----------------------------------------------------

    if dataframe["Model"].duplicated().any():

        duplicated_models = (
            dataframe.loc[
                dataframe["Model"].duplicated(),
                "Model",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"\n{table_name} contains duplicate model rows:\n"
            f"{duplicated_models}\n\n"
            "ablation_summary.csv must contain exactly one "
            "aggregate row per model."
        )

    # -----------------------------------------------------
    # Validate numerical metrics.
    # -----------------------------------------------------

    numeric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    validate_numeric_columns(
        dataframe,
        numeric_columns,
        table_name,
    )

    # -----------------------------------------------------
    # Validate that the summary is not accidentally a
    # per-sample table.
    # -----------------------------------------------------

    if "Sample_ID" in dataframe.columns:

        print(
            "NOTE: Sample_ID column detected in ablation "
            "summary."
        )

        print(
            "The table will be copied unchanged, but the "
            "Model uniqueness requirement remains enforced."
        )

    print(
        f"[VALID] {table_name}"
    )


# =========================================================
# VALIDATE UNCERTAINTY TABLE
# =========================================================

def validate_uncertainty_table(
        dataframe,
):
    """
    Validate the uncertainty statistics table.

    The uncertainty table may contain several uncertainty
    statistics. Because the exact number of summary columns
    can evolve as the uncertainty framework is refined,
    this validator checks for meaningful uncertainty-related
    columns without altering the source table.
    """

    table_name = "Table 4.3 - Uncertainty Statistics"

    if dataframe.empty:

        raise ValueError(
            f"\n{table_name} is empty."
        )

    # -----------------------------------------------------
    # Identify uncertainty-related columns.
    # -----------------------------------------------------

    uncertainty_keywords = (
        "aleatoric",
        "epistemic",
        "predictive",
        "uncertainty",
        "variance",
        "std",
    )

    uncertainty_columns = [
        column
        for column in dataframe.columns
        if any(
            keyword in str(column).lower()
            for keyword in uncertainty_keywords
        )
    ]

    if not uncertainty_columns:

        raise ValueError(
            f"\n{table_name} does not contain any recognized "
            "uncertainty-statistics columns."
        )

    # -----------------------------------------------------
    # Validate uncertainty columns that contain numeric
    # statistics.
    # -----------------------------------------------------

    for column in uncertainty_columns:

        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        # A column may be a descriptive text column whose
        # name happens to contain a keyword. Only enforce
        # numeric validation when at least one value can be
        # interpreted numerically.
        if numeric_values.notna().any():

            if numeric_values.isna().any():

                raise ValueError(
                    f"\n{table_name} contains invalid or "
                    f"missing values in uncertainty column "
                    f"'{column}'."
                )

            if not numeric_values.map(
                lambda value:
                    pd.notna(value)
                    and abs(float(value)) != float("inf")
            ).all():

                raise ValueError(
                    f"\n{table_name} contains non-finite "
                    f"values in column '{column}'."
                )

    print(
        f"[VALID] {table_name}"
    )


# =========================================================
# VALIDATE STATISTICAL SIGNIFICANCE TABLE
# =========================================================

def validate_significance_table(
        dataframe,
):
    """
    Validate statistical significance results.

    Expected core fields are based on the finalized paired
    statistical-analysis workflow.
    """

    table_name = "Table 4.4 - Statistical Significance"

    if dataframe.empty:

        raise ValueError(
            f"\n{table_name} is empty."
        )

    required_columns = [
        "Comparison",
        "N_Pairs",
        "Raw_P_Value",
    ]

    validate_required_columns(
        dataframe,
        required_columns,
        table_name,
    )

    numeric_columns = [
        "N_Pairs",
        "Raw_P_Value",
    ]

    # Optional finalized statistical fields.
    optional_numeric_columns = [
        "Adjusted_P_Value",
        "Holm_Adjusted_P_Value",
        "Cohen_dz",
        "Mean_Difference",
        "T_Statistic",
        "Degrees_of_Freedom",
    ]

    numeric_columns.extend(
        [
            column
            for column in optional_numeric_columns
            if column in dataframe.columns
        ]
    )

    validate_numeric_columns(
        dataframe,
        numeric_columns,
        table_name,
    )

    # -----------------------------------------------------
    # N_Pairs must be positive.
    # -----------------------------------------------------

    if (dataframe["N_Pairs"] <= 0).any():

        raise ValueError(
            f"\n{table_name} contains a non-positive "
            "N_Pairs value."
        )

    # -----------------------------------------------------
    # P-values must lie within [0, 1].
    # -----------------------------------------------------

    if (
        (dataframe["Raw_P_Value"] < 0)
        | (dataframe["Raw_P_Value"] > 1)
    ).any():

        raise ValueError(
            f"\n{table_name} contains Raw_P_Value values "
            "outside the valid [0, 1] interval."
        )

    if "Adjusted_P_Value" in dataframe.columns:

        if (
            (dataframe["Adjusted_P_Value"] < 0)
            | (dataframe["Adjusted_P_Value"] > 1)
        ).any():

            raise ValueError(
                f"\n{table_name} contains Adjusted_P_Value "
                "values outside the valid [0, 1] interval."
            )

    if "Holm_Adjusted_P_Value" in dataframe.columns:

        if (
            (dataframe["Holm_Adjusted_P_Value"] < 0)
            | (dataframe["Holm_Adjusted_P_Value"] > 1)
        ).any():

            raise ValueError(
                f"\n{table_name} contains Holm-adjusted "
                "p-values outside the valid [0, 1] interval."
            )

    print(
        f"[VALID] {table_name}"
    )


# =========================================================
# WRITE VALIDATED THESIS TABLE
# =========================================================

def write_thesis_table(
        dataframe,
        output_file,
        table_name,
):
    """
    Write a validated dataframe to a thesis-table CSV.

    No numerical transformation, rounding, sorting, or
    filtering is performed.
    """

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[CREATED] {table_name}"
    )

    print(
        "Output:",
        output_file,
    )

    print(
        "Rows:",
        len(dataframe),
    )

    print(
        "Columns:",
        len(dataframe.columns),
    )


# =========================================================
# GENERATE METADATA
# =========================================================

def generate_metadata(
        created_tables,
        missing_tables,
        failed_tables,
):
    """
    Generate provenance metadata for the thesis-table
    generation run.
    """

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "report_directory": REPORT_DIR,
        "thesis_table_directory": THESIS_DIR,
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_policy": (
            "Source numerical values are copied without "
            "modification."
        ),
        "data_processing": (
            "No rounding, filtering, sorting, aggregation, "
            "or numerical transformation is performed by "
            "this script."
        ),
        "created_tables": created_tables,
        "missing_tables": missing_tables,
        "failed_tables": failed_tables,
        "complete": (
            len(missing_tables) == 0
            and len(failed_tables) == 0
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print(
        "[CREATED] thesis_tables_metadata.json"
    )


# =========================================================
# GENERATE THESIS TABLES
# =========================================================

def generate_thesis_tables():
    """
    Generate all validated thesis tables.

    Returns
    -------
    dict
        Generation summary.
    """

    print()
    print("=" * 70)
    print("FINAL THESIS TABLE GENERATION")
    print("=" * 70)

    print()
    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Report directory:",
        REPORT_DIR,
    )

    print(
        "Thesis table directory:",
        THESIS_DIR,
    )

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        THESIS_DIR,
        exist_ok=True,
    )

    # =====================================================
    # TRACK RESULTS
    # =====================================================

    created_tables = []
    missing_tables = []
    failed_tables = []

    # =====================================================
    # TABLE 4.1
    # MAIN PERFORMANCE
    # =====================================================

    dataframe = load_source_table(
        source_file=MAIN_METRICS_FILE,
        table_name="Table 4.1 - Main Performance",
    )

    if dataframe is None:

        missing_tables.append(
            "Table_4_1_Main_Performance.csv"
        )

    else:

        try:

            validate_main_performance(
                dataframe
            )

            write_thesis_table(
                dataframe,
                TABLE_4_1_FILE,
                "Table 4.1 - Main Performance",
            )

            created_tables.append(
                "Table_4_1_Main_Performance.csv"
            )

        except ValueError as error:

            failed_tables.append(
                "Table_4_1_Main_Performance.csv"
            )

            print(
                "ERROR:",
                error,
            )

    # =====================================================
    # TABLE 4.2
    # ABLATION STUDY
    # =====================================================

    dataframe = load_source_table(
        source_file=ABLATION_SUMMARY_FILE,
        table_name="Table 4.2 - Ablation Study",
    )

    if dataframe is None:

        missing_tables.append(
            "Table_4_2_Ablation_Study.csv"
        )

    else:

        try:

            validate_ablation_summary(
                dataframe
            )

            write_thesis_table(
                dataframe,
                TABLE_4_2_FILE,
                "Table 4.2 - Ablation Study",
            )

            created_tables.append(
                "Table_4_2_Ablation_Study.csv"
            )

        except ValueError as error:

            failed_tables.append(
                "Table_4_2_Ablation_Study.csv"
            )

            print(
                "ERROR:",
                error,
            )

    # =====================================================
    # TABLE 4.3
    # UNCERTAINTY STATISTICS
    # =====================================================

    dataframe = load_source_table(
        source_file=UNCERTAINTY_FILE,
        table_name="Table 4.3 - Uncertainty Statistics",
    )

    if dataframe is None:

        missing_tables.append(
            "Table_4_3_Uncertainty_Statistics.csv"
        )

    else:

        try:

            validate_uncertainty_table(
                dataframe
            )

            write_thesis_table(
                dataframe,
                TABLE_4_3_FILE,
                "Table 4.3 - Uncertainty Statistics",
            )

            created_tables.append(
                "Table_4_3_Uncertainty_Statistics.csv"
            )

        except ValueError as error:

            failed_tables.append(
                "Table_4_3_Uncertainty_Statistics.csv"
            )

            print(
                "ERROR:",
                error,
            )

    # =====================================================
    # TABLE 4.4
    # STATISTICAL SIGNIFICANCE
    # =====================================================

    dataframe = load_source_table(
        source_file=SIGNIFICANCE_FILE,
        table_name="Table 4.4 - Statistical Significance",
    )

    if dataframe is None:

        missing_tables.append(
            "Table_4_4_Statistical_Significance.csv"
        )

    else:

        try:

            validate_significance_table(
                dataframe
            )

            write_thesis_table(
                dataframe,
                TABLE_4_4_FILE,
                "Table 4.4 - Statistical Significance",
            )

            created_tables.append(
                "Table_4_4_Statistical_Significance.csv"
            )

        except ValueError as error:

            failed_tables.append(
                "Table_4_4_Statistical_Significance.csv"
            )

            print(
                "ERROR:",
                error,
            )

    # =====================================================
    # METADATA
    # =====================================================

    generate_metadata(
        created_tables=created_tables,
        missing_tables=missing_tables,
        failed_tables=failed_tables,
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print("THESIS TABLE GENERATION SUMMARY")
    print("=" * 70)

    print()

    print(
        "Tables created:",
        len(created_tables),
    )

    for table in created_tables:

        print(
            "  [CREATED]",
            table,
        )

    if missing_tables:

        print()
        print(
            "Missing source tables:",
            len(missing_tables),
        )

        for table in missing_tables:

            print(
                "  [MISSING SOURCE]",
                table,
            )

    if failed_tables:

        print()
        print(
            "Failed validation:",
            len(failed_tables),
        )

        for table in failed_tables:

            print(
                "  [VALIDATION FAILED]",
                table,
            )

    print()
    print(
        "Thesis tables directory:"
    )

    print(
        THESIS_DIR
    )

    print()

    # =====================================================
    # COMPLETION STATUS
    # =====================================================

    complete = (
        len(missing_tables) == 0
        and len(failed_tables) == 0
    )

    if complete:

        print(
            "[SUCCESS]"
        )

        print(
            "All four thesis tables were generated "
            "successfully from validated source files."
        )

    else:

        print(
            "[INCOMPLETE]"
        )

        print(
            "The thesis-table generation run is not "
            "complete."
        )

        print(
            "Review the missing-source and validation "
            "messages above before using the tables."
        )

    print()
    print("=" * 70)
    print("FINAL THESIS TABLE GENERATION COMPLETE")
    print("=" * 70)

    return {
        "created": created_tables,
        "missing": missing_tables,
        "failed": failed_tables,
        "directory": THESIS_DIR,
        "complete": complete,
    }


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    generate_thesis_tables()