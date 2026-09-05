"""
=========================================================
FINAL THESIS TABLES
=========================================================

Creates thesis-ready CSV tables from the finalized
evaluation outputs.

Source files:

1. evaluation_metrics.csv
2. ablation_summary.csv
3. uncertainty_statistics.csv
4. statistical_significance.csv

Important:

    - All paths are derived from REPORT_DIR.
    - DATASET_MODE is NOT used to construct the experiment
      directory.
    - ablation_study.csv contains per-sample results and is
      retained for statistical analysis.
    - ablation_summary.csv contains model-level mean results
      and is therefore used for the thesis ablation table.
    - Missing source files are reported explicitly.
    - Existing source files are copied without changing
      their numerical values.

Output:

    outputs/
        <EXPERIMENT_NAME>/
            reports/
                thesis_tables/
                    Table_4_1_Main_Performance.csv
                    Table_4_2_Ablation_Study.csv
                    Table_4_3_Uncertainty_Statistics.csv
                    Table_4_4_Statistical_Significance.csv

=========================================================
"""

import os

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR
)


# =========================================================
# SOURCE FILES
# =========================================================

MAIN_METRICS_FILE = os.path.join(
    REPORT_DIR,
    "evaluation_metrics.csv"
)

ABLATION_SUMMARY_FILE = os.path.join(
    REPORT_DIR,
    "ablation_summary.csv"
)

UNCERTAINTY_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_statistics.csv"
)

SIGNIFICANCE_FILE = os.path.join(
    REPORT_DIR,
    "statistical_significance.csv"
)


# =========================================================
# THESIS TABLE DIRECTORY
# =========================================================

THESIS_DIR = os.path.join(
    REPORT_DIR,
    "thesis_tables"
)


# =========================================================
# COPY SOURCE TO THESIS TABLE
# =========================================================

def copy_table(
        source_file,
        output_file,
        table_name
):
    """
    Read a source CSV and save a copy in the thesis_tables
    directory.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """

    print()
    print(
        f"Processing {table_name}"
    )

    print(
        "Source:",
        source_file
    )

    if not os.path.isfile(
        source_file
    ):

        print(
            f"WARNING: Source file not found for "
            f"{table_name}."
        )

        print(
            source_file
        )

        return None

    dataframe = pd.read_csv(
        source_file
    )

    if dataframe.empty:

        print(
            f"WARNING: {table_name} source file is empty."
        )

        return None

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        f"Created: {output_file}"
    )

    print(
        "Rows   :",
        len(dataframe)
    )

    print(
        "Columns:",
        len(dataframe.columns)
    )

    return dataframe


# =========================================================
# VALIDATE MAIN PERFORMANCE TABLE
# =========================================================

def validate_main_performance(
        dataframe
):
    """
    Validate the main model evaluation table.
    """

    required_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "\nTable 4.1 is missing required columns:\n"
            f"{missing}"
        )


# =========================================================
# VALIDATE ABLATION SUMMARY
# =========================================================

def validate_ablation_summary(
        dataframe
):
    """
    Validate the aggregate ablation summary.

    The thesis table should contain one row per model.
    """

    required_columns = [
        "Model",
        "Attention",
        "Residual",
        "Uncertainty",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "\nTable 4.2 is missing required columns:\n"
            f"{missing}"
        )

    if dataframe["Model"].duplicated().any():

        raise ValueError(
            "\nTable 4.2 contains duplicate model rows.\n"
            "ablation_summary.csv should contain exactly "
            "one row per model."
        )


# =========================================================
# VALIDATE UNCERTAINTY TABLE
# =========================================================

def validate_uncertainty_table(
        dataframe
):
    """
    Validate uncertainty statistics when the corresponding
    source file exists.
    """

    if dataframe.empty:

        raise ValueError(
            "\nUncertainty statistics table is empty."
        )


# =========================================================
# VALIDATE SIGNIFICANCE TABLE
# =========================================================

def validate_significance_table(
        dataframe
):
    """
    Validate statistical significance results.
    """

    if dataframe.empty:

        raise ValueError(
            "\nStatistical significance table is empty."
        )

    required_columns = [
        "Comparison",
        "N_Pairs",
        "Raw_P_Value"
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "\nTable 4.4 is missing required columns:\n"
            f"{missing}"
        )


# =========================================================
# GENERATE THESIS TABLES
# =========================================================

def generate_thesis_tables():

    print()
    print("=" * 70)
    print("GENERATING THESIS TABLES")
    print("=" * 70)

    print()
    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Report directory:",
        REPORT_DIR
    )

    print(
        "Thesis table directory:",
        THESIS_DIR
    )

    # =====================================================
    # CREATE THESIS DIRECTORY
    # =====================================================

    os.makedirs(
        THESIS_DIR,
        exist_ok=True
    )

    # =====================================================
    # TRACK RESULTS
    # =====================================================

    created_tables = []
    missing_tables = []

    # =====================================================
    # TABLE 4.1
    # MAIN PERFORMANCE
    # =====================================================

    output_file = os.path.join(
        THESIS_DIR,
        "Table_4_1_Main_Performance.csv"
    )

    dataframe = copy_table(
        source_file=MAIN_METRICS_FILE,
        output_file=output_file,
        table_name="Table 4.1 - Main Performance"
    )

    if dataframe is not None:

        validate_main_performance(
            dataframe
        )

        created_tables.append(
            "Table_4_1_Main_Performance.csv"
        )

    else:

        missing_tables.append(
            "Table_4_1_Main_Performance.csv"
        )

    # =====================================================
    # TABLE 4.2
    # ABLATION STUDY
    # =====================================================

    output_file = os.path.join(
        THESIS_DIR,
        "Table_4_2_Ablation_Study.csv"
    )

    dataframe = copy_table(
        source_file=ABLATION_SUMMARY_FILE,
        output_file=output_file,
        table_name="Table 4.2 - Ablation Study"
    )

    if dataframe is not None:

        validate_ablation_summary(
            dataframe
        )

        created_tables.append(
            "Table_4_2_Ablation_Study.csv"
        )

    else:

        missing_tables.append(
            "Table_4_2_Ablation_Study.csv"
        )

    # =====================================================
    # TABLE 4.3
    # UNCERTAINTY STATISTICS
    # =====================================================

    output_file = os.path.join(
        THESIS_DIR,
        "Table_4_3_Uncertainty_Statistics.csv"
    )

    dataframe = copy_table(
        source_file=UNCERTAINTY_FILE,
        output_file=output_file,
        table_name="Table 4.3 - Uncertainty Statistics"
    )

    if dataframe is not None:

        validate_uncertainty_table(
            dataframe
        )

        created_tables.append(
            "Table_4_3_Uncertainty_Statistics.csv"
        )

    else:

        missing_tables.append(
            "Table_4_3_Uncertainty_Statistics.csv"
        )

    # =====================================================
    # TABLE 4.4
    # STATISTICAL SIGNIFICANCE
    # =====================================================

    output_file = os.path.join(
        THESIS_DIR,
        "Table_4_4_Statistical_Significance.csv"
    )

    dataframe = copy_table(
        source_file=SIGNIFICANCE_FILE,
        output_file=output_file,
        table_name="Table 4.4 - Statistical Significance"
    )

    if dataframe is not None:

        validate_significance_table(
            dataframe
        )

        created_tables.append(
            "Table_4_4_Statistical_Significance.csv"
        )

    else:

        missing_tables.append(
            "Table_4_4_Statistical_Significance.csv"
        )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print()
    print("=" * 70)
    print("THESIS TABLE GENERATION SUMMARY")
    print("=" * 70)

    print()

    if created_tables:

        print(
            "Tables created:"
        )

        for table in created_tables:

            print(
                "  [CREATED]",
                table
            )

    if missing_tables:

        print()
        print(
            "Tables not created:"
        )

        for table in missing_tables:

            print(
                "  [MISSING SOURCE]",
                table
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

    if len(missing_tables) > 0:

        print(
            "WARNING:"
        )

        print(
            "Some thesis tables could not be created "
            "because their source files are missing."
        )

        print()
        print(
            "This is not treated as a complete thesis "
            "table-generation run."
        )

    else:

        print(
            "All four thesis tables were created successfully."
        )

    print()
    print("=" * 70)
    print("THESIS TABLE GENERATION COMPLETE")
    print("=" * 70)

    return {
        "created": created_tables,
        "missing": missing_tables,
        "directory": THESIS_DIR
    }


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    generate_thesis_tables()