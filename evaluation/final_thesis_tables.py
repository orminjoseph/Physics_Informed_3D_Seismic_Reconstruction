"""
=========================================================
FINAL THESIS TABLES
=========================================================

Creates thesis-ready tables from:

1. evaluation_metrics.csv
2. ablation_study.csv
3. uncertainty_statistics.csv
4. statistical_significance.csv

=========================================================
"""

import os

import pandas as pd

from utils.config import DATASET_MODE


def generate_thesis_tables():

    print()
    print("=" * 70)
    print("GENERATING THESIS TABLES")
    print("=" * 70)

    report_dir = os.path.join(

        "outputs",

        DATASET_MODE,

        "reports"

    )

    thesis_dir = os.path.join(

        report_dir,

        "thesis_tables"

    )

    os.makedirs(
        thesis_dir,
        exist_ok=True
    )

    # =====================================================
    # TABLE 4.1
    # Main Performance
    # =====================================================

    metrics_file = os.path.join(

        report_dir,

        "evaluation_metrics.csv"

    )

    if os.path.exists(metrics_file):

        dataframe = pd.read_csv(
            metrics_file
        )

        dataframe.to_csv(

            os.path.join(
                thesis_dir,
                "Table_4_1_Main_Performance.csv"
            ),

            index=False

        )

        print(
            "Created Table_4_1_Main_Performance.csv"
        )

    # =====================================================
    # TABLE 4.2
    # Ablation Study
    # =====================================================

    ablation_file = os.path.join(

        report_dir,

        "ablation_study.csv"

    )

    if os.path.exists(ablation_file):

        dataframe = pd.read_csv(
            ablation_file
        )

        dataframe.to_csv(

            os.path.join(
                thesis_dir,
                "Table_4_2_Ablation_Study.csv"
            ),

            index=False

        )

        print(
            "Created Table_4_2_Ablation_Study.csv"
        )

    # =====================================================
    # TABLE 4.3
    # Uncertainty Statistics
    # =====================================================

    uncertainty_file = os.path.join(

        report_dir,

        "uncertainty_statistics.csv"

    )

    if os.path.exists(uncertainty_file):

        dataframe = pd.read_csv(
            uncertainty_file
        )

        dataframe.to_csv(

            os.path.join(
                thesis_dir,
                "Table_4_3_Uncertainty_Statistics.csv"
            ),

            index=False

        )

        print(
            "Created Table_4_3_Uncertainty_Statistics.csv"
        )

    # =====================================================
    # TABLE 4.4
    # Statistical Significance
    # =====================================================

    significance_file = os.path.join(

        report_dir,

        "statistical_significance.csv"

    )

    if os.path.exists(significance_file):

        dataframe = pd.read_csv(
            significance_file
        )

        dataframe.to_csv(

            os.path.join(
                thesis_dir,
                "Table_4_4_Statistical_Significance.csv"
            ),

            index=False

        )

        print(
            "Created Table_4_4_Statistical_Significance.csv"
        )

    print()
    print("=" * 70)
    print("THESIS TABLE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    generate_thesis_tables()