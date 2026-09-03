"""
=========================================================
Statistical Significance Test
=========================================================

Paired t-test between the Full Model and each ablation
model.

The test is performed on paired SSIM values obtained from
the same evaluation samples/patches.

Results are saved inside the current experiment directory:

outputs/
    <EXPERIMENT_NAME>/
        reports/
            statistical_significance.csv

=========================================================
"""

import os

import pandas as pd

from scipy.stats import ttest_rel

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR
)


# =========================================================
# CONFIGURATION
# =========================================================

ALPHA = 0.05

ABLATION_FILE = os.path.join(
    REPORT_DIR,
    "ablation_study.csv"
)

OUTPUT_FILE = os.path.join(
    REPORT_DIR,
    "statistical_significance.csv"
)


# =========================================================
# STATISTICAL SIGNIFICANCE TEST
# =========================================================

def run_significance_test():

    print()
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 60)

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME
    )

    print(
        "Input file :",
        ABLATION_FILE
    )

    print(
        "Output file:",
        OUTPUT_FILE
    )

    # -----------------------------------------------------
    # Check input file
    # -----------------------------------------------------

    if not os.path.exists(
        ABLATION_FILE
    ):

        raise FileNotFoundError(
            "\nAblation study file not found:\n"
            f"{ABLATION_FILE}\n\n"
            "Run the ablation study first."
        )

    # -----------------------------------------------------
    # Load ablation results
    # -----------------------------------------------------

    dataframe = pd.read_csv(
        ABLATION_FILE
    )

    print()
    print(
        "Rows loaded:",
        len(dataframe)
    )

    # -----------------------------------------------------
    # Check required columns
    # -----------------------------------------------------

    required_columns = [
        "Model",
        "SSIM"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in dataframe.columns

    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{missing_columns}\n\n"
            "The ablation study must contain at least "
            "'Model' and 'SSIM' columns."
        )

    # -----------------------------------------------------
    # Display available models
    # -----------------------------------------------------

    models = dataframe[
        "Model"
    ].unique()

    print()
    print(
        "Models found:"
    )

    for model_name in models:

        print(
            "  -",
            model_name
        )

    # -----------------------------------------------------
    # Full Model
    # -----------------------------------------------------

    full_model = dataframe[
        dataframe["Model"] == "Full_Model"
    ].copy()

    if len(full_model) == 0:

        raise ValueError(
            "\nFull_Model was not found in "
            "ablation_study.csv."
        )

    # -----------------------------------------------------
    # Identify sample/patch column
    # -----------------------------------------------------
    #
    # A paired t-test requires corresponding observations.
    #
    # We therefore look for a column identifying the same
    # sample/patch across models.
    #
    # -----------------------------------------------------

    possible_id_columns = [
        "Sample",
        "Sample_ID",
        "Patch",
        "Patch_ID",
        "Index",
        "sample",
        "sample_id",
        "patch",
        "patch_id",
        "index"
    ]

    id_column = None

    for column in possible_id_columns:

        if column in dataframe.columns:

            id_column = column

            break

    # -----------------------------------------------------
    # If no sample identifier exists
    # -----------------------------------------------------

    if id_column is None:

        raise ValueError(
            "\nNo sample/patch identifier was found.\n\n"
            "A paired t-test requires multiple paired "
            "observations from the same evaluation samples.\n\n"
            "Add a column such as 'Sample_ID' or 'Patch_ID' "
            "to ablation_study.csv."
        )

    print()
    print(
        "Pairing column:",
        id_column
    )

    # -----------------------------------------------------
    # Convert SSIM to numeric
    # -----------------------------------------------------

    dataframe["SSIM"] = pd.to_numeric(
        dataframe["SSIM"],
        errors="coerce"
    )

    # Remove invalid SSIM values

    dataframe = dataframe.dropna(
        subset=["SSIM"]
    )

    # -----------------------------------------------------
    # Statistical results
    # -----------------------------------------------------

    results = []

    for model_name in models:

        # Skip Full Model

        if model_name == "Full_Model":

            continue

        # -----------------------------------------------
        # Select model
        # -----------------------------------------------

        ablation_model = dataframe[
            dataframe["Model"] == model_name
        ].copy()

        # -----------------------------------------------
        # Rename SSIM columns
        # -----------------------------------------------

        full_ssim = full_model[
            [id_column, "SSIM"]
        ].rename(
            columns={
                "SSIM": "Full_Model_SSIM"
            }
        )

        ablation_ssim = ablation_model[
            [id_column, "SSIM"]
        ].rename(
            columns={
                "SSIM": "Ablation_SSIM"
            }
        )

        # -----------------------------------------------
        # Pair observations by sample/patch
        # -----------------------------------------------

        paired = pd.merge(
            full_ssim,
            ablation_ssim,
            on=id_column,
            how="inner"
        )

        # -----------------------------------------------
        # Number of paired observations
        # -----------------------------------------------

        number_of_pairs = len(
            paired
        )

        if number_of_pairs < 2:

            print()
            print(
                f"Skipping {model_name}: "
                f"only {number_of_pairs} paired observation(s)."
            )

            continue

        # -----------------------------------------------
        # Paired t-test
        # -----------------------------------------------

        statistic, p_value = ttest_rel(

            paired[
                "Full_Model_SSIM"
            ].values,

            paired[
                "Ablation_SSIM"
            ].values

        )

        # -----------------------------------------------
        # Statistical decision
        # -----------------------------------------------

        if p_value < ALPHA:

            significance = (
                "Statistically Significant"
            )

        else:

            significance = (
                "Not Statistically Significant"
            )

        # -----------------------------------------------
        # Store result
        # -----------------------------------------------

        results.append({

            "Comparison":
                f"Full_Model vs {model_name}",

            "Metric":
                "SSIM",

            "N_Pairs":
                number_of_pairs,

            "T_Statistic":
                statistic,

            "P_Value":
                p_value,

            "Alpha":
                ALPHA,

            "Significance":
                significance

        })

    # =====================================================
    # CREATE RESULTS DATAFRAME
    # =====================================================

    results = pd.DataFrame(
        results
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    results.to_csv(

        OUTPUT_FILE,

        index=False

    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 60)

    if len(results) == 0:

        print()
        print(
            "No valid paired comparisons were available."
        )

    else:

        print()

        print(
            results.to_string(
                index=False
            )
        )

    print()
    print(
        "Significance level (alpha):",
        ALPHA
    )

    print()
    print(
        "Results saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TEST COMPLETE")
    print("=" * 60)

    return results


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_significance_test()