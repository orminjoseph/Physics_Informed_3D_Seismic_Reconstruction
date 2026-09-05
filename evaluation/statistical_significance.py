"""
=========================================================
Statistical Significance Test
=========================================================

Paired statistical comparison between the Full Model and
each ablation model.

The comparison is performed on paired SSIM values obtained
from the SAME evaluation samples/patches.

Primary statistical test:

    Paired t-test

Multiple-comparison correction:

    Holm-Bonferroni correction

Additional reported statistic:

    Cohen's dz effect size

Results are saved inside the current experiment directory:

outputs/
    <EXPERIMENT_NAME>/
        reports/
            statistical_significance.csv

Expected ablation study structure:

    Model, Sample_ID, SSIM

Example:

    Full_Model, 0, 0.91
    Ablation_A, 0, 0.87
    Full_Model, 1, 0.89
    Ablation_A, 1, 0.84

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
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
# HELPER FUNCTIONS
# =========================================================

def identify_id_column(dataframe):
    """
    Identify the column used to pair observations across
    the Full Model and ablation models.
    """

    possible_id_columns = [
        "Sample_ID",
        "Patch_ID",
        "Sample",
        "Patch",
        "Index",
        "sample_id",
        "patch_id",
        "sample",
        "patch",
        "index"
    ]

    for column in possible_id_columns:

        if column in dataframe.columns:
            return column

    return None


def validate_ssim(dataframe):
    """
    Convert SSIM values to numeric and validate them.
    """

    dataframe = dataframe.copy()

    dataframe["SSIM"] = pd.to_numeric(
        dataframe["SSIM"],
        errors="coerce"
    )

    invalid_count = dataframe["SSIM"].isna().sum()

    if invalid_count > 0:

        print()
        print(
            "Warning:",
            invalid_count,
            "row(s) contain invalid SSIM values "
            "and will be removed."
        )

    dataframe = dataframe.dropna(
        subset=["SSIM"]
    ).copy()

    if len(dataframe) == 0:

        raise ValueError(
            "\nNo valid numeric SSIM observations remain."
        )

    if not np.isfinite(
        dataframe["SSIM"].to_numpy()
    ).all():

        raise ValueError(
            "\nSSIM contains non-finite values."
        )

    return dataframe


def check_duplicate_pairs(
        dataframe,
        model_name,
        id_column
):
    """
    Check whether a model contains duplicate sample IDs.

    A paired comparison requires one SSIM observation per
    sample/model combination.
    """

    model_data = dataframe[
        dataframe["Model"] == model_name
    ]

    duplicate_ids = model_data[
        model_data[id_column].duplicated(
            keep=False
        )
    ][id_column].unique()

    if len(duplicate_ids) > 0:

        raise ValueError(
            f"\nDuplicate {id_column} values detected "
            f"for model '{model_name}':\n"
            f"{duplicate_ids.tolist()}\n\n"
            "Each model must contain exactly one SSIM "
            "value per evaluation sample."
        )


def calculate_cohens_dz(
        full_values,
        ablation_values
):
    """
    Calculate Cohen's dz for paired observations.

    dz = mean(difference) / SD(difference)

    Difference is defined as:

        Full Model SSIM - Ablation SSIM
    """

    differences = (
        full_values - ablation_values
    )

    standard_deviation = np.std(
        differences,
        ddof=1
    )

    if standard_deviation == 0:

        if np.mean(differences) == 0:
            return 0.0

        return np.inf

    return (
        np.mean(differences)
        / standard_deviation
    )


def holm_correction(
        p_values,
        alpha=0.05
):
    """
    Perform Holm-Bonferroni multiple-comparison
    correction.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.

    alpha : float
        Significance level.

    Returns
    -------
    adjusted_p_values : numpy.ndarray
        Holm-adjusted p-values.

    significant : numpy.ndarray
        Boolean significance decisions based on
        adjusted p-values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    number_of_tests = len(
        p_values
    )

    if number_of_tests == 0:

        return (
            np.array([]),
            np.array([], dtype=bool)
        )

    order = np.argsort(
        p_values
    )

    sorted_p_values = p_values[
        order
    ]

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=float
    )

    running_max = 0.0

    for rank, p_value in enumerate(
        sorted_p_values
    ):

        adjusted_value = (
            number_of_tests - rank
        ) * p_value

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted_sorted[rank] = min(
            running_max,
            1.0
        )

    adjusted_p_values = np.empty(
        number_of_tests,
        dtype=float
    )

    adjusted_p_values[
        order
    ] = adjusted_sorted

    significant = (
        adjusted_p_values < alpha
    )

    return (
        adjusted_p_values,
        significant
    )


# =========================================================
# STATISTICAL SIGNIFICANCE TEST
# =========================================================

def run_significance_test():

    print()
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)

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

    print(
        "Alpha      :",
        ALPHA
    )

    # -----------------------------------------------------
    # Check input file
    # -----------------------------------------------------

    if not os.path.isfile(
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

    if len(dataframe) == 0:

        raise ValueError(
            "\nThe ablation study file is empty."
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
    # Identify pairing column
    # -----------------------------------------------------

    id_column = identify_id_column(
        dataframe
    )

    if id_column is None:

        raise ValueError(
            "\nNo sample/patch identifier was found.\n\n"
            "A paired t-test requires corresponding "
            "observations from the same evaluation "
            "samples.\n\n"
            "Add a column such as 'Sample_ID' or "
            "'Patch_ID' to ablation_study.csv."
        )

    print()
    print(
        "Pairing column:",
        id_column
    )

    # -----------------------------------------------------
    # Validate pairing column
    # -----------------------------------------------------

    if dataframe[id_column].isna().any():

        raise ValueError(
            f"\nThe pairing column '{id_column}' "
            "contains missing values."
        )

    # -----------------------------------------------------
    # Convert SSIM to numeric BEFORE creating subsets.
    #
    # This is important because otherwise Full_Model may
    # retain string/object SSIM values.
    # -----------------------------------------------------

    dataframe = validate_ssim(
        dataframe
    )

    # -----------------------------------------------------
    # Display available models
    # -----------------------------------------------------

    models = dataframe[
        "Model"
    ].dropna().unique().tolist()

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

    if "Full_Model" not in models:

        raise ValueError(
            "\nFull_Model was not found in "
            "ablation_study.csv."
        )

    # -----------------------------------------------------
    # Check duplicates for every model.
    #
    # Duplicate sample IDs can cause a many-to-many merge
    # and invalidate the paired analysis.
    # -----------------------------------------------------

    for model_name in models:

        check_duplicate_pairs(
            dataframe,
            model_name,
            id_column
        )

    # -----------------------------------------------------
    # Extract Full Model AFTER SSIM validation.
    # -----------------------------------------------------

    full_model = dataframe[
        dataframe["Model"] == "Full_Model"
    ][
        [id_column, "SSIM"]
    ].copy()

    if len(full_model) < 2:

        raise ValueError(
            "\nThe Full_Model contains fewer than "
            "two valid observations."
        )

    # -----------------------------------------------------
    # Statistical results
    # -----------------------------------------------------

    results = []

    raw_p_values = []

    comparison_indices = []

    for model_name in models:

        # -------------------------------------------------
        # Skip Full Model
        # -------------------------------------------------

        if model_name == "Full_Model":
            continue

        # -------------------------------------------------
        # Select ablation model
        # -------------------------------------------------

        ablation_model = dataframe[
            dataframe["Model"] == model_name
        ][
            [id_column, "SSIM"]
        ].copy()

        if len(ablation_model) < 2:

            print()
            print(
                f"Skipping {model_name}: "
                "fewer than two valid observations."
            )

            continue

        # -------------------------------------------------
        # Rename SSIM columns
        # -------------------------------------------------

        full_ssim = full_model.rename(
            columns={
                "SSIM": "Full_Model_SSIM"
            }
        )

        ablation_ssim = ablation_model.rename(
            columns={
                "SSIM": "Ablation_SSIM"
            }
        )

        # -------------------------------------------------
        # Pair observations by sample/patch
        # -------------------------------------------------

        paired = pd.merge(
            full_ssim,
            ablation_ssim,
            on=id_column,
            how="inner",
            validate="one_to_one"
        )

        # -------------------------------------------------
        # Number of paired observations
        # -------------------------------------------------

        number_of_pairs = len(
            paired
        )

        if number_of_pairs < 2:

            print()
            print(
                f"Skipping {model_name}: "
                f"only {number_of_pairs} paired "
                "observation(s)."
            )

            continue

        # -------------------------------------------------
        # Convert paired values to NumPy arrays
        # -------------------------------------------------

        full_values = paired[
            "Full_Model_SSIM"
        ].to_numpy(
            dtype=np.float64
        )

        ablation_values = paired[
            "Ablation_SSIM"
        ].to_numpy(
            dtype=np.float64
        )

        # -------------------------------------------------
        # Validate finite values
        # -------------------------------------------------

        if not np.isfinite(
            full_values
        ).all():

            raise ValueError(
                f"Non-finite Full_Model SSIM values "
                f"found for comparison with {model_name}."
            )

        if not np.isfinite(
            ablation_values
        ).all():

            raise ValueError(
                f"Non-finite ablation SSIM values "
                f"found for {model_name}."
            )

        # -------------------------------------------------
        # Paired t-test
        # -------------------------------------------------

        statistic, p_value = ttest_rel(
            full_values,
            ablation_values
        )

        # -------------------------------------------------
        # Mean SSIM values
        # -------------------------------------------------

        full_mean = np.mean(
            full_values
        )

        ablation_mean = np.mean(
            ablation_values
        )

        # -------------------------------------------------
        # Difference:
        #
        # Positive -> Full Model has higher SSIM
        # Negative -> Ablation has higher SSIM
        # -------------------------------------------------

        mean_difference = (
            full_mean
            - ablation_mean
        )

        # -------------------------------------------------
        # Effect size
        # -------------------------------------------------

        cohens_dz = calculate_cohens_dz(
            full_values,
            ablation_values
        )

        # -------------------------------------------------
        # Store preliminary result
        # -------------------------------------------------

        results.append({

            "Comparison":
                f"Full_Model vs {model_name}",

            "Metric":
                "SSIM",

            "N_Pairs":
                number_of_pairs,

            "Full_Model_Mean_SSIM":
                full_mean,

            "Ablation_Mean_SSIM":
                ablation_mean,

            "Mean_Difference":
                mean_difference,

            "T_Statistic":
                statistic,

            "Raw_P_Value":
                p_value,

            "Cohens_dz":
                cohens_dz,

            "Alpha":
                ALPHA

        })

        raw_p_values.append(
            p_value
        )

        comparison_indices.append(
            len(results) - 1
        )

    # =====================================================
    # MULTIPLE-COMPARISON CORRECTION
    # =====================================================

    adjusted_p_values, significant = (
        holm_correction(
            raw_p_values,
            alpha=ALPHA
        )
    )

    # -----------------------------------------------------
    # Add adjusted p-values and conclusions
    # -----------------------------------------------------

    for index, adjusted_p, is_significant in zip(
        comparison_indices,
        adjusted_p_values,
        significant
    ):

        results[index][
            "Holm_Adjusted_P_Value"
        ] = adjusted_p

        results[index][
            "Significance"
        ] = (
            "Statistically Significant"
            if is_significant
            else "Not Statistically Significant"
        )

        # -------------------------------------------------
        # Direction of difference
        # -------------------------------------------------

        difference = results[index][
            "Mean_Difference"
        ]

        if difference > 0:

            results[index][
                "Direction"
            ] = "Full Model Higher SSIM"

        elif difference < 0:

            results[index][
                "Direction"
            ] = "Ablation Higher SSIM"

        else:

            results[index][
                "Direction"
            ] = "Equal Mean SSIM"

    # =====================================================
    # CREATE RESULTS DATAFRAME
    # =====================================================

    results = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Handle case where no comparisons were possible.
    # -----------------------------------------------------

    if len(results) == 0:

        print()
        print(
            "No valid paired comparisons were available."
        )

        results = pd.DataFrame(
            columns=[
                "Comparison",
                "Metric",
                "N_Pairs",
                "Full_Model_Mean_SSIM",
                "Ablation_Mean_SSIM",
                "Mean_Difference",
                "T_Statistic",
                "Raw_P_Value",
                "Holm_Adjusted_P_Value",
                "Cohens_dz",
                "Alpha",
                "Direction",
                "Significance"
            ]
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
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 70)

    if len(results) == 0:

        print()
        print(
            "No valid paired comparisons were available."
        )

    else:

        print()

        display_columns = [
            "Comparison",
            "N_Pairs",
            "Full_Model_Mean_SSIM",
            "Ablation_Mean_SSIM",
            "Mean_Difference",
            "T_Statistic",
            "Raw_P_Value",
            "Holm_Adjusted_P_Value",
            "Cohens_dz",
            "Significance"
        ]

        print(
            results[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Significance level (alpha):",
        ALPHA
    )

    print(
        "Multiple-comparison correction:",
        "Holm-Bonferroni"
    )

    print()
    print(
        "Results saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TEST COMPLETE")
    print("=" * 70)

    return results


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_significance_test()