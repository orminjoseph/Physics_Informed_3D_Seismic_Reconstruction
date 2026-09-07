"""
=========================================================
Statistical Significance Test
=========================================================

Paired statistical comparison between the Full Model and
each ablation model.

The comparison is performed on paired SSIM values obtained
from the SAME validation samples.

Primary statistical test:
    Paired t-test

Multiple-comparison correction:
    Holm-Bonferroni correction

Additional reported statistic:
    Cohen's dz effect size

The analysis requires:
    1. A per-sample ablation_study.csv
    2. A common Sample_ID for every model
    3. Exactly one observation per Model/Sample_ID pair
    4. The same validation samples for every model
    5. At least two paired observations for the paired
       t-test

Expected input structure:

    Model,Sample_ID,Attention,Residual,Uncertainty,
    MAE,RMSE,PSNR,SNR,SSIM,Checkpoint

Example:

    Full_Model,0,True,True,True,...,0.91,...
    No_Attention,0,False,True,True,...,0.87,...
    Full_Model,1,True,True,True,...,0.89,...
    No_Attention,1,False,True,True,...,0.84,...

Results are saved inside the current experiment directory:

outputs/
    <EXPERIMENT_NAME>/
        reports/
            statistical_significance.csv

IMPORTANT:
    If fewer than two paired validation samples are
    available, inferential statistical testing is stopped.
    This prevents meaningless p-values from being reported.

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

FULL_MODEL_NAME = "Full_Model"

EXPECTED_ABLATION_MODELS = [
    "No_Attention",
    "No_Residual",
    "No_Uncertainty",
    "Plain_UNet"
]

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

    Sample_ID is preferred because it is the identifier
    produced by the corrected ablation-study pipeline.
    """

    preferred_columns = [
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

    for column in preferred_columns:

        if column in dataframe.columns:
            return column

    return None


def normalize_id_column(
        dataframe,
        id_column
):
    """
    Normalize the pairing identifier.

    Numeric identifiers are converted to integers where
    possible so that values such as 1 and 1.0 represent
    the same sample.

    Non-numeric identifiers are retained as strings.
    """

    dataframe = dataframe.copy()

    numeric_ids = pd.to_numeric(
        dataframe[id_column],
        errors="coerce"
    )

    if numeric_ids.notna().all():

        if np.isfinite(
            numeric_ids.to_numpy(
                dtype=np.float64
            )
        ).all():

            if np.all(
                np.equal(
                    numeric_ids.to_numpy(),
                    np.floor(
                        numeric_ids.to_numpy()
                    )
                )
            ):

                dataframe[id_column] = (
                    numeric_ids.astype(np.int64)
                )

                return dataframe

    dataframe[id_column] = (
        dataframe[id_column]
        .astype(str)
        .str.strip()
    )

    return dataframe


def validate_ssim(dataframe):
    """
    Convert SSIM values to numeric and validate them.

    SSIM must contain finite numerical values.

    The function does not impose an artificial SSIM range
    because the exact SSIM implementation used by the
    evaluation pipeline determines its theoretical range.
    """

    dataframe = dataframe.copy()

    dataframe["SSIM"] = pd.to_numeric(
        dataframe["SSIM"],
        errors="coerce"
    )

    invalid_count = (
        dataframe["SSIM"].isna().sum()
    )

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
        dataframe["SSIM"].to_numpy(
            dtype=np.float64
        )
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

    A valid paired analysis requires exactly one SSIM
    observation for every model/sample combination.
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
            "value per validation sample."
        )


def get_model_ids(
        dataframe,
        model_name,
        id_column
):
    """
    Return the unique sample IDs belonging to a model.
    """

    return set(
        dataframe.loc[
            dataframe["Model"] == model_name,
            id_column
        ].tolist()
    )


def validate_common_pairing(
        dataframe,
        models,
        id_column
):
    """
    Verify that every model was evaluated on exactly the
    same validation samples.

    This is critical.

    We do NOT allow an inner merge to silently discard
    unmatched samples because that could invalidate the
    intended paired experimental design.
    """

    full_ids = get_model_ids(
        dataframe,
        FULL_MODEL_NAME,
        id_column
    )

    if len(full_ids) == 0:

        raise ValueError(
            "\nFull_Model contains no valid sample IDs."
        )

    for model_name in models:

        model_ids = get_model_ids(
            dataframe,
            model_name,
            id_column
        )

        missing_from_model = (
            full_ids - model_ids
        )

        extra_in_model = (
            model_ids - full_ids
        )

        if missing_from_model:

            raise ValueError(
                f"\nPairing mismatch for {model_name}.\n"
                f"Samples missing from {model_name}: "
                f"{sorted(missing_from_model)}\n\n"
                "All models must be evaluated on exactly "
                "the same validation samples."
            )

        if extra_in_model:

            raise ValueError(
                f"\nPairing mismatch for {model_name}.\n"
                f"Extra samples found in {model_name}: "
                f"{sorted(extra_in_model)}\n\n"
                "All models must be evaluated on exactly "
                "the same validation samples."
            )


def calculate_cohens_dz(
        full_values,
        ablation_values
):
    """
    Calculate Cohen's dz for paired observations.

    Definition:

        dz = mean(difference) / SD(difference)

    where:

        difference =
            Full_Model_SSIM - Ablation_SSIM

    Interpretation:

        Positive dz:
            Full Model tends to have higher SSIM.

        Negative dz:
            Ablation model tends to have higher SSIM.
    """

    differences = (
        full_values - ablation_values
    )

    mean_difference = np.mean(
        differences
    )

    standard_deviation = np.std(
        differences,
        ddof=1
    )

    if not np.isfinite(
        standard_deviation
    ):

        return np.nan

    if standard_deviation == 0:

        if mean_difference == 0:
            return 0.0

        return np.inf

    return (
        mean_difference
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
        Boolean significance decisions.
    """

    p_values = np.asarray(
        p_values,
        dtype=np.float64
    )

    number_of_tests = len(
        p_values
    )

    if number_of_tests == 0:

        return (
            np.array(
                [],
                dtype=np.float64
            ),
            np.array(
                [],
                dtype=bool
            )
        )

    if not np.isfinite(
        p_values
    ).all():

        raise ValueError(
            "\nHolm correction received "
            "non-finite p-values."
        )

    order = np.argsort(
        p_values
    )

    sorted_p_values = (
        p_values[order]
    )

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=np.float64
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
        dtype=np.float64
    )

    adjusted_p_values[
        order
    ] = adjusted_sorted

    significant = (
        adjusted_p_values <= alpha
    )

    return (
        adjusted_p_values,
        significant
    )


def create_empty_results_dataframe():
    """
    Create a consistent empty result dataframe.
    """

    return pd.DataFrame(
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
            "Run the corrected ablation study first."
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
            "The ablation study must contain "
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
            "observations from the same validation "
            "samples.\n\n"
            "The corrected ablation study should contain "
            "a 'Sample_ID' column."
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

    dataframe = normalize_id_column(
        dataframe,
        id_column
    )

    # -----------------------------------------------------
    # Validate SSIM
    # -----------------------------------------------------

    dataframe = validate_ssim(
        dataframe
    )

    # -----------------------------------------------------
    # Display available models
    # -----------------------------------------------------

    models = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

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
    # Validate Full Model
    # -----------------------------------------------------

    if FULL_MODEL_NAME not in models:

        raise ValueError(
            "\nFull_Model was not found in "
            "ablation_study.csv."
        )

    # -----------------------------------------------------
    # Validate expected ablation models
    # -----------------------------------------------------

    missing_models = [
        model
        for model in EXPECTED_ABLATION_MODELS
        if model not in models
    ]

    if missing_models:

        raise ValueError(
            "\nExpected ablation model(s) missing:\n"
            f"{missing_models}\n\n"
            "The controlled ablation study should contain "
            "Full_Model plus all four ablation configurations."
        )

    # -----------------------------------------------------
    # Check for unexpected model names.
    #
    # Unexpected models are not automatically included in
    # the statistical family because doing so would change
    # the multiple-comparison correction.
    # -----------------------------------------------------

    expected_models = [
        FULL_MODEL_NAME
    ] + EXPECTED_ABLATION_MODELS

    unexpected_models = [
        model
        for model in models
        if model not in expected_models
    ]

    if unexpected_models:

        raise ValueError(
            "\nUnexpected model(s) found:\n"
            f"{unexpected_models}\n\n"
            "Remove unexpected models or explicitly update "
            "EXPECTED_ABLATION_MODELS before performing "
            "statistical testing."
        )

    # -----------------------------------------------------
    # Check duplicate model/sample pairs
    # -----------------------------------------------------

    for model_name in expected_models:

        check_duplicate_pairs(
            dataframe,
            model_name,
            id_column
        )

    # -----------------------------------------------------
    # Verify identical validation samples
    # -----------------------------------------------------

    validate_common_pairing(
        dataframe,
        EXPECTED_ABLATION_MODELS,
        id_column
    )

    # -----------------------------------------------------
    # Full Model data
    # -----------------------------------------------------

    full_model = dataframe[
        dataframe["Model"] == FULL_MODEL_NAME
    ][
        [id_column, "SSIM"]
    ].copy()

    number_of_full_samples = len(
        full_model
    )

    print()
    print(
        "Full Model validation samples:",
        number_of_full_samples
    )

    # -----------------------------------------------------
    # Minimum requirement for paired t-test
    # -----------------------------------------------------

    if number_of_full_samples < 2:

        raise ValueError(
            "\nSTATISTICAL TEST NOT PERFORMED.\n\n"
            f"The current ablation study contains only "
            f"{number_of_full_samples} paired validation "
            "sample.\n\n"
            "A paired t-test requires at least two paired "
            "observations.\n\n"
            "More importantly, with one validation sample "
            "there is no estimate of between-sample "
            "variability, so a statistical significance "
            "claim would be scientifically invalid.\n\n"
            "Increase the validation-set size and rerun the "
            "corrected ablation study before performing "
            "statistical significance testing."
        )

    # -----------------------------------------------------
    # Statistical results
    # -----------------------------------------------------

    results = []

    raw_p_values = []

    comparison_indices = []

    # -----------------------------------------------------
    # Perform Full Model versus each ablation comparison
    # -----------------------------------------------------

    for model_name in EXPECTED_ABLATION_MODELS:

        print()
        print(
            "-" * 70
        )

        print(
            "Comparison:",
            FULL_MODEL_NAME,
            "vs",
            model_name
        )

        # -------------------------------------------------
        # Select ablation model
        # -------------------------------------------------

        ablation_model = dataframe[
            dataframe["Model"] == model_name
        ][
            [id_column, "SSIM"]
        ].copy()

        # -------------------------------------------------
        # Rename SSIM columns
        # -------------------------------------------------

        full_ssim = full_model.rename(
            columns={
                "SSIM":
                    "Full_Model_SSIM"
            }
        )

        ablation_ssim = ablation_model.rename(
            columns={
                "SSIM":
                    "Ablation_SSIM"
            }
        )

        # -------------------------------------------------
        # Pair by Sample_ID.
        #
        # An outer merge is deliberately NOT used here
        # because pairing completeness was already validated.
        # -------------------------------------------------

        paired = pd.merge(
            full_ssim,
            ablation_ssim,
            on=id_column,
            how="inner",
            validate="one_to_one"
        )

        number_of_pairs = len(
            paired
        )

        # -------------------------------------------------
        # Final pairing check
        # -------------------------------------------------

        if number_of_pairs != number_of_full_samples:

            raise ValueError(
                f"\nPairing failure for {model_name}.\n"
                f"Expected {number_of_full_samples} "
                f"paired samples but found "
                f"{number_of_pairs}.\n\n"
                "The statistical analysis cannot continue."
            )

        # -------------------------------------------------
        # Convert to NumPy
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
                f"\nNon-finite Full_Model SSIM values "
                f"found for comparison with {model_name}."
            )

        if not np.isfinite(
            ablation_values
        ).all():

            raise ValueError(
                f"\nNon-finite SSIM values found for "
                f"{model_name}."
            )

        # -------------------------------------------------
        # Paired differences
        # -------------------------------------------------

        differences = (
            full_values
            - ablation_values
        )

        # -------------------------------------------------
        # Paired t-test
        # -------------------------------------------------

        statistic, p_value = ttest_rel(
            full_values,
            ablation_values
        )

        # -------------------------------------------------
        # Mean SSIM
        # -------------------------------------------------

        full_mean = np.mean(
            full_values
        )

        ablation_mean = np.mean(
            ablation_values
        )

        # -------------------------------------------------
        # Mean paired difference
        #
        # Positive:
        #     Full Model has higher SSIM.
        #
        # Negative:
        #     Ablation has higher SSIM.
        # -------------------------------------------------

        mean_difference = np.mean(
            differences
        )

        # -------------------------------------------------
        # Cohen's dz
        # -------------------------------------------------

        cohens_dz = calculate_cohens_dz(
            full_values,
            ablation_values
        )

        # -------------------------------------------------
        # Handle degenerate paired differences
        # -------------------------------------------------

        if not np.isfinite(p_value):

            print()
            print(
                "Warning:",
                "Paired t-test returned a non-finite "
                "p-value for",
                model_name
            )

        # -------------------------------------------------
        # Store preliminary result
        # -------------------------------------------------

        results.append({

            "Comparison":
                f"{FULL_MODEL_NAME} vs {model_name}",

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

    raw_p_values_array = np.asarray(
        raw_p_values,
        dtype=np.float64
    )

    # -----------------------------------------------------
    # Ensure all p-values are valid before correction.
    # -----------------------------------------------------

    if not np.isfinite(
        raw_p_values_array
    ).all():

        raise ValueError(
            "\nAt least one paired t-test produced a "
            "non-finite p-value.\n\n"
            "Holm-Bonferroni correction cannot be "
            "performed reliably."
        )

    adjusted_p_values, significant = (
        holm_correction(
            raw_p_values_array,
            alpha=ALPHA
        )
    )

    # -----------------------------------------------------
    # Add corrected p-values and conclusions
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
        # Direction
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
        "Primary metric:",
        "SSIM"
    )

    print(
        "Primary statistical test:",
        "Paired t-test"
    )

    print(
        "Multiple-comparison correction:",
        "Holm-Bonferroni"
    )

    print(
        "Effect size:",
        "Cohen's dz"
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