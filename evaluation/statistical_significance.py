"""
=========================================================
Statistical Significance Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Perform paired statistical comparisons between the trained
Full Model and each controlled ablation configuration.

Experimental comparisons
-------------------------
1. Full_Model vs No_Attention
2. Full_Model vs No_Residual
3. Full_Model vs No_Uncertainty
4. Full_Model vs Plain_UNet

Primary metric
--------------
SSIM

Primary statistical test
------------------------
Paired t-test

Multiple-comparison correction
------------------------------
Holm-Bonferroni correction

Effect size
-----------
Cohen's dz

Experimental requirement
------------------------
Every model must be evaluated on exactly the SAME validation
samples.

The ablation study must therefore contain:

    Model
    Sample_ID
    SSIM

with exactly one observation for every:

    Model / Sample_ID

combination.

Example
-------
    Model,Sample_ID,SSIM

    Full_Model,0,0.91
    No_Attention,0,0.87
    No_Residual,0,0.89
    No_Uncertainty,0,0.90
    Plain_UNet,0,0.88

    Full_Model,1,0.89
    No_Attention,1,0.84
    No_Residual,1,0.87
    No_Uncertainty,1,0.88
    Plain_UNet,1,0.86

Important
---------
The paired design compares each model's SSIM on the SAME
validation sample.

Therefore, the statistical comparison is based on paired
differences:

    Difference =
        Full_Model_SSIM - Ablation_SSIM

Cohen's dz is:

    dz =
        mean(Difference) /
        standard_deviation(Difference)

Interpretation of direction
---------------------------
Positive mean difference:
    Full Model has higher SSIM for the paired observations.

Negative mean difference:
    Ablation model has higher SSIM for the paired observations.

This script reports the direction descriptively and does not
rank or declare an overall "best" model.

Multiple-comparison family
---------------------------
Exactly four planned comparisons are tested:

    Full_Model vs No_Attention
    Full_Model vs No_Residual
    Full_Model vs No_Uncertainty
    Full_Model vs Plain_UNet

Holm-Bonferroni correction is applied across these four
planned comparisons.

Output
------
REPORT_DIR/
    statistical_significance.csv
    statistical_significance_metadata.json

Author: Ormin Joseph
=========================================================
"""

import json
import os

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
    STATISTICAL_ALPHA,
    STATISTICAL_METRIC,
    STATISTICAL_TEST,
    MULTIPLE_COMPARISON_CORRECTION,
    EFFECT_SIZE,
)


# =========================================================
# EXPERIMENTAL CONFIGURATION
# =========================================================

FULL_MODEL_NAME = "Full_Model"


EXPECTED_ABLATION_MODELS = (
    "No_Attention",
    "No_Residual",
    "No_Uncertainty",
    "Plain_UNet",
)


ABLATION_FILE = os.path.join(
    REPORT_DIR,
    "ablation_study.csv",
)


OUTPUT_FILE = os.path.join(
    REPORT_DIR,
    "statistical_significance.csv",
)


METADATA_FILE = os.path.join(
    REPORT_DIR,
    "statistical_significance_metadata.json",
)


# =========================================================
# VALIDATION
# =========================================================

def validate_configuration():
    """
    Validate the statistical-analysis configuration.
    """

    if not 0.0 < STATISTICAL_ALPHA < 1.0:

        raise ValueError(
            "STATISTICAL_ALPHA must be between 0 and 1. "
            f"Received: {STATISTICAL_ALPHA}"
        )

    if STATISTICAL_METRIC != "SSIM":

        raise ValueError(
            "The current statistical-significance module "
            "is configured specifically for SSIM."
        )

    if STATISTICAL_TEST != "paired_t_test":

        raise ValueError(
            "The current implementation requires "
            "STATISTICAL_TEST='paired_t_test'."
        )

    if (
        MULTIPLE_COMPARISON_CORRECTION
        != "Holm-Bonferroni"
    ):

        raise ValueError(
            "The current implementation requires "
            "Holm-Bonferroni correction."
        )

    if EFFECT_SIZE != "Cohen_dz":

        raise ValueError(
            "The current implementation requires "
            "Cohen_dz as the effect-size measure."
        )


# =========================================================
# IDENTIFY SAMPLE ID
# =========================================================

def identify_sample_id_column(
    dataframe
):
    """
    Identify the validation-sample identifier.

    The final ablation-study design should contain
    'Sample_ID'.

    Legacy alternatives are accepted only to make the
    analysis robust to older result files.
    """

    preferred_columns = (
        "Sample_ID",
        "Patch_ID",
        "Sample",
        "Patch",
        "Index",
        "sample_id",
        "patch_id",
        "sample",
        "patch",
        "index",
    )

    for column in preferred_columns:

        if column in dataframe.columns:

            return column

    return None


# =========================================================
# NORMALIZE SAMPLE ID
# =========================================================

def normalize_sample_id(
    dataframe,
    id_column
):
    """
    Normalize Sample_ID values.

    Integer-like numerical IDs are represented as integers.

    Non-numeric identifiers are retained as cleaned strings.
    """

    dataframe = dataframe.copy()

    numeric_ids = pd.to_numeric(
        dataframe[id_column],
        errors="coerce",
    )

    if numeric_ids.notna().all():

        numeric_array = numeric_ids.to_numpy(
            dtype=np.float64
        )

        if np.isfinite(
            numeric_array
        ).all():

            integer_like = np.equal(
                numeric_array,
                np.floor(
                    numeric_array
                ),
            )

            if integer_like.all():

                dataframe[id_column] = (
                    numeric_ids.astype(
                        np.int64
                    )
                )

                return dataframe

    dataframe[id_column] = (
        dataframe[id_column]
        .astype(str)
        .str.strip()
    )

    return dataframe


# =========================================================
# VALIDATE SSIM
# =========================================================

def validate_ssim(
    dataframe
):
    """
    Validate SSIM observations.

    Invalid or missing SSIM values are rejected rather than
    silently removed because paired statistical analysis
    requires complete observations for every comparison.
    """

    dataframe = dataframe.copy()

    dataframe["SSIM"] = pd.to_numeric(
        dataframe["SSIM"],
        errors="coerce",
    )

    invalid_mask = (
        dataframe["SSIM"].isna()
        |
        ~np.isfinite(
            dataframe["SSIM"].to_numpy(
                dtype=np.float64
            )
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        raise ValueError(
            "\nInvalid SSIM values detected.\n"
            f"Invalid rows: {invalid_count}\n\n"
            "The paired statistical analysis requires "
            "a valid SSIM observation for every model/"
            "Sample_ID pair.\n"
            "Do not silently remove incomplete observations."
        )

    return dataframe


# =========================================================
# VALIDATE MODEL NAMES
# =========================================================

def validate_model_names(
    dataframe
):
    """
    Verify that the ablation study contains exactly the
    planned statistical-analysis model family.
    """

    models = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    expected_models = [
        FULL_MODEL_NAME,
        *EXPECTED_ABLATION_MODELS,
    ]

    missing_models = [
        model
        for model in expected_models
        if model not in models
    ]

    if missing_models:

        raise ValueError(
            "\nRequired model(s) missing:\n"
            f"{missing_models}\n\n"
            "The statistical analysis requires Full_Model "
            "and all four planned ablation configurations."
        )

    unexpected_models = [
        model
        for model in models
        if model not in expected_models
    ]

    if unexpected_models:

        raise ValueError(
            "\nUnexpected model(s) detected:\n"
            f"{unexpected_models}\n\n"
            "The current statistical-analysis family "
            "contains exactly five models.\n"
            "Remove unexpected models or explicitly update "
            "the planned comparison family."
        )

    return expected_models


# =========================================================
# DUPLICATE CHECK
# =========================================================

def check_duplicate_pairs(
    dataframe,
    model_name,
    id_column
):
    """
    Verify exactly one observation per model/sample pair.
    """

    model_data = dataframe[
        dataframe["Model"] == model_name
    ]

    duplicate_mask = (
        model_data[id_column]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicate_ids = (
            model_data.loc[
                duplicate_mask,
                id_column
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"\nDuplicate observations detected for "
            f"model '{model_name}'.\n\n"
            f"{id_column} values:\n"
            f"{duplicate_ids}\n\n"
            "Each model must contain exactly one SSIM "
            "observation per validation sample."
        )


# =========================================================
# MODEL SAMPLE IDS
# =========================================================

def get_model_ids(
    dataframe,
    model_name,
    id_column
):
    """
    Return the validation sample IDs for one model.
    """

    return set(
        dataframe.loc[
            dataframe["Model"] == model_name,
            id_column,
        ].tolist()
    )


# =========================================================
# COMMON VALIDATION-SAMPLE CHECK
# =========================================================

def validate_common_validation_samples(
    dataframe,
    models,
    id_column
):
    """
    Verify that all models were evaluated on exactly the
    same validation samples.

    No silent inner-join loss is permitted.
    """

    full_ids = get_model_ids(
        dataframe,
        FULL_MODEL_NAME,
        id_column,
    )

    if not full_ids:

        raise ValueError(
            "\nFull_Model contains no validation samples."
        )

    for model_name in models:

        model_ids = get_model_ids(
            dataframe,
            model_name,
            id_column,
        )

        missing_ids = (
            full_ids - model_ids
        )

        extra_ids = (
            model_ids - full_ids
        )

        if missing_ids:

            raise ValueError(
                f"\nValidation-sample mismatch for "
                f"{model_name}.\n\n"
                f"Missing Sample_ID values:\n"
                f"{sorted(missing_ids)}\n\n"
                "All ablation configurations must be "
                "evaluated on exactly the same validation "
                "samples."
            )

        if extra_ids:

            raise ValueError(
                f"\nValidation-sample mismatch for "
                f"{model_name}.\n\n"
                f"Extra Sample_ID values:\n"
                f"{sorted(extra_ids)}\n\n"
                "All ablation configurations must be "
                "evaluated on exactly the same validation "
                "samples."
            )


# =========================================================
# COHEN'S DZ
# =========================================================

def calculate_cohens_dz(
    full_values,
    ablation_values
):
    """
    Calculate Cohen's dz for paired observations.

    dz = mean(difference) / SD(difference)

    Difference:

        Full_Model_SSIM - Ablation_SSIM
    """

    differences = (
        full_values
        -
        ablation_values
    )

    mean_difference = float(
        np.mean(
            differences
        )
    )

    standard_deviation = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    # -----------------------------------------------------
    # Zero-variance paired differences
    # -----------------------------------------------------

    if standard_deviation == 0.0:

        if mean_difference == 0.0:

            return 0.0

        return (
            np.inf
            if mean_difference > 0
            else -np.inf
        )

    return (
        mean_difference
        /
        standard_deviation
    )


# =========================================================
# PAIRED T-TEST
# =========================================================

def run_paired_t_test(
    full_values,
    ablation_values
):
    """
    Perform a paired t-test.

    The degenerate zero-variance case is handled explicitly
    because scipy may return NaN when all paired differences
    are identical.
    """

    differences = (
        full_values
        -
        ablation_values
    )

    difference_mean = float(
        np.mean(
            differences
        )
    )

    difference_std = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    # -----------------------------------------------------
    # All differences are exactly zero.
    # -----------------------------------------------------

    if difference_std == 0.0:

        if difference_mean == 0.0:

            return (
                0.0,
                1.0,
            )

        # If every paired difference is the same nonzero
        # value, the t-statistic tends to ±infinity and
        # the two-sided p-value tends to zero.
        return (
            np.inf
            if difference_mean > 0
            else -np.inf,
            0.0,
        )

    statistic, p_value = ttest_rel(
        full_values,
        ablation_values,
    )

    statistic = float(
        statistic
    )

    p_value = float(
        p_value
    )

    if not np.isfinite(
        statistic
    ):

        raise ValueError(
            "The paired t-test returned a non-finite "
            "test statistic."
        )

    if not np.isfinite(
        p_value
    ):

        raise ValueError(
            "The paired t-test returned a non-finite "
            "p-value."
        )

    return (
        statistic,
        p_value,
    )


# =========================================================
# HOLM-BONFERRONI CORRECTION
# =========================================================

def holm_bonferroni(
    p_values,
    alpha
):
    """
    Apply Holm-Bonferroni correction.

    Parameters
    ----------
    p_values:
        Raw p-values.

    alpha:
        Family-wise significance level.

    Returns
    -------
    adjusted_p_values:
        Holm-adjusted p-values.

    significant:
        Boolean significance decisions.
    """

    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    number_of_tests = len(
        p_values
    )

    if number_of_tests == 0:

        return (
            np.array(
                [],
                dtype=np.float64,
            ),
            np.array(
                [],
                dtype=bool,
            ),
        )

    if not np.isfinite(
        p_values
    ).all():

        raise ValueError(
            "Holm-Bonferroni received "
            "non-finite p-values."
        )

    order = np.argsort(
        p_values,
        kind="stable",
    )

    sorted_p_values = (
        p_values[
            order
        ]
    )

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=np.float64,
    )

    running_max = 0.0

    for rank, p_value in enumerate(
        sorted_p_values
    ):

        multiplier = (
            number_of_tests
            -
            rank
        )

        adjusted_value = (
            multiplier
            *
            p_value
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted_sorted[
            rank
        ] = min(
            running_max,
            1.0,
        )

    adjusted_p_values = np.empty(
        number_of_tests,
        dtype=np.float64,
    )

    adjusted_p_values[
        order
    ] = adjusted_sorted

    significant = (
        adjusted_p_values
        <=
        alpha
    )

    return (
        adjusted_p_values,
        significant,
    )


# =========================================================
# CREATE EMPTY RESULTS
# =========================================================

def create_empty_results_dataframe():
    """
    Return a consistent empty results structure.
    """

    return pd.DataFrame(
        columns=[
            "Experiment",
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
            "Significance",
        ]
    )


# =========================================================
# METADATA
# =========================================================

def save_metadata(
    number_of_pairs,
    number_of_comparisons,
    id_column
):
    """
    Save the statistical-analysis metadata.
    """

    metadata = {
        "experiment":
            EXPERIMENT_NAME,

        "input_file":
            ABLATION_FILE,

        "output_file":
            OUTPUT_FILE,

        "metric":
            STATISTICAL_METRIC,

        "statistical_test":
            STATISTICAL_TEST,

        "multiple_comparison_correction":
            MULTIPLE_COMPARISON_CORRECTION,

        "effect_size":
            EFFECT_SIZE,

        "alpha":
            STATISTICAL_ALPHA,

        "pairing_column":
            id_column,

        "number_of_pairs":
            number_of_pairs,

        "number_of_planned_comparisons":
            number_of_comparisons,

        "full_model":
            FULL_MODEL_NAME,

        "ablation_models":
            list(
                EXPECTED_ABLATION_MODELS
            ),

        "pairing_design":
            "Same validation Sample_ID for every model",

        "analysis_type":
            "Paired comparison of SSIM values",

        "effect_size_definition":
            "Mean paired SSIM difference divided by "
            "the sample standard deviation of paired "
            "differences",

        "direction_definition":
            "Full_Model_SSIM minus Ablation_SSIM",
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# =========================================================
# MAIN ANALYSIS
# =========================================================

def run_significance_test():

    print()
    print("=" * 70)
    print(
        "STATISTICAL SIGNIFICANCE TEST"
    )
    print("=" * 70)

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME,
    )

    print(
        "Input file :",
        ABLATION_FILE,
    )

    print(
        "Output file:",
        OUTPUT_FILE,
    )

    print(
        "Metric     :",
        STATISTICAL_METRIC,
    )

    print(
        "Test       :",
        STATISTICAL_TEST,
    )

    print(
        "Correction :",
        MULTIPLE_COMPARISON_CORRECTION,
    )

    print(
        "Effect size:",
        EFFECT_SIZE,
    )

    print(
        "Alpha      :",
        STATISTICAL_ALPHA,
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Validate configuration
    # -----------------------------------------------------

    validate_configuration()

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
        len(dataframe),
    )

    if dataframe.empty:

        raise ValueError(
            "\nThe ablation-study CSV is empty."
        )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = (
        "Model",
        "SSIM",
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nRequired column(s) missing:\n"
            f"{missing_columns}\n\n"
            "The ablation study must contain "
            "'Model' and 'SSIM'."
        )

    # -----------------------------------------------------
    # Identify Sample_ID
    # -----------------------------------------------------

    id_column = identify_sample_id_column(
        dataframe
    )

    if id_column is None:

        raise ValueError(
            "\nNo validation-sample identifier was found.\n\n"
            "The corrected ablation study must contain "
            "'Sample_ID' so that corresponding validation "
            "samples can be paired."
        )

    print()
    print(
        "Pairing column:",
        id_column,
    )

    # -----------------------------------------------------
    # Missing Sample_ID values
    # -----------------------------------------------------

    if dataframe[
        id_column
    ].isna().any():

        raise ValueError(
            f"\nThe pairing column '{id_column}' "
            "contains missing values."
        )

    dataframe = normalize_sample_id(
        dataframe,
        id_column,
    )

    # -----------------------------------------------------
    # Validate SSIM
    # -----------------------------------------------------

    dataframe = validate_ssim(
        dataframe
    )

    # -----------------------------------------------------
    # Normalize model names
    # -----------------------------------------------------

    dataframe["Model"] = (
        dataframe["Model"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Validate model family
    # -----------------------------------------------------

    expected_models = (
        validate_model_names(
            dataframe
        )
    )

    print()
    print(
        "Planned statistical model family:"
    )

    for model_name in expected_models:

        print(
            f"  - {model_name}"
        )

    # -----------------------------------------------------
    # Duplicate model/sample checks
    # -----------------------------------------------------

    for model_name in expected_models:

        check_duplicate_pairs(
            dataframe,
            model_name,
            id_column,
        )

    # -----------------------------------------------------
    # Verify common validation samples
    # -----------------------------------------------------

    validate_common_validation_samples(
        dataframe,
        expected_models,
        id_column,
    )

    # -----------------------------------------------------
    # Full Model validation IDs
    # -----------------------------------------------------

    full_ids = get_model_ids(
        dataframe,
        FULL_MODEL_NAME,
        id_column,
    )

    number_of_pairs = len(
        full_ids
    )

    print()
    print(
        "Common validation samples:",
        number_of_pairs,
    )

    # -----------------------------------------------------
    # Minimum sample requirement
    # -----------------------------------------------------

    if number_of_pairs < 2:

        raise ValueError(
            "\nSTATISTICAL TEST NOT PERFORMED.\n\n"
            f"Only {number_of_pairs} paired validation "
            "sample(s) are available.\n\n"
            "A paired t-test requires at least two paired "
            "observations, and a meaningful estimate of "
            "between-sample variability requires more than "
            "one observation.\n\n"
            "Increase the validation-set size and rerun "
            "the ablation study."
        )

    # =====================================================
    # PREPARE FULL MODEL DATA
    # =====================================================

    full_model = dataframe[
        dataframe["Model"]
        ==
        FULL_MODEL_NAME
    ][
        [id_column, "SSIM"]
    ].copy()

    full_model = full_model.rename(
        columns={
            "SSIM":
                "Full_Model_SSIM"
        }
    )

    # =====================================================
    # PERFORM FOUR PLANNED COMPARISONS
    # =====================================================

    preliminary_results = []

    raw_p_values = []

    for model_name in (
        EXPECTED_ABLATION_MODELS
    ):

        print()
        print(
            "-" * 70
        )

        print(
            "Comparison:",
            FULL_MODEL_NAME,
            "vs",
            model_name,
        )

        # -------------------------------------------------
        # Select ablation model
        # -------------------------------------------------

        ablation_model = dataframe[
            dataframe["Model"]
            ==
            model_name
        ][
            [id_column, "SSIM"]
        ].copy()

        ablation_model = (
            ablation_model.rename(
                columns={
                    "SSIM":
                        "Ablation_SSIM"
                }
            )
        )

        # -------------------------------------------------
        # Pair observations
        # -------------------------------------------------

        paired = pd.merge(
            full_model,
            ablation_model,
            on=id_column,
            how="inner",
            validate="one_to_one",
        )

        if len(paired) != number_of_pairs:

            raise ValueError(
                f"\nPairing failure for {model_name}.\n"
                f"Expected {number_of_pairs} pairs.\n"
                f"Found {len(paired)} pairs.\n\n"
                "The statistical analysis cannot continue."
            )

        # -------------------------------------------------
        # Sort by Sample_ID
        #
        # This is not mathematically necessary after the
        # merge, but makes the paired data explicit and
        # reproducible.
        # -------------------------------------------------

        paired = paired.sort_values(
            by=id_column
        ).reset_index(
            drop=True
        )

        # -------------------------------------------------
        # Convert SSIM values
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
        # Final finite-value check
        # -------------------------------------------------

        if not np.isfinite(
            full_values
        ).all():

            raise ValueError(
                f"\nNon-finite Full_Model SSIM values "
                f"found for {model_name}."
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
            -
            ablation_values
        )

        mean_difference = float(
            np.mean(
                differences
            )
        )

        # -------------------------------------------------
        # Paired t-test
        # -------------------------------------------------

        (
            t_statistic,
            p_value
        ) = run_paired_t_test(
            full_values,
            ablation_values,
        )

        # -------------------------------------------------
        # Cohen's dz
        # -------------------------------------------------

        cohens_dz = (
            calculate_cohens_dz(
                full_values,
                ablation_values,
            )
        )

        # -------------------------------------------------
        # Means
        # -------------------------------------------------

        full_mean = float(
            np.mean(
                full_values
            )
        )

        ablation_mean = float(
            np.mean(
                ablation_values
            )
        )

        # -------------------------------------------------
        # Direction
        # -------------------------------------------------

        if mean_difference > 0:

            direction = (
                "Full Model Higher SSIM"
            )

        elif mean_difference < 0:

            direction = (
                "Ablation Higher SSIM"
            )

        else:

            direction = (
                "Equal Mean SSIM"
            )

        preliminary_results.append(
            {
                "Experiment":
                    EXPERIMENT_NAME,

                "Comparison":
                    (
                        f"{FULL_MODEL_NAME} "
                        f"vs {model_name}"
                    ),

                "Metric":
                    STATISTICAL_METRIC,

                "N_Pairs":
                    number_of_pairs,

                "Full_Model_Mean_SSIM":
                    full_mean,

                "Ablation_Mean_SSIM":
                    ablation_mean,

                "Mean_Difference":
                    mean_difference,

                "T_Statistic":
                    t_statistic,

                "Raw_P_Value":
                    p_value,

                "Cohens_dz":
                    cohens_dz,

                "Alpha":
                    STATISTICAL_ALPHA,

                "Direction":
                    direction,
            }
        )

        raw_p_values.append(
            p_value
        )

        print(
            f"N pairs       : {number_of_pairs}"
        )

        print(
            f"Full mean SSIM: {full_mean:.6f}"
        )

        print(
            f"Ablation mean : {ablation_mean:.6f}"
        )

        print(
            f"Mean difference: "
            f"{mean_difference:.6f}"
        )

        print(
            f"t-statistic   : "
            f"{t_statistic:.6f}"
        )

        print(
            f"Raw p-value   : "
            f"{p_value:.6e}"
        )

        print(
            f"Cohen's dz   : "
            f"{cohens_dz:.6f}"
        )

    # =====================================================
    # HOLM-BONFERRONI CORRECTION
    # =====================================================

    raw_p_values = np.asarray(
        raw_p_values,
        dtype=np.float64,
    )

    if not np.isfinite(
        raw_p_values
    ).all():

        raise ValueError(
            "\nAt least one paired t-test produced "
            "a non-finite p-value.\n\n"
            "Holm-Bonferroni correction cannot be applied."
        )

    (
        adjusted_p_values,
        significant
    ) = holm_bonferroni(
        raw_p_values,
        STATISTICAL_ALPHA,
    )

    # -----------------------------------------------------
    # Add corrected results
    # -----------------------------------------------------

    for index in range(
        len(
            preliminary_results
        )
    ):

        preliminary_results[index][
            "Holm_Adjusted_P_Value"
        ] = float(
            adjusted_p_values[index]
        )

        preliminary_results[index][
            "Significance"
        ] = (
            "Statistically Significant"
            if significant[index]
            else
            "Not Statistically Significant"
        )

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results = pd.DataFrame(
        preliminary_results
    )

    # -----------------------------------------------------
    # Column order
    # -----------------------------------------------------

    results = results[
        [
            "Experiment",
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
            "Significance",
        ]
    ]

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    save_metadata(
        number_of_pairs=
            number_of_pairs,

        number_of_comparisons=
            len(
                EXPECTED_ABLATION_MODELS
            ),

        id_column=
            id_column,
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "STATISTICAL SIGNIFICANCE RESULTS"
    )
    print("=" * 70)

    print()

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "Significance level:",
        STATISTICAL_ALPHA,
    )

    print(
        "Metric:",
        STATISTICAL_METRIC,
    )

    print(
        "Test:",
        STATISTICAL_TEST,
    )

    print(
        "Correction:",
        MULTIPLE_COMPARISON_CORRECTION,
    )

    print(
        "Effect size:",
        EFFECT_SIZE,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Metadata saved:"
    )

    print(
        METADATA_FILE
    )

    print()
    print("=" * 70)
    print(
        "STATISTICAL SIGNIFICANCE TEST COMPLETE"
    )
    print("=" * 70)

    return results


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_significance_test()