"""
=========================================================
Statistical Significance Test
=========================================================

Paired t-test between models

=========================================================
"""

import os

import pandas as pd

from scipy.stats import ttest_rel

from utils.config import DATASET_MODE


def run_significance_test():

    report_dir = os.path.join(

        "outputs",

        DATASET_MODE,

        "reports"

    )

    ablation_file = os.path.join(

        report_dir,

        "ablation_study.csv"

    )

    dataframe = pd.read_csv(
        ablation_file
    )

    full_model = dataframe[
        dataframe["Model"] == "Full_Model"
    ]

    results = []

    for _, row in dataframe.iterrows():

        if row["Model"] == "Full_Model":

            continue

        statistic, p_value = ttest_rel(

            [full_model["SSIM"].values[0]],

            [row["SSIM"]]

        )

        results.append({

            "Comparison":

                f"Full_Model vs {row['Model']}",

            "p_value":

                p_value

        })

    results = pd.DataFrame(results)

    results.to_csv(

        os.path.join(
            report_dir,
            "statistical_significance.csv"
        ),

        index=False

    )

    print()
    print(results)

    return results


if __name__ == "__main__":

    run_significance_test()