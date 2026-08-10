"""
Uncertainty Calibration Summary

Computes correlation between predictive uncertainty
and reconstruction quality metrics.

Author: Ormin Joseph
"""

import os
import pandas as pd

REPORT_FILE = (
    "outputs/reports/"
    "uncertainty_evaluation.csv"
)

OUTPUT_FILE = (
    "outputs/reports/"
    "uncertainty_calibration_summary.csv"
)


def main():

    print()
    print("=" * 60)
    print("UNCERTAINTY CALIBRATION SUMMARY")
    print("=" * 60)

    df = pd.read_csv(REPORT_FILE)

    correlations = []

    metrics = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"
    ]

    for metric in metrics:

        corr = df["Mean_Uncertainty"].corr(
            df[metric]
        )

        correlations.append(
            [metric, corr]
        )

        print(
            f"{metric:5s} : {corr:.4f}"
        )

    result_df = pd.DataFrame(
        correlations,
        columns=[
            "Metric",
            "Correlation"
        ]
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()