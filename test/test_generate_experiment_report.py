"""
=========================================================
EXPERIMENT REPORT GENERATOR
=========================================================

Collects all evaluation CSV files and produces
a single summary report.

Author: Ormin Joseph
=========================================================
"""

import os
import pandas as pd


REPORT_DIR = "outputs/reports"


def main():

    print("=" * 60)
    print("EXPERIMENT REPORT GENERATOR")
    print("=" * 60)

    summary = []

    # -----------------------------------------
    # Patch Size Sensitivity
    # -----------------------------------------

    patch_file = os.path.join(
        REPORT_DIR,
        "patch_size_sensitivity.csv"
    )

    if os.path.exists(patch_file):

        df = pd.read_csv(patch_file)

        best_row = df.loc[df["SSIM"].idxmax()]

        summary.append([
            "Best Patch Size",
            str(best_row["Patch_Size"])
        ])

        summary.append([
            "Best Patch SSIM",
            round(float(best_row["SSIM"]), 4)
        ])

    # -----------------------------------------
    # Missing Data Sensitivity
    # -----------------------------------------

    missing_file = os.path.join(
        REPORT_DIR,
        "missing_data_sensitivity.csv"
    )

    if os.path.exists(missing_file):

        df = pd.read_csv(missing_file)

        best_row = df.loc[df["SSIM"].idxmax()]

        summary.append([
            "Best Missing %",
            str(best_row["Missing_Percentage"])
        ])

    # -----------------------------------------
    # Mask Robustness
    # -----------------------------------------

    mask_file = os.path.join(
        REPORT_DIR,
        "mask_robustness.csv"
    )

    if os.path.exists(mask_file):
        df = pd.read_csv(mask_file)

        best_row = df.loc[df["SSIM"].idxmax()]

        mask_column = df.columns[0]

        summary.append([
            "Best Mask",
            str(best_row[mask_column])
        ])

        summary.append([
            "Best Mask SSIM",
            round(float(best_row["SSIM"]), 4)
        ])

    # -----------------------------------------
    # Variance Head Calibration
    # -----------------------------------------

    variance_file = os.path.join(
        REPORT_DIR,
        "uncertainty_calibration.csv"
    )

    if os.path.exists(variance_file):

        df = pd.read_csv(variance_file)

        summary.append([
            "Variance Correlation",
            round(
                float(
                    df["Correlation"][0]
                ),
                4
            )
        ])

    # -----------------------------------------
    # MC Dropout Calibration
    # -----------------------------------------

    mc_file = os.path.join(
        REPORT_DIR,
        "mc_dropout_calibration.csv"
    )

    if os.path.exists(mc_file):

        df = pd.read_csv(mc_file)

        summary.append([
            "MC Dropout Correlation",
            round(
                float(
                    df["Correlation"][0]
                ),
                4
            )
        ])

    # -----------------------------------------
    # Save CSV
    # -----------------------------------------

    summary_df = pd.DataFrame(
        summary,
        columns=[
            "Metric",
            "Value"
        ]
    )

    output_csv = os.path.join(
        REPORT_DIR,
        "experiment_summary.csv"
    )

    summary_df.to_csv(
        output_csv,
        index=False
    )

    # -----------------------------------------
    # Save TXT
    # -----------------------------------------

    output_txt = os.path.join(
        REPORT_DIR,
        "experiment_summary.txt"
    )

    with open(output_txt, "w") as file:

        file.write(
            "EXPERIMENT SUMMARY\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        for metric, value in summary:

            file.write(
                f"{metric}: {value}\n"
            )

    print()
    print("Summary CSV saved to:")
    print(output_csv)

    print()
    print("Summary TXT saved to:")
    print(output_txt)


if __name__ == "__main__":
    main()