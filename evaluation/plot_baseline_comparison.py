"""
Baseline Comparison Plot

Creates publication-quality comparison charts.

Author: Ormin Joseph
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_FILE = (
    "outputs/reports/"
    "baseline_comparison.csv"
)

OUTPUT_FILE = (
    "outputs/reports/"
    "baseline_metrics_comparison.png"
)


def main():

    df = pd.read_csv(CSV_FILE)

    metrics = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM"
    ]

    methods = df["Method"].tolist()

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i in range(len(methods)):

        values = [
            df.loc[i, metric]
            for metric in metrics
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=methods[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_title(
        "Baseline Comparison of Reconstruction Methods"
    )

    ax.set_ylabel("Metric Value")

    ax.legend()

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()