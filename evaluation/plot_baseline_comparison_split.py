import os
import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = "outputs/reports"

CSV_FILE = os.path.join(
    REPORT_DIR,
    "baseline_comparison.csv"
)

df = pd.read_csv(CSV_FILE)

methods = df["Method"]

# -----------------------------------
# Error Metrics
# -----------------------------------

plt.figure(figsize=(8, 5))

x = range(len(methods))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    df["MAE"],
    width=width,
    label="MAE"
)

plt.bar(
    [i + width/2 for i in x],
    df["RMSE"],
    width=width,
    label="RMSE"
)

plt.xticks(
    x,
    methods,
    rotation=15
)

plt.ylabel("Error")
plt.title("Baseline Comparison: Error Metrics")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "baseline_error_metrics.png"
    )
)

plt.close()

# -----------------------------------
# Quality Metrics
# -----------------------------------

plt.figure(figsize=(8, 5))

width = 0.25

plt.bar(
    [i - width for i in x],
    df["PSNR"],
    width=width,
    label="PSNR"
)

plt.bar(
    x,
    df["SNR"],
    width=width,
    label="SNR"
)

plt.bar(
    [i + width for i in x],
    df["SSIM"],
    width=width,
    label="SSIM"
)

plt.xticks(
    x,
    methods,
    rotation=15
)

plt.ylabel("Quality")
plt.title("Baseline Comparison: Quality Metrics")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "baseline_quality_metrics.png"
    )
)

plt.close()

print()
print("Saved:")
print("outputs/reports/baseline_error_metrics.png")
print("outputs/reports/baseline_quality_metrics.png")