"""
============================================================
UNCERTAINTY METHOD COMPARISON
============================================================
"""

import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

results = [

    ["Variance_Head", -0.0915, 0.0, 0.8897],

    ["MC_Dropout", 0.2825, 0.9697, 0.0772]

]

# ----------------------------------------------------------
# Save CSV
# ----------------------------------------------------------

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

with open(
    "outputs/reports/uncertainty_comparison.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Method",
        "Correlation",
        "Spatial_Ratio",
        "Mean_Uncertainty"
    ])

    writer.writerows(results)

# ----------------------------------------------------------
# Create DataFrame
# ----------------------------------------------------------

df = pd.DataFrame(
    results,
    columns=[
        "Method",
        "Correlation",
        "Spatial_Ratio",
        "Mean_Uncertainty"
    ]
)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Method"],
    df["Correlation"]
)

plt.title(
    "Error–Uncertainty Correlation"
)

plt.ylabel(
    "Correlation"
)

plt.grid(axis="y")

plt.tight_layout()

# ----------------------------------------------------------
# Save Figure
# ----------------------------------------------------------

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

plt.savefig(
    "outputs/figures/uncertainty_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Final messages
# ----------------------------------------------------------

print()
print("CSV saved to:")
print("outputs/reports/uncertainty_comparison.csv")

print()
print("Plot saved to:")
print("outputs/figures/uncertainty_comparison.png")