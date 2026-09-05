"""
=========================================================
Uncertainty CSV Diagnostic
=========================================================

Checks whether uncertainty_evaluation.csv is suitable
for uncertainty-vs-reconstruction-error correlation.

Checks:
    1. File existence
    2. Column names
    3. Number of rows
    4. Missing values
    5. Non-numeric values
    6. Infinite values
    7. Duplicate sample IDs (if available)
    8. Variation in uncertainty
    9. Variation in MAE
   10. Pearson correlation feasibility

Author: Ormin Joseph
=========================================================
"""

import os
import numpy as np
import pandas as pd

from utils.config import REPORT_DIR


# ---------------------------------------------------------
# 1. File location
# ---------------------------------------------------------

CSV_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_evaluation.csv"
)


print("=" * 60)
print("UNCERTAINTY CSV DIAGNOSTIC")
print("=" * 60)

print()
print(f"Expected file:")
print(CSV_FILE)


# ---------------------------------------------------------
# 2. Check whether the file exists
# ---------------------------------------------------------

if not os.path.exists(CSV_FILE):

    print()
    print("STATUS: FILE NOT FOUND")
    print()
    print("The expected CSV does not exist:")
    print(CSV_FILE)

    print()
    print("Available CSV files in the report directory:")

    if os.path.exists(REPORT_DIR):

        csv_files = [
            file
            for file in os.listdir(REPORT_DIR)
            if file.lower().endswith(".csv")
        ]

        if csv_files:

            for file in csv_files:
                print(f"  - {file}")

        else:
            print("  No CSV files found.")

    else:
        print("  Report directory does not exist.")

    raise SystemExit


# ---------------------------------------------------------
# 3. Load CSV
# ---------------------------------------------------------

try:

    df = pd.read_csv(CSV_FILE)

except Exception as error:

    print()
    print("STATUS: FAILED TO READ CSV")
    print()
    print(error)

    raise SystemExit


print()
print("STATUS: CSV LOADED SUCCESSFULLY")


# ---------------------------------------------------------
# 4. Basic information
# ---------------------------------------------------------

print()
print("-" * 60)
print("BASIC INFORMATION")
print("-" * 60)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print()
print("Columns:")

for column in df.columns:

    print(f"  - {column}")


# ---------------------------------------------------------
# 5. Display first rows
# ---------------------------------------------------------

print()
print("-" * 60)
print("FIRST FIVE ROWS")
print("-" * 60)

print(df.head())


# ---------------------------------------------------------
# 6. Required columns
# ---------------------------------------------------------

required_columns = [
    "Mean_Uncertainty",
    "MAE"
]

print()
print("-" * 60)
print("REQUIRED COLUMNS")
print("-" * 60)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("STATUS: FAILED")

    print()
    print("Missing columns:")

    for column in missing_columns:
        print(f"  - {column}")

    print()
    print("The correlation analysis cannot proceed.")

    raise SystemExit

else:

    print("STATUS: PASSED")
    print("Required columns are present.")


# ---------------------------------------------------------
# 7. Check data types
# ---------------------------------------------------------

print()
print("-" * 60)
print("DATA TYPES")
print("-" * 60)

print(df[required_columns].dtypes)


# ---------------------------------------------------------
# 8. Convert required columns to numeric
# ---------------------------------------------------------

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# 9. Check missing values
# ---------------------------------------------------------

print()
print("-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing_counts = df[required_columns].isna().sum()

for column, count in missing_counts.items():

    print(
        f"{column}: {count}"
    )


if missing_counts.sum() > 0:

    print()
    print("WARNING:")
    print("Some required values are missing.")

else:

    print()
    print("STATUS: PASSED")
    print("No missing values in required columns.")


# ---------------------------------------------------------
# 10. Check infinite values
# ---------------------------------------------------------

print()
print("-" * 60)
print("INFINITE VALUES")
print("-" * 60)

infinite_counts = {}

for column in required_columns:

    infinite_counts[column] = np.isinf(
        df[column].to_numpy()
    ).sum()

    print(
        f"{column}: {infinite_counts[column]}"
    )


if sum(infinite_counts.values()) > 0:

    print()
    print("WARNING:")
    print("Infinite values were detected.")

else:

    print()
    print("STATUS: PASSED")
    print("No infinite values detected.")


# ---------------------------------------------------------
# 11. Remove invalid rows temporarily
# ---------------------------------------------------------
# This does NOT modify the original CSV.
# It is only used for diagnostic calculations.

valid_df = df[
    np.isfinite(df["Mean_Uncertainty"])
    &
    np.isfinite(df["MAE"])
].copy()


print()
print("-" * 60)
print("VALID DATA")
print("-" * 60)

print(
    f"Valid paired rows: {len(valid_df)}"
)


# ---------------------------------------------------------
# 12. Check number of paired observations
# ---------------------------------------------------------

print()
print("-" * 60)
print("CORRELATION SAMPLE SIZE")
print("-" * 60)

if len(valid_df) < 2:

    print("STATUS: FAILED")
    print(
        "At least 2 valid paired observations are required."
    )

    raise SystemExit

else:

    print("STATUS: PASSED")
    print(
        f"{len(valid_df)} valid paired observations available."
    )


# ---------------------------------------------------------
# 13. Descriptive statistics
# ---------------------------------------------------------

print()
print("-" * 60)
print("DESCRIPTIVE STATISTICS")
print("-" * 60)

print()

print("Mean Uncertainty:")
print(
    valid_df["Mean_Uncertainty"].describe()
)

print()

print("MAE:")
print(
    valid_df["MAE"].describe()
)


# ---------------------------------------------------------
# 14. Check variation
# ---------------------------------------------------------

uncertainty_unique = (
    valid_df["Mean_Uncertainty"]
    .nunique()
)

mae_unique = (
    valid_df["MAE"]
    .nunique()
)

print()
print("-" * 60)
print("VARIATION CHECK")
print("-" * 60)

print(
    f"Unique Mean_Uncertainty values: {uncertainty_unique}"
)

print(
    f"Unique MAE values              : {mae_unique}"
)


if uncertainty_unique < 2:

    print()
    print(
        "WARNING: Mean_Uncertainty has no variation."
    )

if mae_unique < 2:

    print()
    print(
        "WARNING: MAE has no variation."
    )


# ---------------------------------------------------------
# 15. Check duplicate Sample_ID if available
# ---------------------------------------------------------

print()
print("-" * 60)
print("SAMPLE ID CHECK")
print("-" * 60)

possible_id_columns = [
    "Sample_ID",
    "sample_id",
    "ID",
    "id"
]

id_column = None

for column in possible_id_columns:

    if column in df.columns:

        id_column = column
        break


if id_column is None:

    print(
        "No Sample_ID column detected."
    )

    print(
        "This is acceptable for simple correlation,"
    )

    print(
        "but Sample_ID is recommended for verifying"
    )

    print(
        "that uncertainty and MAE are correctly paired."
    )

else:

    print(
        f"ID column detected: {id_column}"
    )

    duplicate_count = (
        df[id_column]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate IDs: {duplicate_count}"
    )

    if duplicate_count > 0:

        print()
        print(
            "WARNING: Duplicate sample IDs detected."
        )

    else:

        print(
            "STATUS: No duplicate sample IDs."
        )


# ---------------------------------------------------------
# 16. Calculate Pearson correlation
# ---------------------------------------------------------

print()
print("-" * 60)
print("PEARSON CORRELATION")
print("-" * 60)

x = valid_df["Mean_Uncertainty"].to_numpy()
y = valid_df["MAE"].to_numpy()


if np.std(x) == 0:

    print(
        "Correlation cannot be calculated because"
        " Mean_Uncertainty is constant."
    )

elif np.std(y) == 0:

    print(
        "Correlation cannot be calculated because"
        " MAE is constant."
    )

else:

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    print(
        f"Pearson correlation (r): {correlation:.6f}"
    )

    print(
        f"R-squared (r²)          : {correlation ** 2:.6f}"
    )


# ---------------------------------------------------------
# 17. Final diagnostic conclusion
# ---------------------------------------------------------

print()
print("=" * 60)
print("FINAL DIAGNOSTIC")
print("=" * 60)

problems = []

if missing_columns:

    problems.append(
        "Required columns are missing."
    )

if len(valid_df) < 2:

    problems.append(
        "Insufficient valid paired observations."
    )

if uncertainty_unique < 2:

    problems.append(
        "Mean_Uncertainty has no variation."
    )

if mae_unique < 2:

    problems.append(
        "MAE has no variation."
    )

if sum(infinite_counts.values()) > 0:

    problems.append(
        "Infinite values are present."
    )


if problems:

    print()
    print("STATUS: NOT READY")

    print()

    for problem in problems:

        print(f" - {problem}")

else:

    print()
    print("STATUS: READY FOR CORRELATION ANALYSIS")

    print()
    print(
        "The CSV contains valid paired uncertainty"
    )

    print(
        "and MAE observations suitable for the"
    )

    print(
        "uncertainty-vs-error correlation analysis."
    )


print()
print("=" * 60)