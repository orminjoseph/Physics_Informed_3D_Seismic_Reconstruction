import pandas as pd

df = pd.read_csv(
    "outputs/reports/uncertainty_calibration.csv"
)

print(df.head())
print(df.columns)