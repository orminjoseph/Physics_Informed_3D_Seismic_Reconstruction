import pandas as pd

df = pd.read_csv(
    "outputs/reports/mc_dropout_calibration.csv"
)

print(df.head())
print(df.columns)