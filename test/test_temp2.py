import pandas as pd

df = pd.read_csv(
    "outputs/reports/mask_robustness.csv"
)

print(df.head())
print(df.columns)