import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = (
    "outputs/reports/"
    "uncertainty_evaluation.csv"
)

df = pd.read_csv(CSV_FILE)

x = df["Mean_Uncertainty"].values
y = df["MAE"].values

correlation = np.corrcoef(x, y)[0, 1]

m, b = np.polyfit(x, y, 1)

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = m * x_line + b

plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    alpha=0.8
)

plt.plot(
    x_line,
    y_line,
    linewidth=2
)

plt.xlabel("Mean Uncertainty")
plt.ylabel("MAE")

plt.title(
    f"Uncertainty vs Reconstruction Error\n"
    f"Correlation = {correlation:.4f}"
)

plt.grid(True)

plt.tight_layout()

output_file = (
    "outputs/reports/"
    "uncertainty_vs_error.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.close()

print()
print("Saved:")
print(output_file)

print()
print(
    f"Correlation = {correlation:.4f}"
)