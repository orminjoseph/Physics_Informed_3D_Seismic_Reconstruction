"""
============================================================
MC DROPOUT CALIBRATION TEST
============================================================

Checks whether MC Dropout uncertainty
correlates with reconstruction error.

Author: Ormin Joseph
============================================================
"""

import numpy as np
import torch
import csv
import os
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from inference.mc_dropout_predictor import MCDropoutPredictor


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"

print()
print("=" * 60)
print("MC DROPOUT CALIBRATION TEST")
print("=" * 60)

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0]

device = torch.device("cpu")

predictor = MCDropoutPredictor(
    model=Network3D(),
    checkpoint=CHECKPOINT,
    device=device,
    num_samples=20
)

mean_prediction, epistemic_uncertainty = predictor.predict(
    corrupted
)

error_map = torch.abs(
    mean_prediction.squeeze(0)
    - target
)

error_flat = (
    error_map.cpu()
    .numpy()
    .flatten()
)

uncertainty_flat = (
    epistemic_uncertainty.squeeze(0)
    .cpu()
    .numpy()
    .flatten()
)

correlation = np.corrcoef(
    error_flat,
    uncertainty_flat
)[0, 1]

print()
print("=" * 60)
print("MC DROPOUT CALIBRATION")
print("=" * 60)

print(
    f"Correlation(Error, Uncertainty): "
    f"{correlation:.4f}"
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

with open(
        "outputs/reports/mc_dropout_calibration.csv",
        "w",
        newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Correlation"
        ]
    )

    writer.writerow(
        [
            correlation
        ]
    )

plt.figure(figsize=(6, 6))

sample = np.random.choice(
    len(error_flat),
    5000,
    replace=False
)

plt.scatter(
    uncertainty_flat[sample],
    error_flat[sample],
    alpha=0.3,
    s=5
)

plt.xlabel("MC Dropout Uncertainty")

plt.ylabel("Absolute Error")

plt.title(
    f"MC Dropout Calibration\n"
    f"Correlation={correlation:.4f}"
)

plt.grid(True)

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

plot_file = (
    "outputs/figures/"
    "mc_dropout_calibration.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("CSV saved to:")
print("outputs/reports/mc_dropout_calibration.csv")

print()
print("Plot saved to:")
print(plot_file)