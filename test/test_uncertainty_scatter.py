"""
=========================================================
F3 Error vs Uncertainty Scatter Plot
=========================================================
"""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from inference.predictor import Predictor


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = (
    r"outputs"
    r"\experiment_20260731_040411"
    r"\checkpoints"
    r"\best_model.pth"
)

print()
print("=" * 60)
print("ERROR VS UNCERTAINTY SCATTER")
print("=" * 60)

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0][:4]

device = torch.device("cpu")

model = Network3D(
    in_channels=1,
    out_channels=1
)

predictor = Predictor(
    model=model,
    checkpoint=CHECKPOINT,
    device=device
)

reconstruction, uncertainty = predictor.predict(
    corrupted
)

# -----------------------------------
# Convert to numpy
# -----------------------------------

target = target.cpu().numpy()
reconstruction = reconstruction.cpu().numpy()
uncertainty = uncertainty.cpu().numpy()

# -----------------------------------
# Error
# -----------------------------------

error = np.abs(
    target - reconstruction
)

# -----------------------------------
# Flatten
# -----------------------------------

error = error.flatten()
uncertainty = uncertainty.flatten()

# -----------------------------------
# Sample points
# -----------------------------------

num_points = 5000

indices = np.random.choice(
    len(error),
    num_points,
    replace=False
)

error_sample = error[indices]
uncertainty_sample = uncertainty[indices]

# -----------------------------------
# Correlation
# -----------------------------------

correlation = np.corrcoef(
    error_sample,
    uncertainty_sample
)[0, 1]

print()
print("Correlation :", round(correlation, 4))

# -----------------------------------
# Scatter Plot
# -----------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    error_sample,
    uncertainty_sample,
    alpha=0.3,
    s=5
)

plt.xlabel("Absolute Error")
plt.ylabel("Predictive Uncertainty")

plt.title(
    f"Error vs Uncertainty\nCorrelation = {correlation:.4f}"
)

plt.tight_layout()

output_dir = os.path.join(
    "outputs",
    "uncertainty"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

save_path = os.path.join(
    output_dir,
    "error_vs_uncertainty_scatter.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.close()

print()
print("Figure Saved:")
print(save_path)