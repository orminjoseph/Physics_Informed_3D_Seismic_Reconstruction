"""
=========================================================
F3 Uncertainty Visualization Test
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
print("F3 UNCERTAINTY VISUALIZATION")
print("=" * 60)

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0]

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

# -----------------------------------------
# Convert to numpy
# -----------------------------------------

target = np.squeeze(target.cpu().numpy())
reconstruction = np.squeeze(reconstruction.cpu().numpy())
uncertainty = np.squeeze(uncertainty.cpu().numpy())

# -----------------------------------------
# Error map
# -----------------------------------------

error_map = np.abs(
    target - reconstruction
)

# -----------------------------------------
# Central slice
# -----------------------------------------

slice_index = target.shape[0] // 2

target_slice = target[slice_index]
reconstruction_slice = reconstruction[slice_index]
error_slice = error_map[slice_index]
uncertainty_slice = uncertainty[slice_index]

# -----------------------------------------
# Figure
# -----------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes[0, 0].imshow(
    target_slice,
    cmap="gray",
    aspect="auto"
)
axes[0, 0].set_title("Ground Truth")

axes[0, 1].imshow(
    reconstruction_slice,
    cmap="gray",
    aspect="auto"
)
axes[0, 1].set_title("Reconstruction")

axes[1, 0].imshow(
    error_slice,
    cmap="hot",
    aspect="auto"
)
axes[1, 0].set_title("Absolute Error")

axes[1, 1].imshow(
    uncertainty_slice,
    cmap="viridis",
    aspect="auto"
)
axes[1, 1].set_title("Predictive Uncertainty")

plt.tight_layout()

# -----------------------------------------
# Save
# -----------------------------------------

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
    "uncertainty_error_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Figure Saved:")
print(save_path)