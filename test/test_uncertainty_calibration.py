"""
=========================================================
F3 Uncertainty Calibration Test
=========================================================

Checks whether predictive uncertainty correlates
with reconstruction error.

Author: Ormin Joseph
=========================================================
"""

import numpy as np
import torch

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
print("F3 UNCERTAINTY CALIBRATION TEST")
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

# --------------------------------------
# Error Map
# --------------------------------------

error_map = torch.abs(
    reconstruction.squeeze(0)
    - target
)

# --------------------------------------
# Flatten
# --------------------------------------

error_flat = (
    error_map
    .cpu()
    .numpy()
    .flatten()
)

uncertainty_flat = (
    uncertainty.squeeze(0)
    .cpu()
    .numpy()
    .flatten()
)

# --------------------------------------
# Correlation
# --------------------------------------

correlation = np.corrcoef(
    error_flat,
    uncertainty_flat
)[0, 1]

print()
print("=" * 60)
print("UNCERTAINTY CALIBRATION")
print("=" * 60)

print(
    f"Correlation (Error vs Uncertainty): "
    f"{correlation:.4f}"
)

if correlation > 0.70:
    print("Excellent uncertainty calibration")

elif correlation > 0.40:
    print("Good uncertainty calibration")

elif correlation > 0.00:
    print("Weak uncertainty calibration")

else:
    print("Poor uncertainty calibration")