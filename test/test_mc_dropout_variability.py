"""
============================================================
MC DROPOUT VARIABILITY TEST
============================================================

Verifies that MC Dropout produces different
predictions across stochastic forward passes.

Author: Ormin Joseph
============================================================
"""

import torch

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
print("MC DROPOUT VARIABILITY TEST")
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

print()
print("=" * 60)
print("MC DROPOUT STATISTICS")
print("=" * 60)

print(
    "Mean Uncertainty:",
    epistemic_uncertainty.mean().item()
)

print(
    "Std Uncertainty:",
    epistemic_uncertainty.std().item()
)

print(
    "Min Uncertainty:",
    epistemic_uncertainty.min().item()
)

print(
    "Max Uncertainty:",
    epistemic_uncertainty.max().item()
)
