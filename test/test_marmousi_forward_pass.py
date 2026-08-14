import torch

from dataset.marmousi2_patch_dataset import Marmousi2PatchDataset
from models.network import Network3D


dataset = Marmousi2PatchDataset(
    segy_path=r"C:\Users\ormin\Desktop\SEG_FILES\elastic-marmousi-model\elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy\MODEL_P-WAVE_VELOCITY_1.25m.segy"
)

inputs, targets, masks, velocities = dataset[0]

print("Original Shape:", inputs.shape)

# Add batch dimension
inputs = inputs.unsqueeze(0).unsqueeze(0)

print("Batch Shape:", inputs.shape)

model = Network3D()

reconstruction, log_variance = model(inputs)

print("Reconstruction Shape:", reconstruction.shape)
print("Log Variance Shape:", log_variance.shape)