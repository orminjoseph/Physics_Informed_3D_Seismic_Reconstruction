import torch

from dataset.build_dataset import build_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D

dataset = build_dataset()

loader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=False
)

input_cube, target_cube, mask, velocity_cube = next(iter(loader))

model = Network3D()

reconstruction, log_variance = model(input_cube)

print()
print("Input:", input_cube.shape)
print("Target:", target_cube.shape)
print("Velocity:", velocity_cube.shape)

print()
print("Reconstruction:", reconstruction.shape)
print("Log Variance:", log_variance.shape)