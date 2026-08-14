from dataset.build_dataset import build_dataset
from torch.utils.data import DataLoader

from models.network import Network3D
from losses.total_loss import TotalLoss

import torch

dataset = build_dataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

inputs, targets, masks, velocity = next(iter(loader))

inputs = inputs.unsqueeze(1)
targets = targets.unsqueeze(1)
velocity = velocity.unsqueeze(1)

model = Network3D()

criterion = TotalLoss()

reconstruction, log_variance = model(inputs)

losses = criterion(
    reconstruction,
    targets,
    velocity,
    log_variance
)

print("MAE:", losses["mae"].item())
print("Physics:", losses["physics"].item())
print("Uncertainty:", losses["uncertainty"].item())
print("SSIM:", losses["ssim"].item())
print("Total:", losses["total"].item())

losses["total"].backward()

print("Backward Pass Successful")