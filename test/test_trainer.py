import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from models.network import PhysicsInformed3DUNet

model = PhysicsInformed3DUNet()

from losses.total_loss import TotalLoss
from trainer.trainer import Trainer

from utils.config import DEVICE

model = PhysicsInformed3DUNet()

loss_function = TotalLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    device=DEVICE
)

input_cube = torch.randn(
    1,
    1,
    64,
    128,
    128
)

target_cube = torch.randn(
    1,
    1,
    64,
    128,
    128
)

losses = trainer.train_step(
    input_cube,
    target_cube
)

print("=" * 60)

for name, value in losses.items():
    print(
        f"{name:<15}: {value:.6f}"
    )

print("=" * 60)