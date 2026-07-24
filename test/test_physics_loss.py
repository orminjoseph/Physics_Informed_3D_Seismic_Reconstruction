import torch

from losses.physics_loss import PhysicsLoss

prediction = torch.randn(
    1,
    1,
    32,
    32,
    32
)

target = torch.randn(
    1,
    1,
    32,
    32,
    32
)

criterion = PhysicsLoss()

loss = criterion(
    prediction,
    target
)

print("="*60)
print("Physics Loss :", loss.item())
print("="*60)