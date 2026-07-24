import torch

from losses.reconstruction_loss import ReconstructionLoss

prediction = torch.randn(
    1,
    1,
    64,
    128,
    128
)

target = torch.randn(
    1,
    1,
    64,
    128,
    128
)

criterion = ReconstructionLoss()

loss = criterion(
    prediction,
    target
)

print("=" * 60)
print("MAE Loss:", loss.item())
print("=" * 60)