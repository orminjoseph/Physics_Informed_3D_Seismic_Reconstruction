import torch

from losses.uncertainty_loss import UncertaintyLoss

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

log_variance = torch.zeros_like(prediction)

criterion = UncertaintyLoss()

loss = criterion(
    prediction,
    target,
    log_variance
)

print("=" * 60)
print("Uncertainty Loss:", loss.item())
print("=" * 60)