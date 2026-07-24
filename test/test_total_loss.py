import torch

from losses.total_loss import TotalLoss

prediction = torch.rand(
    1,
    1,
    32,
    32,
    32
)

target = torch.rand(
    1,
    1,
    32,
    32,
    32
)

log_variance = torch.zeros_like(prediction)

criterion = TotalLoss()

loss, loss_dict = criterion(
    prediction,
    target,
    log_variance
)

print("=" * 60)

for key, value in loss_dict.items():
    print(f"{key:15s}: {value:.6f}")

print("=" * 60)