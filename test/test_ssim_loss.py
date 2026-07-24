import torch

from losses.ssim_loss import SSIMLoss

prediction = torch.rand(
    1,
    1,
    64,
    128,
    128
)

target = torch.rand(
    1,
    1,
    64,
    128,
    128
)

criterion = SSIMLoss()

loss = criterion(
    prediction,
    target
)

print("="*60)
print("SSIM Loss :", loss.item())
print("="*60)