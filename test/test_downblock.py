import torch

from models.blocks import DownBlock3D

# Dummy seismic features
x = torch.randn(1, 16, 64, 128, 128)

model = DownBlock3D(
    in_channels=16,
    out_channels=32
)

output = model(x)

print("=" * 60)
print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
print("=" * 60)