import torch

from models.bottleneck import Bottleneck3D

x = torch.randn(
    1,
    512,
    4,
    8,
    8
)

model = Bottleneck3D()

output = model(x)

print("=" * 60)
print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
print("=" * 60)