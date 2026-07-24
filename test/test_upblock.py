import torch

from models.blocks import UpBlock3D

# Decoder feature map (coming from a deeper level)
decoder = torch.randn(1, 32, 32, 64, 64)

# Skip connection from the encoder
skip = torch.randn(1, 16, 64, 128, 128)

model = UpBlock3D(
    in_channels=32,
    skip_channels=16,
    out_channels=16
)

output = model(decoder, skip)

print("=" * 60)
print("Decoder Input :", decoder.shape)
print("Skip Shape    :", skip.shape)
print("Output Shape  :", output.shape)
print("=" * 60)