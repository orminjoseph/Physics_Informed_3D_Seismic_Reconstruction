import torch

from models.network import PhysicsInformed3DUNet

model = PhysicsInformed3DUNet()

x = torch.randn(
    1,
    1,
    64,
    128,
    128
)

reconstruction, uncertainty = model(x)

print("=" * 70)
print("Input Shape          :", x.shape)
print("Reconstruction Shape :", reconstruction.shape)
print("Log Variance Shape   :", uncertainty.shape)
print("=" * 70)