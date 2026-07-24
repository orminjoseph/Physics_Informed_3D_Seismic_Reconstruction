import torch

from models.attention import AttentionGate3D

encoder = torch.randn(
    1,
    64,
    16,
    32,
    32
)

decoder = torch.randn(
    1,
    64,
    16,
    32,
    32
)

model = AttentionGate3D(
    encoder_channels=64,
    decoder_channels=64,
    inter_channels=32
)

output = model(
    encoder,
    decoder
)

print("=" * 60)
print("Encoder :", encoder.shape)
print("Decoder :", decoder.shape)
print("Output  :", output.shape)
print("=" * 60)