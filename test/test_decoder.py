import torch

from models.encoder import Encoder3D
from models.bottleneck import Bottleneck3D
from models.decoder import Decoder3D

encoder = Encoder3D()

bottleneck = Bottleneck3D()

decoder = Decoder3D()

x = torch.randn(
    1,
    1,
    64,
    128,
    128
)

x1, x2, x3, x4, x5 = encoder(x)

latent = bottleneck(x5)

output = decoder(
    x1,
    x2,
    x3,
    x4,
    latent
)

print("=" * 60)
print("Decoder Output:", output.shape)
print("=" * 60)