"""
=========================================================
Test Full Network With Attention Gates
=========================================================
"""

import torch

from models.network import Network3D


def main():

    model = Network3D()

    x = torch.randn(
        1,
        1,
        64,
        128,
        128
    )

    reconstruction, uncertainty = model(x)

    print()
    print("Input Shape        :", x.shape)
    print("Reconstruction     :", reconstruction.shape)
    print("Uncertainty Shape  :", uncertainty.shape)


if __name__ == "__main__":
    main()