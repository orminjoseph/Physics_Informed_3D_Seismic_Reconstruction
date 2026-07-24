"""
=========================================================
Test Bottleneck3D
=========================================================
"""

import torch

from models.bottleneck import Bottleneck3D


def test_bottleneck():

    print()

    print("=" * 60)

    print("Testing Bottleneck3D")

    print("=" * 60)

    x = torch.randn(
        2,
        512,
        4,
        4,
        4,
        requires_grad=True
    )

    model = Bottleneck3D(
        channels=512
    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    print("Backward pass: OK")

    print()

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)

    assert y.shape == (

        2,

        512,

        4,

        4,

        4

    )

    print()

    print("Bottleneck Shape Test: PASSED")


def main():

    test_bottleneck()


if __name__ == "__main__":

    main()