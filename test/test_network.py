"""
=========================================================
Test Complete Network3D
=========================================================
"""

import torch

from models.network import Network3D


def test_network():

    print()

    print("=" * 60)

    print("Testing Complete Network3D")

    print("=" * 60)

    x = torch.randn(

        2,

        1,

        64,

        64,

        64,

        requires_grad=True

    )

    model = Network3D(

        in_channels=1,

        out_channels=1

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

        1,

        64,

        64,

        64

    )

    print()

    print("Network Shape Test: PASSED")


def main():

    test_network()


if __name__ == "__main__":

    main()