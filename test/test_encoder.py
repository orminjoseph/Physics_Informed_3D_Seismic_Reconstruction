"""
=========================================================
Test 3D Encoder
=========================================================
"""

import torch

from models.encoder import Encoder3D
def test_encoder():

    print()

    print("=" * 60)

    print("Testing Encoder3D")

    print("=" * 60)

    # ------------------------------------------
    # Synthetic seismic cube
    # ------------------------------------------

    x = torch.randn(

        2,
        1,
        64,
        64,
        64,
        requires_grad=True

    )

    model = Encoder3D(

        in_channels=1

    )

    x1, x2, x3, x4, x5 = model(x)

    # ------------------------------------------
    # Backpropagation test
    # ------------------------------------------

    loss = (

        x1.mean() +

        x2.mean() +

        x3.mean() +

        x4.mean() +

        x5.mean()

    )

    loss.backward()

    print("Backward pass: OK")

    print()

    print("Input Shape :", x.shape)

    print()

    print("x1 :", x1.shape)

    print("x2 :", x2.shape)

    print("x3 :", x3.shape)

    print("x4 :", x4.shape)

    print("x5 :", x5.shape)

    assert x1.shape == (

        2,
        32,
        64,
        64,
        64

    )

    assert x2.shape == (

        2,
        64,
        32,
        32,
        32

    )

    assert x3.shape == (

        2,
        128,
        16,
        16,
        16

    )

    assert x4.shape == (

        2,
        256,
        8,
        8,
        8

    )

    assert x5.shape == (

        2,
        512,
        4,
        4,
        4

    )

    print()

    print("Encoder Shape Test: PASSED")

def main():

    test_encoder()


if __name__ == "__main__":

    main()