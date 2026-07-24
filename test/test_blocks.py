"""
=========================================================
Test Neural Network Building Blocks
=========================================================
"""

import torch

from models.blocks import (

    DoubleConv3D,

    ResidualBlock3D,

    DownBlock3D,

    UpBlock3D

)
def test_double_conv():

    print()

    print("=" * 60)

    print("Testing DoubleConv3D")

    print("=" * 60)

    x = torch.randn(
        2,
        1,
        64,
        64,
        64,
        requires_grad=True
    )


    model = DoubleConv3D(

        in_channels=1,

        out_channels=16

    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    print("Backward pass: OK")

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)



def test_residual_block():

    print()

    print("=" * 60)

    print("Testing ResidualBlock3D")

    print("=" * 60)

    x = torch.randn(
        2,
        16,
        64,
        64,
        64,
        requires_grad=True
    )


    model = ResidualBlock3D(

        channels=16

    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    print("Backward pass: OK")

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)



def test_down_block():

    print()

    print("=" * 60)

    print("Testing DownBlock3D")

    print("=" * 60)

    x = torch.randn(
        2,
        16,
        64,
        64,
        64,
        requires_grad=True
    )


    model = DownBlock3D(

        in_channels=16,

        out_channels=32

    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    print("Backward pass: OK")

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)


def test_up_block():
    print()

    print("=" * 60)

    print("Testing UpBlock3D")

    print("=" * 60)

    # ------------------------------------------
    # Decoder feature (deep feature map)
    # ------------------------------------------

    decoder_feature = torch.randn(

        2,
        64,
        16,
        16,
        16,
        requires_grad=True

    )

    # ------------------------------------------
    # Encoder feature (skip connection)
    # ------------------------------------------

    encoder_feature = torch.randn(

        2,
        32,
        32,
        32,
        32,
        requires_grad=True

    )

    model = UpBlock3D(

        in_channels=64,

        skip_channels=32,

        out_channels=32

    )

    y = model(

        decoder_feature,

        encoder_feature

    )

    loss = y.mean()

    loss.backward()

    print("Backward pass: OK")

    print("Decoder Shape :", decoder_feature.shape)

    print("Encoder Shape :", encoder_feature.shape)

    print("Output Shape  :", y.shape)


def main():

    test_double_conv()

    test_residual_block()

    test_down_block()

    test_up_block()


if __name__ == "__main__":

    main()
