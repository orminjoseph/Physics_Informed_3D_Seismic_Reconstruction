"""
=========================================================
Test Composite Total Loss
=========================================================
"""

import torch

from losses.total_loss import TotalLoss


def test_total_loss():

    print()

    print("=" * 60)

    print("Testing Composite Total Loss")

    print("=" * 60)

    prediction = torch.randn(

        2,
        1,
        64,
        64,
        64,

        requires_grad=True

    )

    target = torch.randn(

        2,
        1,
        64,
        64,
        64

    )

    velocity_model = torch.ones(

        2,
        1,
        64,
        64,
        64

    )


    log_variance = torch.zeros(

        2,
        1,
        64,
        64,
        64,

        requires_grad=True

    )

    criterion = TotalLoss()

    losses = criterion(

        prediction,

        target,

        velocity_model,

        log_variance

    )


    losses["total"].backward()

    print("Backward pass: OK")

    print()

    print("MAE Loss         :", losses["mae"].item())

    print("Physics Loss     :", losses["physics"].item())

    print("Uncertainty Loss :", losses["uncertainty"].item())

    print("SSIM Loss        :", losses["ssim"].item())

    print("--------------------------------------")

    print("Total Loss       :", losses["total"].item())

    print()

    print("Composite Loss Test: PASSED")


def main():

    test_total_loss()


if __name__ == "__main__":

    main()