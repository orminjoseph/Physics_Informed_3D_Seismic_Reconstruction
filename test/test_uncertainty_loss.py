"""
=========================================================
Test Predictive Uncertainty Loss
=========================================================
"""

import torch

from losses.uncertainty_loss import UncertaintyLoss


def test_uncertainty_loss():

    print()

    print("=" * 60)

    print("Testing Predictive Uncertainty Loss")

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

    log_variance = torch.zeros(

        2,
        1,
        64,
        64,
        64,

        requires_grad=True

    )

    criterion = UncertaintyLoss()

    loss = criterion(

        prediction,

        target,

        log_variance

    )

    loss.backward()

    print("Backward pass: OK")

    print()

    print("Loss Value :", loss.item())

    print()

    print("Uncertainty Loss Test: PASSED")


def main():

    test_uncertainty_loss()


if __name__ == "__main__":

    main()