"""
=========================================================
Test Composite Total Loss
=========================================================

Tests the complete composite loss for the
Physics-Informed 3D Encoder-Decoder Framework.

Author: Ormin Joseph
=========================================================
"""

import torch

from losses.total_loss import TotalLoss

from utils.config import (
    DX,
    DY,
    DZ
)


def main():

    print("=" * 60)
    print("TESTING COMPOSITE TOTAL LOSS")
    print("=" * 60)

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    torch.manual_seed(42)

    # =====================================================
    # TEST TENSOR DIMENSIONS
    # =====================================================

    batch_size = 1
    channels = 1

    depth = 16
    crossline = 32
    inline = 32

    shape = (
        batch_size,
        channels,
        depth,
        crossline,
        inline
    )

    print("\nTest tensor shape:")

    print(shape)

    # =====================================================
    # CREATE PREDICTION
    # =====================================================

    prediction = torch.randn(
        shape,
        requires_grad=True
    )

    # =====================================================
    # CREATE TARGET
    # =====================================================

    target = torch.randn(
        shape
    )

    # =====================================================
    # CREATE TRAVEL TIME
    # =====================================================

    travel_time = torch.rand(
        shape,
        requires_grad=True
    )

    # =====================================================
    # CREATE VELOCITY MODEL
    # =====================================================

    # Use physically meaningful positive
    # P-wave velocity.

    velocity_model = (
        1500.0
        +
        3000.0
        *
        torch.rand(shape)
    )

    # =====================================================
    # CREATE LOG VARIANCE
    # =====================================================

    log_variance = torch.randn(
        shape,
        requires_grad=True
    )

    # =====================================================
    # INITIALIZE TOTAL LOSS
    # =====================================================

    total_loss_function = TotalLoss(

        dx=DX,

        dy=DY,

        dz=DZ

    )

    # =====================================================
    # COMPUTE LOSS
    # =====================================================

    losses = total_loss_function(

        prediction=prediction,

        target=target,

        travel_time=travel_time,

        velocity_model=velocity_model,

        log_variance=log_variance

    )

    # =====================================================
    # PRINT LOSS COMPONENTS
    # =====================================================

    print("\nLoss Components:\n")

    for name, value in losses.items():

        print(

            f"{name:25s}: "

            f"{value.item():.10e}"

        )

    # =====================================================
    # CHECK TOTAL LOSS
    # =====================================================

    print("\nChecking total loss...")

    total = losses["total"]

    if not torch.isfinite(total):

        raise RuntimeError(

            "Total loss contains NaN or Inf."
        )

    print(

        "Total loss is finite."
    )

    # =====================================================
    # BACKWARD PROPAGATION
    # =====================================================

    print("\nTesting backward propagation...")

    total.backward()

    # =====================================================
    # CHECK PREDICTION GRADIENT
    # =====================================================

    if prediction.grad is None:

        raise RuntimeError(

            "Prediction gradient was not computed."
        )

    # =====================================================
    # CHECK TRAVEL-TIME GRADIENT
    # =====================================================

    if travel_time.grad is None:

        raise RuntimeError(

            "Travel-time gradient was not computed."
        )

    # =====================================================
    # CHECK LOG-VARIANCE GRADIENT
    # =====================================================

    if log_variance.grad is None:

        raise RuntimeError(

            "Log-variance gradient was not computed."
        )

    # =====================================================
    # PRINT GRADIENT INFORMATION
    # =====================================================

    print("\nPrediction gradient:")

    print(

        "Mean:",

        prediction.grad.abs().mean().item()

    )

    print(

        "Maximum:",

        prediction.grad.abs().max().item()

    )

    print("\nTravel-time gradient:")

    print(

        "Mean:",

        travel_time.grad.abs().mean().item()

    )

    print(

        "Maximum:",

        travel_time.grad.abs().max().item()

    )

    print("\nLog-variance gradient:")

    print(

        "Mean:",

        log_variance.grad.abs().mean().item()

    )

    print(

        "Maximum:",

        log_variance.grad.abs().max().item()

    )

    # =====================================================
    # SUCCESS
    # =====================================================

    print("\n" + "=" * 60)

    print(

        "COMPOSITE TOTAL LOSS TEST PASSED."

    )

    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    main()