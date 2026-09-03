"""
=========================================================
Test MC Dropout Epistemic Uncertainty
=========================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D


def main():

    print()
    print("=" * 60)
    print("TESTING MC DROPOUT EPISTEMIC UNCERTAINTY")
    print("=" * 60)

    # =====================================================
    # TEST INPUT
    # =====================================================

    x = torch.randn(
        1,
        1,
        64,
        64,
        64
    )

    # =====================================================
    # CREATE NETWORK
    # =====================================================

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    )

    # =====================================================
    # CREATE MC DROPOUT ESTIMATOR
    # =====================================================

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=20
    )

    # =====================================================
    # RUN MC DROPOUT
    # =====================================================

    results = mc_dropout.predict(x)

    # =====================================================
    # EXTRACT RESULTS
    # =====================================================

    reconstruction_samples = (
        results["reconstruction_samples"]
    )

    reconstruction_mean = (
        results["reconstruction_mean"]
    )

    epistemic_variance = (
        results[
            "reconstruction_epistemic_variance"
        ]
    )

    travel_time_mean = (
        results["travel_time_mean"]
    )

    log_variance_mean = (
        results["log_variance_mean"]
    )

    # =====================================================
    # PRINT SHAPES
    # =====================================================

    print()
    print("Input Shape:")
    print(x.shape)

    print()
    print("MC Reconstruction Samples:")
    print(reconstruction_samples.shape)

    print()
    print("Reconstruction Mean:")
    print(reconstruction_mean.shape)

    print()
    print("Epistemic Variance:")
    print(epistemic_variance.shape)

    print()
    print("Travel Time Mean:")
    print(travel_time_mean.shape)

    print()
    print("Log Variance Mean:")
    print(log_variance_mean.shape)

    # =====================================================
    # EXPECTED SHAPES
    # =====================================================

    assert reconstruction_samples.shape == (
        20,
        1,
        1,
        64,
        64,
        64
    )

    assert reconstruction_mean.shape == (
        1,
        1,
        64,
        64,
        64
    )

    assert epistemic_variance.shape == (
        1,
        1,
        64,
        64,
        64
    )

    assert travel_time_mean.shape == (
        1,
        1,
        64,
        64,
        64
    )

    assert log_variance_mean.shape == (
        1,
        1,
        64,
        64,
        64
    )

    # =====================================================
    # FINITE-VALUE CHECK
    # =====================================================

    assert torch.isfinite(
        reconstruction_mean
    ).all()

    assert torch.isfinite(
        epistemic_variance
    ).all()

    assert torch.isfinite(
        travel_time_mean
    ).all()

    assert torch.isfinite(
        log_variance_mean
    ).all()

    # =====================================================
    # NON-NEGATIVE VARIANCE CHECK
    # =====================================================

    assert torch.all(
        epistemic_variance >= 0.0
    )

    # =====================================================
    # STOCHASTICITY CHECK
    # =====================================================

    first_prediction = (
        reconstruction_samples[0]
    )

    second_prediction = (
        reconstruction_samples[1]
    )

    stochastic_difference = (
        first_prediction - second_prediction
    ).abs().mean()

    print()
    print(
        "Mean difference between "
        "MC samples:"
    )

    print(
        stochastic_difference.item()
    )

    assert stochastic_difference > 0.0

    # =====================================================
    # UNCERTAINTY STATISTICS
    # =====================================================

    print()
    print(
        "Epistemic Variance Minimum:"
    )

    print(
        epistemic_variance.min().item()
    )

    print(
        "Epistemic Variance Maximum:"
    )

    print(
        epistemic_variance.max().item()
    )

    print(
        "Epistemic Variance Mean:"
    )

    print(
        epistemic_variance.mean().item()
    )

    # =====================================================
    # TEST PASSED
    # =====================================================

    print()
    print("=" * 60)
    print("MC DROPOUT EPISTEMIC UNCERTAINTY TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()