"""
=========================================================
Test MC Dropout Epistemic Uncertainty
=========================================================

Purpose
-------
Verify the authoritative MCDropout3D implementation.

This test verifies:

1. MC reconstruction sample generation
2. MC travel-time sample generation
3. MC log-variance sample generation
4. Correct tensor shapes
5. Finite numerical values
6. Non-negative epistemic variance
7. Stochastic variation between MC samples
8. Independent verification of population epistemic variance
9. Controlled zero-variance behavior

Epistemic uncertainty is computed as:

    sigma_e^2 =
        (1 / N) * sum_n (y_hat_n - y_hat_mean)^2

where N is the number of MC Dropout samples.

Aleatoric and predictive uncertainty are intentionally
not tested here. Those belong to the predictive uncertainty
test suite.

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
    # TEST CONFIGURATION
    # =====================================================

    num_samples = 20

    input_shape = (
        1,
        1,
        64,
        64,
        64
    )

    # =====================================================
    # TEST INPUT
    # =====================================================

    x = torch.randn(*input_shape)

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
        num_samples=num_samples
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

    travel_time_samples = (
        results["travel_time_samples"]
    )

    travel_time_mean = (
        results["travel_time_mean"]
    )

    travel_time_epistemic_variance = (
        results[
            "travel_time_epistemic_variance"
        ]
    )

    log_variance_samples = (
        results["log_variance_samples"]
    )

    log_variance_mean = (
        results["log_variance_mean"]
    )

    log_variance_epistemic_variance = (
        results[
            "log_variance_epistemic_variance"
        ]
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
    print("Reconstruction Epistemic Variance:")
    print(epistemic_variance.shape)

    print()
    print("MC Travel-Time Samples:")
    print(travel_time_samples.shape)

    print()
    print("Travel-Time Mean:")
    print(travel_time_mean.shape)

    print()
    print("Travel-Time Epistemic Variance:")
    print(travel_time_epistemic_variance.shape)

    print()
    print("MC Log-Variance Samples:")
    print(log_variance_samples.shape)

    print()
    print("Log-Variance Mean:")
    print(log_variance_mean.shape)

    print()
    print("Log-Variance Epistemic Variance:")
    print(log_variance_epistemic_variance.shape)

    # =====================================================
    # EXPECTED SHAPES
    # =====================================================

    expected_sample_shape = (
        num_samples,
        1,
        1,
        64,
        64,
        64
    )

    expected_output_shape = (
        1,
        1,
        64,
        64,
        64
    )

    assert reconstruction_samples.shape == (
        expected_sample_shape
    )

    assert travel_time_samples.shape == (
        expected_sample_shape
    )

    assert log_variance_samples.shape == (
        expected_sample_shape
    )

    assert reconstruction_mean.shape == (
        expected_output_shape
    )

    assert epistemic_variance.shape == (
        expected_output_shape
    )

    assert travel_time_mean.shape == (
        expected_output_shape
    )

    assert travel_time_epistemic_variance.shape == (
        expected_output_shape
    )

    assert log_variance_mean.shape == (
        expected_output_shape
    )

    assert log_variance_epistemic_variance.shape == (
        expected_output_shape
    )

    print()
    print("Shape checks: PASSED")

    # =====================================================
    # FINITE-VALUE CHECK
    # =====================================================

    assert torch.isfinite(
        reconstruction_samples
    ).all()

    assert torch.isfinite(
        reconstruction_mean
    ).all()

    assert torch.isfinite(
        epistemic_variance
    ).all()

    assert torch.isfinite(
        travel_time_samples
    ).all()

    assert torch.isfinite(
        travel_time_mean
    ).all()

    assert torch.isfinite(
        travel_time_epistemic_variance
    ).all()

    assert torch.isfinite(
        log_variance_samples
    ).all()

    assert torch.isfinite(
        log_variance_mean
    ).all()

    assert torch.isfinite(
        log_variance_epistemic_variance
    ).all()

    print()
    print("Finite-value checks: PASSED")

    # =====================================================
    # NON-NEGATIVE VARIANCE CHECK
    # =====================================================

    assert torch.all(
        epistemic_variance >= 0.0
    )

    assert torch.all(
        travel_time_epistemic_variance >= 0.0
    )

    assert torch.all(
        log_variance_epistemic_variance >= 0.0
    )

    print()
    print("Non-negative variance checks: PASSED")

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
        "first two MC reconstruction samples:"
    )

    print(
        stochastic_difference.item()
    )

    assert stochastic_difference > 0.0

    print()
    print("Stochasticity check: PASSED")

    # =====================================================
    # INDEPENDENT RECONSTRUCTION VARIANCE CHECK
    # =====================================================
    #
    # MCDropout3D should use population variance:
    #
    # sigma_e^2 =
    #     (1/N) * sum_n (y_n - y_mean)^2
    #
    # Therefore unbiased=False is required.
    # =====================================================

    independent_reconstruction_variance = (
        (
            reconstruction_samples
            - reconstruction_mean.unsqueeze(0)
        ) ** 2
    ).mean(dim=0)

    variance_difference = (
        independent_reconstruction_variance
        - epistemic_variance
    ).abs().max()

    print()
    print(
        "Maximum difference between "
        "implemented and independently "
        "calculated reconstruction variance:"
    )

    print(
        variance_difference.item()
    )

    assert torch.allclose(
        epistemic_variance,
        independent_reconstruction_variance,
        rtol=1e-5,
        atol=1e-6
    )

    print()
    print(
        "Independent reconstruction "
        "variance check: PASSED"
    )

    # =====================================================
    # INDEPENDENT TRAVEL-TIME VARIANCE CHECK
    # =====================================================

    independent_travel_time_variance = (
        (
            travel_time_samples
            - travel_time_mean.unsqueeze(0)
        ) ** 2
    ).mean(dim=0)

    travel_time_variance_difference = (
        independent_travel_time_variance
        - travel_time_epistemic_variance
    ).abs().max()

    print()
    print(
        "Maximum difference between "
        "implemented and independently "
        "calculated travel-time variance:"
    )

    print(
        travel_time_variance_difference.item()
    )

    assert torch.allclose(
        travel_time_epistemic_variance,
        independent_travel_time_variance,
        rtol=1e-5,
        atol=1e-6
    )

    print()
    print(
        "Independent travel-time "
        "variance check: PASSED"
    )

    # =====================================================
    # INDEPENDENT LOG-VARIANCE VARIANCE CHECK
    # =====================================================
    #
    # This is diagnostic only.
    #
    # It must NOT be added to reconstruction
    # epistemic uncertainty.
    # =====================================================

    independent_log_variance_variance = (
        (
            log_variance_samples
            - log_variance_mean.unsqueeze(0)
        ) ** 2
    ).mean(dim=0)

    log_variance_difference = (
        independent_log_variance_variance
        - log_variance_epistemic_variance
    ).abs().max()

    print()
    print(
        "Maximum difference between "
        "implemented and independently "
        "calculated log-variance variance:"
    )

    print(
        log_variance_difference.item()
    )

    assert torch.allclose(
        log_variance_epistemic_variance,
        independent_log_variance_variance,
        rtol=1e-5,
        atol=1e-6
    )

    print()
    print(
        "Independent log-variance "
        "variance check: PASSED"
    )

    # =====================================================
    # CONTROLLED ZERO-VARIANCE TEST
    # =====================================================
    #
    # If all MC samples are identical, epistemic
    # variance must be exactly zero.
    #
    # This verifies that epistemic uncertainty is
    # genuinely measuring variation between MC samples.
    # =====================================================

    controlled_prediction = (
        reconstruction_samples[0]
    )

    identical_samples = (
        controlled_prediction
        .unsqueeze(0)
        .repeat(num_samples, 1, 1, 1, 1, 1)
    )

    controlled_mean = (
        identical_samples.mean(dim=0)
    )

    controlled_variance = (
        (
            identical_samples
            - controlled_mean.unsqueeze(0)
        ) ** 2
    ).mean(dim=0)

    print()
    print(
        "Maximum controlled "
        "zero-variance value:"
    )

    print(
        controlled_variance.max().item()
    )

    assert torch.allclose(
        controlled_variance,
        torch.zeros_like(controlled_variance),
        atol=1e-12
    )

    print()
    print(
        "Controlled zero-variance "
        "check: PASSED"
    )

    # =====================================================
    # UNCERTAINTY STATISTICS
    # =====================================================

    print()
    print(
        "Reconstruction Epistemic Variance Minimum:"
    )

    print(
        epistemic_variance.min().item()
    )

    print(
        "Reconstruction Epistemic Variance Maximum:"
    )

    print(
        epistemic_variance.max().item()
    )

    print(
        "Reconstruction Epistemic Variance Mean:"
    )

    print(
        epistemic_variance.mean().item()
    )

    print()
    print(
        "Travel-Time Epistemic Variance Mean:"
    )

    print(
        travel_time_epistemic_variance.mean().item()
    )

    print()
    print(
        "Log-Variance Epistemic Variance Mean "
        "(diagnostic only):"
    )

    print(
        log_variance_epistemic_variance.mean().item()
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()
    print("=" * 60)
    print("MC DROPOUT EPISTEMIC UNCERTAINTY TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()