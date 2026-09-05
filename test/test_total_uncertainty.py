"""
====================================================================
Total Uncertainty Integration Test
====================================================================

Data-independent validation of the complete predictive uncertainty
pipeline for the Physics-Informed 3D Seismic Reconstruction framework.

This test deliberately does NOT load:

    - F3
    - Synthetic seismic data
    - Marmousi
    - SEG data
    - Any other dataset

The purpose is to verify that the uncertainty architecture itself
works for ANY data mode.

Uncertainty decomposition:

    Aleatoric variance:
        sigma_a^2 = mean(exp(log_variance_samples))

    Epistemic variance:
        sigma_e^2 = Var(MC reconstruction samples)

    Predictive variance:
        sigma_pred^2 = sigma_a^2 + sigma_e^2

    Predictive standard deviation:
        sigma_pred = sqrt(sigma_pred^2)

Author: Ormin Joseph
====================================================================
"""


# ==================================================================
# IMPORTS
# ==================================================================

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import PredictiveUncertaintyEstimator
from utils.config import MC_DROPOUT_SAMPLES


# ==================================================================
# TEST CONFIGURATION
# ==================================================================

# Use the project's configured MC Dropout sample count.
MC_SAMPLES = MC_DROPOUT_SAMPLES

# Use a small controlled 3D tensor.
# This test is deliberately independent of all dataset modes.
INPUT_SHAPE = (1, 1, 64, 128, 128)


# ==================================================================
# MAIN TEST
# ==================================================================

def main():
    """
    Run the complete data-independent predictive uncertainty audit.
    """

    print("=" * 70)
    print("TOTAL UNCERTAINTY INTEGRATION TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Display test configuration.
    # --------------------------------------------------------------

    print("\nTest Configuration")
    print("------------------")
    print(f"Input Shape: {INPUT_SHAPE}")
    print(f"MC Dropout Samples: {MC_SAMPLES}")

    # --------------------------------------------------------------
    # Validate MC sample configuration.
    #
    # At least two stochastic samples are required to estimate
    # epistemic variance.
    # --------------------------------------------------------------

    if MC_SAMPLES < 2:
        raise ValueError(
            "MC_DROPOUT_SAMPLES must be at least 2 "
            "for epistemic variance estimation."
        )

    # --------------------------------------------------------------
    # Create a controlled input tensor.
    #
    # No seismic dataset is loaded.
    # --------------------------------------------------------------

    torch.manual_seed(42)

    seismic_input = torch.randn(INPUT_SHAPE)

    print("\nInput Tensor Shape:")
    print(seismic_input.shape)

    # --------------------------------------------------------------
    # Create the 3D physics-informed network.
    # --------------------------------------------------------------

    model = Network3D()

    # --------------------------------------------------------------
    # Create the authoritative MC Dropout engine.
    # --------------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    # --------------------------------------------------------------
    # Create the authoritative predictive uncertainty estimator.
    #
    # IMPORTANT:
    #
    # PredictiveUncertaintyEstimator is an nn.Module and its
    # uncertainty calculation methods are instance methods.
    # Therefore, we must instantiate it before calling:
    #
    #     aleatoric_variance()
    #     epistemic_variance()
    #     predictive_variance()
    # --------------------------------------------------------------

    uncertainty_estimator = PredictiveUncertaintyEstimator()

    # --------------------------------------------------------------
    # Generate stochastic predictions.
    #
    # The MC Dropout engine returns:
    #
    #     reconstruction_samples
    #     travel_time_samples
    #     log_variance_samples
    # --------------------------------------------------------------

    predictions = mc_dropout.predict(seismic_input)

    reconstruction_samples = predictions["reconstruction_samples"]
    travel_time_samples = predictions["travel_time_samples"]
    log_variance_samples = predictions["log_variance_samples"]

    # --------------------------------------------------------------
    # Display MC sample shapes.
    # --------------------------------------------------------------

    print("\nMC Reconstruction Samples:")
    print(reconstruction_samples.shape)

    print("\nMC Travel-Time Samples:")
    print(travel_time_samples.shape)

    print("\nMC Log-Variance Samples:")
    print(log_variance_samples.shape)

    # --------------------------------------------------------------
    # Compute reconstruction mean.
    #
    # The MC mean is the central reconstruction estimate.
    # --------------------------------------------------------------

    reconstruction_mean = reconstruction_samples.mean(dim=0)

    # --------------------------------------------------------------
    # Compute aleatoric variance.
    #
    # IMPORTANT:
    #
    # Every MC pass predicts its own log variance:
    #
    #     s_n = log(sigma_a,n^2)
    #
    # Therefore the correct MC aggregation is:
    #
    #     sigma_a^2
    #         = mean(exp(s_n))
    #
    # NOT:
    #
    #     exp(mean(s_n))
    #
    # The estimator handles this aggregation internally.
    # --------------------------------------------------------------

    aleatoric_variance = (
        uncertainty_estimator
        .aleatoric_variance(log_variance_samples)
    )

    # --------------------------------------------------------------
    # Compute epistemic variance.
    #
    # IMPORTANT:
    #
    # Epistemic uncertainty must be calculated from the stochastic
    # reconstruction predictions, NOT from log variance.
    #
    #     sigma_e^2
    #         = Var(y_hat^(1), ..., y_hat^(N))
    # --------------------------------------------------------------

    epistemic_variance = (
        uncertainty_estimator
        .epistemic_variance(reconstruction_samples)
    )

    # --------------------------------------------------------------
    # Compute predictive variance.
    #
    # The finalized estimator expects:
    #
    #     predictive_variance(
    #         log_variance,
    #         mc_predictions
    #     )
    #
    # It internally computes:
    #
    #     predictive variance
    #         = aleatoric variance
    #         + epistemic variance
    # --------------------------------------------------------------

    predictive_variance = (
        uncertainty_estimator
        .predictive_variance(
            log_variance_samples,
            reconstruction_samples,
        )
    )

    # --------------------------------------------------------------
    # Compute predictive standard deviation.
    #
    # The finalized estimator provides the standard deviation
    # calculation through standard_deviation().
    # --------------------------------------------------------------

    predictive_std = (
        uncertainty_estimator
        .standard_deviation(predictive_variance)
    )

    # ==================================================================
    # SHAPE CHECKS
    # ==================================================================

    expected_shape = INPUT_SHAPE

    assert reconstruction_mean.shape == expected_shape, (
        "Reconstruction mean shape mismatch."
    )

    assert aleatoric_variance.shape == expected_shape, (
        "Aleatoric variance shape mismatch."
    )

    assert epistemic_variance.shape == expected_shape, (
        "Epistemic variance shape mismatch."
    )

    assert predictive_variance.shape == expected_shape, (
        "Predictive variance shape mismatch."
    )

    assert predictive_std.shape == expected_shape, (
        "Predictive standard deviation shape mismatch."
    )

    print("\nShape Checks: PASSED")

    # ==================================================================
    # FINITE-VALUE CHECKS
    # ==================================================================

    assert torch.isfinite(reconstruction_mean).all(), (
        "Reconstruction mean contains NaN or Inf."
    )

    assert torch.isfinite(aleatoric_variance).all(), (
        "Aleatoric variance contains NaN or Inf."
    )

    assert torch.isfinite(epistemic_variance).all(), (
        "Epistemic variance contains NaN or Inf."
    )

    assert torch.isfinite(predictive_variance).all(), (
        "Predictive variance contains NaN or Inf."
    )

    assert torch.isfinite(predictive_std).all(), (
        "Predictive standard deviation contains NaN or Inf."
    )

    print("Finite-Value Checks: PASSED")

    # ==================================================================
    # NON-NEGATIVE VARIANCE CHECKS
    # ==================================================================

    assert (aleatoric_variance >= 0).all(), (
        "Aleatoric variance contains negative values."
    )

    assert (epistemic_variance >= 0).all(), (
        "Epistemic variance contains negative values."
    )

    assert (predictive_variance >= 0).all(), (
        "Predictive variance contains negative values."
    )

    assert (predictive_std >= 0).all(), (
        "Predictive standard deviation contains negative values."
    )

    print("Non-Negative Variance Checks: PASSED")

    # ==================================================================
    # INDEPENDENT ALEATORIC VARIANCE CHECK
    # ==================================================================

    # Independently reproduce the finalized aleatoric calculation.
    #
    # The log variance is clamped before exponentiation to match
    # the estimator's numerical-safety behavior.
    independent_aleatoric = (
        torch.exp(
            torch.clamp(
                log_variance_samples,
                min=-10.0,
                max=10.0,
            )
        )
        .mean(dim=0)
        .clamp(min=1e-8)
    )

    aleatoric_difference = torch.max(
        torch.abs(
            aleatoric_variance - independent_aleatoric
        )
    ).item()

    print(
        "\nMaximum Difference Between Implemented and "
        f"Independent Aleatoric Variance: "
        f"{aleatoric_difference}"
    )

    assert torch.allclose(
        aleatoric_variance,
        independent_aleatoric,
        atol=1e-6,
        rtol=1e-5,
    ), "Aleatoric variance calculation mismatch."

    print("Independent Aleatoric Variance Check: PASSED")

    # ==================================================================
    # INDEPENDENT EPISTEMIC VARIANCE CHECK
    # ==================================================================

    # Independently calculate population variance across the MC
    # dimension.
    #
    # unbiased=False is intentional because the MC uncertainty
    # estimator treats the available stochastic predictions as the
    # Monte Carlo population used for the predictive estimate.
    independent_epistemic = torch.var(
        reconstruction_samples,
        dim=0,
        unbiased=False,
    ).clamp(min=0.0)

    epistemic_difference = torch.max(
        torch.abs(
            epistemic_variance - independent_epistemic
        )
    ).item()

    print(
        "\nMaximum Difference Between Implemented and "
        f"Independent Epistemic Variance: "
        f"{epistemic_difference}"
    )

    assert torch.allclose(
        epistemic_variance,
        independent_epistemic,
        atol=1e-6,
        rtol=1e-5,
    ), "Epistemic variance calculation mismatch."

    print("Independent Epistemic Variance Check: PASSED")

    # ==================================================================
    # INDEPENDENT PREDICTIVE VARIANCE DECOMPOSITION CHECK
    # ==================================================================

    # Predictive uncertainty is the sum of:
    #
    #     aleatoric variance
    #     +
    #     epistemic variance
    #
    # This is the central uncertainty decomposition being audited.
    independent_predictive = (
        aleatoric_variance
        + epistemic_variance
    ).clamp(min=1e-8)

    predictive_difference = torch.max(
        torch.abs(
            predictive_variance - independent_predictive
        )
    ).item()

    print(
        "\nMaximum Difference Between Implemented and "
        f"Independent Predictive Variance: "
        f"{predictive_difference}"
    )

    assert torch.allclose(
        predictive_variance,
        independent_predictive,
        atol=1e-6,
        rtol=1e-5,
    ), "Predictive variance decomposition mismatch."

    print("Predictive Variance Decomposition Check: PASSED")

    # ==================================================================
    # INDEPENDENT PREDICTIVE STANDARD DEVIATION CHECK
    # ==================================================================

    independent_predictive_std = torch.sqrt(
        torch.clamp(
            independent_predictive,
            min=0.0,
        )
    )

    std_difference = torch.max(
        torch.abs(
            predictive_std - independent_predictive_std
        )
    ).item()

    print(
        "\nMaximum Difference Between Implemented and "
        f"Independent Predictive Standard Deviation: "
        f"{std_difference}"
    )

    assert torch.allclose(
        predictive_std,
        independent_predictive_std,
        atol=1e-6,
        rtol=1e-5,
    ), "Predictive standard deviation mismatch."

    print("Predictive Standard Deviation Check: PASSED")

    # ==================================================================
    # BASIC CONSISTENCY CHECKS
    # ==================================================================

    # Predictive variance must not be smaller than either component.
    assert torch.all(
        predictive_variance >= aleatoric_variance
    ), (
        "Predictive variance is smaller than aleatoric variance."
    )

    assert torch.all(
        predictive_variance >= epistemic_variance
    ), (
        "Predictive variance is smaller than epistemic variance."
    )

    print("Predictive Variance Consistency Checks: PASSED")

    # ==================================================================
    # REPORT UNCERTAINTY STATISTICS
    # ==================================================================

    print("\n" + "=" * 70)
    print("UNCERTAINTY STATISTICS")
    print("=" * 70)

    print("\nAleatoric Variance")
    print(f"Minimum: {aleatoric_variance.min().item()}")
    print(f"Maximum: {aleatoric_variance.max().item()}")
    print(f"Mean:    {aleatoric_variance.mean().item()}")

    print("\nEpistemic Variance")
    print(f"Minimum: {epistemic_variance.min().item()}")
    print(f"Maximum: {epistemic_variance.max().item()}")
    print(f"Mean:    {epistemic_variance.mean().item()}")

    print("\nPredictive Variance")
    print(f"Minimum: {predictive_variance.min().item()}")
    print(f"Maximum: {predictive_variance.max().item()}")
    print(f"Mean:    {predictive_variance.mean().item()}")

    print("\nPredictive Standard Deviation")
    print(f"Minimum: {predictive_std.min().item()}")
    print(f"Maximum: {predictive_std.max().item()}")
    print(f"Mean:    {predictive_std.mean().item()}")

    # ==================================================================
    # FINAL RESULT
    # ==================================================================

    print("\n" + "=" * 70)
    print("TOTAL UNCERTAINTY INTEGRATION TEST: PASSED")
    print("=" * 70)

    print(
        "\nThe complete predictive uncertainty architecture "
        "is data-mode independent."
    )


# ==================================================================
# SCRIPT ENTRY POINT
# ==================================================================

if __name__ == "__main__":
    main()