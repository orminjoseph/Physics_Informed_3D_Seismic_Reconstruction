"""
=========================================================
Test Predictive Uncertainty Estimator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Tests the complete uncertainty decomposition:

    Aleatoric uncertainty
            +
    Epistemic uncertainty
            =
    Predictive uncertainty

Mathematical relationship
-------------------------

    sigma²_predictive
        =
    sigma²_aleatoric
        +
    sigma²_epistemic

where:

    sigma²_aleatoric
        = exp(log_variance)

and:

    sigma²_epistemic
        = Var(MC Dropout predictions)

Tensor convention
-----------------

Network output:

    [B, C, D, H, W]

MC Dropout predictions:

    [N, B, C, D, H, W]

where:

    N = number of stochastic MC samples

Author: Ormin Joseph
=========================================================
"""

import torch

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)


def test_predictive_uncertainty():

    print()

    print("=" * 60)

    print("TESTING PREDICTIVE UNCERTAINTY ESTIMATOR")

    print("=" * 60)

    # =====================================================
    # TEST CONFIGURATION
    # =====================================================

    batch_size = 1

    channels = 1

    depth = 16

    height = 32

    width = 32

    num_mc_samples = 20

    # =====================================================
    # CREATE SYNTHETIC LOG-VARIANCE
    # =====================================================

    log_variance = torch.randn(
        batch_size,
        channels,
        depth,
        height,
        width
    )

    # =====================================================
    # CREATE SYNTHETIC MC DROPOUT PREDICTIONS
    # =====================================================

    mc_predictions = torch.randn(
        num_mc_samples,
        batch_size,
        channels,
        depth,
        height,
        width
    )

    # =====================================================
    # CREATE ESTIMATOR
    # =====================================================

    estimator = PredictiveUncertaintyEstimator()

    # =====================================================
    # FORWARD PASS
    # =====================================================

    result = estimator(
        log_variance,
        mc_predictions
    )

    # =====================================================
    # DISPLAY INPUT SHAPES
    # =====================================================

    print()

    print("Log Variance Shape:")
    print(log_variance.shape)

    print()

    print("MC Predictions Shape:")
    print(mc_predictions.shape)

    # =====================================================
    # DISPLAY OUTPUT SHAPES
    # =====================================================

    print()

    print("Aleatoric Variance Shape:")
    print(
        result["aleatoric_variance"].shape
    )

    print()

    print("Epistemic Variance Shape:")
    print(
        result["epistemic_variance"].shape
    )

    print()

    print("Predictive Variance Shape:")
    print(
        result["predictive_variance"].shape
    )

    print()

    print("Aleatoric Standard Deviation Shape:")
    print(
        result["aleatoric_std"].shape
    )

    print()

    print("Epistemic Standard Deviation Shape:")
    print(
        result["epistemic_std"].shape
    )

    print()

    print("Predictive Standard Deviation Shape:")
    print(
        result["predictive_std"].shape
    )

    # =====================================================
    # EXPECTED SHAPE
    # =====================================================

    expected_shape = (
        batch_size,
        channels,
        depth,
        height,
        width
    )

    # =====================================================
    # SHAPE TESTS
    # =====================================================

    assert (
        result["aleatoric_variance"].shape
        == expected_shape
    )

    assert (
        result["epistemic_variance"].shape
        == expected_shape
    )

    assert (
        result["predictive_variance"].shape
        == expected_shape
    )

    assert (
        result["aleatoric_std"].shape
        == expected_shape
    )

    assert (
        result["epistemic_std"].shape
        == expected_shape
    )

    assert (
        result["predictive_std"].shape
        == expected_shape
    )

    # =====================================================
    # EXPECTED ALEATORIC VARIANCE
    # =====================================================

    expected_aleatoric_variance = torch.exp(
        torch.clamp(
            log_variance,
            min=-10.0,
            max=10.0
        )
    )

    # =====================================================
    # EXPECTED EPISTEMIC VARIANCE
    # =====================================================

    expected_epistemic_variance = (
        mc_predictions.var(
            dim=0,
            unbiased=False
        )
    )

    # =====================================================
    # EXPECTED PREDICTIVE VARIANCE
    # =====================================================

    expected_predictive_variance = (
        expected_aleatoric_variance
        +
        expected_epistemic_variance
    )

    # =====================================================
    # MATHEMATICAL CONSISTENCY
    # =====================================================

    assert torch.allclose(
        result["aleatoric_variance"],
        expected_aleatoric_variance,
        atol=1.0e-6
    )

    assert torch.allclose(
        result["epistemic_variance"],
        expected_epistemic_variance,
        atol=1.0e-6
    )

    assert torch.allclose(
        result["predictive_variance"],
        expected_predictive_variance,
        atol=1.0e-6
    )

    # =====================================================
    # STANDARD DEVIATION TEST
    # =====================================================

    expected_aleatoric_std = torch.sqrt(
        expected_aleatoric_variance
    )

    expected_epistemic_std = torch.sqrt(
        expected_epistemic_variance
    )

    expected_predictive_std = torch.sqrt(
        expected_predictive_variance
    )

    assert torch.allclose(
        result["aleatoric_std"],
        expected_aleatoric_std,
        atol=1.0e-6
    )

    assert torch.allclose(
        result["epistemic_std"],
        expected_epistemic_std,
        atol=1.0e-6
    )

    assert torch.allclose(
        result["predictive_std"],
        expected_predictive_std,
        atol=1.0e-6
    )

    # =====================================================
    # POSITIVITY TEST
    # =====================================================

    assert torch.all(
        result["aleatoric_variance"] >= 0.0
    )

    assert torch.all(
        result["epistemic_variance"] >= 0.0
    )

    assert torch.all(
        result["predictive_variance"] >= 0.0
    )

    assert torch.all(
        result["aleatoric_std"] >= 0.0
    )

    assert torch.all(
        result["epistemic_std"] >= 0.0
    )

    assert torch.all(
        result["predictive_std"] >= 0.0
    )

    # =====================================================
    # FINITE-VALUE TEST
    # =====================================================

    assert torch.isfinite(
        result["aleatoric_variance"]
    ).all()

    assert torch.isfinite(
        result["epistemic_variance"]
    ).all()

    assert torch.isfinite(
        result["predictive_variance"]
    ).all()

    assert torch.isfinite(
        result["aleatoric_std"]
    ).all()

    assert torch.isfinite(
        result["epistemic_std"]
    ).all()

    assert torch.isfinite(
        result["predictive_std"]
    ).all()

    # =====================================================
    # PREDICTIVE VARIANCE DOMINANCE TEST
    # =====================================================

    assert torch.all(
        result["predictive_variance"]
        >=
        result["aleatoric_variance"]
    )

    assert torch.all(
        result["predictive_variance"]
        >=
        result["epistemic_variance"]
    )

    # =====================================================
    # DISPLAY STATISTICS
    # =====================================================

    print()

    print("Aleatoric Variance Minimum:")
    print(
        result["aleatoric_variance"].min().item()
    )

    print()

    print("Aleatoric Variance Maximum:")
    print(
        result["aleatoric_variance"].max().item()
    )

    print()

    print("Aleatoric Variance Mean:")
    print(
        result["aleatoric_variance"].mean().item()
    )

    print()

    print("Epistemic Variance Minimum:")
    print(
        result["epistemic_variance"].min().item()
    )

    print()

    print("Epistemic Variance Maximum:")
    print(
        result["epistemic_variance"].max().item()
    )

    print()

    print("Epistemic Variance Mean:")
    print(
        result["epistemic_variance"].mean().item()
    )

    print()

    print("Predictive Variance Minimum:")
    print(
        result["predictive_variance"].min().item()
    )

    print()

    print("Predictive Variance Maximum:")
    print(
        result["predictive_variance"].max().item()
    )

    print()

    print("Predictive Variance Mean:")
    print(
        result["predictive_variance"].mean().item()
    )

    print()

    print("Predictive Standard Deviation Minimum:")
    print(
        result["predictive_std"].min().item()
    )

    print()

    print("Predictive Standard Deviation Maximum:")
    print(
        result["predictive_std"].max().item()
    )

    # =====================================================
    # SUCCESS
    # =====================================================

    print()

    print("=" * 60)

    print(
        "PREDICTIVE UNCERTAINTY TEST: PASSED"
    )

    print("=" * 60)


def main():

    test_predictive_uncertainty()


if __name__ == "__main__":

    main()