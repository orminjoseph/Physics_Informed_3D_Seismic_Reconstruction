"""
=========================================================
Integration Test
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Tests integration of:

    1. Network3D
    2. TotalLoss
    3. MCDropout3D
    4. PredictiveUncertaintyEstimator

The test verifies:

    - Forward propagation
    - Tensor shape consistency
    - Physics loss integration
    - Aleatoric uncertainty
    - Epistemic uncertainty
    - Predictive uncertainty
    - Backward propagation
    - Numerical finiteness

Author:
Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from losses.total_loss import TotalLoss


# =========================================================
# TEST CONFIGURATION
# =========================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

NUM_MC_SAMPLES = 20

DX = 25.0
DY = 25.0
DZ = 10.0


# =========================================================
# MAIN INTEGRATION TEST
# =========================================================

def test_integration():

    print()
    print("=" * 60)
    print("TESTING COMPLETE MODEL INTEGRATION")
    print("=" * 60)

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device("cpu")

    print()
    print("Device:", device)

    # =====================================================
    # CREATE INPUT
    # =====================================================

    x = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device,
        requires_grad=True
    )

    # =====================================================
    # CREATE TARGET
    # =====================================================

    target = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    # =====================================================
    # CREATE VELOCITY MODEL
    # =====================================================
    #
    # P-wave velocity in m/s.
    #
    # This is a synthetic integration-test velocity
    # field only. It is NOT a replacement for the
    # actual dataset velocity model.
    # =====================================================

    velocity_model = torch.full(
        (
            BATCH_SIZE,
            CHANNELS,
            DEPTH,
            HEIGHT,
            WIDTH
        ),
        2000.0,
        device=device
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
    ).to(device)

    # =====================================================
    # NORMAL FORWARD PASS
    # =====================================================

    (
        prediction,
        travel_time,
        log_variance
    ) = model(x)

    print()
    print("Network Outputs:")
    print(
        "Prediction     :",
        prediction.shape
    )
    print(
        "Travel Time    :",
        travel_time.shape
    )
    print(
        "Log Variance   :",
        log_variance.shape
    )

    # =====================================================
    # SHAPE CHECKS
    # =====================================================

    expected_shape = (
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH
    )

    assert prediction.shape == expected_shape

    assert travel_time.shape == expected_shape

    assert log_variance.shape == expected_shape

    # =====================================================
    # FINITE OUTPUT CHECK
    # =====================================================

    assert torch.isfinite(
        prediction
    ).all()

    assert torch.isfinite(
        travel_time
    ).all()

    assert torch.isfinite(
        log_variance
    ).all()

    # =====================================================
    # TRAVEL-TIME POSITIVITY
    # =====================================================

    assert torch.all(
        travel_time >= 0.0
    )

    print()
    print("Network forward pass: PASSED")

    # =====================================================
    # TOTAL LOSS
    # =====================================================

    total_loss_function = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    )

    loss_components = total_loss_function(
        prediction=prediction,
        target=target,
        travel_time=travel_time,
        velocity_model=velocity_model,
        log_variance=log_variance
    )

    print()
    print("Loss Components:")

    for name, value in loss_components.items():

        if isinstance(
            value,
            torch.Tensor
        ):

            print(
                f"{name:25s}: "
                f"{value.detach().item():.6e}"
            )

    # =====================================================
    # TOTAL LOSS CHECK
    # =====================================================

    total_loss = loss_components["total"]

    assert torch.isfinite(
        total_loss
    )

    assert total_loss.ndim == 0

    print()
    print("Total loss finite: PASSED")

    # =====================================================
    # BACKWARD PROPAGATION
    # =====================================================

    model.zero_grad()

    if x.grad is not None:

        x.grad.zero_()

    total_loss.backward()

    print()
    print("Backward propagation: PASSED")

    # =====================================================
    # MC DROPOUT
    # =====================================================

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=NUM_MC_SAMPLES
    )

    mc_results = mc_dropout.predict(
        x.detach()
    )

    reconstruction_samples = (
        mc_results[
            "reconstruction_samples"
        ]
    )

    reconstruction_mean = (
        mc_results[
            "reconstruction_mean"
        ]
    )

    reconstruction_epistemic_variance = (
        mc_results[
            "reconstruction_epistemic_variance"
        ]
    )

    log_variance_mean = (
        mc_results[
            "log_variance_mean"
        ]
    )

    # =====================================================
    # MC SAMPLE SHAPE
    # =====================================================

    expected_mc_shape = (
        NUM_MC_SAMPLES,
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH
    )

    assert (
        reconstruction_samples.shape
        ==
        expected_mc_shape
    )

    assert (
        reconstruction_mean.shape
        ==
        expected_shape
    )

    assert (
        reconstruction_epistemic_variance.shape
        ==
        expected_shape
    )

    print()
    print("MC Dropout:")
    print(
        "Samples       :",
        reconstruction_samples.shape
    )
    print(
        "Mean          :",
        reconstruction_mean.shape
    )
    print(
        "Epistemic Var.:",
        reconstruction_epistemic_variance.shape
    )

    # =====================================================
    # MC FINITE CHECK
    # =====================================================

    assert torch.isfinite(
        reconstruction_samples
    ).all()

    assert torch.isfinite(
        reconstruction_mean
    ).all()

    assert torch.isfinite(
        reconstruction_epistemic_variance
    ).all()

    # =====================================================
    # PREDICTIVE UNCERTAINTY
    # =====================================================

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
    )

    uncertainty = uncertainty_estimator(
        log_variance=log_variance_mean,
        mc_predictions=reconstruction_samples
    )

    # =====================================================
    # EXTRACT UNCERTAINTY COMPONENTS
    # =====================================================

    aleatoric_variance = (
        uncertainty[
            "aleatoric_variance"
        ]
    )

    epistemic_variance = (
        uncertainty[
            "epistemic_variance"
        ]
    )

    predictive_variance = (
        uncertainty[
            "predictive_variance"
        ]
    )

    aleatoric_std = (
        uncertainty[
            "aleatoric_std"
        ]
    )

    epistemic_std = (
        uncertainty[
            "epistemic_std"
        ]
    )

    predictive_std = (
        uncertainty[
            "predictive_std"
        ]
    )

    # =====================================================
    # UNCERTAINTY SHAPE CHECKS
    # =====================================================

    assert (
        aleatoric_variance.shape
        ==
        expected_shape
    )

    assert (
        epistemic_variance.shape
        ==
        expected_shape
    )

    assert (
        predictive_variance.shape
        ==
        expected_shape
    )

    assert (
        aleatoric_std.shape
        ==
        expected_shape
    )

    assert (
        epistemic_std.shape
        ==
        expected_shape
    )

    assert (
        predictive_std.shape
        ==
        expected_shape
    )

    # =====================================================
    # UNCERTAINTY FINITE CHECK
    # =====================================================

    assert torch.isfinite(
        aleatoric_variance
    ).all()

    assert torch.isfinite(
        epistemic_variance
    ).all()

    assert torch.isfinite(
        predictive_variance
    ).all()

    assert torch.isfinite(
        predictive_std
    ).all()

    # =====================================================
    # NON-NEGATIVITY CHECK
    # =====================================================

    assert torch.all(
        aleatoric_variance >= 0.0
    )

    assert torch.all(
        epistemic_variance >= 0.0
    )

    assert torch.all(
        predictive_variance >= 0.0
    )

    # =====================================================
    # PREDICTIVE VARIANCE CONSISTENCY
    # =====================================================
    #
    # Predictive variance:
    #
    #     Var_predictive
    #
    #       =
    #
    #     Var_aleatoric
    #
    #       +
    #
    #     Var_epistemic
    #
    # =====================================================

    expected_predictive_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    assert torch.allclose(
        predictive_variance,
        expected_predictive_variance,
        atol=1.0e-6,
        rtol=1.0e-5
    )

    # =====================================================
    # STANDARD DEVIATION CONSISTENCY
    # =====================================================

    assert torch.allclose(
        predictive_std,
        torch.sqrt(
            predictive_variance
        ),
        atol=1.0e-6,
        rtol=1.0e-5
    )

    print()
    print("Predictive Uncertainty:")
    print(
        "Aleatoric Var.:",
        aleatoric_variance.shape
    )
    print(
        "Epistemic Var.:",
        epistemic_variance.shape
    )
    print(
        "Predictive Var.:",
        predictive_variance.shape
    )

    print()
    print(
        "Predictive variance consistency: PASSED"
    )

    # =====================================================
    # DISPLAY SUMMARY STATISTICS
    # =====================================================

    print()
    print("Uncertainty Statistics:")

    print(
        "Aleatoric variance mean :",
        aleatoric_variance.mean().item()
    )

    print(
        "Epistemic variance mean :",
        epistemic_variance.mean().item()
    )

    print(
        "Predictive variance mean:",
        predictive_variance.mean().item()
    )

    print(
        "Predictive std mean     :",
        predictive_std.mean().item()
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()
    print("=" * 60)
    print(
        "COMPLETE MODEL INTEGRATION TEST: PASSED"
    )
    print("=" * 60)


# =========================================================
# MAIN
# =========================================================

def main():

    test_integration()


if __name__ == "__main__":

    main()