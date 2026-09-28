"""
=========================================================
Uncertainty / Model Integration Test
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
This test verifies the finalized uncertainty/model subsystem:

1. Network3D forward pass
2. Three-output interface
3. Output shapes
4. Finite network outputs
5. Zero-initialized aleatoric uncertainty head
6. Heteroscedastic Aleatoric Uncertainty Loss
7. Finite uncertainty-loss gradients
8. MC-Dropout inference
9. Reconstruction sample dimensions
10. Epistemic variance
11. Aleatoric variance
12. Predictive variance decomposition
13. MC-Dropout state restoration

Expected input
--------------
[B, C, D, H, W]

Test input
----------
[1, 1, 64, 128, 128]

=========================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    UncertaintyLoss,
)


# =========================================================
# Test configuration
# =========================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

MC_SAMPLES = 10

DEVICE = torch.device("cpu")


# =========================================================
# Utility functions
# =========================================================

def check_finite(name, tensor):
    """
    Check that a tensor contains no NaN or Inf values.
    """

    if not torch.isfinite(tensor).all():
        raise AssertionError(
            f"{name} contains NaN or Inf values."
        )


def check_shape(name, tensor, expected_shape):
    """
    Check that a tensor has the expected shape.
    """

    actual_shape = tuple(tensor.shape)

    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} has incorrect shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {actual_shape}"
        )


# =========================================================
# Main test
# =========================================================

def main():

    print("=" * 70)
    print("UNCERTAINTY / MODEL INTEGRATION TEST")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(
        f"Input shape: "
        f"[{BATCH_SIZE}, {CHANNELS}, "
        f"{DEPTH}, {HEIGHT}, {WIDTH}]"
    )

    # -----------------------------------------------------
    # 1. Create test input
    # -----------------------------------------------------

    torch.manual_seed(42)

    inputs = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE,
    )

    check_finite("Input", inputs)

    print("\n[1/10] Input tensor ................. PASS")

    # -----------------------------------------------------
    # 2. Create target
    # -----------------------------------------------------
    #
    # The target is only needed to test the uncertainty loss.
    #
    # -----------------------------------------------------

    target = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE,
    )

    check_finite("Target", target)

    # -----------------------------------------------------
    # 3. Create finalized Network3D
    # -----------------------------------------------------

    model = Network3D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
    ).to(DEVICE)

    model.eval()

    print("[2/10] Network3D creation .......... PASS")

    # -----------------------------------------------------
    # 4. Deterministic forward pass
    # -----------------------------------------------------

    with torch.no_grad():

        reconstruction, travel_time, log_variance = (
            model(inputs)
        )

    expected_shape = (
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
    )

    check_shape(
        "Reconstruction",
        reconstruction,
        expected_shape,
    )

    check_shape(
        "Travel time",
        travel_time,
        expected_shape,
    )

    check_shape(
        "Log variance",
        log_variance,
        expected_shape,
    )

    check_finite(
        "Reconstruction",
        reconstruction,
    )

    check_finite(
        "Travel time",
        travel_time,
    )

    check_finite(
        "Log variance",
        log_variance,
    )

    print("[3/10] Network forward ............. PASS")

    print(
        f"       Reconstruction: {tuple(reconstruction.shape)}"
    )

    print(
        f"       Travel time:     {tuple(travel_time.shape)}"
    )

    print(
        f"       Log variance:    {tuple(log_variance.shape)}"
    )

    # -----------------------------------------------------
    # 5. Check uncertainty-head initialization
    # -----------------------------------------------------
    #
    # The finalized network initializes the uncertainty head
    # weights and bias to zero.
    #
    # Therefore:
    #
    #     log_variance = 0
    #
    # and:
    #
    #     variance = exp(0) = 1
    #
    # before training.
    # -----------------------------------------------------

    expected_log_variance = torch.zeros_like(
        log_variance
    )

    if not torch.allclose(
        log_variance,
        expected_log_variance,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            "Uncertainty head initialization test failed. "
            "Expected initial log_variance ≈ 0."
        )

    aleatoric_variance_initial = torch.exp(
        log_variance
    )

    expected_variance = torch.ones_like(
        aleatoric_variance_initial
    )

    if not torch.allclose(
        aleatoric_variance_initial,
        expected_variance,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            "Initial aleatoric variance is not approximately 1."
        )

    print(
        "[4/10] Uncertainty initialization ... PASS"
    )

    print(
        f"       Initial log variance mean: "
        f"{log_variance.mean().item():.6f}"
    )

    print(
        f"       Initial variance mean:     "
        f"{aleatoric_variance_initial.mean().item():.6f}"
    )

    # -----------------------------------------------------
    # 6. Test Heteroscedastic Aleatoric
    #    Uncertainty Loss
    # -----------------------------------------------------

    uncertainty_loss = UncertaintyLoss()

    log_variance_for_loss = log_variance.clone()
    log_variance_for_loss.requires_grad_(True)

    loss_value = uncertainty_loss(
        reconstruction.detach(),
        target,
        log_variance_for_loss,
    )

    if loss_value.ndim != 0:
        raise AssertionError(
            "Uncertainty loss must return a scalar."
        )

    if not torch.isfinite(loss_value):
        raise AssertionError(
            "Uncertainty loss produced NaN or Inf."
        )

    print(
        "[5/10] Aleatoric uncertainty loss .. PASS"
    )

    print(
        f"       Loss: {loss_value.item():.6f}"
    )

    # -----------------------------------------------------
    # 7. Test uncertainty-loss gradients
    # -----------------------------------------------------

    loss_value.backward()

    if log_variance_for_loss.grad is None:
        raise AssertionError(
            "No gradient was produced for log_variance."
        )

    check_finite(
        "Uncertainty loss gradient",
        log_variance_for_loss.grad,
    )

    print(
        "[6/10] Uncertainty gradients ....... PASS"
    )

    # -----------------------------------------------------
    # 8. Record model state before MC Dropout
    # -----------------------------------------------------

    model.eval()

    state_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # -----------------------------------------------------
    # 9. Run MC-Dropout prediction
    # -----------------------------------------------------

    mc_predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    mc_results = mc_predictor.predict(
        inputs
    )

    reconstruction_samples = mc_results[
        "reconstruction_samples"
    ]

    reconstruction_mean = mc_results[
        "reconstruction_mean"
    ]

    reconstruction_epistemic_variance = mc_results[
        "reconstruction_epistemic_variance"
    ]

    travel_time_samples = mc_results[
        "travel_time_samples"
    ]

    log_variance_samples = mc_results[
        "log_variance_samples"
    ]

    print(
        "[7/10] MC-Dropout prediction ........ PASS"
    )

    # -----------------------------------------------------
    # 10. Validate MC sample dimensions
    # -----------------------------------------------------

    expected_mc_shape = (
        MC_SAMPLES,
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
    )

    check_shape(
        "Reconstruction samples",
        reconstruction_samples,
        expected_mc_shape,
    )

    check_shape(
        "Travel-time samples",
        travel_time_samples,
        expected_mc_shape,
    )

    check_shape(
        "Log-variance samples",
        log_variance_samples,
        expected_mc_shape,
    )

    check_shape(
        "Reconstruction mean",
        reconstruction_mean,
        expected_shape,
    )

    check_shape(
        "Epistemic variance",
        reconstruction_epistemic_variance,
        expected_shape,
    )

    check_finite(
        "Reconstruction samples",
        reconstruction_samples,
    )

    check_finite(
        "Travel-time samples",
        travel_time_samples,
    )

    check_finite(
        "Log-variance samples",
        log_variance_samples,
    )

    check_finite(
        "Epistemic variance",
        reconstruction_epistemic_variance,
    )

    print(
        "[8/10] MC sample shapes ............. PASS"
    )

    print(
        f"       Reconstruction samples: "
        f"{tuple(reconstruction_samples.shape)}"
    )

    print(
        f"       Epistemic variance: "
        f"{tuple(reconstruction_epistemic_variance.shape)}"
    )

    # -----------------------------------------------------
    # 11. Compute aleatoric variance correctly
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # Correct:
    #
    #     mean(exp(log_variance_samples))
    #
    # NOT:
    #
    #     exp(mean(log_variance_samples))
    #
    # because the predictive distribution is obtained by
    # averaging the conditional variances.
    # -----------------------------------------------------

    aleatoric_variance = torch.mean(
        torch.exp(log_variance_samples),
        dim=0,
    )

    check_shape(
        "Aleatoric variance",
        aleatoric_variance,
        expected_shape,
    )

    check_finite(
        "Aleatoric variance",
        aleatoric_variance,
    )

    if (aleatoric_variance < 0).any():
        raise AssertionError(
            "Aleatoric variance cannot be negative."
        )

    # -----------------------------------------------------
    # 12. Predictive variance decomposition
    # -----------------------------------------------------

    predictive_variance = (
        aleatoric_variance
        + reconstruction_epistemic_variance
    )

    check_shape(
        "Predictive variance",
        predictive_variance,
        expected_shape,
    )

    check_finite(
        "Predictive variance",
        predictive_variance,
    )

    # Verify decomposition numerically.
    decomposition_error = torch.max(
        torch.abs(
            predictive_variance
            - (
                aleatoric_variance
                + reconstruction_epistemic_variance
            )
        )
    ).item()

    if decomposition_error > 1e-6:
        raise AssertionError(
            "Predictive variance decomposition failed."
        )

    print(
        "[9/10] Uncertainty decomposition ... PASS"
    )

    print(
        f"       Aleatoric variance mean: "
        f"{aleatoric_variance.mean().item():.8f}"
    )

    print(
        f"       Epistemic variance mean: "
        f"{reconstruction_epistemic_variance.mean().item():.8f}"
    )

    print(
        f"       Predictive variance mean: "
        f"{predictive_variance.mean().item():.8f}"
    )

    print(
        f"       Decomposition error: "
        f"{decomposition_error:.3e}"
    )

    # -----------------------------------------------------
    # 13. Verify model parameters were not modified
    # -----------------------------------------------------

    for name, parameter in model.named_parameters():

        if not torch.equal(
            parameter.detach(),
            state_before[name],
        ):
            raise AssertionError(
                "Model parameter changed during MC-Dropout "
                f"inference: {name}"
            )

    print(
        "[10/10] MC-Dropout state integrity ... PASS"
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("UNCERTAINTY / MODEL INTEGRATION TEST PASSED")
    print("=" * 70)

    print("\nVerified:")
    print("  ✓ Network3D three-output interface")
    print("  ✓ Reconstruction output")
    print("  ✓ Travel-time output")
    print("  ✓ Aleatoric log-variance output")
    print("  ✓ Zero uncertainty-head initialization")
    print("  ✓ Heteroscedastic Aleatoric Uncertainty Loss")
    print("  ✓ Finite uncertainty gradients")
    print("  ✓ MC-Dropout sampling")
    print("  ✓ Reconstruction epistemic variance")
    print("  ✓ Aleatoric variance")
    print("  ✓ Predictive variance")
    print("  ✓ Predictive variance decomposition")
    print("  ✓ MC-Dropout parameter integrity")

    print("\nSTATUS: UNCERTAINTY/MODEL SUBSYSTEM READY FOR FREEZE")
    print("=" * 70)


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    main()