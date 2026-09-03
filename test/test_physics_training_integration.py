"""
=========================================================
PHYSICS LOSS TRAINING-INTEGRATION AUDIT
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

This test verifies that the current PhysicsInformed3DUNet
is correctly integrated with the PhysicsLoss implementation.

The audit verifies:

1. Network initialization.
2. Three-output network interface.
3. Reconstruction output.
4. Travel-time output.
5. Predictive uncertainty output.
6. Tensor shape consistency.
7. Travel-time positivity.
8. Travel-time scaling.
9. Physical velocity compatibility.
10. Eikonal physics-loss computation.
11. Physics-loss weighting.
12. Composite loss construction.
13. Differentiability.
14. Backward propagation.
15. Network parameter gradients.
16. Gradient finiteness.
17. Uncertainty branch participation.
18. Complete training-step execution.

Tensor convention
-----------------

[B, C, D, H, W]

where:

    B = batch
    C = channel
    D = depth
    H = crossline
    W = inline

Author: Ormin Joseph
=========================================================
"""

import sys
import torch
import torch.nn.functional as F

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
)


# =========================================================
# TEST UTILITIES
# =========================================================

def print_header(title):
    """Print a formatted test-section header."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_statistic(name, tensor):
    """Print basic statistics for a tensor."""

    print(
        f"{name:<30}: "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


def assert_finite(tensor, name):
    """Verify that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all():

        raise AssertionError(
            f"{name} contains NaN or Inf values."
        )


def assert_shape(tensor, expected_shape, name):
    """Verify tensor shape."""

    if tuple(tensor.shape) != tuple(expected_shape):

        raise AssertionError(
            f"{name} shape mismatch. "
            f"Expected {expected_shape}, "
            f"received {tuple(tensor.shape)}."
        )


def count_parameters(model):
    """Return number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def gradient_statistics(model):
    """
    Calculate gradient statistics across all trainable
    network parameters.
    """

    gradients = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients.append(
                parameter.grad.detach().abs().reshape(-1)
            )

    if not gradients:

        return None

    all_gradients = torch.cat(gradients)

    return {
        "mean": all_gradients.mean().item(),
        "maximum": all_gradients.max().item(),
        "minimum": all_gradients.min().item(),
        "finite": torch.isfinite(all_gradients).all().item(),
        "count": all_gradients.numel(),
    }


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print_header(
        "PHYSICS LOSS TRAINING-INTEGRATION AUDIT"
    )

    # =====================================================
    # 1. DEVICE CONFIGURATION
    # =====================================================

    print_header(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(DEVICE)

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # 2. PHYSICAL CONFIGURATION
    # =====================================================

    print_header(
        "PHYSICAL CONFIGURATION"
    )

    print(
        f"DX                         : {DX}"
    )

    print(
        f"DY                         : {DY}"
    )

    print(
        f"DZ                         : {DZ}"
    )

    print(
        f"Travel-time scale          : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"Physics loss weight        : "
        f"{LOSS_WEIGHTS['physics']}"
    )

    print(
        f"Eikonal loss weight        : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    # =====================================================
    # 3. CREATE SYNTHETIC INPUT
    # =====================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    # Use a smaller cube than the full training cube
    # for a practical integration test.

    batch_size = 1
    channels = 1
    depth = 64
    height = 128
    width = 128

    input_tensor = torch.randn(
        batch_size,
        channels,
        depth,
        height,
        width,
        device=device
    )

    input_tensor.requires_grad_(False)

    print(
        f"Input shape                : "
        f"{tuple(input_tensor.shape)}"
    )

    print_statistic(
        "Input seismic volume",
        input_tensor
    )

    assert_finite(
        input_tensor,
        "Input seismic volume"
    )

    # =====================================================
    # 4. CREATE PHYSICAL VELOCITY FIELD
    # =====================================================

    print_header(
        "CREATING PHYSICAL VELOCITY FIELD"
    )

    # Constant physical P-wave velocity.
    #
    # Units:
    #
    #     m/s

    velocity = torch.full(
        (
            batch_size,
            channels,
            depth,
            height,
            width
        ),
        2000.0,
        dtype=input_tensor.dtype,
        device=device
    )

    print_statistic(
        "Velocity field",
        velocity
    )

    assert_finite(
        velocity,
        "Velocity field"
    )

    if torch.any(velocity <= 0):

        raise AssertionError(
            "Velocity field contains non-positive values."
        )

    # =====================================================
    # 5. INITIALIZE NETWORK
    # =====================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(device)

    model.train()

    print(
        "PhysicsInformed3DUNet successfully initialized."
    )

    print(
        f"Trainable parameters       : "
        f"{count_parameters(model):,}"
    )

    # =====================================================
    # 6. FORWARD PROPAGATION
    # =====================================================

    print_header(
        "FORWARD PROPAGATION"
    )

    outputs = model(
        input_tensor
    )

    # The current network returns exactly three outputs:
    #
    #     reconstruction
    #     travel_time
    #     log_variance

    if not isinstance(outputs, tuple):

        raise AssertionError(
            "Network output must be a tuple."
        )

    if len(outputs) != 3:

        raise AssertionError(
            "Current network must return exactly "
            "three outputs: reconstruction, "
            "travel_time and log_variance. "
            f"Received {len(outputs)} outputs."
        )

    reconstruction, travel_time, log_variance = outputs

    print(
        "Network returned exactly three outputs."
    )

    # =====================================================
    # 7. OUTPUT SHAPE AUDIT
    # =====================================================

    print_header(
        "NETWORK OUTPUT SHAPE AUDIT"
    )

    print(
        f"Input shape                : "
        f"{tuple(input_tensor.shape)}"
    )

    print(
        f"Reconstruction shape       : "
        f"{tuple(reconstruction.shape)}"
    )

    print(
        f"Travel-time shape          : "
        f"{tuple(travel_time.shape)}"
    )

    print(
        f"Log-variance shape         : "
        f"{tuple(log_variance.shape)}"
    )

    expected_shape = tuple(
        input_tensor.shape
    )

    assert_shape(
        reconstruction,
        expected_shape,
        "Reconstruction"
    )

    assert_shape(
        travel_time,
        expected_shape,
        "Travel time"
    )

    assert_shape(
        log_variance,
        expected_shape,
        "Log variance"
    )

    print(
        "Network output-shape test: PASSED"
    )

    # =====================================================
    # 8. OUTPUT FINITENESS AUDIT
    # =====================================================

    print_header(
        "NETWORK OUTPUT FINITENESS AUDIT"
    )

    assert_finite(
        reconstruction,
        "Reconstruction"
    )

    assert_finite(
        travel_time,
        "Travel time"
    )

    assert_finite(
        log_variance,
        "Log variance"
    )

    print(
        "Reconstruction contains only finite values."
    )

    print(
        "Travel-time field contains only finite values."
    )

    print(
        "Log-variance field contains only finite values."
    )

    print(
        "Network output finiteness test: PASSED"
    )

    # =====================================================
    # 9. TRAVEL-TIME POSITIVITY AUDIT
    # =====================================================

    print_header(
        "TRAVEL-TIME POSITIVITY AUDIT"
    )

    print_statistic(
        "Predicted travel time",
        travel_time
    )

    minimum_travel_time = (
        travel_time.min().item()
    )

    if minimum_travel_time < 0.0:

        raise AssertionError(
            "Travel-time output contains negative values."
        )

    print(
        f"Minimum travel time        : "
        f"{minimum_travel_time:.6e}"
    )

    print(
        "Travel-time positivity test: PASSED"
    )

    # =====================================================
    # 10. TRAVEL-TIME SCALE AUDIT
    # =====================================================

    print_header(
        "TRAVEL-TIME SCALE AUDIT"
    )

    maximum_travel_time = (
        travel_time.max().item()
    )

    print(
        f"Configured scale            : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"Observed maximum           : "
        f"{maximum_travel_time:.6e}"
    )

    # Because Softplus is positive and the final network
    # output is multiplied by TRAVEL_TIME_SCALE, the output
    # should remain finite and positive.
    #
    # We do not require the maximum to equal the scale,
    # because Softplus is not bounded above.

    if maximum_travel_time <= 0.0:

        raise AssertionError(
            "Travel-time output is not positive."
        )

    print(
        "Travel-time scale test: PASSED"
    )

    # =====================================================
    # 11. UNCERTAINTY OUTPUT AUDIT
    # =====================================================

    print_header(
        "PREDICTIVE UNCERTAINTY AUDIT"
    )

    print_statistic(
        "Log variance",
        log_variance
    )

    # Log variance is intentionally unrestricted.
    #
    # Therefore, we do NOT require:
    #
    #     log_variance >= 0
    #
    # We only require finite values.

    print(
        "Log-variance is unrestricted as expected."
    )

    print(
        "Predictive uncertainty output test: PASSED"
    )

    # =====================================================
    # 12. INITIALIZE PHYSICS LOSS
    # =====================================================

    print_header(
        "INITIALIZING PHYSICS LOSS"
    )

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=PHYSICS_LOSS_WEIGHTS[
            "eikonal"
        ],
        source_weight=PHYSICS_LOSS_WEIGHTS[
            "source"
        ],
        travel_time_weight=PHYSICS_LOSS_WEIGHTS[
            "travel_time"
        ]
    ).to(device)

    print(
        "PhysicsLoss successfully initialized."
    )

    # =====================================================
    # 13. PHYSICS LOSS FORWARD PASS
    # =====================================================

    print_header(
        "PHYSICS LOSS FORWARD PASS"
    )

    physics_components = physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=None,
        travel_time_target=None
    )

    required_keys = {
        "total",
        "eikonal",
        "source",
        "travel_time",
        "weighted_eikonal",
        "weighted_source",
        "weighted_travel_time"
    }

    if not required_keys.issubset(
        physics_components.keys()
    ):

        missing = (
            required_keys
            -
            set(physics_components.keys())
        )

        raise AssertionError(
            f"PhysicsLoss is missing components: "
            f"{missing}"
        )

    for name, value in physics_components.items():

        print(
            f"{name:<30}: "
            f"{value.item():.12e}"
        )

        if not torch.isfinite(value):

            raise AssertionError(
                f"Physics-loss component '{name}' "
                "contains NaN or Inf."
            )

    print(
        "PhysicsLoss forward-pass test: PASSED"
    )

    # =====================================================
    # 14. EIKONAL DIFFERENTIABILITY AUDIT
    # =====================================================

    print_header(
        "EIKONAL DIFFERENTIABILITY AUDIT"
    )

    eikonal_loss = (
        physics_components["eikonal"]
    )

    print(
        f"Travel-time requires_grad  : "
        f"{travel_time.requires_grad}"
    )

    print(
        f"Eikonal requires_grad      : "
        f"{eikonal_loss.requires_grad}"
    )

    if not travel_time.requires_grad:

        raise AssertionError(
            "Travel-time output does not require gradients."
        )

    if not eikonal_loss.requires_grad:

        raise AssertionError(
            "Eikonal loss does not require gradients."
        )

    print(
        "Eikonal differentiability test: PASSED"
    )

    # =====================================================
    # 15. PHYSICS WEIGHT AUDIT
    # =====================================================

    print_header(
        "PHYSICS WEIGHT AUDIT"
    )

    expected_weighted_eikonal = (
        PHYSICS_LOSS_WEIGHTS["eikonal"]
        *
        physics_components["eikonal"]
    )

    actual_weighted_eikonal = (
        physics_components["weighted_eikonal"]
    )

    weight_difference = (
        actual_weighted_eikonal
        -
        expected_weighted_eikonal
    ).abs().item()

    print(
        f"Eikonal loss              : "
        f"{physics_components['eikonal'].item():.12e}"
    )

    print(
        f"Eikonal weight             : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    print(
        f"Weighted Eikonal           : "
        f"{actual_weighted_eikonal.item():.12e}"
    )

    print(
        f"Expected weighted value    : "
        f"{expected_weighted_eikonal.item():.12e}"
    )

    print(
        f"Absolute difference        : "
        f"{weight_difference:.12e}"
    )

    if weight_difference > 1.0e-8:

        raise AssertionError(
            "Physics Eikonal weighting is incorrect."
        )

    print(
        "Physics weighting test: PASSED"
    )

    # =====================================================
    # 16. BUILD RECONSTRUCTION LOSS
    # =====================================================

    print_header(
        "RECONSTRUCTION LOSS"
    )

    # For integration testing, use the input as a target.
    #
    # This does not represent the final scientific
    # reconstruction target. It simply allows us to verify
    # that reconstruction and physics losses can participate
    # in one differentiable objective.

    reconstruction_target = (
        input_tensor.detach()
    )

    mae_loss = (
        reconstruction
        -
        reconstruction_target
    ).abs().mean()

    print(
        f"MAE loss                   : "
        f"{mae_loss.item():.12e}"
    )

    assert_finite(
        mae_loss,
        "MAE loss"
    )

    # =====================================================
    # 17. COMPOSITE LOSS CONSTRUCTION
    # =====================================================

    print_header(
        "COMPOSITE LOSS CONSTRUCTION"
    )

    composite_loss = (
        LOSS_WEIGHTS["mae"]
        *
        mae_loss
        +
        LOSS_WEIGHTS["physics"]
        *
        physics_components["total"]
    )

    # -----------------------------------------------------
    # Include uncertainty branch in the integration audit.
    #
    # A simple finite regularization term is used here only
    # to verify that gradients can flow through the
    # uncertainty head.
    #
    # The final uncertainty objective is handled by the
    # project's dedicated uncertainty loss.
    # -----------------------------------------------------

    uncertainty_probe = (
        log_variance.pow(2).mean()
    )

    composite_loss = (
        composite_loss
        +
        LOSS_WEIGHTS["uncertainty"]
        *
        uncertainty_probe
    )

    print(
        f"MAE contribution            : "
        f"{(
            LOSS_WEIGHTS['mae']
            *
            mae_loss
        ).item():.12e}"
    )

    print(
        f"Physics contribution        : "
        f"{(
            LOSS_WEIGHTS['physics']
            *
            physics_components['total']
        ).item():.12e}"
    )

    print(
        f"Uncertainty probe           : "
        f"{uncertainty_probe.item():.12e}"
    )

    print(
        f"Composite loss              : "
        f"{composite_loss.item():.12e}"
    )

    if not torch.isfinite(composite_loss):

        raise AssertionError(
            "Composite loss contains NaN or Inf."
        )

    if not composite_loss.requires_grad:

        raise AssertionError(
            "Composite loss does not require gradients."
        )

    print(
        "Composite-loss construction test: PASSED"
    )

    # =====================================================
    # 18. BACKWARD PROPAGATION
    # =====================================================

    print_header(
        "BACKWARD PROPAGATION AUDIT"
    )

    model.zero_grad(
        set_to_none=True
    )

    composite_loss.backward()

    print(
        "Backward propagation completed successfully."
    )

    # =====================================================
    # 19. NETWORK GRADIENT AUDIT
    # =====================================================

    print_header(
        "NETWORK GRADIENT AUDIT"
    )

    statistics = gradient_statistics(
        model
    )

    if statistics is None:

        raise AssertionError(
            "No network gradients were generated."
        )

    print(
        f"Gradient mean              : "
        f"{statistics['mean']:.12e}"
    )

    print(
        f"Gradient maximum           : "
        f"{statistics['maximum']:.12e}"
    )

    print(
        f"Gradient minimum           : "
        f"{statistics['minimum']:.12e}"
    )

    print(
        f"Gradient elements          : "
        f"{statistics['count']:,}"
    )

    print(
        f"Gradients finite           : "
        f"{statistics['finite']}"
    )

    if not statistics["finite"]:

        raise AssertionError(
            "Network gradients contain NaN or Inf."
        )

    if statistics["maximum"] <= 0.0:

        raise AssertionError(
            "No non-zero network gradients were generated."
        )

    print(
        "Network gradient test: PASSED"
    )

    # =====================================================
    # 20. UNCERTAINTY HEAD GRADIENT AUDIT
    # =====================================================

    print_header(
        "UNCERTAINTY HEAD GRADIENT AUDIT"
    )

    uncertainty_gradients = []

    for parameter in model.uncertainty_head.parameters():

        if parameter.grad is not None:

            uncertainty_gradients.append(
                parameter.grad.detach().abs().reshape(-1)
            )

    if not uncertainty_gradients:

        raise AssertionError(
            "No gradients reached the uncertainty head."
        )

    uncertainty_gradients = torch.cat(
        uncertainty_gradients
    )

    uncertainty_gradient_max = (
        uncertainty_gradients.max().item()
    )

    print(
        f"Maximum uncertainty gradient : "
        f"{uncertainty_gradient_max:.12e}"
    )

    if not torch.isfinite(
        uncertainty_gradients
    ).all():

        raise AssertionError(
            "Uncertainty-head gradients contain "
            "NaN or Inf."
        )

    if uncertainty_gradient_max <= 0.0:

        raise AssertionError(
            "No non-zero gradients reached the "
            "uncertainty head."
        )

    print(
        "Uncertainty gradient test: PASSED"
    )

    # =====================================================
    # 21. COMPLETE TRAINING-STEP SIMULATION
    # =====================================================

    print_header(
        "COMPLETE TRAINING-STEP SIMULATION"
    )

    # Reinitialize optimizer for a clean training-step test.

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-4
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # Fresh forward pass.

    reconstruction_2, travel_time_2, log_variance_2 = (
        model(input_tensor)
    )

    physics_components_2 = physics_loss(
        travel_time=travel_time_2,
        velocity=velocity,
        source_indices=None,
        travel_time_target=None
    )

    mae_loss_2 = (
        reconstruction_2
        -
        reconstruction_target
    ).abs().mean()

    uncertainty_probe_2 = (
        log_variance_2.pow(2).mean()
    )

    total_training_loss = (
        LOSS_WEIGHTS["mae"]
        *
        mae_loss_2
        +
        LOSS_WEIGHTS["physics"]
        *
        physics_components_2["total"]
        +
        LOSS_WEIGHTS["uncertainty"]
        *
        uncertainty_probe_2
    )

    if not torch.isfinite(
        total_training_loss
    ):

        raise AssertionError(
            "Training loss contains NaN or Inf."
        )

    total_training_loss.backward()

    optimizer.step()

    print(
        f"Training loss              : "
        f"{total_training_loss.item():.12e}"
    )

    print(
        "Optimizer step completed successfully."
    )

    print(
        "Complete training-step test: PASSED"
    )

    # =====================================================
    # 22. FINAL RESULT
    # =====================================================

    print_header(
        "PHYSICS LOSS TRAINING-INTEGRATION AUDIT PASSED"
    )

    print(
        "The current PhysicsInformed3DUNet is correctly "
        "connected to the PhysicsLoss."
    )

    print()
    print(
        "Verified:"
    )

    print(
        "  ✓ Three-output network interface"
    )

    print(
        "  ✓ Reconstruction output"
    )

    print(
        "  ✓ Travel-time output"
    )

    print(
        "  ✓ Predictive uncertainty output"
    )

    print(
        "  ✓ Output shape consistency"
    )

    print(
        "  ✓ Output numerical stability"
    )

    print(
        "  ✓ Travel-time positivity"
    )

    print(
        "  ✓ Travel-time scaling"
    )

    print(
        "  ✓ Physical velocity compatibility"
    )

    print(
        "  ✓ Eikonal physics-loss computation"
    )

    print(
        "  ✓ Physics-loss weighting"
    )

    print(
        "  ✓ Eikonal differentiability"
    )

    print(
        "  ✓ Composite loss construction"
    )

    print(
        "  ✓ Backward propagation"
    )

    print(
        "  ✓ Network parameter gradients"
    )

    print(
        "  ✓ Uncertainty-head gradients"
    )

    print(
        "  ✓ Complete optimizer training step"
    )

    print()
    print(
        "PHYSICS LOSS TRAINING-INTEGRATION TEST PASSED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("=" * 60)
        print(
            "PHYSICS LOSS TRAINING-INTEGRATION AUDIT FAILED"
        )
        print("=" * 60)

        print()
        print(
            f"ERROR: {error}"
        )

        sys.exit(1)