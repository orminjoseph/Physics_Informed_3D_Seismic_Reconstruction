"""
======================================================================
Physics Gradient Scale Audit
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Diagnose the physical and numerical scale of the Eikonal physics loss
and determine why the physics component may dominate parameter
gradients despite having a relatively small loss weight.

This audit examines:

1. Physical sampling:
       DX, DY, DZ

2. Velocity magnitude:
       V

3. Travel-time output scale:
       T

4. Travel-time gradients:
       |grad T|

5. Eikonal quantity:
       V |grad T|

6. Eikonal residual:
       V |grad T| - 1

7. Eikonal loss

8. Physics-loss parameter gradient norm

9. Sensitivity to:
       - spatial sampling
       - velocity magnitude
       - travel-time output scale

10. Numerical dominance of the epsilon used in:
       sqrt(|grad T|^2 + eps)

IMPORTANT
---------
This file is an AUDIT ONLY.

It does NOT:
    - modify LOSS_WEIGHTS
    - modify PhysicsLoss
    - modify Network3D
    - modify training configuration
    - enable/disable losses
    - perform optimizer updates

======================================================================
"""

import random
import numpy as np
import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    VELOCITY_MIN,
    VELOCITY_MAX,
)


# ======================================================================
# Configuration
# ======================================================================

SEED = 42

TEST_SHAPE = (1, 1, 16, 32, 32)

# Sensitivity factors
SPACING_FACTORS = [0.5, 1.0, 2.0]
VELOCITY_FACTORS = [0.5, 1.0, 2.0]
TRAVEL_TIME_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Small analytic floor used ONLY to investigate the configured
# epsilon. It is not substituted into the actual PhysicsLoss.
ANALYTIC_FLOOR = 1.0e-30


# ======================================================================
# Reproducibility
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ======================================================================
# Utility functions
# ======================================================================

def tensor_stats(name, tensor):
    """
    Print basic finite statistics for a tensor.
    """

    tensor = tensor.detach()

    finite_mask = torch.isfinite(tensor)

    if not finite_mask.all():
        print(f"{name:<30} NON-FINITE VALUES DETECTED")
        return

    values = tensor[finite_mask]

    print(
        f"{name:<30} "
        f"mean={values.mean().item():.6e} | "
        f"std={values.std().item():.6e} | "
        f"min={values.min().item():.6e} | "
        f"max={values.max().item():.6e}"
    )


def gradient_statistics(model):
    """
    Calculate total parameter gradient norm and maximum gradient.
    """

    total_squared = 0.0
    max_gradient = 0.0
    parameter_count = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            return {
                "finite": False,
                "norm": float("nan"),
                "max": float("nan"),
                "count": parameter_count,
            }

        total_squared += gradient.pow(2).sum().item()
        max_gradient = max(
            max_gradient,
            gradient.abs().max().item()
        )

        parameter_count += gradient.numel()

    return {
        "finite": True,
        "norm": total_squared ** 0.5,
        "max": max_gradient,
        "count": parameter_count,
    }


def clear_gradients(model):
    """
    Clear all model gradients.
    """

    for parameter in model.parameters():

        if parameter.grad is not None:
            parameter.grad.zero_()


# ======================================================================
# Derivative calculation
# ======================================================================

def calculate_gradient_components(
    physics_loss,
    travel_time,
    dx,
    dy,
    dz,
):
    """
    Calculate the three spatial travel-time derivatives.

    Tensor convention:

        [B, C, D, H, W]

    Therefore:

        z -> dimension 2
        y -> dimension 3
        x -> dimension 4
    """

    dt_dx = physics_loss._derivative(
        travel_time,
        spacing=dx,
        dimension=4,
    )

    dt_dy = physics_loss._derivative(
        travel_time,
        spacing=dy,
        dimension=3,
    )

    dt_dz = physics_loss._derivative(
        travel_time,
        spacing=dz,
        dimension=2,
    )

    return dt_dx, dt_dy, dt_dz


# ======================================================================
# Eikonal diagnostics
# ======================================================================

def eikonal_diagnostics(
    physics_loss,
    travel_time,
    velocity,
    dx,
    dy,
    dz,
):
    """
    Calculate detailed Eikonal quantities.

    Stabilized formulation:

        |grad T| = sqrt(
            dT/dx^2 +
            dT/dy^2 +
            dT/dz^2 +
            eps
        )

        residual = V |grad T| - 1

        loss = mean(residual^2)
    """

    dt_dx, dt_dy, dt_dz = calculate_gradient_components(
        physics_loss=physics_loss,
        travel_time=travel_time,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    gradient_squared = (
        dt_dx.pow(2)
        + dt_dy.pow(2)
        + dt_dz.pow(2)
    )

    # Actual configured epsilon
    configured_eps = physics_loss.eps

    gradient_magnitude = torch.sqrt(
        gradient_squared + configured_eps
    )

    # Analytic-floor version.
    # This is only used to expose the influence of epsilon.
    gradient_magnitude_no_eps = torch.sqrt(
        gradient_squared + ANALYTIC_FLOOR
    )

    v_grad_t = velocity * gradient_magnitude

    residual = v_grad_t - 1.0

    loss = residual.pow(2).mean()

    physical_target_gradient = 1.0 / velocity

    return {
        "dt_dx": dt_dx,
        "dt_dy": dt_dy,
        "dt_dz": dt_dz,
        "gradient_squared": gradient_squared,
        "gradient_magnitude": gradient_magnitude,
        "gradient_magnitude_no_eps": gradient_magnitude_no_eps,
        "v_grad_t": v_grad_t,
        "residual": residual,
        "loss": loss,
        "physical_target_gradient": physical_target_gradient,
    }


# ======================================================================
# Main audit
# ======================================================================

def main():

    print("=" * 78)
    print("PHYSICS GRADIENT SCALE AUDIT")
    print("=" * 78)

    print("\n[1] Configuration")
    print("-" * 78)

    print(f"Device                  : {DEVICE}")
    print(f"Test shape              : {TEST_SHAPE}")
    print(f"Seed                    : {SEED}")

    print("\nPhysical sampling:")
    print(f"DX                      : {DX}")
    print(f"DY                      : {DY}")
    print(f"DZ                      : {DZ}")

    print("\nTravel-time configuration:")
    print(f"TRAVEL_TIME_SCALE       : {TRAVEL_TIME_SCALE}")

    print("\nVelocity configuration:")
    print(f"VELOCITY_MIN            : {VELOCITY_MIN}")
    print(f"VELOCITY_MAX            : {VELOCITY_MAX}")

    print("\nNOTE:")
    print("LOSS_WEIGHTS are intentionally NOT modified.")
    print("This is a diagnostic audit only.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\n[2] Creating model")
    print("-" * 78)

    model = Network3D().to(DEVICE)

    # Disable dropout for deterministic physical-scale analysis.
    model.eval()

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Synthetic input
    # ------------------------------------------------------------------

    input_tensor = torch.randn(
        TEST_SHAPE,
        device=DEVICE,
    )

    # Positive physical velocity model.
    velocity_midpoint = (
        VELOCITY_MIN + VELOCITY_MAX
    ) / 2.0

    velocity_model = torch.full(
        TEST_SHAPE,
        velocity_midpoint,
        dtype=torch.float32,
        device=DEVICE,
    )

    # ------------------------------------------------------------------
    # Baseline forward pass
    # ------------------------------------------------------------------

    print("\n[3] Baseline forward pass")
    print("-" * 78)

    with torch.no_grad():

        reconstruction, travel_time, log_variance = model(
            input_tensor
        )

    tensor_stats(
        "Travel time",
        travel_time,
    )

    tensor_stats(
        "Reconstruction",
        reconstruction,
    )

    tensor_stats(
        "Log variance",
        log_variance,
    )

    tensor_stats(
        "Velocity",
        velocity_model,
    )

    # ------------------------------------------------------------------
    # Baseline Eikonal diagnostic
    # ------------------------------------------------------------------

    print("\n[4] Baseline Eikonal diagnostics")
    print("-" * 78)

    diagnostics = eikonal_diagnostics(
        physics_loss=physics_loss,
        travel_time=travel_time,
        velocity=velocity_model,
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    tensor_stats(
        "dT/dx",
        diagnostics["dt_dx"],
    )

    tensor_stats(
        "dT/dy",
        diagnostics["dt_dy"],
    )

    tensor_stats(
        "dT/dz",
        diagnostics["dt_dz"],
    )

    tensor_stats(
        "|grad T|",
        diagnostics["gradient_magnitude"],
    )

    tensor_stats(
        "1 / V",
        diagnostics["physical_target_gradient"],
    )

    tensor_stats(
        "V |grad T|",
        diagnostics["v_grad_t"],
    )

    tensor_stats(
        "Eikonal residual",
        diagnostics["residual"],
    )

    print(
        f"\nEikonal loss              : "
        f"{diagnostics['loss'].item():.6e}"
    )

    # ------------------------------------------------------------------
    # Compare gradient against physical target
    # ------------------------------------------------------------------

    actual_gradient = diagnostics[
        "gradient_magnitude"
    ].detach()

    target_gradient = diagnostics[
        "physical_target_gradient"
    ].detach()

    gradient_ratio = (
        actual_gradient /
        target_gradient.clamp_min(ANALYTIC_FLOOR)
    )

    tensor_stats(
        "|grad T| / (1/V)",
        gradient_ratio,
    )

    print(
        "\nInterpretation:"
    )

    print(
        "The physically required first-order Eikonal scale is "
        "approximately |grad T| = 1/V."
    )

    # ------------------------------------------------------------------
    # Compare actual implementation with manual calculation
    # ------------------------------------------------------------------

    print("\n[5] PhysicsLoss implementation consistency")
    print("-" * 78)

    implementation_residual = physics_loss.eikonal_residual(
        travel_time=travel_time,
        velocity=velocity_model,
    )

    implementation_loss = physics_loss.eikonal_loss(
        travel_time=travel_time,
        velocity=velocity_model,
    )

    residual_difference = (
        implementation_residual -
        diagnostics["residual"]
    ).abs().max().item()

    loss_difference = abs(
        implementation_loss.item()
        - diagnostics["loss"].item()
    )

    print(
        f"Maximum residual difference : "
        f"{residual_difference:.6e}"
    )

    print(
        f"Loss difference             : "
        f"{loss_difference:.6e}"
    )

    if residual_difference < 1.0e-6:
        print("Residual consistency         : PASS")
    else:
        print("Residual consistency         : FAIL")

    if loss_difference < 1.0e-6:
        print("Loss consistency             : PASS")
    else:
        print("Loss consistency             : FAIL")

    # ------------------------------------------------------------------
    # Epsilon dominance audit
    # ------------------------------------------------------------------

    print("\n[6] Epsilon dominance audit")
    print("-" * 78)

    configured_eps = physics_loss.eps

    gradient_squared = diagnostics[
        "gradient_squared"
    ].detach()

    epsilon_dominated_fraction = (
        gradient_squared < configured_eps
    ).float().mean().item()

    print(
        f"Configured epsilon           : "
        f"{configured_eps:.6e}"
    )

    print(
        f"Mean gradient squared       : "
        f"{gradient_squared.mean().item():.6e}"
    )

    print(
        f"Median gradient squared     : "
        f"{gradient_squared.median().item():.6e}"
    )

    print(
        f"Fraction grad² < epsilon    : "
        f"{epsilon_dominated_fraction * 100:.4f}%"
    )

    print(
        f"Mean |grad T| with epsilon   : "
        f"{diagnostics['gradient_magnitude'].mean().item():.6e}"
    )

    print(
        f"Mean |grad T| tiny floor     : "
        f"{diagnostics['gradient_magnitude_no_eps'].mean().item():.6e}"
    )

    epsilon_difference = (
        diagnostics["gradient_magnitude"]
        - diagnostics["gradient_magnitude_no_eps"]
    ).abs()

    print(
        f"Mean epsilon-induced change  : "
        f"{epsilon_difference.mean().item():.6e}"
    )

    if epsilon_dominated_fraction > 0.50:
        print(
            "\nWARNING: More than 50% of voxels have "
            "gradient² below configured epsilon."
        )
        print(
            "The epsilon may materially influence "
            "the calculated gradient magnitude."
        )
    else:
        print(
            "\nEpsilon dominance assessment : PASS"
        )

    # ------------------------------------------------------------------
    # Physics gradient audit
    # ------------------------------------------------------------------

    print("\n[7] Baseline physics parameter gradient")
    print("-" * 78)

    clear_gradients(model)

    # Fresh forward pass because the previous graph was under no_grad.
    _, travel_time_grad, _ = model(
        input_tensor
    )

    physics_value = physics_loss.eikonal_loss(
        travel_time=travel_time_grad,
        velocity=velocity_model,
    )

    physics_value.backward()

    gradient_info = gradient_statistics(model)

    print(
        f"Physics loss                : "
        f"{physics_value.item():.6e}"
    )

    print(
        f"Gradient norm               : "
        f"{gradient_info['norm']:.6e}"
    )

    print(
        f"Maximum parameter gradient  : "
        f"{gradient_info['max']:.6e}"
    )

    print(
        f"Gradient elements           : "
        f"{gradient_info['count']}"
    )

    # ------------------------------------------------------------------
    # Spatial sampling sensitivity
    # ------------------------------------------------------------------

    print("\n[8] Spatial sampling sensitivity")
    print("-" * 78)

    print(
        "\nFactor | DX       DY       DZ       "
        "mean|gradT|       mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    for factor in SPACING_FACTORS:

        test_dx = DX * factor
        test_dy = DY * factor
        test_dz = DZ * factor

        # Physical diagnostics
        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            physics_loss=physics_loss,
            travel_time=t_test,
            velocity=velocity_model,
            dx=test_dx,
            dy=test_dy,
            dz=test_dz,
        )

        clear_gradients(model)

        _, t_grad, _ = model(
            input_tensor
        )

        loss_test = physics_loss.eikonal_loss(
            travel_time=t_grad,
            velocity=velocity_model,
        )

        loss_test.backward()

        ginfo = gradient_statistics(model)

        print(
            f"{factor:5.2f} | "
            f"{test_dx:7.3f} "
            f"{test_dy:7.3f} "
            f"{test_dz:7.3f} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{ginfo['norm']:.6e}"
        )

    # ------------------------------------------------------------------
    # Velocity sensitivity
    # ------------------------------------------------------------------

    print("\n[9] Velocity magnitude sensitivity")
    print("-" * 78)

    print(
        "\nFactor | mean V           mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    for factor in VELOCITY_FACTORS:

        test_velocity = velocity_model * factor

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            physics_loss=physics_loss,
            travel_time=t_test,
            velocity=test_velocity,
            dx=DX,
            dy=DY,
            dz=DZ,
        )

        clear_gradients(model)

        _, t_grad, _ = model(
            input_tensor
        )

        loss_test = physics_loss.eikonal_loss(
            travel_time=t_grad,
            velocity=test_velocity,
        )

        loss_test.backward()

        ginfo = gradient_statistics(model)

        print(
            f"{factor:5.2f} | "
            f"{test_velocity.mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{ginfo['norm']:.6e}"
        )

    # ------------------------------------------------------------------
    # Travel-time output sensitivity
    # ------------------------------------------------------------------

    print("\n[10] Travel-time output scale sensitivity")
    print("-" * 78)

    print(
        "\nFactor | mean T            mean|gradT|        "
        "mean(V|gradT|)       loss           grad_norm"
    )

    print("-" * 78)

    with torch.no_grad():

        _, base_travel_time, _ = model(
            input_tensor
        )

    for factor in TRAVEL_TIME_FACTORS:

        scaled_travel_time = (
            base_travel_time * factor
        )

        diag = eikonal_diagnostics(
            physics_loss=physics_loss,
            travel_time=scaled_travel_time,
            velocity=velocity_model,
            dx=DX,
            dy=DY,
            dz=DZ,
        )

        clear_gradients(model)

        _, t_grad, _ = model(
            input_tensor
        )

        scaled_t_grad = (
            t_grad * factor
        )

        loss_test = physics_loss.eikonal_loss(
            travel_time=scaled_t_grad,
            velocity=velocity_model,
        )

        loss_test.backward()

        ginfo = gradient_statistics(model)

        print(
            f"{factor:5.2f} | "
            f"{scaled_travel_time.mean().item():.6e} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{ginfo['norm']:.6e}"
        )

    # ------------------------------------------------------------------
    # Characteristic travel-time scale
    # ------------------------------------------------------------------

    print("\n[11] Characteristic physical travel-time scale")
    print("-" * 78)

    depth = (TEST_SHAPE[2] - 1) * DZ
    height = (TEST_SHAPE[3] - 1) * DY
    width = (TEST_SHAPE[4] - 1) * DX

    diagonal_distance = (
        depth ** 2
        + height ** 2
        + width ** 2
    ) ** 0.5

    characteristic_velocity = velocity_midpoint

    characteristic_time = (
        diagonal_distance /
        characteristic_velocity
    )

    print(
        f"Depth extent               : "
        f"{depth:.6e} m"
    )

    print(
        f"Y extent                   : "
        f"{height:.6e} m"
    )

    print(
        f"X extent                   : "
        f"{width:.6e} m"
    )

    print(
        f"3D diagonal distance       : "
        f"{diagonal_distance:.6e} m"
    )

    print(
        f"Characteristic velocity    : "
        f"{characteristic_velocity:.6e} m/s"
    )

    print(
        f"Characteristic travel time : "
        f"{characteristic_time:.6e} s"
    )

    print(
        f"Configured T scale         : "
        f"{TRAVEL_TIME_SCALE:.6e} s"
    )

    scale_ratio = (
        TRAVEL_TIME_SCALE /
        max(characteristic_time, ANALYTIC_FLOOR)
    )

    print(
        f"T-scale / physical scale  : "
        f"{scale_ratio:.6e}"
    )

    # ------------------------------------------------------------------
    # Final assessment
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("PHYSICS GRADIENT SCALE AUDIT COMPLETE")
    print("=" * 78)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT change LOSS_WEIGHTS yet."
    )

    print(
        "The next decision should be based on:"
    )

    print(
        "  1. Whether epsilon dominates |grad T|"
    )

    print(
        "  2. Whether |grad T| is near 1/V"
    )

    print(
        "  3. Whether V|grad T| is substantially below 1"
    )

    print(
        "  4. How strongly spacing changes the physics gradient"
    )

    print(
        "  5. How strongly travel-time scaling changes the "
        "physics gradient"
    )

    print(
        "  6. Whether the configured travel-time scale is "
        "physically appropriate"
    )

    print(
        "  7. Whether the physics gradient dominance is caused "
        "by the physical formulation or merely by initialization"
    )

    print(
        "\nNo model parameters were updated."
    )
    print(
        "No optimizer step was performed."
    )
    print(
        "No loss weights were changed."
    )


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    main()