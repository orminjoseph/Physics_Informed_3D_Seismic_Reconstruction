"""
======================================================================
Physics Gradient Scale Audit v2.2
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

9. Isotropic spatial-sampling sensitivity

10. Directional spatial-sampling sensitivity:
       DX independently
       DY independently
       DZ independently

11. Velocity magnitude sensitivity

12. Absolute travel-time output-scale sensitivity

13. Numerical dominance of epsilon in:
       sqrt(|grad T|^2 + eps)

14. Parameter-group gradient statistics

15. Loss-component gradient contribution

16. Characteristic physical travel-time scale

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
    - change model parameters

======================================================================
"""

import random
import numpy as np
import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from losses.mae_loss import MAELoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import UncertaintyLoss
from losses.ssim_loss import SSIMLoss

from utils.config import (
    DEVICE,
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    VELOCITY_MIN,
    VELOCITY_MAX,
    LOSS_WEIGHTS,
)


# ======================================================================
# Audit configuration
# ======================================================================

SEED = 42

TEST_SHAPE = (1, 1, 16, 32, 32)

# Isotropic spatial-sampling factors.
SPACING_FACTORS = [0.5, 1.0, 2.0]

# Independent directional spacing factors.
DIRECTIONAL_SPACING_FACTORS = [0.5, 1.0, 2.0]

# Velocity sensitivity.
VELOCITY_FACTORS = [0.5, 1.0, 2.0]

# Absolute physical travel-time scales in seconds.
#
# These are NOT configuration changes.
# They are diagnostic rescalings of the baseline predicted field.
TRAVEL_TIME_SCALES = [
    0.01,
    0.02,
    0.05,
    0.10,
]

# Epsilon sensitivity.
EPSILON_VALUES = [
    1.0e-16,
    1.0e-14,
    1.0e-12,
    1.0e-10,
]

# Tiny analytic floor used only for diagnostics.
ANALYTIC_FLOOR = 1.0e-30

# Numerical tolerance for implementation-consistency checks.
CONSISTENCY_TOLERANCE = 1.0e-6


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
    Print finite descriptive statistics for a tensor.
    """

    tensor = tensor.detach()

    finite_mask = torch.isfinite(tensor)

    if not finite_mask.all():
        print(
            f"{name:<32} "
            f"NON-FINITE VALUES DETECTED"
        )
        return

    values = tensor[finite_mask]

    print(
        f"{name:<32} "
        f"mean={values.mean().item():.6e} | "
        f"std={values.std().item():.6e} | "
        f"min={values.min().item():.6e} | "
        f"max={values.max().item():.6e}"
    )


def clear_gradients(model):
    """
    Clear all parameter gradients.
    """

    for parameter in model.parameters():

        if parameter.grad is not None:
            parameter.grad.zero_()


def gradient_statistics(model):
    """
    Calculate total parameter-gradient statistics.

    Returns
    -------
    dict
        finite:
            Whether all observed gradients are finite.

        norm:
            Global L2 parameter-gradient norm.

        max:
            Maximum absolute parameter gradient.

        count:
            Number of gradient elements.
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
            gradient.abs().max().item(),
        )

        parameter_count += gradient.numel()

    return {
        "finite": True,
        "norm": total_squared ** 0.5,
        "max": max_gradient,
        "count": parameter_count,
    }


def parameter_group_gradient_statistics(model):
    """
    Calculate gradient norms for major parameter groups.

    The grouping is based on parameter names.

    Important:
    Shared encoder/decoder parameters may not belong exclusively
    to one output head. Therefore, the group norms do NOT necessarily
    sum to the global gradient norm.
    """

    groups = {
        "reconstruction_head": [],
        "travel_time_head": [],
        "uncertainty_head": [],
        "encoder": [],
        "decoder": [],
        "bottleneck": [],
        "other": [],
    }

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        name_lower = name.lower()

        if "reconstruction" in name_lower:
            group = "reconstruction_head"

        elif "travel_time" in name_lower:
            group = "travel_time_head"

        elif "uncertainty" in name_lower:
            group = "uncertainty_head"

        elif "encoder" in name_lower:
            group = "encoder"

        elif "decoder" in name_lower:
            group = "decoder"

        elif "bottleneck" in name_lower:
            group = "bottleneck"

        else:
            group = "other"

        groups[group].append(parameter.grad.detach())

    statistics = {}

    for group_name, gradients in groups.items():

        if not gradients:

            statistics[group_name] = {
                "norm": 0.0,
                "max": 0.0,
                "elements": 0,
            }

            continue

        total_squared = 0.0
        maximum = 0.0
        elements = 0

        for gradient in gradients:

            total_squared += gradient.pow(2).sum().item()

            maximum = max(
                maximum,
                gradient.abs().max().item(),
            )

            elements += gradient.numel()

        statistics[group_name] = {
            "norm": total_squared ** 0.5,
            "max": maximum,
            "elements": elements,
        }

    return statistics


def make_physics_loss(dx, dy, dz, eps=None):
    """
    Construct a NEW PhysicsLoss object.

    This is critical for the spatial-sampling audit.

    Every spacing experiment must backpropagate through a PhysicsLoss
    configured with the experimental DX, DY, and DZ values.
    """

    if eps is None:

        return PhysicsLoss(
            dx=dx,
            dy=dy,
            dz=dz,
        ).to(DEVICE)

    return PhysicsLoss(
        dx=dx,
        dy=dy,
        dz=dz,
        eps=eps,
    ).to(DEVICE)


# ======================================================================
# Derivative calculation
# ======================================================================

def calculate_gradient_components(
    physics_loss,
    travel_time,
):
    """
    Calculate spatial travel-time derivatives.

    Tensor convention:

        [B, C, D, H, W]

    Therefore:

        z -> dimension 2
        y -> dimension 3
        x -> dimension 4
    """

    dt_dx = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dx,
        dimension=4,
    )

    dt_dy = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dy,
        dimension=3,
    )

    dt_dz = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dz,
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
):
    """
    Calculate detailed Eikonal quantities.

    Stabilized formulation:

        |grad T| =
            sqrt(
                dT/dx^2
                + dT/dy^2
                + dT/dz^2
                + eps
            )

        residual =
            V |grad T| - 1

        loss =
            mean(residual^2)
    """

    dt_dx, dt_dy, dt_dz = calculate_gradient_components(
        physics_loss=physics_loss,
        travel_time=travel_time,
    )

    gradient_squared = (
        dt_dx.pow(2)
        + dt_dy.pow(2)
        + dt_dz.pow(2)
    )

    configured_eps = physics_loss.eps

    gradient_magnitude = torch.sqrt(
        gradient_squared + configured_eps
    )

    # Tiny analytic floor used only to estimate epsilon influence.
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
# Physics-gradient audit helper
# ======================================================================

def physics_gradient_audit(
    model,
    physics_loss,
    input_tensor,
    velocity_model,
):
    """
    Perform an actual backpropagation through the supplied PhysicsLoss.

    IMPORTANT:
    The PhysicsLoss supplied here must be the same PhysicsLoss used
    to calculate the corresponding physical diagnostic.

    Returns
    -------
    dict
        Physics loss and gradient statistics.
    """

    clear_gradients(model)

    _, travel_time, _ = model(
        input_tensor
    )

    loss = physics_loss.eikonal_loss(
        travel_time=travel_time,
        velocity=velocity_model,
    )

    loss.backward()

    gradient_info = gradient_statistics(model)

    group_info = parameter_group_gradient_statistics(model)

    return {
        "loss": loss.item(),
        "gradient": gradient_info,
        "groups": group_info,
    }


# ======================================================================
# Component-gradient audit
# ======================================================================

def component_gradient_audit(
    model,
    input_tensor,
    velocity_model,
):
    """
    Measure gradient contributions from:

        MAE
        Physics
        Uncertainty
        SSIM

    The purpose is NOT to declare a correct training weighting.

    The purpose is to determine which loss component produces the
    strongest parameter gradients under the current initialization.

    Weighted gradient contribution is calculated as:

        weighted_gradient_norm
            =
        raw_gradient_norm * configured_weight

    and:

        relative_share =
            weighted_gradient_norm
            /
            sum(all weighted_gradient_norms)
    """

    mae_loss = MAELoss().to(DEVICE)
    uncertainty_loss = UncertaintyLoss().to(DEVICE)
    ssim_loss = SSIMLoss().to(DEVICE)

    physics_loss = make_physics_loss(
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    model.eval()

    # --------------------------------------------------------------
    # One common forward graph
    # --------------------------------------------------------------

    reconstruction, travel_time, log_variance = model(
        input_tensor
    )

    target = torch.randn_like(
        reconstruction
    )

    results = {}

    components = {
        "mae": mae_loss(
            reconstruction,
            target,
        ),

        "physics": physics_loss.eikonal_loss(
            travel_time,
            velocity_model,
        ),

        "uncertainty": uncertainty_loss(
            reconstruction,
            target,
            log_variance,
        ),

        "ssim": ssim_loss(
            reconstruction,
            target,
        ),
    }

    # --------------------------------------------------------------
    # Backpropagate each component independently
    # --------------------------------------------------------------

    component_names = list(components.keys())

    for index, name in enumerate(component_names):

        clear_gradients(model)

        retain_graph = (
            index < len(component_names) - 1
        )

        components[name].backward(
            retain_graph=retain_graph
        )

        gradient_info = gradient_statistics(model)

        results[name] = {
            "raw_loss": components[name].item(),
            "weight": LOSS_WEIGHTS[name],
            "weighted_loss":
                components[name].item()
                * LOSS_WEIGHTS[name],
            "grad_norm": gradient_info["norm"],
            "max_grad": gradient_info["max"],
            "finite": gradient_info["finite"],
        }

    # --------------------------------------------------------------
    # Correct weighted-gradient calculation
    # --------------------------------------------------------------

    weighted_gradient_total = sum(
        value["grad_norm"] * value["weight"]
        for value in results.values()
    )

    for name in component_names:

        raw_gradient = results[name]["grad_norm"]

        weight = results[name]["weight"]

        weighted_gradient = (
            raw_gradient * weight
        )

        results[name]["weighted_grad_norm"] = (
            weighted_gradient
        )

        if weighted_gradient_total > 0.0:

            results[name]["weighted_grad_share"] = (
                100.0
                * weighted_gradient
                / weighted_gradient_total
            )

        else:

            results[name]["weighted_grad_share"] = 0.0

    return results


# ======================================================================
# Main audit
# ======================================================================

def main():

    print("=" * 78)
    print("PHYSICS GRADIENT SCALE AUDIT v2.2")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

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

    print("\nLoss weights:")
    for name, weight in LOSS_WEIGHTS.items():
        print(
            f"{name:<24}: {weight}"
        )

    print("\nIMPORTANT:")
    print("LOSS_WEIGHTS are NOT modified.")
    print("This is a diagnostic audit only.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\n[2] Creating model")
    print("-" * 78)

    model = Network3D().to(DEVICE)

    # Disable dropout for deterministic physical-scale analysis.
    model.eval()

    physics_loss = make_physics_loss(
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    # ------------------------------------------------------------------
    # Synthetic input
    # ------------------------------------------------------------------

    input_tensor = torch.randn(
        TEST_SHAPE,
        device=DEVICE,
    )

    velocity_midpoint = (
        VELOCITY_MIN
        + VELOCITY_MAX
    ) / 2.0

    velocity_model = torch.full(
        TEST_SHAPE,
        velocity_midpoint,
        dtype=torch.float32,
        device=DEVICE,
    )

    # ==================================================================
    # BASELINE
    # ==================================================================

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
    # Baseline Eikonal diagnostics
    # ------------------------------------------------------------------

    print("\n[4] Baseline Eikonal diagnostics")
    print("-" * 78)

    diagnostics = eikonal_diagnostics(
        physics_loss=physics_loss,
        travel_time=travel_time,
        velocity=velocity_model,
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
    # Gradient / physical target comparison
    # ------------------------------------------------------------------

    actual_gradient = diagnostics[
        "gradient_magnitude"
    ].detach()

    target_gradient = diagnostics[
        "physical_target_gradient"
    ].detach()

    gradient_ratio = (
        actual_gradient
        / target_gradient.clamp_min(
            ANALYTIC_FLOOR
        )
    )

    tensor_stats(
        "|grad T| / (1/V)",
        gradient_ratio,
    )

    print("\nInterpretation:")
    print(
        "The first-order Eikonal target requires "
        "|grad T| approximately equal to 1/V."
    )

    # ==================================================================
    # IMPLEMENTATION CONSISTENCY
    # ==================================================================

    print(
        "\n[5] PhysicsLoss implementation consistency"
    )
    print("-" * 78)

    implementation_residual = (
        physics_loss.eikonal_residual(
            travel_time=travel_time,
            velocity=velocity_model,
        )
    )

    implementation_loss = (
        physics_loss.eikonal_loss(
            travel_time=travel_time,
            velocity=velocity_model,
        )
    )

    residual_difference = (
        implementation_residual
        - diagnostics["residual"]
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

    if residual_difference < CONSISTENCY_TOLERANCE:
        print(
            "Residual consistency         : PASS"
        )
    else:
        print(
            "Residual consistency         : FAIL"
        )

    if loss_difference < CONSISTENCY_TOLERANCE:
        print(
            "Loss consistency             : PASS"
        )
    else:
        print(
            "Loss consistency             : FAIL"
        )

    # ==================================================================
    # EPSILON AUDIT
    # ==================================================================

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
        f"Mean gradient squared        : "
        f"{gradient_squared.mean().item():.6e}"
    )

    print(
        f"Median gradient squared      : "
        f"{gradient_squared.median().item():.6e}"
    )

    print(
        f"Fraction grad² < epsilon     : "
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
            "the gradient magnitude."
        )

    else:

        print(
            "\nEpsilon dominance assessment : PASS"
        )

    # ==================================================================
    # EPSILON SENSITIVITY
    # ==================================================================

    print("\n[7] Explicit epsilon sensitivity")
    print("-" * 78)

    print(
        "\nEpsilon       mean|gradT|       "
        "mean(V|gradT|)       loss"
    )

    print("-" * 78)

    for epsilon in EPSILON_VALUES:

        epsilon_loss = make_physics_loss(
            dx=DX,
            dy=DY,
            dz=DZ,
            eps=epsilon,
        )

        diag = eikonal_diagnostics(
            physics_loss=epsilon_loss,
            travel_time=travel_time,
            velocity=velocity_model,
        )

        print(
            f"{epsilon:10.2e} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e}"
        )

    # ==================================================================
    # BASELINE PHYSICS GRADIENT
    # ==================================================================

    print(
        "\n[8] Baseline physics parameter gradient"
    )
    print("-" * 78)

    baseline_gradient = physics_gradient_audit(
        model=model,
        physics_loss=physics_loss,
        input_tensor=input_tensor,
        velocity_model=velocity_model,
    )

    print(
        f"Physics loss                : "
        f"{baseline_gradient['loss']:.6e}"
    )

    print(
        f"Gradient norm               : "
        f"{baseline_gradient['gradient']['norm']:.6e}"
    )

    print(
        f"Maximum parameter gradient  : "
        f"{baseline_gradient['gradient']['max']:.6e}"
    )

    print(
        f"Gradient elements           : "
        f"{baseline_gradient['gradient']['count']}"
    )

    print("\nParameter-group gradients:")

    for name, values in (
        baseline_gradient["groups"].items()
    ):

        print(
            f"{name:<24} "
            f"norm={values['norm']:.6e} | "
            f"max={values['max']:.6e} | "
            f"elements={values['elements']}"
        )

    # ==================================================================
    # ISOTROPIC SPATIAL SAMPLING
    # ==================================================================

    print(
        "\n[9] Isotropic spatial-sampling sensitivity"
    )
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

        # --------------------------------------------------------------
        # CRITICAL FIX:
        # Create a NEW PhysicsLoss using the experimental spacing.
        # --------------------------------------------------------------

        test_physics_loss = make_physics_loss(
            dx=test_dx,
            dy=test_dy,
            dz=test_dz,
        )

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            physics_loss=test_physics_loss,
            travel_time=t_test,
            velocity=velocity_model,
        )

        gradient_result = physics_gradient_audit(
            model=model,
            physics_loss=test_physics_loss,
            input_tensor=input_tensor,
            velocity_model=velocity_model,
        )

        print(
            f"{factor:5.2f} | "
            f"{test_dx:7.3f} "
            f"{test_dy:7.3f} "
            f"{test_dz:7.3f} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_result['gradient']['norm']:.6e}"
        )

    # ==================================================================
    # DIRECTIONAL DX SENSITIVITY
    # ==================================================================

    print(
        "\n[10] Independent DX sensitivity"
    )
    print("-" * 78)

    print(
        "\nFactor | DX       DY       DZ       "
        "mean|gradT|       mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    for factor in DIRECTIONAL_SPACING_FACTORS:

        test_dx = DX * factor
        test_dy = DY
        test_dz = DZ

        test_physics_loss = make_physics_loss(
            dx=test_dx,
            dy=test_dy,
            dz=test_dz,
        )

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            test_physics_loss,
            t_test,
            velocity_model,
        )

        gradient_result = physics_gradient_audit(
            model,
            test_physics_loss,
            input_tensor,
            velocity_model,
        )

        print(
            f"{factor:5.2f} | "
            f"{test_dx:7.3f} "
            f"{test_dy:7.3f} "
            f"{test_dz:7.3f} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_result['gradient']['norm']:.6e}"
        )

    # ==================================================================
    # DIRECTIONAL DY SENSITIVITY
    # ==================================================================

    print(
        "\n[11] Independent DY sensitivity"
    )
    print("-" * 78)

    print(
        "\nFactor | DX       DY       DZ       "
        "mean|gradT|       mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    for factor in DIRECTIONAL_SPACING_FACTORS:

        test_dx = DX
        test_dy = DY * factor
        test_dz = DZ

        test_physics_loss = make_physics_loss(
            dx=test_dx,
            dy=test_dy,
            dz=test_dz,
        )

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            test_physics_loss,
            t_test,
            velocity_model,
        )

        gradient_result = physics_gradient_audit(
            model,
            test_physics_loss,
            input_tensor,
            velocity_model,
        )

        print(
            f"{factor:5.2f} | "
            f"{test_dx:7.3f} "
            f"{test_dy:7.3f} "
            f"{test_dz:7.3f} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_result['gradient']['norm']:.6e}"
        )

    # ==================================================================
    # DIRECTIONAL DZ SENSITIVITY
    # ==================================================================

    print(
        "\n[12] Independent DZ sensitivity"
    )
    print("-" * 78)

    print(
        "\nFactor | DX       DY       DZ       "
        "mean|gradT|       mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    for factor in DIRECTIONAL_SPACING_FACTORS:

        test_dx = DX
        test_dy = DY
        test_dz = DZ * factor

        test_physics_loss = make_physics_loss(
            dx=test_dx,
            dy=test_dy,
            dz=test_dz,
        )

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            test_physics_loss,
            t_test,
            velocity_model,
        )

        gradient_result = physics_gradient_audit(
            model,
            test_physics_loss,
            input_tensor,
            velocity_model,
        )

        print(
            f"{factor:5.2f} | "
            f"{test_dx:7.3f} "
            f"{test_dy:7.3f} "
            f"{test_dz:7.3f} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_result['gradient']['norm']:.6e}"
        )

    # ==================================================================
    # VELOCITY SENSITIVITY
    # ==================================================================

    print(
        "\n[13] Velocity magnitude sensitivity"
    )
    print("-" * 78)

    print(
        "\nFactor | mean V           "
        "mean(V|gradT|)       loss           grad_norm"
    )

    print("-" * 78)

    for factor in VELOCITY_FACTORS:

        test_velocity = (
            velocity_model * factor
        )

        with torch.no_grad():

            _, t_test, _ = model(
                input_tensor
            )

        diag = eikonal_diagnostics(
            physics_loss,
            t_test,
            test_velocity,
        )

        gradient_result = physics_gradient_audit(
            model,
            physics_loss,
            input_tensor,
            test_velocity,
        )

        print(
            f"{factor:5.2f} | "
            f"{test_velocity.mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_result['gradient']['norm']:.6e}"
        )

    # ==================================================================
    # ABSOLUTE TRAVEL-TIME SCALE SENSITIVITY
    # ==================================================================

    print(
        "\n[14] Absolute travel-time output-scale sensitivity"
    )
    print("-" * 78)

    print(
        "\nTarget T scale | actual mean T       "
        "mean|gradT|       mean(V|gradT|)       "
        "loss           grad_norm"
    )

    print("-" * 78)

    with torch.no_grad():

        _, base_travel_time, _ = model(
            input_tensor
        )

    baseline_mean_t = (
        base_travel_time.abs().mean().item()
    )

    print(
        f"\nBaseline predicted |T| mean : "
        f"{baseline_mean_t:.6e} s"
    )

    for target_scale in TRAVEL_TIME_SCALES:

        # --------------------------------------------------------------
        # Diagnostic rescaling only.
        #
        # This does NOT modify TRAVEL_TIME_SCALE in config.py.
        # --------------------------------------------------------------

        current_mean = (
            base_travel_time.abs().mean()
        ).clamp_min(ANALYTIC_FLOOR)

        scale_factor = (
            target_scale / current_mean
        )

        scaled_travel_time = (
            base_travel_time
            * scale_factor
        )

        diag = eikonal_diagnostics(
            physics_loss,
            scaled_travel_time,
            velocity_model,
        )

        # --------------------------------------------------------------
        # Actual parameter-gradient experiment.
        #
        # Recreate the model forward graph.
        # --------------------------------------------------------------

        clear_gradients(model)

        _, t_grad, _ = model(
            input_tensor
        )

        current_mean_grad = (
            t_grad.abs().mean()
        ).clamp_min(ANALYTIC_FLOOR)

        scale_factor_grad = (
            target_scale
            / current_mean_grad
        )

        scaled_t_grad = (
            t_grad
            * scale_factor_grad
        )

        loss_test = physics_loss.eikonal_loss(
            travel_time=scaled_t_grad,
            velocity=velocity_model,
        )

        loss_test.backward()

        gradient_info = gradient_statistics(
            model
        )

        print(
            f"{target_scale:13.4e} | "
            f"{scaled_travel_time.mean().item():.6e} | "
            f"{diag['gradient_magnitude'].mean().item():.6e} | "
            f"{diag['v_grad_t'].mean().item():.6e} | "
            f"{diag['loss'].item():.6e} | "
            f"{gradient_info['norm']:.6e}"
        )

    # ==================================================================
    # CHARACTERISTIC PHYSICAL TRAVEL-TIME SCALE
    # ==================================================================

    print(
        "\n[15] Characteristic physical travel-time scale"
    )
    print("-" * 78)

    depth = (
        TEST_SHAPE[2] - 1
    ) * DZ

    height = (
        TEST_SHAPE[3] - 1
    ) * DY

    width = (
        TEST_SHAPE[4] - 1
    ) * DX

    diagonal_distance = (
        depth ** 2
        + height ** 2
        + width ** 2
    ) ** 0.5

    characteristic_velocity = (
        velocity_midpoint
    )

    characteristic_time = (
        diagonal_distance
        / characteristic_velocity
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
        TRAVEL_TIME_SCALE
        / max(
            characteristic_time,
            ANALYTIC_FLOOR,
        )
    )

    print(
        f"T-scale / physical scale  : "
        f"{scale_ratio:.6e}"
    )

    # ==================================================================
    # LOSS-COMPONENT GRADIENT CONTRIBUTION
    # ==================================================================

    print(
        "\n[16] Loss-component gradient contribution"
    )
    print("-" * 78)

    component_results = component_gradient_audit(
        model=model,
        input_tensor=input_tensor,
        velocity_model=velocity_model,
    )

    print(
        "\nComponent             Raw loss       Weight      "
        "Weighted loss       Raw grad norm       "
        "Weighted grad norm       Share"
    )

    print("-" * 110)

    for name, values in component_results.items():

        print(
            f"{name:<20} "
            f"{values['raw_loss']:<13.6e} "
            f"{values['weight']:<10.4f} "
            f"{values['weighted_loss']:<17.6e} "
            f"{values['grad_norm']:<18.6e} "
            f"{values['weighted_grad_norm']:<23.6e} "
            f"{values['weighted_grad_share']:>8.3f}%"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Weighted gradient share is based on "
        "gradient norm × configured loss weight."
    )

    print(
        "It is a diagnostic measure, not a training recommendation."
    )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print("\n" + "=" * 78)
    print("PHYSICS GRADIENT SCALE AUDIT v2.2 COMPLETE")
    print("=" * 78)

    print(
        "\nThis audit does NOT change:"
    )

    print(
        "  - Network3D"
    )

    print(
        "  - PhysicsLoss"
    )

    print(
        "  - LOSS_WEIGHTS"
    )

    print(
        "  - training configuration"
    )

    print(
        "  - model parameters"
    )

    print(
        "  - optimizer state"
    )

    print(
        "\nNo optimizer step was performed."
    )

    print(
        "\nThe results should be interpreted using:"
    )

    print(
        "  1. |grad T| compared with 1/V"
    )

    print(
        "  2. V|grad T| compared with 1"
    )

    print(
        "  3. Eikonal residual magnitude"
    )

    print(
        "  4. Eikonal loss magnitude"
    )

    print(
        "  5. epsilon sensitivity"
    )

    print(
        "  6. isotropic spatial sensitivity"
    )

    print(
        "  7. independent DX, DY and DZ sensitivity"
    )

    print(
        "  8. velocity sensitivity"
    )

    print(
        "  9. absolute travel-time scale sensitivity"
    )

    print(
        " 10. parameter-group gradient distribution"
    )

    print(
        " 11. correctly weighted loss-component gradients"
    )

    print(
        "\nDO NOT change LOSS_WEIGHTS solely from this audit."
    )

    print(
        "First determine whether the observed physics-gradient "
        "dominance originates from physical scaling, travel-time "
        "parameterization/initialization, spatial sampling, velocity "
        "magnitude, or the Eikonal pathway itself."
    )


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    main()