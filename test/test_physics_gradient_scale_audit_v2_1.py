"""
==============================================================================
PHYSICS GRADIENT SCALE AUDIT v2.1
==============================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

PURPOSE
-------
This audit investigates the physical gradient scale and the contribution
of the Eikonal physics loss to the Network3D training gradients.

IMPORTANT
---------
This is a DIAGNOSTIC AUDIT ONLY.

It does NOT modify:
    - Network3D
    - PhysicsLoss
    - TotalLoss
    - config.py
    - LOSS_WEIGHTS
    - TRAVEL_TIME_SCALE

The audit specifically investigates:

    1. Baseline Network3D travel-time field
    2. Exact PhysicsLoss implementation consistency
    3. Spatial sampling sensitivity
    4. Travel-time scale sensitivity
    5. Travel-time parameterization behavior
    6. Physics gradient pathways
    7. Epsilon sensitivity
    8. Physical-scale comparison
    9. Final classification

v2.1 CORRECTIONS
----------------
Compared with v2:

    - Manual Eikonal calculation uses the EXACT derivative implementation
      from PhysicsLoss.

    - Every spatial-spacing experiment creates a NEW PhysicsLoss instance
      with the tested dx, dy, dz values.

    - Actual backward gradients therefore use the tested spacing.

    - No production code is modified.

Author:
    Ormin Joseph
==============================================================================
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import numpy as np
import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cpu")

SEED = 42

TEST_SHAPE = (1, 1, 16, 32, 32)

DX = 1.0
DY = 1.0
DZ = 1.0

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 5000.0

TRAVEL_TIME_SCALE = 0.1

EPSILON = 1.0e-12

TOLERANCE = 1.0e-6


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set deterministic random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# PRINTING UTILITIES
# ============================================================================

def section(title: str) -> None:
    """
    Print a formatted audit section.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def status(name: str, passed: bool) -> None:
    """
    Print PASS/FAIL status.
    """

    print(f"{name:<35}: {'PASS' if passed else 'FAIL'}")


def tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """
    Print basic tensor statistics.
    """

    tensor_detached = tensor.detach()

    print()
    print(name)

    print(
        f"    min   : {tensor_detached.min().item():.8e}"
    )

    print(
        f"    max   : {tensor_detached.max().item():.8e}"
    )

    print(
        f"    mean  : {tensor_detached.mean().item():.8e}"
    )

    print(
        f"    std   : {tensor_detached.std().item():.8e}"
    )


# ============================================================================
# FINITE CHECK
# ============================================================================

def is_finite(tensor: torch.Tensor) -> bool:
    """
    Check whether every element is finite.
    """

    return bool(torch.isfinite(tensor).all().item())


# ============================================================================
# SYNTHETIC INPUT
# ============================================================================

def create_diagnostic_data() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create deterministic diagnostic seismic and velocity fields.
    """

    B, C, D, H, W = TEST_SHAPE

    # ---------------------------------------------------------------
    # Deterministic seismic input.
    # ---------------------------------------------------------------

    z = torch.linspace(-1.0, 1.0, D)
    y = torch.linspace(-1.0, 1.0, H)
    x = torch.linspace(-1.0, 1.0, W)

    zz, yy, xx = torch.meshgrid(
        z,
        y,
        x,
        indexing="ij",
    )

    seismic = (
        0.20 * torch.sin(2.0 * math.pi * xx)
        + 0.15 * torch.cos(2.0 * math.pi * yy)
        + 0.10 * torch.sin(math.pi * zz)
    )

    seismic = seismic.unsqueeze(0).unsqueeze(0)

    # ---------------------------------------------------------------
    # Target.
    # ---------------------------------------------------------------

    target = (
        0.8 * seismic
        + 0.05 * torch.sin(3.0 * math.pi * xx)
        * torch.cos(2.0 * math.pi * yy)
    )

    target = target.unsqueeze(0).unsqueeze(0) \
        if target.dim() == 3 else target

    # ---------------------------------------------------------------
    # Velocity model.
    #
    # Smooth velocity variation between approximately 1500 and 5000 m/s.
    # ---------------------------------------------------------------

    velocity_normalized = (
        0.5
        + 0.5 * torch.sin(math.pi * xx)
        * torch.cos(math.pi * yy)
    )

    velocity = (
        VELOCITY_MIN
        + (
            VELOCITY_MAX - VELOCITY_MIN
        ) * velocity_normalized
    )

    velocity = velocity.clamp(
        min=VELOCITY_MIN,
        max=VELOCITY_MAX,
    )

    velocity = velocity.unsqueeze(0).unsqueeze(0)

    return (
        seismic.float().to(DEVICE),
        target.float().to(DEVICE),
        velocity.float().to(DEVICE),
    )


# ============================================================================
# PHYSICSLOSS CREATION
# ============================================================================

def create_physics_loss(
    dx: float,
    dy: float,
    dz: float,
) -> PhysicsLoss:
    """
    Create a PhysicsLoss using the specified physical sampling.

    IMPORTANT:
    Every sensitivity experiment creates a fresh PhysicsLoss object.
    """

    return PhysicsLoss(
        dx=dx,
        dy=dy,
        dz=dz,
    )


# ============================================================================
# EXACT PHYSICS DERIVATIVES
# ============================================================================

def exact_gradient_components(
    physics: PhysicsLoss,
    travel_time: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Calculate derivatives using PhysicsLoss._derivative().

    Dimension convention:

        dimension 2 -> z
        dimension 3 -> y
        dimension 4 -> x
    """

    dT_dz = physics._derivative(
        travel_time,
        spacing=physics.dz,
        dimension=2,
    )

    dT_dy = physics._derivative(
        travel_time,
        spacing=physics.dy,
        dimension=3,
    )

    dT_dx = physics._derivative(
        travel_time,
        spacing=physics.dx,
        dimension=4,
    )

    return dT_dx, dT_dy, dT_dz


# ============================================================================
# EXACT STABILIZED EIKONAL
# ============================================================================

def exact_stabilized_eikonal(
    physics: PhysicsLoss,
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Reproduce the stabilized PhysicsLoss formulation:

        |grad T| = sqrt(
            dT/dx^2 +
            dT/dy^2 +
            dT/dz^2
        )

        residual = V |grad T| - 1

        loss = mean(residual^2)

    The derivatives are obtained from the production PhysicsLoss.
    """

    dT_dx, dT_dy, dT_dz = exact_gradient_components(
        physics,
        travel_time,
    )

    gradient_squared = (
        dT_dx.pow(2)
        + dT_dy.pow(2)
        + dT_dz.pow(2)
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared.clamp_min(physics.eps)
    )

    residual = (
        velocity * gradient_magnitude
        - 1.0
    )

    loss = residual.pow(2).mean()

    return (
        gradient_magnitude,
        residual,
        loss,
    )


# ============================================================================
# PHYSICSLOSS ACTUAL EIKONAL
# ============================================================================

def actual_eikonal_loss(
    physics: PhysicsLoss,
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """
    Obtain the actual Eikonal loss from PhysicsLoss.

    The preferred route is PhysicsLoss.eikonal_loss().
    """

    if hasattr(physics, "eikonal_loss"):

        return physics.eikonal_loss(
            travel_time,
            velocity,
        )

    # ---------------------------------------------------------------
    # Fallback.
    # ---------------------------------------------------------------

    _, _, loss = exact_stabilized_eikonal(
        physics,
        travel_time,
        velocity,
    )

    return loss


# ============================================================================
# PHYSICSLOSS ACTUAL RESIDUAL
# ============================================================================

def actual_eikonal_residual(
    physics: PhysicsLoss,
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """
    Obtain the actual PhysicsLoss Eikonal residual if exposed.
    """

    if hasattr(physics, "eikonal_residual"):

        return physics.eikonal_residual(
            travel_time,
            velocity,
        )

    # ---------------------------------------------------------------
    # Fallback to exact stabilized implementation.
    # ---------------------------------------------------------------

    _, residual, _ = exact_stabilized_eikonal(
        physics,
        travel_time,
        velocity,
    )

    return residual


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_network() -> Network3D:
    """
    Initialize Network3D deterministically.
    """

    set_seed(SEED)

    model = Network3D()

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================================
# NETWORK FORWARD
# ============================================================================

@torch.no_grad()
def network_forward(
    model: Network3D,
    seismic: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    reconstruction, travel_time, log_variance = model(
        seismic
    )

    return (
        reconstruction,
        travel_time,
        log_variance,
    )


# ============================================================================
# BASELINE AUDIT
# ============================================================================

def baseline_audit(
    model: Network3D,
    seismic: torch.Tensor,
    velocity: torch.Tensor,
    physics: PhysicsLoss,
) -> Dict[str, torch.Tensor]:
    """
    Perform baseline network and Eikonal audit.
    """

    section(
        "1. BASELINE NETWORK AND EIKONAL AUDIT"
    )

    reconstruction, travel_time, log_variance = network_forward(
        model,
        seismic,
    )

    status(
        "Network output shapes",
        reconstruction.shape == TEST_SHAPE
        and travel_time.shape == TEST_SHAPE
        and log_variance.shape == TEST_SHAPE,
    )

    tensor_stats(
        "Reconstruction",
        reconstruction,
    )

    tensor_stats(
        "Travel Time",
        travel_time,
    )

    tensor_stats(
        "Log Variance",
        log_variance,
    )

    status(
        "Reconstruction",
        is_finite(reconstruction),
    )

    status(
        "Travel time",
        is_finite(travel_time),
    )

    status(
        "Log variance",
        is_finite(log_variance),
    )

    status(
        "Travel-time positivity check",
        bool((travel_time > 0.0).all().item()),
    )

    with torch.enable_grad():

        travel_time_grad = travel_time.detach().clone()
        travel_time_grad.requires_grad_(True)

        gradient_magnitude, residual, manual_loss = (
            exact_stabilized_eikonal(
                physics,
                travel_time_grad,
                velocity,
            )
        )

    mean_grad = gradient_magnitude.mean().item()

    mean_inv_velocity = (
        (1.0 / velocity).mean().item()
    )

    mean_v_grad = (
        (velocity * gradient_magnitude)
        .mean()
        .item()
    )

    residual_mean = residual.mean().item()

    ratio = (
        mean_grad
        / max(mean_inv_velocity, 1.0e-30)
    )

    print()
    print("Baseline Eikonal diagnostics:")

    print(
        f"    dT/dx mean        : "
        f"{physics._derivative(travel_time.detach(), physics.dx, 4).mean().item():.6e}"
    )

    print(
        f"    dT/dy mean        : "
        f"{physics._derivative(travel_time.detach(), physics.dy, 3).mean().item():.6e}"
    )

    print(
        f"    dT/dz mean        : "
        f"{physics._derivative(travel_time.detach(), physics.dz, 2).mean().item():.6e}"
    )

    print(
        f"    |grad T| mean     : "
        f"{mean_grad:.6e}"
    )

    print(
        f"    1/V mean          : "
        f"{mean_inv_velocity:.6e}"
    )

    print(
        f"    V|grad T| mean    : "
        f"{mean_v_grad:.6e}"
    )

    print(
        f"    residual mean      : "
        f"{residual_mean:.6e}"
    )

    print(
        f"    Eikonal loss       : "
        f"{manual_loss.item():.6e}"
    )

    print(
        f"    |gradT|/(1/V)      : "
        f"{ratio:.6e}"
    )

    return {
        "reconstruction": reconstruction,
        "travel_time": travel_time,
        "log_variance": log_variance,
    }


# ============================================================================
# IMPLEMENTATION CONSISTENCY
# ============================================================================

def implementation_consistency_audit(
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> bool:
    """
    Verify that the audit reproduces PhysicsLoss exactly.
    """

    section(
        "2. EXACT PHYSICSLOSS IMPLEMENTATION CONSISTENCY"
    )

    physics = create_physics_loss(
        DX,
        DY,
        DZ,
    )

    with torch.enable_grad():

        T = travel_time.detach().clone()
        T.requires_grad_(True)

        (
            gradient_magnitude,
            manual_residual,
            manual_loss,
        ) = exact_stabilized_eikonal(
            physics,
            T,
            velocity,
        )

        actual_residual = actual_eikonal_residual(
            physics,
            T,
            velocity,
        )

        actual_loss = actual_eikonal_loss(
            physics,
            T,
            velocity,
        )

    residual_difference = (
        manual_residual
        - actual_residual
    ).abs().max().item()

    loss_difference = abs(
        manual_loss.item()
        - actual_loss.item()
    )

    print(
        f"Manual Eikonal loss        : "
        f"{manual_loss.item():.8e}"
    )

    print(
        f"PhysicsLoss Eikonal loss   : "
        f"{actual_loss.item():.8e}"
    )

    print(
        f"Loss absolute difference   : "
        f"{loss_difference:.8e}"
    )

    print(
        f"Residual max difference    : "
        f"{residual_difference:.8e}"
    )

    passed = (
        loss_difference <= TOLERANCE
        and residual_difference <= TOLERANCE
    )

    status(
        "Implementation consistency",
        passed,
    )

    if not passed:

        print()
        print(
            "ERROR: The audit calculation does not "
            "match the production PhysicsLoss."
        )

        raise RuntimeError(
            "PhysicsLoss implementation consistency failed."
        )

    return True


# ============================================================================
# SPATIAL SAMPLING SENSITIVITY
# ============================================================================

def spatial_sampling_sensitivity(
    model: Network3D,
    seismic: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Test independent dx, dy and dz sensitivity.

    IMPORTANT:
    Each test creates a NEW PhysicsLoss object.

    This means the actual backward pass uses the tested spacing.
    """

    section(
        "3. CORRECTED SPATIAL SAMPLING SENSITIVITY"
    )

    with torch.enable_grad():

        # ---------------------------------------------------------------
        # Obtain one deterministic network travel-time field.
        # ---------------------------------------------------------------

        _, travel_time, _ = model(
            seismic,
        )

        base_travel_time = (
            travel_time.detach()
        )

        # ---------------------------------------------------------------
        # DX sensitivity.
        # ---------------------------------------------------------------

        print()
        print("DX sensitivity")

        print(
            f"{'Factor':>10}"
            f"{'DX':>12}"
            f"{'Loss':>18}"
            f"{'GradNorm':>18}"
            f"{'Mean |gradT|':>18}"
        )

        for factor in (0.5, 1.0, 2.0):

            test_dx = DX * factor

            physics = create_physics_loss(
                test_dx,
                DY,
                DZ,
            )

            T = base_travel_time.clone()
            T.requires_grad_(True)

            _, _, loss = exact_stabilized_eikonal(
                physics,
                T,
                velocity,
            )

            gradient_T = torch.autograd.grad(
                loss,
                T,
                retain_graph=False,
                create_graph=False,
            )[0]

            gradient_norm = (
                gradient_T.norm().item()
            )

            with torch.no_grad():

                grad_mag, _, _ = (
                    exact_stabilized_eikonal(
                        physics,
                        base_travel_time,
                        velocity,
                    )
                )

                mean_grad = (
                    grad_mag.mean().item()
                )

            print(
                f"{factor:>10.2f}"
                f"{test_dx:>12.4f}"
                f"{loss.item():>18.8e}"
                f"{gradient_norm:>18.8e}"
                f"{mean_grad:>18.8e}"
            )

        # ---------------------------------------------------------------
        # DY sensitivity.
        # ---------------------------------------------------------------

        print()
        print("DY sensitivity")

        print(
            f"{'Factor':>10}"
            f"{'DY':>12}"
            f"{'Loss':>18}"
            f"{'GradNorm':>18}"
            f"{'Mean |gradT|':>18}"
        )

        for factor in (0.5, 1.0, 2.0):

            test_dy = DY * factor

            physics = create_physics_loss(
                DX,
                test_dy,
                DZ,
            )

            T = base_travel_time.clone()
            T.requires_grad_(True)

            _, _, loss = exact_stabilized_eikonal(
                physics,
                T,
                velocity,
            )

            gradient_T = torch.autograd.grad(
                loss,
                T,
                retain_graph=False,
                create_graph=False,
            )[0]

            gradient_norm = (
                gradient_T.norm().item()
            )

            with torch.no_grad():

                grad_mag, _, _ = (
                    exact_stabilized_eikonal(
                        physics,
                        base_travel_time,
                        velocity,
                    )
                )

                mean_grad = (
                    grad_mag.mean().item()
                )

            print(
                f"{factor:>10.2f}"
                f"{test_dy:>12.4f}"
                f"{loss.item():>18.8e}"
                f"{gradient_norm:>18.8e}"
                f"{mean_grad:>18.8e}"
            )

        # ---------------------------------------------------------------
        # DZ sensitivity.
        # ---------------------------------------------------------------

        print()
        print("DZ sensitivity")

        print(
            f"{'Factor':>10}"
            f"{'DZ':>12}"
            f"{'Loss':>18}"
            f"{'GradNorm':>18}"
            f"{'Mean |gradT|':>18}"
        )

        for factor in (0.5, 1.0, 2.0):

            test_dz = DZ * factor

            physics = create_physics_loss(
                DX,
                DY,
                test_dz,
            )

            T = base_travel_time.clone()
            T.requires_grad_(True)

            _, _, loss = exact_stabilized_eikonal(
                physics,
                T,
                velocity,
            )

            gradient_T = torch.autograd.grad(
                loss,
                T,
                retain_graph=False,
                create_graph=False,
            )[0]

            gradient_norm = (
                gradient_T.norm().item()
            )

            with torch.no_grad():

                grad_mag, _, _ = (
                    exact_stabilized_eikonal(
                        physics,
                        base_travel_time,
                        velocity,
                    )
                )

                mean_grad = (
                    grad_mag.mean().item()
                )

            print(
                f"{factor:>10.2f}"
                f"{test_dz:>12.4f}"
                f"{loss.item():>18.8e}"
                f"{gradient_norm:>18.8e}"
                f"{mean_grad:>18.8e}"
            )

    print()
    print(
        "RESULT: Spatial sensitivity uses the actual "
        "tested PhysicsLoss spacing."
    )


# ============================================================================
# TRAVEL-TIME SCALE SENSITIVITY
# ============================================================================

def travel_time_scale_sensitivity(
    model: Network3D,
    seismic: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Investigate how multiplying the network travel-time output changes
    the Eikonal loss and gradient scale.
    """

    section(
        "4. TRAVEL-TIME SCALE SENSITIVITY"
    )

    physics = create_physics_loss(
        DX,
        DY,
        DZ,
    )

    with torch.no_grad():

        _, base_T, _ = model(
            seismic,
        )

        base_T = base_T.detach()

    print()
    print(
        f"{'Scale':>10}"
        f"{'Mean T':>18}"
        f"{'Mean |gradT|':>18}"
        f"{'Mean V|gradT|':>18}"
        f"{'Eikonal Loss':>18}"
        f"{'GradNorm(T)':>18}"
    )

    for scale in (
        0.01,
        0.02,
        0.05,
        0.10,
    ):

        T = base_T * (
            scale / TRAVEL_TIME_SCALE
        )

        T.requires_grad_(True)

        grad_mag, residual, loss = (
            exact_stabilized_eikonal(
                physics,
                T,
                velocity,
            )
        )

        grad_T = torch.autograd.grad(
            loss,
            T,
            retain_graph=False,
            create_graph=False,
        )[0]

        print(
            f"{scale:>10.4f}"
            f"{T.mean().item():>18.8e}"
            f"{grad_mag.mean().item():>18.8e}"
            f"{(velocity * grad_mag).mean().item():>18.8e}"
            f"{loss.item():>18.8e}"
            f"{grad_T.norm().item():>18.8e}"
        )


# ============================================================================
# TRAVEL-TIME PARAMETERIZATION
# ============================================================================

def travel_time_parameterization_audit(
    model: Network3D,
    seismic: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Examine the current Network3D travel-time parameterization.

    This does not modify the model.

    It reports:

        - positivity
        - output range
        - output variation
        - relationship to characteristic physical travel time
    """

    section(
        "5. TRAVEL-TIME PARAMETERIZATION AUDIT"
    )

    with torch.no_grad():

        _, travel_time, _ = model(
            seismic,
        )

    mean_T = travel_time.mean().item()

    std_T = travel_time.std().item()

    min_T = travel_time.min().item()

    max_T = travel_time.max().item()

    print(
        f"Travel-time mean       : {mean_T:.8e}"
    )

    print(
        f"Travel-time std        : {std_T:.8e}"
    )

    print(
        f"Travel-time minimum    : {min_T:.8e}"
    )

    print(
        f"Travel-time maximum    : {max_T:.8e}"
    )

    status(
        "Travel-time positivity",
        bool((travel_time > 0.0).all().item()),
    )

    # ---------------------------------------------------------------
    # Physical domain dimensions.
    # ---------------------------------------------------------------

    _, _, D, H, W = TEST_SHAPE

    depth_extent = (
        max(D - 1, 1) * DZ
    )

    y_extent = (
        max(H - 1, 1) * DY
    )

    x_extent = (
        max(W - 1, 1) * DX
    )

    diagonal_distance = math.sqrt(
        depth_extent ** 2
        + y_extent ** 2
        + x_extent ** 2
    )

    characteristic_velocity = (
        velocity.mean().item()
    )

    characteristic_time = (
        diagonal_distance
        / characteristic_velocity
    )

    print()
    print(
        f"Depth extent           : "
        f"{depth_extent:.8e} m"
    )

    print(
        f"Y extent               : "
        f"{y_extent:.8e} m"
    )

    print(
        f"X extent               : "
        f"{x_extent:.8e} m"
    )

    print(
        f"3D diagonal distance   : "
        f"{diagonal_distance:.8e} m"
    )

    print(
        f"Mean velocity          : "
        f"{characteristic_velocity:.8e} m/s"
    )

    print(
        f"Characteristic T       : "
        f"{characteristic_time:.8e} s"
    )

    print(
        f"Configured T scale     : "
        f"{TRAVEL_TIME_SCALE:.8e} s"
    )

    print(
        f"T-scale / characteristic T : "
        f"{TRAVEL_TIME_SCALE / characteristic_time:.8e}"
    )


# ============================================================================
# PHYSICS GRADIENT PATHWAY
# ============================================================================

def physics_gradient_pathway_audit(
    model: Network3D,
    seismic: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Measure the actual parameter gradient generated by the PhysicsLoss.

    This checks whether the physics gradient reaches:

        - reconstruction head
        - travel-time head
        - uncertainty head

    The physics loss should primarily affect the travel-time pathway.
    """

    section(
        "6. PHYSICS GRADIENT PATHWAY AUDIT"
    )

    model.zero_grad(set_to_none=True)

    physics = create_physics_loss(
        DX,
        DY,
        DZ,
    )

    reconstruction, travel_time, log_variance = (
        model(seismic)
    )

    physics_loss = actual_eikonal_loss(
        physics,
        travel_time,
        velocity,
    )

    physics_loss.backward()

    total_norm_squared = 0.0

    recon_norm_squared = 0.0

    travel_norm_squared = 0.0

    uncertainty_norm_squared = 0.0

    max_gradient = 0.0

    finite = True

    recon_count = 0
    travel_count = 0
    uncertainty_count = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if not torch.isfinite(gradient).all():
            finite = False

        norm = gradient.norm().item()

        total_norm_squared += norm ** 2

        max_gradient = max(
            max_gradient,
            gradient.abs().max().item(),
        )

        name_lower = name.lower()

        if "travel" in name_lower:

            travel_norm_squared += norm ** 2
            travel_count += 1

        elif (
            "uncertainty" in name_lower
            or "variance" in name_lower
            or "logvar" in name_lower
        ):

            uncertainty_norm_squared += norm ** 2
            uncertainty_count += 1

        else:

            recon_norm_squared += norm ** 2
            recon_count += 1

    total_norm = math.sqrt(
        total_norm_squared
    )

    recon_norm = math.sqrt(
        recon_norm_squared
    )

    travel_norm = math.sqrt(
        travel_norm_squared
    )

    uncertainty_norm = math.sqrt(
        uncertainty_norm_squared
    )

    print(
        f"Physics loss           : "
        f"{physics_loss.item():.8e}"
    )

    print(
        f"Total gradient norm    : "
        f"{total_norm:.8e}"
    )

    print(
        f"Maximum parameter grad : "
        f"{max_gradient:.8e}"
    )

    print(
        f"Reconstruction path    : "
        f"{recon_norm:.8e}"
    )

    print(
        f"Travel-time path       : "
        f"{travel_norm:.8e}"
    )

    print(
        f"Uncertainty path       : "
        f"{uncertainty_norm:.8e}"
    )

    print(
        f"Finite gradients       : "
        f"{finite}"
    )

    print(
        f"Recon parameter groups : "
        f"{recon_count}"
    )

    print(
        f"Travel parameter groups: "
        f"{travel_count}"
    )

    print(
        f"Uncertainty groups     : "
        f"{uncertainty_count}"
    )

    status(
        "Physics gradients finite",
        finite,
    )

    model.zero_grad(set_to_none=True)


# ============================================================================
# EPSILON SENSITIVITY
# ============================================================================

def epsilon_sensitivity(
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Investigate the effect of the numerical gradient epsilon.

    No production epsilon is changed.
    """

    section(
        "7. EPSILON SENSITIVITY"
    )

    print(
        f"{'Epsilon':>14}"
        f"{'Mean grad²':>18}"
        f"{'Median grad²':>18}"
        f"{'Fraction < eps':>18}"
        f"{'Mean |gradT|':>18}"
        f"{'Loss':>18}"
    )

    for eps in (
        1.0e-16,
        1.0e-12,
        1.0e-10,
        1.0e-8,
    ):

        physics = create_physics_loss(
            DX,
            DY,
            DZ,
        )

        physics.eps = eps

        T = travel_time.detach().clone()
        T.requires_grad_(True)

        dT_dx, dT_dy, dT_dz = (
            exact_gradient_components(
                physics,
                T,
            )
        )

        gradient_squared = (
            dT_dx.pow(2)
            + dT_dy.pow(2)
            + dT_dz.pow(2)
        )

        gradient_magnitude = torch.sqrt(
            gradient_squared.clamp_min(eps)
        )

        residual = (
            velocity * gradient_magnitude
            - 1.0
        )

        loss = residual.pow(2).mean()

        fraction_below = (
            gradient_squared < eps
        ).float().mean().item()

        print(
            f"{eps:>14.2e}"
            f"{gradient_squared.mean().item():>18.8e}"
            f"{gradient_squared.median().item():>18.8e}"
            f"{fraction_below:>18.6%}"
            f"{gradient_magnitude.mean().item():>18.8e}"
            f"{loss.item():>18.8e}"
        )


# ============================================================================
# PHYSICAL SCALE COMPARISON
# ============================================================================

def physical_scale_comparison(
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Compare network gradient magnitude against the physical Eikonal target.
    """

    section(
        "8. PHYSICAL-SCALE COMPARISON"
    )

    physics = create_physics_loss(
        DX,
        DY,
        DZ,
    )

    with torch.no_grad():

        grad_mag, residual, loss = (
            exact_stabilized_eikonal(
                physics,
                travel_time.detach(),
                velocity,
            )
        )

        required_gradient = (
            1.0 / velocity
        )

        ratio = (
            grad_mag
            / required_gradient.clamp_min(1.0e-30)
        )

    print(
        f"Actual |gradT| mean       : "
        f"{grad_mag.mean().item():.8e}"
    )

    print(
        f"Required 1/V mean         : "
        f"{required_gradient.mean().item():.8e}"
    )

    print(
        f"Mean ratio actual/required: "
        f"{ratio.mean().item():.8e}"
    )

    print(
        f"Median ratio              : "
        f"{ratio.median().item():.8e}"
    )

    print(
        f"Eikonal residual mean     : "
        f"{residual.mean().item():.8e}"
    )

    print(
        f"Eikonal loss              : "
        f"{loss.item():.8e}"
    )

    print()
    print(
        "Interpretation:"
    )

    ratio_mean = ratio.mean().item()

    if ratio_mean < 0.1:

        print(
            "    The network travel-time gradient is "
            "far below the physical Eikonal target."
        )

    elif ratio_mean < 0.5:

        print(
            "    The network travel-time gradient is "
            "below the physical target."
        )

    elif ratio_mean <= 2.0:

        print(
            "    The network travel-time gradient is "
            "of the correct physical order."
        )

    else:

        print(
            "    The network travel-time gradient is "
            "above the physical target."
        )


# ============================================================================
# FINAL CLASSIFICATION
# ============================================================================

def final_classification(
    travel_time: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Provide a diagnostic classification.

    This does NOT alter any configuration.
    """

    section(
        "9. FINAL CLASSIFICATION"
    )

    physics = create_physics_loss(
        DX,
        DY,
        DZ,
    )

    with torch.no_grad():

        grad_mag, residual, loss = (
            exact_stabilized_eikonal(
                physics,
                travel_time,
                velocity,
            )
        )

        target_gradient = (
            1.0 / velocity
        )

        ratio = (
            grad_mag
            / target_gradient.clamp_min(1.0e-30)
        )

    ratio_mean = ratio.mean().item()

    loss_value = loss.item()

    print(
        f"Mean physical gradient ratio : "
        f"{ratio_mean:.8e}"
    )

    print(
        f"Eikonal loss                 : "
        f"{loss_value:.8e}"
    )

    print()

    # ---------------------------------------------------------------
    # Classification.
    # ---------------------------------------------------------------

    if ratio_mean < 0.05:

        print(
            "CLASSIFICATION:"
        )

        print(
            "    SEVERE TRAVEL-TIME GRADIENT UNDER-SCALE"
        )

        print(
            "    The initialized travel-time field is "
            "nearly spatially constant."
        )

    elif ratio_mean < 0.25:

        print(
            "CLASSIFICATION:"
        )

        print(
            "    STRONG TRAVEL-TIME GRADIENT UNDER-SCALE"
        )

        print(
            "    The network gradient is substantially "
            "below the physical Eikonal requirement."
        )

    elif ratio_mean < 0.75:

        print(
            "CLASSIFICATION:"
        )

        print(
            "    MODERATE TRAVEL-TIME GRADIENT UNDER-SCALE"
        )

    elif ratio_mean <= 1.5:

        print(
            "CLASSIFICATION:"
        )

        print(
            "    PHYSICALLY REASONABLE TRAVEL-TIME GRADIENT SCALE"
        )

    else:

        print(
            "CLASSIFICATION:"
        )

        print(
            "    TRAVEL-TIME GRADIENT MAY BE OVER-SCALED"
        )

    print()
    print(
        "NO MODEL OR CONFIGURATION WAS MODIFIED."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run complete Physics Gradient Scale Audit v2.1.
    """

    section(
        "PHYSICS GRADIENT SCALE AUDIT v2.1"
    )

    print(
        "Physics-Informed 3D Encoder-Decoder Framework"
    )

    print(
        "Predictive Uncertainty for Seismic Data Reconstruction"
    )

    print()

    print(
        f"Device                 : {DEVICE}"
    )

    print(
        f"Input shape            : {TEST_SHAPE}"
    )

    print(
        f"Seed                   : {SEED}"
    )

    print(
        f"DX                     : {DX}"
    )

    print(
        f"DY                     : {DY}"
    )

    print(
        f"DZ                     : {DZ}"
    )

    print(
        f"Velocity minimum       : {VELOCITY_MIN}"
    )

    print(
        f"Velocity maximum       : {VELOCITY_MAX}"
    )

    print(
        f"Travel-time scale      : "
        f"{TRAVEL_TIME_SCALE}"
    )

    # ------------------------------------------------------------------------
    # Seed.
    # ------------------------------------------------------------------------

    set_seed(SEED)

    # ------------------------------------------------------------------------
    # Data.
    # ------------------------------------------------------------------------

    section(
        "CREATING DETERMINISTIC SYNTHETIC DATA"
    )

    seismic, target, velocity = (
        create_diagnostic_data()
    )

    status(
        "Input",
        seismic.shape == TEST_SHAPE
        and is_finite(seismic),
    )

    status(
        "Target",
        target.shape == TEST_SHAPE
        and is_finite(target),
    )

    status(
        "Velocity",
        velocity.shape == TEST_SHAPE
        and is_finite(velocity),
    )

    status(
        "Velocity positivity check",
        bool((velocity > 0.0).all().item()),
    )

    tensor_stats(
        "Velocity",
        velocity,
    )

    # ------------------------------------------------------------------------
    # Network.
    # ------------------------------------------------------------------------

    section(
        "INITIALIZING NETWORK3D"
    )

    model = initialize_network()

    print(
        "Network3D initialized successfully."
    )

    # ------------------------------------------------------------------------
    # Baseline.
    # ------------------------------------------------------------------------

    baseline_outputs = baseline_audit(
        model,
        seismic,
        velocity,
        create_physics_loss(
            DX,
            DY,
            DZ,
        ),
    )

    travel_time = baseline_outputs[
        "travel_time"
    ]

    # ------------------------------------------------------------------------
    # Exact implementation consistency.
    # ------------------------------------------------------------------------

    implementation_consistency_audit(
        travel_time,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Corrected spatial sensitivity.
    # ------------------------------------------------------------------------

    spatial_sampling_sensitivity(
        model,
        seismic,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Travel-time scale.
    # ------------------------------------------------------------------------

    travel_time_scale_sensitivity(
        model,
        seismic,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Parameterization.
    # ------------------------------------------------------------------------

    travel_time_parameterization_audit(
        model,
        seismic,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Gradient pathways.
    # ------------------------------------------------------------------------

    physics_gradient_pathway_audit(
        model,
        seismic,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Epsilon.
    # ------------------------------------------------------------------------

    epsilon_sensitivity(
        travel_time,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Physical scale.
    # ------------------------------------------------------------------------

    physical_scale_comparison(
        travel_time,
        velocity,
    )

    # ------------------------------------------------------------------------
    # Final classification.
    # ------------------------------------------------------------------------

    final_classification(
        travel_time,
        velocity,
    )

    # ------------------------------------------------------------------------
    # End.
    # ------------------------------------------------------------------------

    section(
        "AUDIT COMPLETE"
    )

    print(
        "Physics Gradient Scale Audit v2.1 completed."
    )

    print(
        "No production model, loss, or configuration was modified."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()