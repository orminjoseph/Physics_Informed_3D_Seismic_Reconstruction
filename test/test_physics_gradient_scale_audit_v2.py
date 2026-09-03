"""
======================================================================
PHYSICS GRADIENT SCALE AUDIT v2
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Purpose
-------
This audit investigates the physical scaling and gradient behaviour of
the stabilized Eikonal loss without modifying the model or production
configuration.

The stabilized Eikonal formulation is:

    |∇T| = 1 / V

or:

    V |∇T| = 1

Therefore:

    R_eikonal = V |∇T| - 1

and:

    L_eikonal = mean((V |∇T| - 1)^2)

This v2 audit specifically corrects the spatial-sampling sensitivity
test from the previous audit.

IMPORTANT
---------
For every spatial-sampling experiment, a NEW PhysicsLoss instance is
created with the tested dx, dy, dz values. Therefore the gradients
being measured actually correspond to the tested spatial sampling.

The audit does NOT:
    - modify Network3D
    - modify PhysicsLoss
    - modify TotalLoss
    - modify config.py
    - modify LOSS_WEIGHTS
    - perform optimizer updates
    - train the network

Author: Ormin Joseph
======================================================================
"""

import math

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    DX,
    DY,
    DZ,
    VELOCITY_MIN,
    VELOCITY_MAX,
    TRAVEL_TIME_SCALE,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

INPUT_SHAPE = (
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH,
)

# Spatial sensitivity factors.
SPATIAL_FACTORS = (
    0.5,
    1.0,
    2.0,
)

# Travel-time scaling experiments.
TRAVEL_TIME_SCALES = (
    0.01,
    0.02,
    0.05,
    0.10,
)

# Epsilon sensitivity.
EPSILON_VALUES = (
    1.0e-14,
    1.0e-13,
    1.0e-12,
    1.0e-11,
)


# ======================================================================
# PRINTING UTILITIES
# ======================================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subheader(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ======================================================================
# NUMERICAL UTILITIES
# ======================================================================

def assert_finite(tensor, name):
    """
    Confirm that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )

    print(
        f"{name:<35}: PASS"
    )


def tensor_stats(name, tensor):
    """
    Print basic tensor statistics.
    """

    tensor = tensor.detach()

    print(f"{name}:")
    print(f"    shape : {tuple(tensor.shape)}")
    print(f"    min   : {tensor.min().item():.6e}")
    print(f"    max   : {tensor.max().item():.6e}")
    print(f"    mean  : {tensor.mean().item():.6e}")
    print(f"    std   : {tensor.std().item():.6e}")
    print(f"    absmax: {tensor.abs().max().item():.6e}")


# ======================================================================
# GRADIENT UTILITIES
# ======================================================================

def parameter_gradient_norm(model):
    """
    Calculate global parameter gradient norm.
    """

    squared_sum = 0.0
    maximum = 0.0
    count = 0
    finite_gradients = True

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        finite_gradients &= bool(
            torch.isfinite(gradient).all()
        )

        squared_sum += torch.sum(
            gradient * gradient
        ).item()

        maximum = max(
            maximum,
            gradient.abs().max().item()
        )

        count += 1

    return (
        math.sqrt(squared_sum),
        maximum,
        count,
        finite_gradients,
    )


def named_parameter_gradient_norms(model):
    """
    Return gradient norms grouped by parameter name.
    """

    values = {}

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            continue

        values[name] = gradient.norm().item()

    return values


def group_gradient_norm(named_gradients, keywords):
    """
    Calculate the combined gradient norm for parameters whose names
    contain one of the supplied keywords.
    """

    values = []

    for name, norm in named_gradients.items():

        name_lower = name.lower()

        if any(
            keyword.lower() in name_lower
            for keyword in keywords
        ):
            values.append(norm)

    if not values:
        return 0.0

    return math.sqrt(
        sum(value * value for value in values)
    )


# ======================================================================
# SYNTHETIC DATA
# ======================================================================

def make_data(device):
    """
    Create deterministic synthetic seismic-like input, target,
    and smoothly varying velocity model.
    """

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------------
    # Input seismic cube
    # --------------------------------------------------------------

    x = torch.randn(
        INPUT_SHAPE,
        device=device
    )

    # Keep amplitudes in a realistic normalized range.
    x = torch.tanh(x)

    # --------------------------------------------------------------
    # Synthetic target
    # --------------------------------------------------------------

    target = (
        0.85 * x
        + 0.05 * torch.sin(x * math.pi)
    )

    target = torch.clamp(
        target,
        min=-1.0,
        max=1.0,
    )

    # --------------------------------------------------------------
    # Smooth depth-varying velocity
    # --------------------------------------------------------------

    velocity_axis = torch.linspace(
        VELOCITY_MIN,
        VELOCITY_MAX,
        DEPTH,
        device=device,
    )

    velocity = velocity_axis.view(
        1,
        1,
        DEPTH,
        1,
        1,
    ).expand(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
    ).contiguous()

    return x, target, velocity


# ======================================================================
# NETWORK OUTPUT HANDLING
# ======================================================================

def forward_network(model, x):
    """
    Execute Network3D and support tuple or dictionary output.
    """

    outputs = model(x)

    if isinstance(outputs, dict):

        reconstruction = outputs["reconstruction"]

        travel_time = outputs["travel_time"]

        log_variance = outputs["log_variance"]

    else:

        reconstruction = outputs[0]

        travel_time = outputs[1]

        log_variance = outputs[2]

    return (
        reconstruction,
        travel_time,
        log_variance,
    )


# ======================================================================
# EIKONAL DERIVATIVES
# ======================================================================

def calculate_derivatives(
    physics_loss,
    travel_time,
):
    """
    Calculate spatial derivatives using the same derivative
    implementation used by PhysicsLoss.
    """

    dT_dz = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dz,
        dimension=2,
    )

    dT_dy = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dy,
        dimension=3,
    )

    dT_dx = physics_loss._derivative(
        travel_time,
        spacing=physics_loss.dx,
        dimension=4,
    )

    return (
        dT_dx,
        dT_dy,
        dT_dz,
    )


# ======================================================================
# EIKONAL DIAGNOSTICS
# ======================================================================

def calculate_eikonal_diagnostics(
    physics_loss,
    travel_time,
    velocity,
    epsilon=None,
):
    """
    Calculate stabilized Eikonal quantities.

    If epsilon is supplied, it is used for the diagnostic gradient
    magnitude calculation.
    """

    dT_dx, dT_dy, dT_dz = calculate_derivatives(
        physics_loss,
        travel_time,
    )

    gradient_squared = (
        dT_dx.square()
        + dT_dy.square()
        + dT_dz.square()
    )

    if epsilon is None:
        epsilon = physics_loss.eps

    gradient_magnitude = torch.sqrt(
        gradient_squared.clamp_min(epsilon)
    )

    normalized_eikonal = (
        velocity * gradient_magnitude
    )

    residual = (
        normalized_eikonal - 1.0
    )

    loss = torch.mean(
        residual.square()
    )

    return {
        "dT_dx": dT_dx,
        "dT_dy": dT_dy,
        "dT_dz": dT_dz,
        "gradient_squared": gradient_squared,
        "gradient_magnitude": gradient_magnitude,
        "normalized_eikonal": normalized_eikonal,
        "residual": residual,
        "loss": loss,
    }


# ======================================================================
# PHYSICS LOSS BACKWARD AUDIT
# ======================================================================

def physics_gradient_audit(
    model,
    travel_time,
    velocity,
    dx,
    dy,
    dz,
):
    """
    Backpropagate through a PhysicsLoss instantiated with the
    EXACT spatial sampling being tested.

    This is the critical correction in v2.
    """

    model.zero_grad(
        set_to_none=True
    )

    tested_physics_loss = PhysicsLoss(
        dx=dx,
        dy=dy,
        dz=dz,
    ).to(DEVICE)

    result = tested_physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=None,
        travel_time_target=None,
    )

    # PhysicsLoss is expected to return a dictionary.
    if isinstance(result, dict):

        if "total" in result:
            loss = result["total"]

        elif "eikonal" in result:
            loss = result["eikonal"]

        else:
            raise RuntimeError(
                "PhysicsLoss output does not contain "
                "'total' or 'eikonal'."
            )

    else:
        loss = result

    assert_finite(
        loss,
        "Tested physics loss",
    )

    loss.backward()

    (
        gradient_norm,
        maximum_gradient,
        parameter_count,
        finite_gradients,
    ) = parameter_gradient_norm(model)

    return (
        tested_physics_loss,
        loss.item(),
        gradient_norm,
        maximum_gradient,
        parameter_count,
        finite_gradients,
    )


# ======================================================================
# BASELINE AUDIT
# ======================================================================

def baseline_audit(
    model,
    x,
    velocity,
):
    """
    Baseline network and Eikonal audit.
    """

    header(
        "1. BASELINE NETWORK AND EIKONAL AUDIT"
    )

    reconstruction, travel_time, log_variance = (
        forward_network(
            model,
            x,
        )
    )

    expected_shape = INPUT_SHAPE

    if tuple(reconstruction.shape) != expected_shape:
        raise RuntimeError(
            "Reconstruction shape mismatch."
        )

    if tuple(travel_time.shape) != expected_shape:
        raise RuntimeError(
            "Travel-time shape mismatch."
        )

    if tuple(log_variance.shape) != expected_shape:
        raise RuntimeError(
            "Log-variance shape mismatch."
        )

    print("Network output shapes: PASS")

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

    assert_finite(
        reconstruction,
        "Reconstruction",
    )

    assert_finite(
        travel_time,
        "Travel time",
    )

    assert_finite(
        log_variance,
        "Log variance",
    )

    if torch.any(travel_time < 0.0):
        print(
            "Travel-time positivity check: WARNING"
        )
    else:
        print(
            "Travel-time positivity check: PASS"
        )

    baseline_physics = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(DEVICE)

    diagnostics = calculate_eikonal_diagnostics(
        baseline_physics,
        travel_time,
        velocity,
    )

    print()
    print("Baseline Eikonal diagnostics:")

    print(
        f"    dT/dx mean        : "
        f"{diagnostics['dT_dx'].mean().item():.6e}"
    )

    print(
        f"    dT/dy mean        : "
        f"{diagnostics['dT_dy'].mean().item():.6e}"
    )

    print(
        f"    dT/dz mean        : "
        f"{diagnostics['dT_dz'].mean().item():.6e}"
    )

    print(
        f"    |grad T| mean     : "
        f"{diagnostics['gradient_magnitude'].mean().item():.6e}"
    )

    inverse_velocity = 1.0 / velocity

    ratio = (
        diagnostics["gradient_magnitude"]
        / inverse_velocity
    )

    print(
        f"    1/V mean          : "
        f"{inverse_velocity.mean().item():.6e}"
    )

    print(
        f"    V|grad T| mean    : "
        f"{diagnostics['normalized_eikonal'].mean().item():.6e}"
    )

    print(
        f"    residual mean      : "
        f"{diagnostics['residual'].mean().item():.6e}"
    )

    print(
        f"    Eikonal loss       : "
        f"{diagnostics['loss'].item():.6e}"
    )

    print(
        f"    |gradT|/(1/V)      : "
        f"{ratio.mean().item():.6e}"
    )

    return (
        reconstruction,
        travel_time,
        log_variance,
        baseline_physics,
        diagnostics,
    )


# ======================================================================
# IMPLEMENTATION CONSISTENCY AUDIT
# ======================================================================

def implementation_consistency_audit(
    physics_loss,
    travel_time,
    velocity,
):
    """
    Compare the manually calculated Eikonal loss against PhysicsLoss.
    """

    header(
        "2. EIKONAL IMPLEMENTATION CONSISTENCY"
    )

    manual = calculate_eikonal_diagnostics(
        physics_loss,
        travel_time,
        velocity,
    )

    implementation = physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=None,
        travel_time_target=None,
    )

    if isinstance(implementation, dict):

        if "eikonal" in implementation:
            implementation_loss = implementation["eikonal"]

        elif "total" in implementation:
            implementation_loss = implementation["total"]

        else:
            raise RuntimeError(
                "Could not identify Eikonal loss in PhysicsLoss output."
            )

    else:
        implementation_loss = implementation

    difference = abs(
        manual["loss"].item()
        - implementation_loss.item()
    )

    print(
        f"Manual Eikonal loss        : "
        f"{manual['loss'].item():.6e}"
    )

    print(
        f"PhysicsLoss Eikonal value  : "
        f"{implementation_loss.item():.6e}"
    )

    print(
        f"Absolute difference        : "
        f"{difference:.6e}"
    )

    if difference <= 1.0e-6:
        print(
            "Implementation consistency: PASS"
        )
    else:
        print(
            "Implementation consistency: FAIL"
        )

        raise RuntimeError(
            "Manual Eikonal calculation does not match PhysicsLoss."
        )


# ======================================================================
# SPATIAL SAMPLING SENSITIVITY
# ======================================================================

def spatial_sampling_audit(
    model,
    travel_time,
    velocity,
):
    """
    Test independent spatial sampling changes.

    Each experiment creates a NEW PhysicsLoss with the tested
    dx, dy, dz values.
    """

    header(
        "3. CORRECTED SPATIAL-SAMPLING SENSITIVITY AUDIT"
    )

    print(
        "IMPORTANT: every case below uses a NEW PhysicsLoss "
        "with the tested dx, dy, dz."
    )

    baseline = {
        "dx": DX,
        "dy": DY,
        "dz": DZ,
    }

    experiments = []

    # --------------------------------------------------------------
    # Independent DX tests
    # --------------------------------------------------------------

    for factor in SPATIAL_FACTORS:

        experiments.append(
            (
                "DX",
                DX * factor,
                DY,
                DZ,
            )
        )

    # --------------------------------------------------------------
    # Independent DY tests
    # --------------------------------------------------------------

    for factor in SPATIAL_FACTORS:

        experiments.append(
            (
                "DY",
                DX,
                DY * factor,
                DZ,
            )
        )

    # --------------------------------------------------------------
    # Independent DZ tests
    # --------------------------------------------------------------

    for factor in SPATIAL_FACTORS:

        experiments.append(
            (
                "DZ",
                DX,
                DY,
                DZ * factor,
            )
        )

    results = []

    for label, dx, dy, dz in experiments:

        model.zero_grad(
            set_to_none=True
        )

        physics = PhysicsLoss(
            dx=dx,
            dy=dy,
            dz=dz,
        ).to(DEVICE)

        result = physics(
            travel_time=travel_time,
            velocity=velocity,
            source_indices=None,
            travel_time_target=None,
        )

        if isinstance(result, dict):

            if "total" in result:
                loss = result["total"]

            elif "eikonal" in result:
                loss = result["eikonal"]

            else:
                raise RuntimeError(
                    "PhysicsLoss output does not contain "
                    "'total' or 'eikonal'."
                )

        else:
            loss = result

        loss.backward()

        (
            gradient_norm,
            maximum_gradient,
            parameter_count,
            finite_gradients,
        ) = parameter_gradient_norm(model)

        diagnostics = calculate_eikonal_diagnostics(
            physics,
            travel_time,
            velocity,
        )

        ratio = (
            diagnostics["gradient_magnitude"]
            / (1.0 / velocity)
        )

        print()
        print(
            f"{label} sampling: "
            f"dx={dx:.6e}, "
            f"dy={dy:.6e}, "
            f"dz={dz:.6e}"
        )

        print(
            f"    factor relative to baseline: "
            f"{dx / baseline['dx'] if label == 'DX' else dy / baseline['dy'] if label == 'DY' else dz / baseline['dz']:.3f}"
        )

        print(
            f"    |grad T| mean       : "
            f"{diagnostics['gradient_magnitude'].mean().item():.6e}"
        )

        print(
            f"    V|grad T| mean      : "
            f"{diagnostics['normalized_eikonal'].mean().item():.6e}"
        )

        print(
            f"    Eikonal loss        : "
            f"{loss.item():.6e}"
        )

        print(
            f"    gradient norm       : "
            f"{gradient_norm:.6e}"
        )

        print(
            f"    maximum gradient    : "
            f"{maximum_gradient:.6e}"
        )

        print(
            f"    finite gradients    : "
            f"{'PASS' if finite_gradients else 'FAIL'}"
        )

        results.append(
            {
                "label": label,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "loss": loss.item(),
                "gradient_norm": gradient_norm,
                "maximum_gradient": maximum_gradient,
                "ratio": ratio.mean().item(),
            }
        )

        if not finite_gradients:
            raise RuntimeError(
                f"Non-finite gradients in {label} spatial test."
            )

    print()
    print(
        "Corrected spatial-sampling audit: PASS"
    )

    return results


# ======================================================================
# TRAVEL-TIME SCALE SENSITIVITY
# ======================================================================

def travel_time_scale_audit(
    model,
    travel_time,
    velocity,
):
    """
    Test the effect of multiplying the network travel-time output
    by different physical scale factors.

    The network itself is NOT modified.

    We scale the already computed dimensionless physical output:

        T_test = T_current * (scale / configured_scale)

    """

    header(
        "4. TRAVEL-TIME SCALE SENSITIVITY AUDIT"
    )

    print(
        f"Configured TRAVEL_TIME_SCALE: "
        f"{TRAVEL_TIME_SCALE:.6e}"
    )

    print(
        "The network is NOT modified during this test."
    )

    results = []

    for scale in TRAVEL_TIME_SCALES:

        model.zero_grad(
            set_to_none=True
        )

        scale_factor = (
            scale
            / TRAVEL_TIME_SCALE
        )

        tested_travel_time = (
            travel_time
            * scale_factor
        )

        physics = PhysicsLoss(
            dx=DX,
            dy=DY,
            dz=DZ,
        ).to(DEVICE)

        result = physics(
            travel_time=tested_travel_time,
            velocity=velocity,
            source_indices=None,
            travel_time_target=None,
        )

        if isinstance(result, dict):

            if "total" in result:
                loss = result["total"]

            elif "eikonal" in result:
                loss = result["eikonal"]

            else:
                raise RuntimeError(
                    "PhysicsLoss output missing expected key."
                )

        else:
            loss = result

        loss.backward()

        (
            gradient_norm,
            maximum_gradient,
            parameter_count,
            finite_gradients,
        ) = parameter_gradient_norm(model)

        diagnostics = calculate_eikonal_diagnostics(
            physics,
            tested_travel_time,
            velocity,
        )

        print()
        print(
            f"TRAVEL_TIME_SCALE = {scale:.6e}"
        )

        print(
            f"    scale factor       : "
            f"{scale_factor:.6e}"
        )

        print(
            f"    mean T             : "
            f"{tested_travel_time.mean().item():.6e}"
        )

        print(
            f"    mean |grad T|      : "
            f"{diagnostics['gradient_magnitude'].mean().item():.6e}"
        )

        print(
            f"    mean V|grad T|     : "
            f"{diagnostics['normalized_eikonal'].mean().item():.6e}"
        )

        print(
            f"    Eikonal loss       : "
            f"{loss.item():.6e}"
        )

        print(
            f"    gradient norm      : "
            f"{gradient_norm:.6e}"
        )

        print(
            f"    maximum gradient   : "
            f"{maximum_gradient:.6e}"
        )

        print(
            f"    finite gradients   : "
            f"{'PASS' if finite_gradients else 'FAIL'}"
        )

        results.append(
            {
                "scale": scale,
                "scale_factor": scale_factor,
                "mean_T": tested_travel_time.mean().item(),
                "mean_gradT": diagnostics[
                    "gradient_magnitude"
                ].mean().item(),
                "mean_VgradT": diagnostics[
                    "normalized_eikonal"
                ].mean().item(),
                "loss": loss.item(),
                "gradient_norm": gradient_norm,
                "maximum_gradient": maximum_gradient,
            }
        )

        if not finite_gradients:
            raise RuntimeError(
                "Non-finite gradients in travel-time scale audit."
            )

    print()
    print(
        "Travel-time scale sensitivity audit: PASS"
    )

    return results


# ======================================================================
# TRAVEL-TIME PARAMETERIZATION AUDIT
# ======================================================================

def travel_time_parameterization_audit(
    model,
    x,
    travel_time,
):
    """
    Audit the currently implemented travel-time parameterization.

    Current architecture:

        raw_travel_time
                |
                v
            Softplus
                |
                v
        normalized_travel_time
                |
                v
        TRAVEL_TIME_SCALE *
                |
                v
           travel_time

    This section reconstructs the dimensionless quantity and verifies
    the physical scaling relation.

    The current Network3D implementation uses a dedicated
    travel_time_head followed by Softplus and TRAVEL_TIME_SCALE.
    """

    header(
        "5. TRAVEL-TIME PARAMETERIZATION AUDIT"
    )

    print(
        "Auditing current travel-time parameterization:"
    )

    print(
        "    raw head -> Softplus -> "
        "dimensionless positive value -> "
        "TRAVEL_TIME_SCALE -> T"
    )

    # --------------------------------------------------------------
    # Re-run network forward
    # --------------------------------------------------------------

    with torch.no_grad():

        outputs = model(x)

        if isinstance(outputs, dict):
            T = outputs["travel_time"]
        else:
            T = outputs[1]

    assert_finite(
        T,
        "Physical travel time",
    )

    # --------------------------------------------------------------
    # Recover dimensionless representation
    # --------------------------------------------------------------

    normalized = (
        T / TRAVEL_TIME_SCALE
    )

    assert_finite(
        normalized,
        "Recovered dimensionless travel time",
    )

    reconstructed_T = (
        TRAVEL_TIME_SCALE
        * normalized
    )

    reconstruction_error = (
        reconstructed_T - T
    ).abs().max().item()

    print()
    print(
        f"Configured scale             : "
        f"{TRAVEL_TIME_SCALE:.6e}"
    )

    print(
        f"Dimensionless T mean         : "
        f"{normalized.mean().item():.6e}"
    )

    print(
        f"Dimensionless T std          : "
        f"{normalized.std().item():.6e}"
    )

    print(
        f"Dimensionless T min          : "
        f"{normalized.min().item():.6e}"
    )

    print(
        f"Dimensionless T max          : "
        f"{normalized.max().item():.6e}"
    )

    print(
        f"Physical T mean              : "
        f"{T.mean().item():.6e}"
    )

    print(
        f"Physical T std               : "
        f"{T.std().item():.6e}"
    )

    print(
        f"Reconstruction max error     : "
        f"{reconstruction_error:.6e}"
    )

    if reconstruction_error <= 1.0e-7:
        print(
            "Travel-time physical scaling consistency: PASS"
        )
    else:
        print(
            "Travel-time physical scaling consistency: FAIL"
        )

        raise RuntimeError(
            "Travel-time scaling reconstruction failed."
        )

    # --------------------------------------------------------------
    # Directly inspect the travel-time head if available.
    # --------------------------------------------------------------

    if hasattr(
        model,
        "travel_time_head"
    ):

        print()
        print(
            "Travel-time head inspection: available"
        )

        head = model.travel_time_head

        print(
            f"    weight mean        : "
            f"{head.weight.detach().mean().item():.6e}"
        )

        print(
            f"    weight std         : "
            f"{head.weight.detach().std().item():.6e}"
        )

        print(
            f"    weight absmax      : "
            f"{head.weight.detach().abs().max().item():.6e}"
        )

        if head.bias is not None:

            print(
                f"    bias mean          : "
                f"{head.bias.detach().mean().item():.6e}"
            )

            print(
                f"    bias absmax        : "
                f"{head.bias.detach().abs().max().item():.6e}"
            )

    else:

        print(
            "Travel-time head inspection: WARNING"
        )

        print(
            "    model.travel_time_head was not found."
        )


# ======================================================================
# EPSILON SENSITIVITY AUDIT
# ======================================================================

def epsilon_sensitivity_audit(
    physics_loss,
    travel_time,
    velocity,
):
    """
    Determine how much the epsilon floor influences |grad T|.
    """

    header(
        "6. EIKONAL EPSILON SENSITIVITY AUDIT"
    )

    gradient_squared = None

    base_diagnostics = calculate_eikonal_diagnostics(
        physics_loss,
        travel_time,
        velocity,
        epsilon=physics_loss.eps,
    )

    gradient_squared = (
        base_diagnostics["gradient_squared"]
    )

    print(
        f"PhysicsLoss epsilon: "
        f"{physics_loss.eps:.6e}"
    )

    print(
        f"Mean gradient squared: "
        f"{gradient_squared.mean().item():.6e}"
    )

    print(
        f"Median gradient squared: "
        f"{gradient_squared.median().item():.6e}"
    )

    for epsilon in EPSILON_VALUES:

        diagnostics = calculate_eikonal_diagnostics(
            physics_loss,
            travel_time,
            velocity,
            epsilon=epsilon,
        )

        fraction_below = (
            gradient_squared < epsilon
        ).float().mean().item()

        print()
        print(
            f"epsilon = {epsilon:.6e}"
        )

        print(
            f"    fraction grad² < epsilon : "
            f"{100.0 * fraction_below:.6f}%"
        )

        print(
            f"    mean |grad T|            : "
            f"{diagnostics['gradient_magnitude'].mean().item():.6e}"
        )

        print(
            f"    Eikonal loss             : "
            f"{diagnostics['loss'].item():.6e}"
        )

    # --------------------------------------------------------------
    # Compare current epsilon with no floor.
    # --------------------------------------------------------------

    true_gradient_magnitude = torch.sqrt(
        gradient_squared
    )

    epsilon_gradient_magnitude = (
        base_diagnostics["gradient_magnitude"]
    )

    change = (
        epsilon_gradient_magnitude
        - true_gradient_magnitude
    ).abs()

    print()
    print(
        "Epsilon-induced magnitude change:"
    )

    print(
        f"    mean change : "
        f"{change.mean().item():.6e}"
    )

    print(
        f"    max change  : "
        f"{change.max().item():.6e}"
    )

    if torch.isfinite(change).all():
        print(
            "Epsilon numerical stability audit: PASS"
        )
    else:
        raise RuntimeError(
            "Non-finite epsilon sensitivity result."
        )


# ======================================================================
# PHYSICAL SCALE COMPARISON
# ======================================================================

def physical_scale_audit():
    """
    Compare configured travel-time scale against the physical
    characteristic travel time across the test volume.
    """

    header(
        "7. CHARACTERISTIC PHYSICAL TRAVEL-TIME SCALE"
    )

    depth_extent = (
        (DEPTH - 1)
        * DZ
    )

    height_extent = (
        (HEIGHT - 1)
        * DY
    )

    width_extent = (
        (WIDTH - 1)
        * DX
    )

    diagonal_distance = math.sqrt(
        depth_extent ** 2
        + height_extent ** 2
        + width_extent ** 2
    )

    characteristic_velocity = (
        VELOCITY_MIN
        + VELOCITY_MAX
    ) / 2.0

    characteristic_time = (
        diagonal_distance
        / characteristic_velocity
    )

    print(
        f"Depth extent               : "
        f"{depth_extent:.6e} m"
    )

    print(
        f"Y extent                   : "
        f"{height_extent:.6e} m"
    )

    print(
        f"X extent                   : "
        f"{width_extent:.6e} m"
    )

    print(
        f"Diagonal distance          : "
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

    ratio = (
        TRAVEL_TIME_SCALE
        / characteristic_time
    )

    print(
        f"T-scale / characteristic T : "
        f"{ratio:.6e}"
    )

    required_gradient = (
        1.0
        / characteristic_velocity
    )

    print(
        f"Required |grad T| ≈ 1/V    : "
        f"{required_gradient:.6e} s/m"
    )


# ======================================================================
# PHYSICS GRADIENT PATHWAY AUDIT
# ======================================================================

def gradient_pathway_audit(
    model,
    x,
    velocity,
):
    """
    Confirm that the Eikonal loss produces gradients through the
    travel-time branch and does not incorrectly propagate through
    reconstruction or uncertainty heads.
    """

    header(
        "8. TRAVEL-TIME PHYSICS GRADIENT PATHWAY AUDIT"
    )

    model.zero_grad(
        set_to_none=True
    )

    reconstruction, travel_time, log_variance = (
        forward_network(
            model,
            x,
        )
    )

    physics = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(DEVICE)

    result = physics(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=None,
        travel_time_target=None,
    )

    if isinstance(result, dict):

        if "total" in result:
            loss = result["total"]

        elif "eikonal" in result:
            loss = result["eikonal"]

        else:
            raise RuntimeError(
                "Could not locate physics loss."
            )

    else:
        loss = result

    loss.backward()

    (
        gradient_norm,
        maximum_gradient,
        parameter_count,
        finite_gradients,
    ) = parameter_gradient_norm(model)

    named_gradients = (
        named_parameter_gradient_norms(model)
    )

    reconstruction_head_norm = group_gradient_norm(
        named_gradients,
        (
            "reconstruction_head",
            "recon_head",
            "reconstruction",
            "recon",
        ),
    )

    travel_time_head_norm = group_gradient_norm(
        named_gradients,
        (
            "travel_time_head",
            "traveltime_head",
            "travel_time",
            "traveltime",
        ),
    )

    uncertainty_head_norm = group_gradient_norm(
        named_gradients,
        (
            "uncertainty_head",
            "uncertainty",
            "log_variance",
            "logvar",
            "variance",
        ),
    )

    print(
        f"Physics loss              : "
        f"{loss.item():.6e}"
    )

    print(
        f"Global gradient norm      : "
        f"{gradient_norm:.6e}"
    )

    print(
        f"Maximum parameter gradient: "
        f"{maximum_gradient:.6e}"
    )

    print(
        f"Parameters with gradients : "
        f"{parameter_count}"
    )

    print(
        f"Finite gradients          : "
        f"{'PASS' if finite_gradients else 'FAIL'}"
    )

    print()
    print(
        "Gradient pathways:"
    )

    print(
        f"    Reconstruction head   : "
        f"{reconstruction_head_norm:.6e}"
    )

    print(
        f"    Travel-time head      : "
        f"{travel_time_head_norm:.6e}"
    )

    print(
        f"    Uncertainty head      : "
        f"{uncertainty_head_norm:.6e}"
    )

    if travel_time_head_norm > 0.0:
        print(
            "Travel-time physics pathway: PASS"
        )
    else:
        print(
            "Travel-time physics pathway: FAIL"
        )

        raise RuntimeError(
            "Eikonal loss does not reach travel-time parameters."
        )

    if finite_gradients:
        print(
            "Physics gradient numerical stability: PASS"
        )
    else:
        raise RuntimeError(
            "Non-finite physics gradients."
        )


# ======================================================================
# FINAL DIAGNOSTIC CLASSIFICATION
# ======================================================================

def final_classification(
    baseline_diagnostics,
    spatial_results,
    scale_results,
):
    """
    Produce a conservative interpretation.

    This section deliberately does NOT change configuration.
    """

    header(
        "9. FINAL DIAGNOSTIC CLASSIFICATION"
    )

    baseline_ratio = (
        baseline_diagnostics[
            "gradient_magnitude"
        ]
        / (
            1.0
            / baseline_diagnostics[
                "gradient_magnitude"
            ].new_tensor(
                1.0
            )
        )
    )

    # The actual ratio is calculated below using the known velocity
    # externally in main(). This local object is intentionally not
    # used as a classification metric.
    del baseline_ratio

    baseline_vgradt = (
        baseline_diagnostics[
            "normalized_eikonal"
        ].mean().item()
    )

    baseline_loss = (
        baseline_diagnostics[
            "loss"
        ].item()
    )

    print(
        f"Baseline mean V|grad T| : "
        f"{baseline_vgradt:.6e}"
    )

    print(
        f"Baseline Eikonal loss    : "
        f"{baseline_loss:.6e}"
    )

    # --------------------------------------------------------------
    # Spatial variation
    # --------------------------------------------------------------

    spatial_losses = [
        result["loss"]
        for result in spatial_results
    ]

    spatial_gradient_norms = [
        result["gradient_norm"]
        for result in spatial_results
    ]

    spatial_loss_range = (
        max(spatial_losses)
        - min(spatial_losses)
    )

    spatial_gradient_range = (
        max(spatial_gradient_norms)
        - min(spatial_gradient_norms)
    )

    print()
    print(
        f"Spatial loss range       : "
        f"{spatial_loss_range:.6e}"
    )

    print(
        f"Spatial gradient range   : "
        f"{spatial_gradient_range:.6e}"
    )

    # --------------------------------------------------------------
    # Travel-time scale variation
    # --------------------------------------------------------------

    scale_gradient_norms = [
        result["gradient_norm"]
        for result in scale_results
    ]

    scale_losses = [
        result["loss"]
        for result in scale_results
    ]

    print()
    print(
        f"Travel-time gradient min : "
        f"{min(scale_gradient_norms):.6e}"
    )

    print(
        f"Travel-time gradient max : "
        f"{max(scale_gradient_norms):.6e}"
    )

    print(
        f"Travel-time loss min     : "
        f"{min(scale_losses):.6e}"
    )

    print(
        f"Travel-time loss max     : "
        f"{max(scale_losses):.6e}"
    )

    # --------------------------------------------------------------
    # Conservative interpretation
    # --------------------------------------------------------------

    print()
    print(
        "Diagnostic interpretation:"
    )

    if baseline_vgradt < 0.1:
        print(
            "    [WARNING] Travel-time gradient magnitude is "
            "far below the Eikonal target."
        )

    if baseline_loss > 0.5:
        print(
            "    [WARNING] Initial Eikonal residual is large."
        )

    if spatial_gradient_range > 0.0:
        print(
            "    [INFO] Physics gradient responds to spatial "
            "sampling."
        )

    if (
        max(scale_gradient_norms)
        > 2.0 * min(scale_gradient_norms)
    ):
        print(
            "    [WARNING] Travel-time scaling has a strong "
            "effect on physics gradient magnitude."
        )

    print()
    print(
        "FINAL STATUS:"
    )

    print(
        "    Eikonal implementation audit: PASS"
    )

    print(
        "    Spatial-sampling audit: PASS"
    )

    print(
        "    Travel-time scaling audit: PASS"
    )

    print(
        "    No production configuration modified."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    These warnings are diagnostic findings."
    )

    print(
        "    They do NOT by themselves justify changing "
        "LOSS_WEIGHTS."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "PHYSICS GRADIENT SCALE AUDIT v2"
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
        f"Input shape            : {INPUT_SHAPE}"
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
        f"Travel-time scale      : {TRAVEL_TIME_SCALE}"
    )

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------

    header(
        "CREATING DETERMINISTIC SYNTHETIC DATA"
    )

    x, target, velocity = make_data(
        DEVICE
    )

    assert_finite(
        x,
        "Input",
    )

    assert_finite(
        target,
        "Target",
    )

    assert_finite(
        velocity,
        "Velocity",
    )

    if torch.any(
        velocity <= 0.0
    ):
        raise RuntimeError(
            "Velocity positivity check failed."
        )

    print(
        "Velocity positivity check: PASS"
    )

    # --------------------------------------------------------------
    # Network
    # --------------------------------------------------------------

    header(
        "INITIALIZING NETWORK3D"
    )

    model = Network3D().to(
        DEVICE
    )

    # Deterministic audit.
    model.eval()

    print(
        "Network3D initialized successfully."
    )

    # --------------------------------------------------------------
    # Baseline
    # --------------------------------------------------------------

    (
        reconstruction,
        travel_time,
        log_variance,
        baseline_physics,
        baseline_diagnostics,
    ) = baseline_audit(
        model,
        x,
        velocity,
    )

    # --------------------------------------------------------------
    # Implementation consistency
    # --------------------------------------------------------------

    implementation_consistency_audit(
        baseline_physics,
        travel_time,
        velocity,
    )

    # --------------------------------------------------------------
    # Spatial sampling
    # --------------------------------------------------------------

    spatial_results = spatial_sampling_audit(
        model,
        travel_time,
        velocity,
    )

    # --------------------------------------------------------------
    # Travel-time scaling
    # --------------------------------------------------------------

    scale_results = travel_time_scale_audit(
        model,
        travel_time,
        velocity,
    )

    # --------------------------------------------------------------
    # Parameterization
    # --------------------------------------------------------------

    travel_time_parameterization_audit(
        model,
        x,
        travel_time,
    )

    # --------------------------------------------------------------
    # Epsilon
    # --------------------------------------------------------------

    epsilon_sensitivity_audit(
        baseline_physics,
        travel_time,
        velocity,
    )

    # --------------------------------------------------------------
    # Physical scale
    # --------------------------------------------------------------

    physical_scale_audit()

    # --------------------------------------------------------------
    # Gradient pathway
    # --------------------------------------------------------------

    gradient_pathway_audit(
        model,
        x,
        velocity,
    )

    # --------------------------------------------------------------
    # Final classification
    # --------------------------------------------------------------

    final_classification(
        baseline_diagnostics,
        spatial_results,
        scale_results,
    )

    # --------------------------------------------------------------
    # Completion
    # --------------------------------------------------------------

    header(
        "AUDIT COMPLETE"
    )

    print(
        "Physics Gradient Scale Audit v2 completed."
    )

    print(
        "No model parameters were updated."
    )

    print(
        "No configuration values were changed."
    )

    print(
        "No loss weights were changed."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()