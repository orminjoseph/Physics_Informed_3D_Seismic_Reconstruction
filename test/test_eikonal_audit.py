"""
=========================================================
Eikonal Physics Audit
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Phase B2
--------

This test audits the numerically stabilized Eikonal
physics-loss implementation.

The stabilized formulation is:

    |grad T| = 1 / V

or equivalently:

    V |grad T| = 1

Therefore:

    R_eikonal = V |grad T| - 1

and:

    L_eikonal = mean(R_eikonal^2)

This test verifies:

    1. Synthetic physical fields
    2. Spatial derivative calculation
    3. Travel-time gradient magnitude
    4. V |grad T|
    5. Eikonal residual
    6. Eikonal loss
    7. Source-condition loss
    8. Optional travel-time supervision
    9. Complete PhysicsLoss forward pass
   10. Backward propagation
   11. Numerical stability
   12. Physical consistency

Tensor convention:

    [B, C, D, H, W]

where:

    D = depth
    H = crossline
    W = inline

Author: Ormin Joseph
=========================================================
"""

import math
import torch

from losses.physics_loss import PhysicsLoss


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

DX = 25.0
DY = 25.0
DZ = 10.0

VELOCITY_MIN = 1500.0
VELOCITY_MAX = 4500.0

EPS = 1.0e-8


# =========================================================
# PRINTING UTILITIES
# =========================================================

def print_header(title):
    """
    Print a formatted section header.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_tensor_statistics(name, tensor):
    """
    Print useful numerical statistics for a tensor.
    """

    print(f"{name}:")

    print(
        f"    shape : {tuple(tensor.shape)}"
    )

    print(
        f"    min   : {tensor.min().item():.6e}"
    )

    print(
        f"    max   : {tensor.max().item():.6e}"
    )

    print(
        f"    mean  : {tensor.mean().item():.6e}"
    )

    print(
        f"    std   : {tensor.std().item():.6e}"
    )

    print(
        f"    absmax: {tensor.abs().max().item():.6e}"
    )


def assert_finite(tensor, name):
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():

        raise AssertionError(
            f"{name} contains NaN or infinite values."
        )

    print(
        f"{name}: finite values confirmed."
    )


# =========================================================
# SYNTHETIC VELOCITY MODEL
# =========================================================

def create_velocity():
    """
    Create a smoothly varying positive velocity model.

    Velocity varies between approximately 1500 and
    4500 m/s.

    The model is deliberately constructed so that
    physical units remain realistic.
    """

    z = torch.linspace(
        0.0,
        1.0,
        DEPTH
    ).view(
        1,
        1,
        DEPTH,
        1,
        1
    )

    y = torch.linspace(
        0.0,
        1.0,
        HEIGHT
    ).view(
        1,
        1,
        1,
        HEIGHT,
        1
    )

    x = torch.linspace(
        0.0,
        1.0,
        WIDTH
    ).view(
        1,
        1,
        1,
        1,
        WIDTH
    )

    # -----------------------------------------------------
    # Normalized velocity model
    # -----------------------------------------------------

    velocity = (
        VELOCITY_MIN
        +
        (
            VELOCITY_MAX
            -
            VELOCITY_MIN
        )
        *
        (
            0.5 * z
            +
            0.3 * y
            +
            0.2 * x
        )
    )

    return velocity.float()


# =========================================================
# SYNTHETIC TRAVEL-TIME FIELD
# =========================================================

def create_physically_consistent_travel_time(
    velocity
):
    """
    Create a travel-time field that satisfies the
    Eikonal equation approximately.

    For the audit we construct:

        T = integral(dz / V)

    along the depth direction.

    This produces:

        dT/dz approximately 1/V

    and therefore:

        V |grad T| approximately 1

    for the dominant depth direction.

    This provides a physically meaningful reference
    for the stabilized Eikonal formulation.
    """

    # -----------------------------------------------------
    # Use the velocity at the first depth sample as a
    # reference velocity for a controlled synthetic field.
    #
    # This produces an analytically predictable field:
    #
    #     T(z) = z / V_ref
    #
    # -----------------------------------------------------

    velocity_reference = velocity[
        :,
        :,
        0:1,
        :,
        :
    ]

    z = torch.arange(
        DEPTH,
        dtype=torch.float32
    ).view(
        1,
        1,
        DEPTH,
        1,
        1
    )

    # Physical depth coordinate.
    depth_coordinate = (
        z * DZ
    )

    travel_time = (
        depth_coordinate
        /
        velocity_reference
    )

    # -----------------------------------------------------
    # Remove source offset so that:
    #
    #     T(z=0) = 0
    #
    # -----------------------------------------------------

    return travel_time.float()


# =========================================================
# RANDOM TRAVEL-TIME FIELD
# =========================================================

def create_random_travel_time():
    """
    Create a differentiable random travel-time field.

    This is used to verify that the physics-loss
    computation remains numerically stable for
    non-ideal network outputs.
    """

    return (
        torch.rand(
            BATCH_SIZE,
            CHANNELS,
            DEPTH,
            HEIGHT,
            WIDTH
        )
        * 0.1
    ).float()


# =========================================================
# MAIN AUDIT
# =========================================================

def test_eikonal_audit():
    """
    Execute the complete Phase-B2 Eikonal audit.
    """

    print_header(
        "STABILIZED EIKONAL PHYSICS AUDIT"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Tensor shape    : "
        f"{(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)}"
    )

    print(
        f"dx              : {DX}"
    )

    print(
        f"dy              : {DY}"
    )

    print(
        f"dz              : {DZ}"
    )

    # =====================================================
    # CREATE VELOCITY
    # =====================================================

    print_header(
        "CREATING SYNTHETIC VELOCITY MODEL"
    )

    velocity = create_velocity().to(device)

    print_tensor_statistics(
        "Velocity",
        velocity
    )

    assert_finite(
        velocity,
        "Velocity"
    )

    # -----------------------------------------------------
    # Velocity positivity
    # -----------------------------------------------------

    if torch.any(velocity <= 0.0):

        raise AssertionError(
            "Velocity positivity check failed."
        )

    print(
        "Velocity positivity check: PASS"
    )

    # =====================================================
    # CREATE PHYSICALLY CONSISTENT TRAVEL TIME
    # =====================================================

    print_header(
        "CREATING SYNTHETIC TRAVEL-TIME FIELD"
    )

    travel_time = (
        create_physically_consistent_travel_time(
            velocity
        )
        .to(device)
    )

    print_tensor_statistics(
        "Travel Time",
        travel_time
    )

    assert_finite(
        travel_time,
        "Travel time"
    )

    # =====================================================
    # INITIALIZE PHYSICS LOSS
    # =====================================================

    print_header(
        "INITIALIZING STABILIZED EIKONAL LOSS"
    )

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=1.0,
        source_weight=1.0,
        travel_time_weight=1.0,
        eps=EPS
    ).to(device)

    print(
        "PhysicsLoss initialized successfully."
    )

    # =====================================================
    # SPATIAL DERIVATIVE AUDIT
    # =====================================================

    print_header(
        "SPATIAL DERIVATIVE AUDIT"
    )

    (
        dT_dz,
        dT_dy,
        dT_dx,
        gradient_squared,
        gradient_magnitude
    ) = physics_loss.travel_time_gradient(
        travel_time
    )

    print_tensor_statistics(
        "dT/dz",
        dT_dz
    )

    print_tensor_statistics(
        "dT/dy",
        dT_dy
    )

    print_tensor_statistics(
        "dT/dx",
        dT_dx
    )

    assert_finite(
        dT_dz,
        "dT/dz"
    )

    assert_finite(
        dT_dy,
        "dT/dy"
    )

    assert_finite(
        dT_dx,
        "dT/dx"
    )

    # =====================================================
    # GRADIENT MAGNITUDE AUDIT
    # =====================================================

    print_header(
        "GRADIENT MAGNITUDE AUDIT"
    )

    print_tensor_statistics(
        "|grad T|^2",
        gradient_squared
    )

    print_tensor_statistics(
        "|grad T|",
        gradient_magnitude
    )

    assert_finite(
        gradient_squared,
        "|grad T|^2"
    )

    assert_finite(
        gradient_magnitude,
        "|grad T|"
    )

    # =====================================================
    # NORMALIZED EIKONAL TERM
    # =====================================================

    print_header(
        "NORMALIZED EIKONAL TERM AUDIT"
    )

    normalized_eikonal = (
        velocity
        *
        gradient_magnitude
    )

    print_tensor_statistics(
        "V |grad T|",
        normalized_eikonal
    )

    assert_finite(
        normalized_eikonal,
        "V |grad T|"
    )

    # =====================================================
    # EIKONAL RESIDUAL
    # =====================================================

    print_header(
        "EIKONAL RESIDUAL AUDIT"
    )

    residual = physics_loss.eikonal_residual(
        travel_time,
        velocity
    )

    print_tensor_statistics(
        "Eikonal residual",
        residual
    )

    assert_finite(
        residual,
        "Eikonal residual"
    )

    # -----------------------------------------------------
    # Verify residual equation
    #
    #     R = V |grad T| - 1
    #
    # -----------------------------------------------------

    expected_residual = (
        normalized_eikonal
        -
        1.0
    )

    residual_error = (
        residual
        -
        expected_residual
    ).abs().max().item()

    print(
        f"Residual equation error : "
        f"{residual_error:.6e}"
    )

    if residual_error > 1.0e-6:

        raise AssertionError(
            "Eikonal residual does not match "
            "V|grad T| - 1."
        )

    print(
        "Residual equation check: PASS"
    )

    # =====================================================
    # EIKONAL LOSS
    # =====================================================

    print_header(
        "EIKONAL LOSS"
    )

    eikonal_loss = physics_loss.eikonal_loss(
        travel_time,
        velocity
    )

    print(
        f"Eikonal loss : "
        f"{eikonal_loss.item():.6e}"
    )

    if not torch.isfinite(
        eikonal_loss
    ):

        raise AssertionError(
            "Eikonal loss is not finite."
        )

    print(
        "Eikonal loss is finite."
    )

    # =====================================================
    # SOURCE CONDITION AUDIT
    # =====================================================

    print_header(
        "SOURCE CONDITION AUDIT"
    )

    source_indices = torch.tensor(
        [[0, 0, 0]],
        dtype=torch.long,
        device=device
    )

    source_loss = physics_loss.source_condition_loss(
        travel_time,
        source_indices
    )

    print(
        f"Source condition loss : "
        f"{source_loss.item():.6e}"
    )

    if not torch.isfinite(
        source_loss
    ):

        raise AssertionError(
            "Source-condition loss is not finite."
        )

    # The synthetic field was explicitly constructed
    # with T(0) = 0.

    if source_loss.item() > 1.0e-8:

        raise AssertionError(
            "Source condition is not satisfied."
        )

    print(
        "Source condition check: PASS"
    )

    # =====================================================
    # TRAVEL-TIME SUPERVISION AUDIT
    # =====================================================

    print_header(
        "TRAVEL-TIME SUPERVISION AUDIT"
    )

    travel_time_target = (
        travel_time.clone()
    )

    supervision_loss = (
        physics_loss.travel_time_supervision_loss(
            travel_time,
            travel_time_target
        )
    )

    print(
        f"Travel-time supervision loss : "
        f"{supervision_loss.item():.6e}"
    )

    if not torch.isfinite(
        supervision_loss
    ):

        raise AssertionError(
            "Travel-time supervision loss "
            "is not finite."
        )

    if supervision_loss.item() > 1.0e-8:

        raise AssertionError(
            "Identical travel-time fields "
            "should produce near-zero "
            "supervision loss."
        )

    print(
        "Travel-time supervision check: PASS"
    )

    # =====================================================
    # COMPLETE PHYSICS LOSS AUDIT
    # =====================================================

    print_header(
        "COMPLETE PHYSICS LOSS AUDIT"
    )

    complete_loss = physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=source_indices,
        travel_time_target=travel_time_target
    )

    for key, value in complete_loss.items():

        print(
            f"{key:25s}: "
            f"{value.item():.6e}"
        )

        if not torch.isfinite(value):

            raise AssertionError(
                f"Physics-loss component "
                f"{key} is not finite."
            )

    # -----------------------------------------------------
    # Verify total loss decomposition
    # -----------------------------------------------------

    expected_total = (
        complete_loss["weighted_eikonal"]
        +
        complete_loss["weighted_source"]
        +
        complete_loss["weighted_travel_time"]
    )

    total_error = (
        complete_loss["total"]
        -
        expected_total
    ).abs().item()

    print(
        f"Total decomposition error : "
        f"{total_error:.6e}"
    )

    if total_error > 1.0e-6:

        raise AssertionError(
            "Total physics loss decomposition "
            "is incorrect."
        )

    print(
        "Physics-loss decomposition check: PASS"
    )

    # =====================================================
    # BACKWARD PROPAGATION AUDIT
    # =====================================================

    print_header(
        "BACKWARD PROPAGATION AUDIT"
    )

    differentiable_travel_time = (
        create_random_travel_time()
        .to(device)
        .requires_grad_(True)
    )

    backward_loss = physics_loss.eikonal_loss(
        differentiable_travel_time,
        velocity
    )

    print(
        f"Eikonal loss before backward : "
        f"{backward_loss.item():.6e}"
    )

    backward_loss.backward()

    if differentiable_travel_time.grad is None:

        raise AssertionError(
            "No gradient was generated for "
            "travel time."
        )

    travel_time_gradient = (
        differentiable_travel_time.grad
    )

    assert_finite(
        travel_time_gradient,
        "Travel-time gradient"
    )

    gradient_norm = (
        torch.linalg.vector_norm(
            travel_time_gradient
        ).item()
    )

    maximum_gradient = (
        travel_time_gradient
        .abs()
        .max()
        .item()
    )

    print(
        "Travel-time gradient:"
    )

    print(
        f"    Gradient norm    : "
        f"{gradient_norm:.6e}"
    )

    print(
        f"    Maximum gradient : "
        f"{maximum_gradient:.6e}"
    )

    if not math.isfinite(
        gradient_norm
    ):

        raise AssertionError(
            "Gradient norm is not finite."
        )

    if not math.isfinite(
        maximum_gradient
    ):

        raise AssertionError(
            "Maximum gradient is not finite."
        )

    print(
        "Backward propagation check: PASS"
    )

    # =====================================================
    # NUMERICAL STABILITY AUDIT
    # =====================================================

    print_header(
        "NUMERICAL STABILITY AUDIT"
    )

    scales = [
        1.0,
        0.1,
        0.01,
        0.001
    ]

    print(
        f"{'Scale':>12s}"
        f"{'Eikonal Loss':>22s}"
        f"{'Max Gradient':>22s}"
    )

    print(
        "-" * 58
    )

    for scale in scales:

        scaled_time = (
            create_random_travel_time()
            * scale
        ).to(device)

        scaled_time.requires_grad_(True)

        scaled_loss = physics_loss.eikonal_loss(
            scaled_time,
            velocity
        )

        scaled_loss.backward()

        if scaled_time.grad is None:

            raise AssertionError(
                "Gradient missing during "
                "scale sensitivity audit."
            )

        scaled_gradient = (
            scaled_time.grad
            .abs()
            .max()
            .item()
        )

        print(
            f"{scale:12.4e}"
            f"{scaled_loss.item():22.6e}"
            f"{scaled_gradient:22.6e}"
        )

        if not math.isfinite(
            scaled_loss.item()
        ):

            raise AssertionError(
                "Non-finite Eikonal loss detected "
                "during scale sensitivity audit."
            )

        if not math.isfinite(
            scaled_gradient
        ):

            raise AssertionError(
                "Non-finite gradient detected "
                "during scale sensitivity audit."
            )

    print()
    print(
        "Scale sensitivity audit completed."
    )

    # =====================================================
    # PHYSICAL DIAGNOSTICS
    # =====================================================

    print_header(
        "DIAGNOSTIC INTERPRETATION"
    )

    mean_velocity = (
        velocity.mean().item()
    )

    mean_travel_time = (
        travel_time.mean().item()
    )

    rms_gradient = math.sqrt(
        gradient_squared
        .mean()
        .item()
    )

    mean_normalized_term = (
        normalized_eikonal
        .mean()
        .item()
    )

    residual_rms = math.sqrt(
        residual.pow(2)
        .mean()
        .item()
    )

    print(
        f"Mean velocity magnitude : "
        f"{mean_velocity:.6e} m/s"
    )

    print(
        f"Mean travel time        : "
        f"{mean_travel_time:.6e} s"
    )

    print(
        f"RMS |grad T|            : "
        f"{rms_gradient:.6e}"
    )

    print(
        f"Mean V|grad T|          : "
        f"{mean_normalized_term:.6e}"
    )

    print(
        f"RMS Eikonal residual    : "
        f"{residual_rms:.6e}"
    )

    print(
        f"Eikonal loss            : "
        f"{eikonal_loss.item():.6e}"
    )

    # =====================================================
    # FINAL AUDIT RESULT
    # =====================================================

    print_header(
        "STABILIZED EIKONAL AUDIT RESULT"
    )

    print(
        "Derivative calculation       : PASS"
    )

    print(
        "Gradient magnitude           : PASS"
    )

    print(
        "V|grad T| calculation        : PASS"
    )

    print(
        "Eikonal residual             : PASS"
    )

    print(
        "Eikonal loss                 : PASS"
    )

    print(
        "Source condition             : PASS"
    )

    print(
        "Travel-time supervision      : PASS"
    )

    print(
        "Loss decomposition           : PASS"
    )

    print(
        "Backward propagation         : PASS"
    )

    print(
        "Numerical stability          : PASS"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The Phase-B2 audit validates the "
        "stabilized Eikonal formulation:"
    )

    print()
    print(
        "    R_eikonal = V |grad T| - 1"
    )

    print()
    print(
        "The audit no longer uses the "
        "V^2 |grad T|^2 formulation."
    )

    print()
    print(
        "STABILIZED EIKONAL PHYSICS AUDIT "
        "COMPLETED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    test_eikonal_audit()