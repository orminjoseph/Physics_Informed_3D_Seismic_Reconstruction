"""
=========================================================
PHYSICS LOSS NUMERICAL AUDIT
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Purpose
-------

This test performs a numerical audit of the PhysicsLoss
implementation used by the seismic reconstruction framework.

The audit verifies:

    1. Analytical Eikonal solution
    2. Non-Eikonal field detection
    3. Spatial derivative accuracy
    4. Direct Eikonal / PhysicsLoss consistency
    5. Physics-loss weighting
    6. Disabled source-condition loss
    7. Disabled travel-time supervision
    8. Gradient propagation
    9. NaN/Inf protection
    10. Overall numerical consistency

Tensor convention
-----------------

    [B, C, D, H, W]

where:

    B = batch
    C = channel
    D = depth
    H = crossline
    W = inline

Eikonal equation
----------------

    |∇T|² = 1 / V²

Normalized form:

    V² |∇T|² = 1

Residual:

    R = V² |∇T|² - 1

Loss:

    L_eikonal = mean(R²)

Author: Ormin Joseph
=========================================================
"""

import torch

from losses.physics_loss import PhysicsLoss

from utils.config import (
    DX,
    DY,
    DZ,
    PHYSICS_LOSS_WEIGHTS,
)


# =========================================================
# TEST CONFIGURATION
# =========================================================

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

VELOCITY_VALUE = 2000.0

TOLERANCE = 1.0e-5


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_header(title):
    """
    Print a formatted audit section header.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_finite(tensor, name):
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():

        raise AssertionError(
            f"{name} contains NaN or infinite values."
        )


def tensor_statistics(tensor, name):
    """
    Print basic tensor statistics.
    """

    print(
        f"{name:<30}: "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


# =========================================================
# CREATE COORDINATE GRID
# =========================================================

def create_coordinate_grid():
    """
    Create physical coordinate grids.

    Coordinates follow:

        z = depth
        y = crossline
        x = inline

    Returns
    -------

    z, y, x : torch.Tensor

        Each tensor has shape:

            [1, 1, D, H, W]
    """

    z = (
        torch.arange(
            DEPTH,
            dtype=torch.float32
        )
        * DZ
    )

    y = (
        torch.arange(
            HEIGHT,
            dtype=torch.float32
        )
        * DY
    )

    x = (
        torch.arange(
            WIDTH,
            dtype=torch.float32
        )
        * DX
    )

    zz, yy, xx = torch.meshgrid(
        z,
        y,
        x,
        indexing="ij"
    )

    zz = zz.unsqueeze(0).unsqueeze(0)

    yy = yy.unsqueeze(0).unsqueeze(0)

    xx = xx.unsqueeze(0).unsqueeze(0)

    return zz, yy, xx


# =========================================================
# CREATE VELOCITY FIELD
# =========================================================

def create_velocity():
    """
    Create a constant physical P-wave velocity field.
    """

    return torch.full(
        (
            BATCH_SIZE,
            CHANNELS,
            DEPTH,
            HEIGHT,
            WIDTH
        ),
        VELOCITY_VALUE,
        dtype=torch.float32
    )


# =========================================================
# ANALYTICAL EIKONAL FIELD
# =========================================================

def create_analytical_eikonal_field(
    velocity
):
    """
    Create an analytical travel-time field satisfying
    the normalized Eikonal equation.

    We use:

        T = (x + y + z) / (sqrt(3) V)

    Therefore:

        dT/dx = 1 / (sqrt(3) V)

        dT/dy = 1 / (sqrt(3) V)

        dT/dz = 1 / (sqrt(3) V)

    Consequently:

        |∇T|²
        =
        3 * [1 / (3V²)]

        =
        1 / V²

    Therefore:

        V² |∇T|² = 1

    and the Eikonal residual should be approximately zero.
    """

    z, y, x = create_coordinate_grid()

    travel_time = (
        x
        +
        y
        +
        z
    ) / (
        torch.sqrt(
            torch.tensor(
                3.0,
                dtype=torch.float32
            )
        )
        *
        velocity
    )

    return travel_time


# =========================================================
# NON-EIKONAL FIELD
# =========================================================

def create_non_eikonal_field():
    """
    Create a deliberately non-Eikonal travel-time field.

    The field is:

        T = a(x + y + z)

    with a chosen so that:

        V² |∇T|² != 1

    This should produce a clearly non-zero Eikonal loss.
    """

    z, y, x = create_coordinate_grid()

    coefficient = 1.0e-5

    travel_time = coefficient * (
        x
        +
        y
        +
        z
    )

    return travel_time


# =========================================================
# ANALYTICAL DERIVATIVE
# =========================================================

def analytical_derivative():
    """
    Return the analytical derivative of the Eikonal field.

    For:

        T = (x + y + z)/(sqrt(3)V)

    all first derivatives are:

        1/(sqrt(3)V)
    """

    return (
        1.0
        /
        (
            torch.sqrt(
                torch.tensor(
                    3.0,
                    dtype=torch.float32
                )
            )
            *
            VELOCITY_VALUE
        )
    )


# =========================================================
# INITIALIZE PHYSICS LOSS
# =========================================================

def create_physics_loss():
    """
    Initialize PhysicsLoss using the global configuration.
    """

    return PhysicsLoss(
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
    )


# =========================================================
# TEST 1
# ANALYTICAL EIKONAL SOLUTION
# =========================================================

def test_analytical_eikonal(
    physics_loss,
    velocity
):
    """
    Verify that the analytical solution satisfies
    the Eikonal equation.
    """

    print_header(
        "TESTING ANALYTICAL EIKONAL SOLUTION"
    )

    travel_time = (
        create_analytical_eikonal_field(
            velocity
        )
    )

    residual = physics_loss.eikonal_residual(
        travel_time,
        velocity
    )

    loss = residual.pow(2).mean()

    check_finite(
        residual,
        "Analytical Eikonal residual"
    )

    print(
        f"Analytical Eikonal loss: "
        f"{loss.item():.12e}"
    )

    print(
        f"Maximum residual: "
        f"{residual.abs().max().item():.12e}"
    )

    if loss.item() > TOLERANCE:

        raise AssertionError(
            "Analytical Eikonal solution produced "
            "an unexpectedly large loss."
        )

    print()
    print(
        "ANALYTICAL EIKONAL TEST PASSED."
    )


# =========================================================
# TEST 2
# NON-EIKONAL FIELD
# =========================================================

def test_non_eikonal_field(
    physics_loss,
    velocity
):
    """
    Verify that a deliberately non-Eikonal field
    produces a non-zero physics loss.
    """

    print_header(
        "TESTING NON-EIKONAL FIELD DETECTION"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    loss = physics_loss.eikonal_loss(
        travel_time,
        velocity
    )

    check_finite(
        loss,
        "Non-Eikonal loss"
    )

    print(
        f"Non-Eikonal loss: "
        f"{loss.item():.12e}"
    )

    if loss.item() <= 1.0e-3:

        raise AssertionError(
            "Non-Eikonal field did not produce "
            "a sufficiently large physics loss."
        )

    print()
    print(
        "NON-EIKONAL FIELD TEST PASSED."
    )


# =========================================================
# TEST 3
# SPATIAL DERIVATIVE ACCURACY
# =========================================================

def test_derivative_accuracy(
    physics_loss
):
    """
    Verify the numerical derivative against
    the analytical derivative.
    """

    print_header(
        "SPATIAL DERIVATIVE ACCURACY AUDIT"
    )

    velocity = create_velocity()

    travel_time = (
        create_analytical_eikonal_field(
            velocity
        )
    )

    expected = analytical_derivative()

    dT_dz = physics_loss._derivative(
        travel_time,
        spacing=DZ,
        dimension=2
    )

    dT_dy = physics_loss._derivative(
        travel_time,
        spacing=DY,
        dimension=3
    )

    dT_dx = physics_loss._derivative(
        travel_time,
        spacing=DX,
        dimension=4
    )

    check_finite(
        dT_dz,
        "dT/dz"
    )

    check_finite(
        dT_dy,
        "dT/dy"
    )

    check_finite(
        dT_dx,
        "dT/dx"
    )

    error_z = (
        dT_dz - expected
    ).abs().max().item()

    error_y = (
        dT_dy - expected
    ).abs().max().item()

    error_x = (
        dT_dx - expected
    ).abs().max().item()

    print(
        f"Analytical derivative   : "
        f"{expected.item():.12e}"
    )

    print(
        f"Maximum dT/dz error     : "
        f"{error_z:.12e}"
    )

    print(
        f"Maximum dT/dy error     : "
        f"{error_y:.12e}"
    )

    print(
        f"Maximum dT/dx error     : "
        f"{error_x:.12e}"
    )

    if max(
        error_x,
        error_y,
        error_z
    ) > TOLERANCE:

        raise AssertionError(
            "Spatial derivative accuracy test failed."
        )

    print()
    print(
        "SPATIAL DERIVATIVE TEST PASSED."
    )


# =========================================================
# TEST 4
# DIRECT VS PHYSICSLOSS CONSISTENCY
# =========================================================

def test_eikonal_consistency(
    physics_loss,
    velocity
):
    """
    Verify that the Eikonal loss exposed through
    PhysicsLoss matches the direct implementation.
    """

    print_header(
        "EIKONAL CONSISTENCY CHECK"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    residual = physics_loss.eikonal_residual(
        travel_time,
        velocity
    )

    direct_loss = (
        residual.pow(2).mean()
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    physics_eikonal = result[
        "eikonal"
    ]

    difference = (
        direct_loss
        -
        physics_eikonal
    ).abs()

    print(
        f"Direct Eikonal loss       : "
        f"{direct_loss.item():.12e}"
    )

    print(
        f"PhysicsLoss Eikonal       : "
        f"{physics_eikonal.item():.12e}"
    )

    print(
        f"Absolute difference       : "
        f"{difference.item():.12e}"
    )

    if difference.item() > TOLERANCE:

        raise AssertionError(
            "Direct Eikonal loss and PhysicsLoss "
            "Eikonal component are inconsistent."
        )

    print(
        "Eikonal implementation consistency: PASSED"
    )


# =========================================================
# TEST 5
# PHYSICS WEIGHTING
# =========================================================

def test_physics_weighting(
    physics_loss,
    velocity
):
    """
    Verify that the configured Eikonal weight is
    correctly applied.
    """

    print_header(
        "PHYSICS LOSS WEIGHTING AUDIT"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    eikonal = result[
        "eikonal"
    ]

    weighted_eikonal = result[
        "weighted_eikonal"
    ]

    expected = (
        PHYSICS_LOSS_WEIGHTS[
            "eikonal"
        ]
        *
        eikonal
    )

    difference = (
        weighted_eikonal
        -
        expected
    ).abs()

    print(
        f"Eikonal loss             : "
        f"{eikonal.item():.12e}"
    )

    print(
        f"Eikonal weight            : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    print(
        f"Weighted Eikonal          : "
        f"{weighted_eikonal.item():.12e}"
    )

    print(
        f"Expected weighted value   : "
        f"{expected.item():.12e}"
    )

    print(
        f"Absolute difference       : "
        f"{difference.item():.12e}"
    )

    if difference.item() > TOLERANCE:

        raise AssertionError(
            "Eikonal weighting is incorrect."
        )

    print(
        "Physics weighting test: PASSED"
    )


# =========================================================
# TEST 6
# DISABLED SOURCE CONDITION
# =========================================================

def test_source_disabled(
    physics_loss,
    velocity
):
    """
    Verify that source loss is zero when no source
    coordinates are supplied.
    """

    print_header(
        "SOURCE CONDITION AUDIT"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        source_indices=None
    )

    source = result[
        "source"
    ]

    weighted_source = result[
        "weighted_source"
    ]

    print(
        f"Source loss              : "
        f"{source.item():.12e}"
    )

    print(
        f"Weighted source loss     : "
        f"{weighted_source.item():.12e}"
    )

    if source.item() != 0.0:

        raise AssertionError(
            "Source loss should be zero when "
            "source_indices=None."
        )

    print(
        "Source condition test: PASSED"
    )


# =========================================================
# TEST 7
# DISABLED TRAVEL-TIME SUPERVISION
# =========================================================

def test_travel_time_disabled(
    physics_loss,
    velocity
):
    """
    Verify that supervised travel-time loss is zero
    when no target is supplied.
    """

    print_header(
        "TRAVEL-TIME SUPERVISION AUDIT"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity,
        travel_time_target=None
    )

    travel_time_loss = result[
        "travel_time"
    ]

    weighted_travel_time = result[
        "weighted_travel_time"
    ]

    print(
        f"Travel-time loss         : "
        f"{travel_time_loss.item():.12e}"
    )

    print(
        f"Weighted travel-time     : "
        f"{weighted_travel_time.item():.12e}"
    )

    if travel_time_loss.item() != 0.0:

        raise AssertionError(
            "Travel-time supervision loss should "
            "be zero when no target is supplied."
        )

    print(
        "Travel-time supervision test: PASSED"
    )


# =========================================================
# TEST 8
# GRADIENT PROPAGATION
# =========================================================

def test_gradient_propagation(
    physics_loss,
    velocity
):
    """
    Verify that the physics loss is differentiable
    with respect to predicted travel time.

    This is essential because the Eikonal physics loss
    must propagate gradients back into the neural network.
    """

    print_header(
        "EIKONAL GRADIENT AUDIT"
    )

    travel_time = (
        create_non_eikonal_field()
        .clone()
        .detach()
        .requires_grad_(True)
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    physics_eikonal = result[
        "eikonal"
    ]

    print(
        f"requires_grad             : "
        f"{travel_time.requires_grad}"
    )

    print(
        f"Eikonal requires_grad     : "
        f"{physics_eikonal.requires_grad}"
    )

    if not physics_eikonal.requires_grad:

        raise AssertionError(
            "Physics Eikonal loss does not retain "
            "a computational graph."
        )

    physics_eikonal.backward()

    if travel_time.grad is None:

        raise AssertionError(
            "No gradient was produced for "
            "travel_time."
        )

    check_finite(
        travel_time.grad,
        "Travel-time gradient"
    )

    gradient_mean = (
        travel_time.grad
        .abs()
        .mean()
        .item()
    )

    gradient_max = (
        travel_time.grad
        .abs()
        .max()
        .item()
    )

    print(
        f"Gradient mean            : "
        f"{gradient_mean:.12e}"
    )

    print(
        f"Gradient maximum         : "
        f"{gradient_max:.12e}"
    )

    if gradient_max == 0.0:

        raise AssertionError(
            "Physics loss produced zero gradient."
        )

    print(
        "Eikonal gradient test: PASSED"
    )


# =========================================================
# TEST 9
# INVALID INPUT PROTECTION
# =========================================================

def test_invalid_input_protection(
    physics_loss,
    velocity
):
    """
    Verify rejection of NaN and Inf inputs.
    """

    print_header(
        "INVALID INPUT PROTECTION AUDIT"
    )

    # -----------------------------------------------------
    # NaN test
    # -----------------------------------------------------

    travel_time_nan = (
        create_non_eikonal_field()
    )

    travel_time_nan[
        0,
        0,
        0,
        0,
        0
    ] = float("nan")

    try:

        physics_loss(
            travel_time=travel_time_nan,
            velocity=velocity
        )

        raise AssertionError(
            "NaN travel-time input was not rejected."
        )

    except ValueError:

        print(
            "NaN input correctly rejected."
        )

    # -----------------------------------------------------
    # Inf test
    # -----------------------------------------------------

    travel_time_inf = (
        create_non_eikonal_field()
    )

    travel_time_inf[
        0,
        0,
        0,
        0,
        0
    ] = float("inf")

    try:

        physics_loss(
            travel_time=travel_time_inf,
            velocity=velocity
        )

        raise AssertionError(
            "Inf travel-time input was not rejected."
        )

    except ValueError:

        print(
            "Inf input correctly rejected."
        )

    print(
        "Invalid input protection test: PASSED"
    )


# =========================================================
# TEST 10
# TOTAL PHYSICS LOSS CONSISTENCY
# =========================================================

def test_total_physics_loss(
    physics_loss,
    velocity
):
    """
    Verify that the total PhysicsLoss is the sum
    of all weighted components.
    """

    print_header(
        "TOTAL PHYSICS LOSS CONSISTENCY"
    )

    travel_time = (
        create_non_eikonal_field()
    )

    result = physics_loss(
        travel_time=travel_time,
        velocity=velocity
    )

    expected_total = (
        result["weighted_eikonal"]
        +
        result["weighted_source"]
        +
        result["weighted_travel_time"]
    )

    actual_total = result[
        "total"
    ]

    difference = (
        actual_total
        -
        expected_total
    ).abs()

    print(
        f"Actual total             : "
        f"{actual_total.item():.12e}"
    )

    print(
        f"Expected total           : "
        f"{expected_total.item():.12e}"
    )

    print(
        f"Absolute difference      : "
        f"{difference.item():.12e}"
    )

    if difference.item() > TOLERANCE:

        raise AssertionError(
            "Total physics loss is inconsistent "
            "with its weighted components."
        )

    print(
        "Total physics loss test: PASSED"
    )


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print()
    print("=" * 60)
    print("PHYSICS LOSS NUMERICAL AUDIT")
    print("=" * 60)

    # =====================================================
    # PHYSICAL CONFIGURATION
    # =====================================================

    print_header(
        "PHYSICAL CONFIGURATION"
    )

    print(
        f"DX                       : {DX}"
    )

    print(
        f"DY                       : {DY}"
    )

    print(
        f"DZ                       : {DZ}"
    )

    print(
        f"Eikonal weight           : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    print(
        f"Source weight            : "
        f"{PHYSICS_LOSS_WEIGHTS['source']}"
    )

    print(
        f"Travel-time weight       : "
        f"{PHYSICS_LOSS_WEIGHTS['travel_time']}"
    )

    # =====================================================
    # CREATE TEST FIELDS
    # =====================================================

    print_header(
        "CREATING TEST FIELDS"
    )

    velocity = create_velocity()

    analytical_travel_time = (
        create_analytical_eikonal_field(
            velocity
        )
    )

    print(
        f"Tensor shape             : "
        f"{tuple(analytical_travel_time.shape)}"
    )

    tensor_statistics(
        analytical_travel_time,
        "Analytical travel-time"
    )

    tensor_statistics(
        velocity,
        "Velocity"
    )

    # =====================================================
    # INITIALIZE PHYSICS LOSS
    # =====================================================

    print_header(
        "INITIALIZING PHYSICS LOSS"
    )

    physics_loss = (
        create_physics_loss()
    )

    print(
        "PhysicsLoss successfully initialized."
    )

    # =====================================================
    # RUN TESTS
    # =====================================================

    test_analytical_eikonal(
        physics_loss,
        velocity
    )

    test_non_eikonal_field(
        physics_loss,
        velocity
    )

    test_derivative_accuracy(
        physics_loss
    )

    test_eikonal_consistency(
        physics_loss,
        velocity
    )

    test_physics_weighting(
        physics_loss,
        velocity
    )

    test_source_disabled(
        physics_loss,
        velocity
    )

    test_travel_time_disabled(
        physics_loss,
        velocity
    )

    test_gradient_propagation(
        physics_loss,
        velocity
    )

    test_invalid_input_protection(
        physics_loss,
        velocity
    )

    test_total_physics_loss(
        physics_loss,
        velocity
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()
    print("=" * 60)
    print(
        "PHYSICS LOSS NUMERICAL AUDIT PASSED"
    )
    print("=" * 60)
    print()
    print(
        "All numerical, consistency, weighting,"
    )
    print(
        "differentiability and input-validation"
    )
    print(
        "tests passed successfully."
    )
    print()


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()