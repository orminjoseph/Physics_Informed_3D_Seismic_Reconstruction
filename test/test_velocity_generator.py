"""
=========================================================
Test Velocity Model Generator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Tests:
    1. Layered velocity model
    2. Gradient velocity model
    3. Random velocity model
    4. Shape validation
    5. Velocity range validation
    6. Increasing velocity with depth
    7. Reproducibility of dimensions

Author: Ormin Joseph
=========================================================
"""

import torch

from dataset.velocity_generator import VelocityGenerator


# =========================================================
# TEST CONFIGURATION
# =========================================================

CUBE_SIZE = (64, 128, 128)

MIN_VELOCITY = 1800.0
MAX_VELOCITY = 3500.0


# =========================================================
# HELPER
# =========================================================

def check_velocity_cube(
    velocity,
    expected_shape
):
    """
    Validate a generated velocity cube.
    """

    assert isinstance(
        velocity,
        torch.Tensor
    ), (
        "Velocity model must be a torch.Tensor."
    )

    assert velocity.shape == expected_shape, (
        "Incorrect velocity model shape. "
        f"Expected {expected_shape}, "
        f"received {tuple(velocity.shape)}."
    )

    assert torch.isfinite(
        velocity
    ).all(), (
        "Velocity model contains NaN or Inf values."
    )

    assert torch.all(
        velocity >= MIN_VELOCITY
    ), (
        "Velocity model contains values below "
        "the minimum velocity."
    )

    assert torch.all(
        velocity <= MAX_VELOCITY
    ), (
        "Velocity model contains values above "
        "the maximum velocity."
    )


# =========================================================
# TEST LAYERED MODEL
# =========================================================

def test_layered_model():

    print()
    print("=" * 60)
    print("Testing Layered Velocity Model")
    print("=" * 60)

    generator = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY
    )

    velocity = generator.generate_layered_model(
        number_of_layers=4
    )

    expected_shape = (
        1,
        64,
        128,
        128
    )

    check_velocity_cube(
        velocity,
        expected_shape
    )

    print(
        "Shape        :",
        velocity.shape
    )

    print(
        "Minimum      :",
        velocity.min().item()
    )

    print(
        "Maximum      :",
        velocity.max().item()
    )

    print(
        "Unique values:",
        torch.unique(velocity).numel()
    )

    print()
    print("Layered Velocity Test: PASSED")


# =========================================================
# TEST GRADIENT MODEL
# =========================================================

def test_gradient_model():

    print()
    print("=" * 60)
    print("Testing Gradient Velocity Model")
    print("=" * 60)

    generator = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY
    )

    velocity = generator.generate_gradient_model()

    expected_shape = (
        1,
        64,
        128,
        128
    )

    check_velocity_cube(
        velocity,
        expected_shape
    )

    print(
        "Shape   :",
        velocity.shape
    )

    print(
        "Minimum :",
        velocity.min().item()
    )

    print(
        "Maximum :",
        velocity.max().item()
    )

    # -----------------------------------------------------
    # Check that velocity increases with depth
    # -----------------------------------------------------

    top_velocity = velocity[
        0,
        0,
        0,
        0
    ]

    bottom_velocity = velocity[
        0,
        -1,
        0,
        0
    ]

    assert bottom_velocity > top_velocity, (
        "Gradient velocity model should increase "
        "with depth."
    )

    print(
        "Top velocity    :",
        top_velocity.item()
    )

    print(
        "Bottom velocity :",
        bottom_velocity.item()
    )

    print()
    print("Gradient Velocity Test: PASSED")


# =========================================================
# TEST RANDOM GENERATOR
# =========================================================

def test_random_generator():

    print()
    print("=" * 60)
    print("Testing Random Velocity Generator")
    print("=" * 60)

    generator = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY
    )

    for i in range(5):

        velocity = generator.generate()

        expected_shape = (
            1,
            64,
            128,
            128
        )

        check_velocity_cube(
            velocity,
            expected_shape
        )

        print(
            f"Model {i + 1}: "
            f"min={velocity.min().item():.2f}, "
            f"max={velocity.max().item():.2f}"
        )

    print()
    print("Random Velocity Test: PASSED")


# =========================================================
# TEST DIFFERENT NUMBER OF LAYERS
# =========================================================

def test_layer_counts():

    print()
    print("=" * 60)
    print("Testing Different Layer Counts")
    print("=" * 60)

    generator = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY
    )

    for number_of_layers in (
        2,
        3,
        4,
        6
    ):

        velocity = generator.generate_layered_model(
            number_of_layers=number_of_layers
        )

        check_velocity_cube(
            velocity,
            (
                1,
                64,
                128,
                128
            )
        )

        print(
            f"{number_of_layers} layers: "
            "PASSED"
        )

    print()
    print("Layer Count Test: PASSED")


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print()
    print("=" * 60)
    print("TESTING VELOCITY MODEL GENERATOR")
    print("=" * 60)

    test_layered_model()

    test_gradient_model()

    test_random_generator()

    test_layer_counts()

    print()
    print("=" * 60)
    print(
        "VELOCITY MODEL GENERATOR TEST: PASSED"
    )
    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()