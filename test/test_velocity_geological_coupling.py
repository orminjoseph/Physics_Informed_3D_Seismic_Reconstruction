"""
=========================================================
Geological–Velocity Coupling Validation
=========================================================

Validates that the velocity generator produces physically
reasonable and spatially varying velocity fields for all
supported geological modes.

Modes tested:
    horizontal
    dipping
    faulted
    folded
    complex
    highly_complex

=========================================================
"""

import torch

from dataset.velocity_generator import VelocityGenerator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CUBE_SIZE = (64, 128, 128)

MIN_VELOCITY = 1500.0
MAX_VELOCITY = 5000.0

MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("GEOLOGICAL–VELOCITY COUPLING VALIDATION")
    print("=" * 70)

    print(f"\nCube size: {CUBE_SIZE}")
    print(f"Velocity range: {MIN_VELOCITY}–{MAX_VELOCITY} m/s")

    generator = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY,
        seed=42,
    )

    results = {}

    # -----------------------------------------------------
    # Test every geological mode
    # -----------------------------------------------------

    for mode in MODES:

        print("\n" + "-" * 70)
        print(f"TESTING MODE: {mode.upper()}")
        print("-" * 70)

        velocity = generator.generate(mode=mode)

        # -------------------------------------------------
        # Shape
        # -------------------------------------------------

        print("Shape:", velocity.shape)

        assert velocity.shape == (1, *CUBE_SIZE)

        # -------------------------------------------------
        # Data type
        # -------------------------------------------------

        print("Data type:", velocity.dtype)

        assert velocity.dtype == torch.float32

        # -------------------------------------------------
        # Finite values
        # -------------------------------------------------

        assert torch.isfinite(velocity).all()

        print("Finite values: PASS")

        # -------------------------------------------------
        # Physical velocity limits
        # -------------------------------------------------

        vmin = velocity.min().item()
        vmax = velocity.max().item()
        vmean = velocity.mean().item()
        vstd = velocity.std().item()

        print(f"Minimum velocity: {vmin:.3f} m/s")
        print(f"Maximum velocity: {vmax:.3f} m/s")
        print(f"Mean velocity:    {vmean:.3f} m/s")
        print(f"Std velocity:     {vstd:.3f} m/s")

        assert vmin >= MIN_VELOCITY
        assert vmax <= MAX_VELOCITY

        print("Physical bounds: PASS")

        # -------------------------------------------------
        # Spatial variability
        # -------------------------------------------------

        assert vstd > 0.0

        print("Spatial variability: PASS")

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        results[mode] = {
            "min": vmin,
            "max": vmax,
            "mean": vmean,
            "std": vstd,
        }

    # -----------------------------------------------------
    # Compare geological modes
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-MODE COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Mode':<18}"
        f"{'Min':>10}"
        f"{'Max':>10}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
    )

    print("-" * 70)

    for mode, values in results.items():

        print(
            f"{mode:<18}"
            f"{values['min']:>10.1f}"
            f"{values['max']:>10.1f}"
            f"{values['mean']:>12.1f}"
            f"{values['std']:>12.1f}"
        )

    # -----------------------------------------------------
    # Check that modes are not all identical
    # -----------------------------------------------------

    means = [
        round(values["mean"], 3)
        for values in results.values()
    ]

    stds = [
        round(values["std"], 3)
        for values in results.values()
    ]

    print("\nUnique velocity means:", len(set(means)))
    print("Unique velocity standard deviations:", len(set(stds)))

    # We do not require every mode to have a different mean.
    # Geological differences can occur spatially while
    # preserving a similar global mean.
    #
    # Therefore, we require variation in at least one
    # statistical property across the modes.

    assert (
        len(set(means)) > 1
        or len(set(stds)) > 1
    )

    print("Cross-mode variability: PASS")

    # -----------------------------------------------------
    # Reproducibility test
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)

    generator_a = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY,
        seed=123,
    )

    generator_b = VelocityGenerator(
        cube_size=CUBE_SIZE,
        min_velocity=MIN_VELOCITY,
        max_velocity=MAX_VELOCITY,
        seed=123,
    )

    velocity_a = generator_a.generate(mode="highly_complex")
    velocity_b = generator_b.generate(mode="highly_complex")

    identical = torch.equal(
        velocity_a,
        velocity_b
    )

    print("Highly-complex fields identical:", identical)

    assert identical

    print("Reproducibility: PASS")

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL GEOLOGICAL–VELOCITY VALIDATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()