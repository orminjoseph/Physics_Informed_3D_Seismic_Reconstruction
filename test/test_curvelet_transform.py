"""
=========================================================
3-D Curvelet / UDCT Transform Validation
=========================================================

Purpose:
    Verify that the selected Curvelets 1.2 UDCT
    implementation can perform a stable 3-D
    forward and inverse transform.

This is an implementation validation test only.
It is NOT the Curvelet-POCS baseline yet.

=========================================================
"""

import numpy as np

from curvelets.numpy import UDCT


# =========================================================
# TEST CONFIGURATION
# =========================================================

# Small volume for fast validation.
VOLUME_SHAPE = (32, 32, 32)

# Fixed seed for reproducibility.
SEED = 42


# =========================================================
# REPRODUCIBLE TEST VOLUME
# =========================================================

rng = np.random.default_rng(SEED)

volume = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)


print("=" * 60)
print("3-D CURVELET / UDCT TRANSFORM VALIDATION")
print("=" * 60)

print(f"Input shape : {volume.shape}")
print(f"Input dtype : {volume.dtype}")
print(f"Random seed : {SEED}")


# =========================================================
# CREATE 3-D UDCT TRANSFORM
# =========================================================

transform = UDCT(
    shape=VOLUME_SHAPE,
)


print("UDCT creation : PASS")


# =========================================================
# FORWARD TRANSFORM
# =========================================================

coefficients = transform.forward(volume)

print("Forward transform : PASS")


# =========================================================
# BACKWARD TRANSFORM
# =========================================================

reconstructed = transform.backward(coefficients)

print("backward transform : PASS")


# =========================================================
# SHAPE VALIDATION
# =========================================================

if reconstructed.shape != volume.shape:

    raise RuntimeError(
        "Reconstructed volume has an unexpected shape: "
        f"{reconstructed.shape}. "
        f"Expected: {volume.shape}"
    )

print("Shape validation : PASS")


# =========================================================
# FINITE-VALUE VALIDATION
# =========================================================

if not np.all(np.isfinite(reconstructed)):

    raise RuntimeError(
        "Reconstructed volume contains "
        "non-finite values."
    )

print("Finite-value validation : PASS")


# =========================================================
# RECONSTRUCTION ERROR
# =========================================================

absolute_error = np.abs(
    reconstructed - volume
)

max_error = np.max(absolute_error)

mean_error = np.mean(absolute_error)

relative_error = (
    np.linalg.norm(reconstructed - volume)
    /
    np.linalg.norm(volume)
)


print(f"Maximum absolute error : {max_error:.6e}")
print(f"Mean absolute error    : {mean_error:.6e}")
print(f"Relative L2 error      : {relative_error:.6e}")


# =========================================================
# RECONSTRUCTION ACCURACY
# =========================================================

# Numerical transforms should reconstruct the original
# volume to a small floating-point tolerance.

if not np.allclose(
    reconstructed,
    volume,
    rtol=1e-4,
    atol=1e-5,
):

    raise RuntimeError(
        "UDCT forward/backward reconstruction "
        "error exceeds tolerance."
    )

print("Reconstruction accuracy : PASS")


# =========================================================
# REPRODUCIBILITY TEST
# =========================================================

coefficients_2 = transform.forward(volume)

reconstructed_2 = transform.backward(coefficients_2)

reproducibility_difference = np.max(
    np.abs(reconstructed_2 - reconstructed)
)

print(
    "Reproducibility difference : "
    f"{reproducibility_difference:.6e}"
)


if reproducibility_difference > 1e-6:

    raise RuntimeError(
        "UDCT reconstruction is not reproducible."
    )

print("Reproducibility : PASS")


# =========================================================
# FINAL STATUS
# =========================================================

print()
print("=" * 60)
print("3-D UDCT TRANSFORM VALIDATION COMPLETE")
print("=" * 60)

print("Overall status : PASS")
print("=" * 60)