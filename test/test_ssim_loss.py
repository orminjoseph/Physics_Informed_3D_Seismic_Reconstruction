"""
=========================================================
SSIM Loss Test
=========================================================

Tests the 3D SSIM loss used for seismic reconstruction.

The test verifies:

1. Identical seismic volumes produce SSIM = 1.
2. Identical seismic volumes produce SSIM loss = 0.
3. A modified seismic volume produces a larger loss.
4. Backpropagation works.
5. The loss is finite.
6. The tensors follow [B,C,D,H,W] convention.

Author: Ormin Joseph
=========================================================
"""

import torch

from losses.ssim_loss import SSIMLoss


def create_synthetic_seismic_volume():
    """
    Create a simple synthetic 3D seismic volume.

    Shape:

        [B,C,D,H,W]

    Returns
    -------
    torch.Tensor
    """

    depth = 64
    height = 128
    width = 128

    volume = torch.zeros(
        1,
        1,
        depth,
        height,
        width
    )

    # -------------------------------------------------
    # Create synthetic seismic reflectors
    # -------------------------------------------------

    for z in range(
        8,
        depth,
        8
    ):

        volume[
            :,
            :,
            z,
            :,
            :
        ] = 0.5

    return volume


def test_ssim_identical_volumes():

    print()
    print("=" * 60)
    print("TEST 1 - IDENTICAL SEISMIC VOLUMES")
    print("=" * 60)

    target = create_synthetic_seismic_volume()

    prediction = target.clone()

    criterion = SSIMLoss()

    loss = criterion(
        prediction,
        target
    )

    score = 1.0 - loss

    print(
        f"SSIM Score : {score.item():.6f}"
    )

    print(
        f"SSIM Loss  : {loss.item():.6f}"
    )

    assert torch.isfinite(loss)

    assert abs(
        score.item() - 1.0
    ) < 1.0e-5

    assert abs(
        loss.item()
    ) < 1.0e-5


def test_ssim_degraded_reconstruction():

    print()
    print("=" * 60)
    print("TEST 2 - DEGRADED SEISMIC RECONSTRUCTION")
    print("=" * 60)

    target = create_synthetic_seismic_volume()

    prediction = target.clone()

    # -------------------------------------------------
    # Introduce reconstruction error
    # -------------------------------------------------

    prediction[
        :,
        :,
        16:32,
        :,
        :
    ] = -0.5

    criterion = SSIMLoss()

    loss = criterion(
        prediction,
        target
    )

    score = 1.0 - loss

    print(
        f"SSIM Score : {score.item():.6f}"
    )

    print(
        f"SSIM Loss  : {loss.item():.6f}"
    )

    assert torch.isfinite(loss)

    assert loss.item() > 0.0

    assert score.item() < 1.0


def test_ssim_backward():

    print()
    print("=" * 60)
    print("TEST 3 - SSIM BACKWARD PASS")
    print("=" * 60)

    target = create_synthetic_seismic_volume()

    prediction = target.clone()

    prediction.requires_grad_(True)

    criterion = SSIMLoss()

    loss = criterion(
        prediction,
        target
    )

    loss.backward()

    print(
        f"SSIM Loss : {loss.item():.6f}"
    )

    print(
        "Backward pass: OK"
    )

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()


def test_ssim_shape_validation():

    print()
    print("=" * 60)
    print("TEST 4 - SSIM SHAPE VALIDATION")
    print("=" * 60)

    criterion = SSIMLoss()

    prediction = torch.zeros(
        1,
        1,
        64,
        128,
        128
    )

    target = torch.zeros(
        1,
        1,
        32,
        128,
        128
    )

    try:

        criterion(
            prediction,
            target
        )

        assert False, (
            "Expected shape validation error."
        )

    except ValueError:

        print(
            "Shape validation: OK"
        )


def test_ssim_finite():

    print()
    print("=" * 60)
    print("TEST 5 - SSIM FINITE VALUE")
    print("=" * 60)

    target = create_synthetic_seismic_volume()

    prediction = target + (
        torch.randn_like(target)
        * 0.05
    )

    prediction = torch.clamp(
        prediction,
        -1.0,
        1.0
    )

    criterion = SSIMLoss()

    loss = criterion(
        prediction,
        target
    )

    score = 1.0 - loss

    print(
        f"SSIM Score : {score.item():.6f}"
    )

    print(
        f"SSIM Loss  : {loss.item():.6f}"
    )

    assert torch.isfinite(loss)

    assert torch.isfinite(score)