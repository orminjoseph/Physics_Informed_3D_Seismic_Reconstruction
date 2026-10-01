"""
=========================================================
Test: Reconstruction Evaluation Metrics
=========================================================

Canonical metric implementation:

    metrics/reconstruction_metrics.py

This test validates the project's authoritative
reconstruction and uncertainty metrics.

Tensor convention:

    [B, C, D, H, W]

Mask convention:

    1 = observed
    0 = missing

Run:

    python -m test.test_reconstruction_metrics

=========================================================
"""

import math

import pytest
import torch

from metrics.reconstruction_metrics import (
    mae,
    mse,
    rmse,
    relative_error,
    psnr,
    snr,
    ssim,

    missing_mae,
    missing_rmse,
    observed_mae,
    observed_rmse,

    uncertainty,
    aleatoric_variance,
    epistemic_variance,
    predictive_variance,
    predictive_std,

    calculate_reconstruction_metrics,
    calculate_uncertainty_metrics,
)


# =========================================================
# TEST TOLERANCES
# =========================================================

ATOL = 1.0e-5
RTOL = 1.0e-5


# =========================================================
# RECONSTRUCTION DATA FIXTURE
# =========================================================

@pytest.fixture
def data():
    """
    Create a deterministic 3D seismic test volume.

    Shape:

        [B, C, D, H, W]

        [1, 1, 4, 4, 4]
    """

    target = torch.zeros(
        (1, 1, 4, 4, 4),
        dtype=torch.float32
    )

    # Create a simple non-zero target structure.
    target[
        :,
        :,
        1:3,
        1:3,
        1:3
    ] = 1.0

    # Copy target to create reconstruction.
    reconstruction = target.clone()

    # Introduce three reconstruction errors.
    reconstruction[
        :, :, 1, 1, 1
    ] = 0.8

    reconstruction[
        :, :, 1, 1, 2
    ] = 0.6

    reconstruction[
        :, :, 2, 2, 2
    ] = 0.4

    # Observation mask.
    #
    # 1 = observed
    # 0 = missing
    mask = torch.ones_like(
        target
    )

    # The three erroneous voxels are missing.
    mask[
        :, :, 1, 1, 1
    ] = 0.0

    mask[
        :, :, 1, 1, 2
    ] = 0.0

    mask[
        :, :, 2, 2, 2
    ] = 0.0

    return (
        reconstruction,
        target,
        mask
    )


# =========================================================
# UNCERTAINTY DATA FIXTURE
# =========================================================

@pytest.fixture
def uncertainty_data():
    """
    Create deterministic uncertainty data.

    The network predicts:

        log_variance

    MC-Dropout provides:

        reconstruction_samples
    """

    # log variance corresponding to:
    #
    # exp(log_variance) = 0.25
    #
    # Therefore:
    #
    # log_variance = log(0.25)

    log_variance = torch.full(
        (1, 1, 4, 4, 4),
        math.log(0.25),
        dtype=torch.float32
    )

    # Three MC-Dropout reconstruction samples.
    mc_predictions = torch.stack(
        [
            torch.full(
                (1, 1, 4, 4, 4),
                0.8
            ),

            torch.full(
                (1, 1, 4, 4, 4),
                1.0
            ),

            torch.full(
                (1, 1, 4, 4, 4),
                1.2
            ),
        ],
        dim=0
    )

    return (
        log_variance,
        mc_predictions
    )


# =========================================================
# MAE
# =========================================================

def test_mae(data):
    """
    Test Mean Absolute Error.
    """

    reconstruction, target, _ = data

    value = mae(
        reconstruction,
        target
    )

    # Errors:
    #
    # 0.2
    # 0.4
    # 0.6
    #
    # Total = 1.2
    #
    # Total voxels = 64

    expected = 1.2 / 64.0

    assert torch.isfinite(value)

    assert torch.isclose(
        value,
        torch.tensor(expected),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# MSE
# =========================================================

def test_mse(data):
    """
    Test Mean Squared Error.
    """

    reconstruction, target, _ = data

    value = mse(
        reconstruction,
        target
    )

    expected = (
        0.2 ** 2
        +
        0.4 ** 2
        +
        0.6 ** 2
    ) / 64.0

    assert torch.isfinite(value)

    assert torch.isclose(
        value,
        torch.tensor(expected),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# RMSE
# =========================================================

def test_rmse(data):
    """
    Test Root Mean Squared Error.
    """

    reconstruction, target, _ = data

    value = rmse(
        reconstruction,
        target
    )

    expected = math.sqrt(
        (
            0.2 ** 2
            +
            0.4 ** 2
            +
            0.6 ** 2
        ) / 64.0
    )

    assert torch.isfinite(value)

    assert torch.isclose(
        value,
        torch.tensor(expected),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# GLOBAL METRICS
# =========================================================

@pytest.mark.parametrize(
    "metric",
    [
        relative_error,
        psnr,
        snr,
        ssim
    ]
)
def test_global_metrics_are_finite(
    data,
    metric
):
    """
    Verify that global metrics produce finite values.
    """

    reconstruction, target, _ = data

    value = metric(
        reconstruction,
        target
    )

    assert torch.isfinite(
        torch.as_tensor(value)
    )


# =========================================================
# SSIM RANGE
# =========================================================

def test_ssim_range(data):
    """
    Verify that SSIM remains within its expected range.
    """

    reconstruction, target, _ = data

    value = ssim(
        reconstruction,
        target
    )

    assert -1.0 <= float(value) <= 1.0


# =========================================================
# MISSING MAE
# =========================================================

def test_missing_mae(data):
    """
    Test MAE in the missing region only.
    """

    reconstruction, target, mask = data

    value = missing_mae(
        reconstruction,
        target,
        mask
    )

    expected = (
        0.2
        +
        0.4
        +
        0.6
    ) / 3.0

    assert torch.isclose(
        value,
        torch.tensor(expected),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# MISSING RMSE
# =========================================================

def test_missing_rmse(data):
    """
    Test RMSE in the missing region only.
    """

    reconstruction, target, mask = data

    value = missing_rmse(
        reconstruction,
        target,
        mask
    )

    expected = math.sqrt(
        (
            0.2 ** 2
            +
            0.4 ** 2
            +
            0.6 ** 2
        ) / 3.0
    )

    assert torch.isclose(
        value,
        torch.tensor(expected),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# OBSERVED METRICS
# =========================================================

def test_observed_metrics_are_zero(data):
    """
    The reconstruction is identical to the target at every
    observed voxel.

    Therefore both observed MAE and observed RMSE must be zero.
    """

    reconstruction, target, mask = data

    observed_mae_value = observed_mae(
        reconstruction,
        target,
        mask
    )

    observed_rmse_value = observed_rmse(
        reconstruction,
        target,
        mask
    )

    assert torch.isclose(
        observed_mae_value,
        torch.tensor(0.0),
        atol=ATOL
    )

    assert torch.isclose(
        observed_rmse_value,
        torch.tensor(0.0),
        atol=ATOL
    )


# =========================================================
# ALEATORIC VARIANCE
# =========================================================

def test_aleatoric_variance(
    uncertainty_data
):
    """
    Verify:

        aleatoric_variance
            = exp(log_variance)
    """

    log_variance, _ = uncertainty_data

    value = aleatoric_variance(
        log_variance
    )

    expected = torch.full_like(
        log_variance,
        0.25
    )

    assert torch.allclose(
        value,
        expected,
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# UNCERTAINTY COMPATIBILITY FUNCTION
# =========================================================

def test_uncertainty(
    uncertainty_data
):
    """
    Verify the compatibility uncertainty() function.

    The project's API defines uncertainty() as the mean
    aleatoric variance.
    """

    log_variance, _ = uncertainty_data

    value = uncertainty(
        log_variance
    )

    assert torch.isfinite(value)

    assert torch.isclose(
        value,
        torch.tensor(0.25),
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# EPISTEMIC VARIANCE
# =========================================================

def test_epistemic_variance(
    uncertainty_data
):
    """
    Verify MC-Dropout epistemic variance.

    The variance must be calculated over dimension 0,
    representing the MC samples.
    """

    _, mc_predictions = uncertainty_data

    value = epistemic_variance(
        mc_predictions
    )

    expected_variance = (
        (
            (0.8 - 1.0) ** 2
            +
            (1.0 - 1.0) ** 2
            +
            (1.2 - 1.0) ** 2
        ) / 3.0
    )

    expected = torch.full(
        (1, 1, 4, 4, 4),
        expected_variance
    )

    assert torch.allclose(
        value,
        expected,
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# PREDICTIVE VARIANCE
# =========================================================

def test_predictive_variance(
    uncertainty_data
):
    """
    Verify:

        predictive variance
            =
        aleatoric variance
            +
        epistemic variance
    """

    log_variance, mc_predictions = uncertainty_data

    aleatoric = aleatoric_variance(
        log_variance
    )

    epistemic = epistemic_variance(
        mc_predictions
    )

    predictive = predictive_variance(
        aleatoric,
        epistemic
    )

    expected = (
        aleatoric
        +
        epistemic
    )

    assert torch.allclose(
        predictive,
        expected,
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# PREDICTIVE STANDARD DEVIATION
# =========================================================

def test_predictive_std(
    uncertainty_data
):
    """
    Verify:

        predictive_std
            =
        sqrt(predictive_variance)
    """

    log_variance, mc_predictions = uncertainty_data

    aleatoric = aleatoric_variance(
        log_variance
    )

    epistemic = epistemic_variance(
        mc_predictions
    )

    predictive = predictive_variance(
        aleatoric,
        epistemic
    )

    value = predictive_std(
        predictive
    )

    expected = torch.sqrt(
        predictive
    )

    assert torch.allclose(
        value,
        expected,
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# RECONSTRUCTION METRIC COLLECTION
# =========================================================

def test_calculate_reconstruction_metrics(
    data
):
    """
    Test the standardized reconstruction metric collection.
    """

    reconstruction, target, mask = data

    result = calculate_reconstruction_metrics(
        reconstruction,
        target,
        mask=mask
    )

    required_keys = {
        "mae",
        "mse",
        "rmse",
        "relative_error",
        "psnr",
        "snr",
        "ssim",
        "missing_mae",
        "missing_rmse",
        "observed_mae",
        "observed_rmse"
    }

    assert required_keys.issubset(
        result.keys()
    )

    for key in required_keys:

        assert torch.isfinite(
            torch.as_tensor(
                result[key]
            )
        ).all()


# =========================================================
# UNCERTAINTY METRIC COLLECTION
# =========================================================

def test_calculate_uncertainty_metrics(
    uncertainty_data
):
    """
    Test the standardized uncertainty metric collection.
    """

    log_variance, mc_predictions = (
        uncertainty_data
    )

    result = calculate_uncertainty_metrics(
        log_variance,
        mc_predictions
    )

    required_keys = {
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std"
    }

    assert required_keys.issubset(
        result.keys()
    )

    for key in required_keys:

        value = torch.as_tensor(
            result[key]
        )

        assert torch.isfinite(
            value
        ).all()


    # Verify variance decomposition.

    expected_predictive = (
        result["aleatoric_variance"]
        +
        result["epistemic_variance"]
    )

    assert torch.allclose(
        result["predictive_variance"],
        expected_predictive,
        atol=ATOL,
        rtol=RTOL
    )


# =========================================================
# PERFECT RECONSTRUCTION
# =========================================================

def test_perfect_reconstruction():
    """
    Perfect reconstruction must produce exactly zero
    reconstruction error.
    """

    torch.manual_seed(42)

    target = torch.rand(
        (1, 1, 4, 4, 4)
    )

    reconstruction = target.clone()

    assert torch.isclose(
        mae(
            reconstruction,
            target
        ),
        torch.tensor(0.0),
        atol=ATOL
    )

    assert torch.isclose(
        mse(
            reconstruction,
            target
        ),
        torch.tensor(0.0),
        atol=ATOL
    )

    assert torch.isclose(
        rmse(
            reconstruction,
            target
        ),
        torch.tensor(0.0),
        atol=ATOL
    )


# =========================================================
# SHAPE VALIDATION
# =========================================================

def test_shape_mismatch_rejected():
    """
    Prediction and target must have identical shapes.
    """

    reconstruction = torch.zeros(
        (1, 1, 4, 4, 4)
    )

    target = torch.zeros(
        (1, 1, 4, 4, 5)
    )

    with pytest.raises(
        (ValueError, AssertionError)
    ):

        mae(
            reconstruction,
            target
        )


# =========================================================
# NaN VALIDATION
# =========================================================

def test_nan_rejected():
    """
    NaN input must be rejected.
    """

    reconstruction = torch.zeros(
        (1, 1, 4, 4, 4)
    )

    target = torch.zeros_like(
        reconstruction
    )

    reconstruction[
        0, 0, 0, 0, 0
    ] = float("nan")

    with pytest.raises(
        (ValueError, AssertionError)
    ):

        mae(
            reconstruction,
            target
        )


# =========================================================
# INFINITY VALIDATION
# =========================================================

def test_inf_rejected():
    """
    Infinite input must be rejected.
    """

    reconstruction = torch.zeros(
        (1, 1, 4, 4, 4)
    )

    target = torch.zeros_like(
        reconstruction
    )

    reconstruction[
        0, 0, 0, 0, 0
    ] = float("inf")

    with pytest.raises(
        (ValueError, AssertionError)
    ):

        mae(
            reconstruction,
            target
        )


# =========================================================
# INVALID MASK VALUES
# =========================================================

def test_invalid_mask_rejected(
    data
):
    """
    Mask must contain only 0 and 1.
    """

    reconstruction, target, mask = data

    invalid_mask = mask.clone()

    invalid_mask[
        0, 0, 0, 0, 0
    ] = 2.0

    with pytest.raises(
        (ValueError, AssertionError)
    ):

        missing_mae(
            reconstruction,
            target,
            invalid_mask
        )


# =========================================================
# MASK SHAPE VALIDATION
# =========================================================

def test_mask_shape_mismatch_rejected(
    data
):
    """
    Mask shape must match prediction/target shape.
    """

    reconstruction, target, _ = data

    invalid_mask = torch.ones(
        (1, 1, 4, 4, 5)
    )

    with pytest.raises(
        (ValueError, AssertionError)
    ):

        missing_mae(
            reconstruction,
            target,
            invalid_mask
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-v"
            ]
        )
    )