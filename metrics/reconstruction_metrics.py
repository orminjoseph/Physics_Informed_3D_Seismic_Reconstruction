"""
=====================================================================
Reconstruction Metrics
=====================================================================

Canonical metric implementation for the:

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings.

This module provides:

1. Global reconstruction metrics
   - MAE
   - MSE
   - RMSE
   - Relative L2 Error
   - PSNR
   - SNR
   - 3D SSIM

2. Regional reconstruction metrics
   - Missing-region MAE
   - Missing-region RMSE
   - Observed-region MAE
   - Observed-region RMSE

3. Predictive uncertainty metrics
   - Aleatoric variance
   - Epistemic variance
   - Predictive variance
   - Predictive standard deviation

4. Metric collection functions
   - calculate_reconstruction_metrics()
   - calculate_uncertainty_metrics()

Tensor convention:

    Reconstruction / target:
        [B, C, D, H, W]

    Monte Carlo reconstruction samples:
        [N, B, C, D, H, W]

    Observation mask:
        [B, C, D, H, W]

    Mask convention:
        1 = observed
        0 = missing

Author: Ormin Joseph
=====================================================================
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# =====================================================================
# CONSTANTS
# =====================================================================

# Seismic amplitudes are normalized to [-1, 1].
# Therefore:
#
#     DATA_RANGE = 1 - (-1) = 2
#
DATA_RANGE = 2.0

# Numerical safety constant.
#
# IMPORTANT:
# This is NOT added inside RMSE or predictive_std square roots.
# Doing so would make a mathematically zero result equal to 1e-4.
#
# EPSILON is reserved for operations involving division or logarithms.
EPSILON = 1.0e-8

# Default parameters for the custom 3D SSIM implementation.
DEFAULT_SSIM_WINDOW_SIZE = 11
DEFAULT_SSIM_SIGMA = 1.5


# =====================================================================
# VALIDATION UTILITIES
# =====================================================================

def _validate_prediction_target(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> None:
    """
    Validate prediction and target tensors.
    """

    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            "prediction must be a torch.Tensor."
        )

    if not isinstance(target, torch.Tensor):
        raise TypeError(
            "target must be a torch.Tensor."
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes. "
            f"Got {prediction.shape} and {target.shape}."
        )

    if prediction.ndim != 5:
        raise ValueError(
            "prediction and target must have shape "
            "[B, C, D, H, W]. "
            f"Got {prediction.ndim} dimensions."
        )

    if not torch.isfinite(prediction).all():
        raise ValueError(
            "prediction contains NaN or Inf values."
        )

    if not torch.isfinite(target).all():
        raise ValueError(
            "target contains NaN or Inf values."
        )


def _validate_mask(
    mask: torch.Tensor,
    reference: torch.Tensor
) -> None:
    """
    Validate an observation mask.

    Mask convention:

        1 = observed
        0 = missing
    """

    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    if mask.shape != reference.shape:
        raise ValueError(
            "mask and reference tensor must have identical shapes. "
            f"Got {mask.shape} and {reference.shape}."
        )

    if mask.ndim != 5:
        raise ValueError(
            "mask must have shape [B, C, D, H, W]. "
            f"Got {mask.ndim} dimensions."
        )

    if not torch.isfinite(mask).all():
        raise ValueError(
            "mask contains NaN or Inf values."
        )

    unique_values = torch.unique(mask)

    valid_values = torch.tensor(
        [0.0, 1.0],
        dtype=mask.dtype,
        device=mask.device
    )

    for value in unique_values:
        if not torch.any(
            torch.isclose(
                value,
                valid_values,
                atol=EPSILON
            )
        ):
            raise ValueError(
                "mask must contain only 0 and 1 values. "
                f"Found value: {value.item()}."
            )


def _validate_uncertainty_tensor(
    tensor: torch.Tensor,
    name: str
) -> None:
    """
    Validate an uncertainty tensor.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )

    if torch.any(tensor < 0):
        raise ValueError(
            f"{name} contains negative variance values."
        )


# =====================================================================
# MAE
# =====================================================================

def mae(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Mean Absolute Error.

    MAE = mean(|prediction - target|)
    """

    _validate_prediction_target(
        prediction,
        target
    )

    return torch.mean(
        torch.abs(prediction - target)
    )


# =====================================================================
# MSE
# =====================================================================

def mse(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Mean Squared Error.

    MSE = mean((prediction - target)^2)
    """

    _validate_prediction_target(
        prediction,
        target
    )

    return torch.mean(
        (prediction - target) ** 2
    )


# =====================================================================
# RMSE
# =====================================================================

def rmse(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Root Mean Squared Error.

    RMSE = sqrt(MSE)

    No epsilon is added inside the square root because a perfect
    reconstruction must produce exactly zero RMSE.
    """

    mse_value = mse(
        prediction,
        target
    )

    return torch.sqrt(
        mse_value
    )


# =====================================================================
# RELATIVE L2 ERROR
# =====================================================================

def relative_error(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Relative L2 Error.

    Relative L2 Error =
        ||prediction - target||_2 /
        ||target||_2

    EPSILON is used only in the denominator to avoid division by zero.
    """

    _validate_prediction_target(
        prediction,
        target
    )

    numerator = torch.linalg.vector_norm(
        prediction - target
    )

    denominator = torch.linalg.vector_norm(
        target
    )

    return numerator / (
        denominator + EPSILON
    )


# =====================================================================
# PSNR
# =====================================================================

def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = DATA_RANGE
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    A numerical floor is applied to MSE because logarithm of zero
    is undefined.

    Therefore, a perfect reconstruction returns a very large
    finite PSNR rather than infinity.
    """

    _validate_prediction_target(
        prediction,
        target
    )

    if data_range <= 0:
        raise ValueError(
            "data_range must be greater than zero."
        )

    mse_value = torch.mean(
        (prediction - target) ** 2
    )

    mse_safe = torch.clamp(
        mse_value,
        min=EPSILON
    )

    return 10.0 * torch.log10(
        (data_range ** 2) / mse_safe
    )


# =====================================================================
# SNR
# =====================================================================

def snr(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Signal-to-Noise Ratio.

    SNR = 10 * log10(signal_power / noise_power)

    Signal:
        target

    Noise:
        prediction - target
    """

    _validate_prediction_target(
        prediction,
        target
    )

    signal_power = torch.mean(
        target ** 2
    )

    noise_power = torch.mean(
        (prediction - target) ** 2
    )

    signal_power = torch.clamp(
        signal_power,
        min=EPSILON
    )

    noise_power = torch.clamp(
        noise_power,
        min=EPSILON
    )

    return 10.0 * torch.log10(
        signal_power / noise_power
    )


# =====================================================================
# 3D SSIM
# =====================================================================

def _create_3d_gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a separable 3D Gaussian kernel.
    """

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be odd."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than zero."
        )

    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype
    )

    coordinates = (
        coordinates -
        window_size // 2
    )

    gaussian_1d = torch.exp(
        -(coordinates ** 2) /
        (2.0 * sigma ** 2)
    )

    gaussian_1d = (
        gaussian_1d /
        torch.sum(gaussian_1d)
    )

    kernel = (
        gaussian_1d[:, None, None] *
        gaussian_1d[None, :, None] *
        gaussian_1d[None, None, :]
    )

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    kernel = kernel.repeat(
        channels,
        1,
        1,
        1,
        1
    )

    return kernel


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = DATA_RANGE,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA
) -> torch.Tensor:
    """
    Compute 3D Structural Similarity Index (SSIM).

    Expected tensor shape:

        [B, C, D, H, W]
    """

    _validate_prediction_target(
        prediction,
        target
    )

    if data_range <= 0:
        raise ValueError(
            "data_range must be greater than zero."
        )

    batch_size, channels, depth, height, width = (
        prediction.shape
    )

    # Prevent the SSIM window from being larger than the
    # smallest spatial dimension.
    maximum_window = min(
        depth,
        height,
        width
    )

    if maximum_window < 3:
        raise ValueError(
            "All spatial dimensions must be at least 3 "
            "for 3D SSIM."
        )

    actual_window_size = min(
        window_size,
        maximum_window
    )

    if actual_window_size % 2 == 0:
        actual_window_size -= 1

    if actual_window_size < 3:
        raise ValueError(
            "SSIM window size must be at least 3."
        )

    kernel = _create_3d_gaussian_kernel(
        window_size=actual_window_size,
        sigma=sigma,
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype
    )

    padding = actual_window_size // 2

    mu_prediction = F.conv3d(
        prediction,
        kernel,
        padding=padding,
        groups=channels
    )

    mu_target = F.conv3d(
        target,
        kernel,
        padding=padding,
        groups=channels
    )

    mu_prediction_squared = (
        mu_prediction ** 2
    )

    mu_target_squared = (
        mu_target ** 2
    )

    mu_prediction_target = (
        mu_prediction *
        mu_target
    )

    sigma_prediction_squared = (
        F.conv3d(
            prediction ** 2,
            kernel,
            padding=padding,
            groups=channels
        )
        -
        mu_prediction_squared
    )

    sigma_target_squared = (
        F.conv3d(
            target ** 2,
            kernel,
            padding=padding,
            groups=channels
        )
        -
        mu_target_squared
    )

    sigma_prediction_target = (
        F.conv3d(
            prediction * target,
            kernel,
            padding=padding,
            groups=channels
        )
        -
        mu_prediction_target
    )

    # Numerical protection against tiny negative values introduced
    # by floating-point round-off.
    sigma_prediction_squared = torch.clamp(
        sigma_prediction_squared,
        min=0.0
    )

    sigma_target_squared = torch.clamp(
        sigma_target_squared,
        min=0.0
    )

    c1 = (
        0.01 * data_range
    ) ** 2

    c2 = (
        0.03 * data_range
    ) ** 2

    numerator_1 = (
        2.0 * mu_prediction_target +
        c1
    )

    denominator_1 = (
        mu_prediction_squared +
        mu_target_squared +
        c1
    )

    numerator_2 = (
        2.0 * sigma_prediction_target +
        c2
    )

    denominator_2 = (
        sigma_prediction_squared +
        sigma_target_squared +
        c2
    )

    ssim_map = (
        numerator_1 /
        (denominator_1 + EPSILON)
    ) * (
        numerator_2 /
        (denominator_2 + EPSILON)
    )

    return torch.mean(
        ssim_map
    )


# =====================================================================
# REGIONAL ERROR UTILITIES
# =====================================================================

def _regional_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    region_value: float
) -> torch.Tensor:
    """
    Calculate mean absolute error in a selected mask region.
    """

    _validate_prediction_target(
        prediction,
        target
    )

    _validate_mask(
        mask,
        prediction
    )

    region = (
        mask == region_value
    )

    if not torch.any(region):
        return torch.tensor(
            0.0,
            dtype=prediction.dtype,
            device=prediction.device
        )

    return torch.mean(
        torch.abs(
            prediction[region] -
            target[region]
        )
    )


def _regional_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    region_value: float
) -> torch.Tensor:
    """
    Calculate RMSE in a selected mask region.

    No epsilon is added inside sqrt().
    """

    _validate_prediction_target(
        prediction,
        target
    )

    _validate_mask(
        mask,
        prediction
    )

    region = (
        mask == region_value
    )

    if not torch.any(region):
        return torch.tensor(
            0.0,
            dtype=prediction.dtype,
            device=prediction.device
        )

    squared_error = (
        prediction[region] -
        target[region]
    ) ** 2

    return torch.sqrt(
        torch.mean(
            squared_error
        )
    )


# =====================================================================
# MISSING-REGION METRICS
# =====================================================================

def missing_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MAE over missing voxels.

    mask = 0 -> missing.
    """

    return _regional_error(
        prediction,
        target,
        mask,
        region_value=0.0
    )


def missing_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    RMSE over missing voxels.

    mask = 0 -> missing.
    """

    return _regional_rmse(
        prediction,
        target,
        mask,
        region_value=0.0
    )


# =====================================================================
# OBSERVED-REGION METRICS
# =====================================================================

def observed_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MAE over observed voxels.

    mask = 1 -> observed.
    """

    return _regional_error(
        prediction,
        target,
        mask,
        region_value=1.0
    )


def observed_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    RMSE over observed voxels.

    mask = 1 -> observed.
    """

    return _regional_rmse(
        prediction,
        target,
        mask,
        region_value=1.0
    )


# =====================================================================
# ALEATORIC UNCERTAINTY
# =====================================================================

def aleatoric_variance(
    log_variance: torch.Tensor
) -> torch.Tensor:
    """
    Convert predicted log-variance into aleatoric variance.

    variance = exp(log_variance)

    The log-variance is clamped to avoid numerical overflow.
    """

    if not isinstance(
        log_variance,
        torch.Tensor
    ):
        raise TypeError(
            "log_variance must be a torch.Tensor."
        )

    if not torch.isfinite(
        log_variance
    ).all():
        raise ValueError(
            "log_variance contains NaN or Inf values."
        )

    safe_log_variance = torch.clamp(
        log_variance,
        min=-30.0,
        max=30.0
    )

    variance = torch.exp(
        safe_log_variance
    )

    _validate_uncertainty_tensor(
        variance,
        "aleatoric variance"
    )

    return variance


# =====================================================================
# UNCERTAINTY COMPATIBILITY FUNCTION
# =====================================================================

def uncertainty(
    log_variance: torch.Tensor
) -> torch.Tensor:
    """
    Compatibility function for aleatoric uncertainty.

    Returns the mean predicted aleatoric variance.

    For the full uncertainty decomposition, use:

        calculate_uncertainty_metrics()

    instead.
    """

    variance = aleatoric_variance(
        log_variance
    )

    return torch.mean(
        variance
    )


# =====================================================================
# EPISTEMIC UNCERTAINTY
# =====================================================================

def epistemic_variance(
    reconstruction_samples: torch.Tensor
) -> torch.Tensor:
    """
    Calculate epistemic variance from Monte Carlo Dropout samples.

    Expected input:

        [N, B, C, D, H, W]

    where N is the number of stochastic forward passes.
    """

    if not isinstance(
        reconstruction_samples,
        torch.Tensor
    ):
        raise TypeError(
            "reconstruction_samples must be a torch.Tensor."
        )

    if reconstruction_samples.ndim != 6:
        raise ValueError(
            "reconstruction_samples must have shape "
            "[N, B, C, D, H, W]. "
            f"Got {reconstruction_samples.shape}."
        )

    if not torch.isfinite(
        reconstruction_samples
    ).all():
        raise ValueError(
            "reconstruction_samples contains NaN or Inf values."
        )

    number_of_samples = (
        reconstruction_samples.shape[0]
    )

    if number_of_samples < 2:
        return torch.zeros_like(
            reconstruction_samples[0]
        )

    variance = torch.var(
        reconstruction_samples,
        dim=0,
        unbiased=False
    )

    _validate_uncertainty_tensor(
        variance,
        "epistemic variance"
    )

    return variance


# =====================================================================
# PREDICTIVE VARIANCE
# =====================================================================

def predictive_variance(
    aleatoric_variance_value: torch.Tensor,
    epistemic_variance_value: torch.Tensor
) -> torch.Tensor:
    """
    Calculate predictive variance.

    Predictive variance =
        Aleatoric variance +
        Epistemic variance
    """

    if not isinstance(
        aleatoric_variance_value,
        torch.Tensor
    ):
        raise TypeError(
            "aleatoric_variance_value must be a torch.Tensor."
        )

    if not isinstance(
        epistemic_variance_value,
        torch.Tensor
    ):
        raise TypeError(
            "epistemic_variance_value must be a torch.Tensor."
        )

    if (
        aleatoric_variance_value.shape !=
        epistemic_variance_value.shape
    ):
        raise ValueError(
            "Aleatoric and epistemic variance tensors "
            "must have identical shapes. "
            f"Got {aleatoric_variance_value.shape} and "
            f"{epistemic_variance_value.shape}."
        )

    _validate_uncertainty_tensor(
        aleatoric_variance_value,
        "aleatoric variance"
    )

    _validate_uncertainty_tensor(
        epistemic_variance_value,
        "epistemic variance"
    )

    predictive = (
        aleatoric_variance_value +
        epistemic_variance_value
    )

    _validate_uncertainty_tensor(
        predictive,
        "predictive variance"
    )

    return predictive


# =====================================================================
# PREDICTIVE STANDARD DEVIATION
# =====================================================================

def predictive_std(
    predictive_variance_value: torch.Tensor
) -> torch.Tensor:
    """
    Calculate predictive standard deviation.

    Predictive standard deviation =
        sqrt(predictive variance)

    No epsilon is added inside sqrt() because zero variance must
    produce exactly zero standard deviation.
    """

    _validate_uncertainty_tensor(
        predictive_variance_value,
        "predictive variance"
    )

    return torch.sqrt(
        predictive_variance_value
    )


# =====================================================================
# COMPLETE RECONSTRUCTION METRIC COLLECTION
# =====================================================================

def calculate_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate the complete reconstruction metric set.

    Returns:

        mae
        mse
        rmse
        relative_l2
        psnr
        snr
        ssim

    If a mask is supplied, also returns:

        missing_mae
        missing_rmse
        observed_mae
        observed_rmse
    """

    _validate_prediction_target(
        prediction,
        target
    )

    metrics = {}

    # -------------------------------------------------------------
    # Global reconstruction metrics
    # -------------------------------------------------------------

    metrics["mae"] = mae(
        prediction,
        target
    )

    metrics["mse"] = mse(
        prediction,
        target
    )

    metrics["rmse"] = rmse(
        prediction,
        target
    )

    relative_l2_value = relative_error(
        prediction,
        target
    )

    # Preferred reporting name.
    metrics["relative_l2"] = relative_l2_value

    # Backward-compatible alias.
    metrics["relative_error"] = relative_l2_value

    metrics["psnr"] = psnr(
        prediction,
        target
    )

    metrics["snr"] = snr(
        prediction,
        target
    )

    metrics["ssim"] = ssim(
        prediction,
        target
    )

    # -------------------------------------------------------------
    # Regional metrics
    # -------------------------------------------------------------

    if mask is not None:

        _validate_mask(
            mask,
            prediction
        )

        metrics["missing_mae"] = missing_mae(
            prediction,
            target,
            mask
        )

        metrics["missing_rmse"] = missing_rmse(
            prediction,
            target,
            mask
        )

        metrics["observed_mae"] = observed_mae(
            prediction,
            target,
            mask
        )

        metrics["observed_rmse"] = observed_rmse(
            prediction,
            target,
            mask
        )

    return metrics


# =====================================================================
# COMPLETE UNCERTAINTY METRIC COLLECTION
# =====================================================================

def calculate_uncertainty_metrics(
    log_variance: torch.Tensor,
    reconstruction_samples: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate the complete predictive uncertainty decomposition.

    Inputs:

        log_variance:
            [B, C, D, H, W]

        reconstruction_samples:
            [N, B, C, D, H, W]

    Returns:

        aleatoric_variance
        epistemic_variance
        predictive_variance
        predictive_std
    """

    if not isinstance(
        log_variance,
        torch.Tensor
    ):
        raise TypeError(
            "log_variance must be a torch.Tensor."
        )

    if not isinstance(
        reconstruction_samples,
        torch.Tensor
    ):
        raise TypeError(
            "reconstruction_samples must be a torch.Tensor."
        )

    if log_variance.ndim != 5:
        raise ValueError(
            "log_variance must have shape "
            "[B, C, D, H, W]. "
            f"Got {log_variance.shape}."
        )

    if reconstruction_samples.ndim != 6:
        raise ValueError(
            "reconstruction_samples must have shape "
            "[N, B, C, D, H, W]. "
            f"Got {reconstruction_samples.shape}."
        )

    if (
        log_variance.shape !=
        reconstruction_samples.shape[1:]
    ):
        raise ValueError(
            "log_variance and reconstruction_samples must "
            "have compatible spatial/batch/channel shapes. "
            f"Got {log_variance.shape} and "
            f"{reconstruction_samples.shape}."
        )

    aleatoric = aleatoric_variance(
        log_variance
    )

    epistemic = epistemic_variance(
        reconstruction_samples
    )

    predictive = predictive_variance(
        aleatoric,
        epistemic
    )

    predictive_standard_deviation = predictive_std(
        predictive
    )

    return {
        "aleatoric_variance": aleatoric,
        "epistemic_variance": epistemic,
        "predictive_variance": predictive,
        "predictive_std": predictive_standard_deviation,
    }


# =====================================================================
# PUBLIC API
# =====================================================================

__all__ = [
    # Global reconstruction metrics
    "mae",
    "mse",
    "rmse",
    "relative_error",
    "psnr",
    "snr",
    "ssim",

    # Regional metrics
    "missing_mae",
    "missing_rmse",
    "observed_mae",
    "observed_rmse",

    # Uncertainty metrics
    "uncertainty",
    "aleatoric_variance",
    "epistemic_variance",
    "predictive_variance",
    "predictive_std",

    # Metric collections
    "calculate_reconstruction_metrics",
    "calculate_uncertainty_metrics",
]