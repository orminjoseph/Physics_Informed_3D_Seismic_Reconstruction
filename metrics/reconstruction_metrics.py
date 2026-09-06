"""
=========================================================
Reconstruction Evaluation Metrics
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Metrics implemented

1. MAE
2. MSE
3. RMSE
4. PSNR
5. SNR
6. SSIM

Tensor convention:

    [B, C, D, H, W]

Normalized seismic amplitude range:

    [-1, 1]

Therefore:

    data_range = 2.0

Author: Ormin Joseph
=========================================================
"""

import torch


# =======================================================
# Mean Absolute Error
# =======================================================

def mae(prediction, target):
    """
    Calculate Mean Absolute Error (MAE).

    MAE measures the average absolute difference between
    the reconstructed seismic volume and the ground truth.

    MAE = mean(|prediction - target|)
    """

    if not isinstance(prediction, torch.Tensor):

        raise TypeError(
            "Prediction must be a torch.Tensor."
        )

    if not isinstance(target, torch.Tensor):

        raise TypeError(
            "Target must be a torch.Tensor."
        )

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target must have identical shapes "
            "for MAE calculation."
        )

    return torch.mean(
        torch.abs(
            prediction - target
        )
    )


# =======================================================
# Mean Squared Error
# =======================================================

def mse(prediction, target):
    """
    Calculate Mean Squared Error (MSE).

    MSE measures the average squared reconstruction error.

    MSE = mean((prediction - target)^2)
    """

    if not isinstance(prediction, torch.Tensor):

        raise TypeError(
            "Prediction must be a torch.Tensor."
        )

    if not isinstance(target, torch.Tensor):

        raise TypeError(
            "Target must be a torch.Tensor."
        )

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target must have identical shapes "
            "for MSE calculation."
        )

    return torch.mean(
        (
            prediction - target
        ) ** 2
    )


# =======================================================
# Root Mean Squared Error
# =======================================================

def rmse(prediction, target):
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE is the square root of MSE.

    RMSE = sqrt(MSE)
    """

    return torch.sqrt(
        mse(
            prediction,
            target
        )
    )


# =======================================================
# Peak Signal-to-Noise Ratio
# =======================================================

def psnr(
    prediction,
    target,
    data_range=2.0
):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    For normalized seismic amplitudes in [-1, 1]:

        data_range = 2.0

    A small numerical floor is applied to MSE so that
    perfect reconstruction produces a large finite PSNR
    rather than +inf.

    PSNR = 20 log10(data_range)
           - 10 log10(max(MSE, epsilon))
    """

    if data_range <= 0:

        raise ValueError(
            "data_range must be greater than zero."
        )

    mse_value = mse(
        prediction,
        target
    )

    epsilon = torch.finfo(
        prediction.dtype
    ).eps

    mse_safe = torch.clamp(
        mse_value,
        min=epsilon
    )

    return (

        20.0
        * torch.log10(
            torch.tensor(
                data_range,
                device=prediction.device,
                dtype=prediction.dtype
            )
        )

        -

        10.0
        * torch.log10(
            mse_safe
        )
    )

# =======================================================
# Signal-to-Noise Ratio
# =======================================================

def snr(prediction, target):
    """
    Calculate Signal-to-Noise Ratio (SNR).

    Signal power:

        mean(target^2)

    Noise power:

        mean((target - prediction)^2)

    A small numerical floor is applied to both signal
    and noise power to prevent NaN/Inf values during
    automated evaluation.

    SNR = 10 log10(signal_power / noise_power)
    """

    if not isinstance(prediction, torch.Tensor):

        raise TypeError(
            "Prediction must be a torch.Tensor."
        )

    if not isinstance(target, torch.Tensor):

        raise TypeError(
            "Target must be a torch.Tensor."
        )

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target must have identical shapes "
            "for SNR calculation."
        )

    signal_power = torch.mean(
        target ** 2
    )

    noise_power = torch.mean(
        (
            target - prediction
        ) ** 2
    )

    epsilon = torch.finfo(
        prediction.dtype
    ).eps

    signal_power_safe = torch.clamp(
        signal_power,
        min=epsilon
    )

    noise_power_safe = torch.clamp(
        noise_power,
        min=epsilon
    )

    return (

        10.0
        * torch.log10(
            signal_power_safe
            /
            noise_power_safe
        )
    )

# =======================================================
# Structural Similarity Index
# =======================================================

def ssim(
    prediction,
    target,
    data_range=2.0
):
    """
    Calculate Structural Similarity Index (SSIM).

    SSIM evaluates similarity using three components:

        1. Luminance
        2. Contrast
        3. Structure

    Complete formulation:

        SSIM = L * C * S

    where:

        L = (2 * mu_x * mu_y + C1)
            / (mu_x^2 + mu_y^2 + C1)

        C = (2 * sigma_x * sigma_y + C2)
            / (sigma_x^2 + sigma_y^2 + C2)

        S = (sigma_xy + C3)
            / (sigma_x * sigma_y + C3)

    with:

        C1 = (K1 * data_range)^2
        C2 = (K2 * data_range)^2
        C3 = C2 / 2

    This implementation calculates the three SSIM
    components globally over the complete seismic volume.

    Tensor convention:

        [B, C, D, H, W]
    """

    # ---------------------------------------------------
    # Validate input types
    # ---------------------------------------------------

    if not isinstance(
        prediction,
        torch.Tensor
    ):

        raise TypeError(
            "Prediction must be a torch.Tensor."
        )

    if not isinstance(
        target,
        torch.Tensor
    ):

        raise TypeError(
            "Target must be a torch.Tensor."
        )

    # ---------------------------------------------------
    # Validate shapes
    # ---------------------------------------------------

    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target must have "
            "identical shapes for SSIM calculation.\n"
            f"Prediction shape: {tuple(prediction.shape)}\n"
            f"Target shape: {tuple(target.shape)}"
        )

    # ---------------------------------------------------
    # Validate data range
    # ---------------------------------------------------

    if data_range <= 0:

        raise ValueError(
            "data_range must be greater than zero."
        )

    # ---------------------------------------------------
    # Constants
    # ---------------------------------------------------

    K1 = 0.01
    K2 = 0.03

    C1 = (
        K1
        * data_range
    ) ** 2

    C2 = (
        K2
        * data_range
    ) ** 2

    C3 = C2 / 2.0

    # ---------------------------------------------------
    # Means
    # ---------------------------------------------------

    mu_x = torch.mean(
        prediction
    )

    mu_y = torch.mean(
        target
    )

    # ---------------------------------------------------
    # Variances
    # ---------------------------------------------------

    sigma_x = torch.var(
        prediction,
        unbiased=False
    )

    sigma_y = torch.var(
        target,
        unbiased=False
    )

    # ---------------------------------------------------
    # Covariance
    # ---------------------------------------------------

    sigma_xy = torch.mean(
        (
            prediction - mu_x
        )
        *
        (
            target - mu_y
        )
    )

    # ---------------------------------------------------
    # Luminance component
    # ---------------------------------------------------

    luminance = (

        2.0
        * mu_x
        * mu_y
        + C1

    ) / (

        mu_x ** 2
        + mu_y ** 2
        + C1
    )

    # ---------------------------------------------------
    # Contrast component
    # ---------------------------------------------------

    contrast = (

        2.0
        * torch.sqrt(
            sigma_x
            + 1e-12
        )
        * torch.sqrt(
            sigma_y
            + 1e-12
        )
        + C2

    ) / (

        sigma_x
        + sigma_y
        + C2
    )

    # ---------------------------------------------------
    # Structural component
    # ---------------------------------------------------

    structure = (

        sigma_xy
        + C3

    ) / (

        torch.sqrt(
            sigma_x
            + 1e-12
        )
        *
        torch.sqrt(
            sigma_y
            + 1e-12
        )
        + C3
    )

    # ---------------------------------------------------
    # Complete SSIM
    # ---------------------------------------------------

    score = (

        luminance
        * contrast
        * structure
    )

    return score