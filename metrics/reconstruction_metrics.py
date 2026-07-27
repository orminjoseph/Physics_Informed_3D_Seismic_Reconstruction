"""
=========================================================
Reconstruction Evaluation Metrics
=========================================================

This module contains evaluation metrics for
3D seismic reconstruction.

Metrics implemented

1. MAE
2. MSE
3. RMSE
4. PSNR
5. SNR
6. SSIM
=========================================================
"""

import math

import torch
import torch.nn.functional as F


# -------------------------------------------------------
# Mean Absolute Error
# -------------------------------------------------------

def mae(prediction, target):

    return torch.mean(torch.abs(prediction - target))


# -------------------------------------------------------
# Mean Squared Error
# -------------------------------------------------------

def mse(prediction, target):

    return torch.mean((prediction - target) ** 2)


# -------------------------------------------------------
# Root Mean Squared Error
# -------------------------------------------------------

def rmse(prediction, target):

    return torch.sqrt(mse(prediction, target))


# -------------------------------------------------------
# Peak Signal-to-Noise Ratio
# -------------------------------------------------------

def psnr(prediction, target, data_range=2.0):

    mse_value = mse(prediction, target)

    if mse_value.item() == 0:

        return torch.tensor(float("inf"))

    return 20 * torch.log10(
        torch.tensor(data_range)
    ) - 10 * torch.log10(mse_value)


# -------------------------------------------------------
# Signal-to-Noise Ratio
# -------------------------------------------------------

def snr(prediction, target):

    signal_power = torch.mean(target ** 2)

    noise_power = torch.mean((target - prediction) ** 2)

    if noise_power.item() == 0:

        return torch.tensor(float("inf"))

    return 10 * torch.log10(signal_power / noise_power)


# -------------------------------------------------------
# Structural Similarity Index
# -------------------------------------------------------

def ssim(prediction, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = prediction.mean()

    mu_y = target.mean()

    sigma_x = prediction.var()

    sigma_y = target.var()

    sigma_xy = ((prediction - mu_x) *
                (target - mu_y)).mean()

    numerator = (
        (2 * mu_x * mu_y + C1)
        *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        *
        (sigma_x + sigma_y + C2)
    )

    return numerator / denominator