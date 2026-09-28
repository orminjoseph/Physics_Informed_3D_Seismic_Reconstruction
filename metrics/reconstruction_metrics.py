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
    data_range=2.0,
    window_size=11,
    sigma=1.5
):
    """
    Calculate local 3D Structural Similarity Index (SSIM).

    SSIM evaluates similarity using three components:

        1. Luminance
        2. Contrast
        3. Structure

    The local formulation is evaluated over overlapping
    3D windows rather than over the complete seismic
    volume.

    Tensor convention:

        [B, C, D, H, W]

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    data_range : float
        Dynamic range of the seismic amplitudes.

        For normalized amplitudes [-1, 1]:

            data_range = 2.0

    window_size : int
        Requested size of the local cubic SSIM window.

        Default:

            11 x 11 x 11

        For small input volumes, the window is automatically
        reduced to the largest valid odd size that fits
        inside all three spatial dimensions.

        Examples:

            Input 8 x 8 x 8   -> effective window 7
            Input 10 x 10 x 10 -> effective window 9
            Input 16 x 16 x 16 -> effective window 11
            Input 64 x 128 x 128 -> effective window 11

    sigma : float
        Standard deviation of the Gaussian window.

    Returns
    -------
    torch.Tensor
        Scalar mean SSIM score across batch, channels,
        and spatial locations.
    """

    # ===================================================
    # Validate input types
    # ===================================================

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

    # ===================================================
    # Validate shapes
    # ===================================================

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have "
            "identical shapes for SSIM calculation.\n"
            f"Prediction shape: {tuple(prediction.shape)}\n"
            f"Target shape: {tuple(target.shape)}"
        )

    # ===================================================
    # Validate dimensionality
    #
    # Expected:
    #
    # [B, C, D, H, W]
    # ===================================================

    if prediction.ndim != 5:
        raise ValueError(
            "SSIM expects 5D tensors with shape "
            "[B, C, D, H, W].\n"
            f"Received shape: {tuple(prediction.shape)}"
        )

    # ===================================================
    # Validate finite values
    # ===================================================

    if not torch.isfinite(prediction).all():
        raise ValueError(
            "Prediction contains NaN or Inf values."
        )

    if not torch.isfinite(target).all():
        raise ValueError(
            "Target contains NaN or Inf values."
        )

    # ===================================================
    # Validate data range
    # ===================================================

    if not isinstance(
        data_range,
        (int, float)
    ):
        raise TypeError(
            "data_range must be a numeric value."
        )

    if not torch.isfinite(
        torch.tensor(
            float(data_range)
        )
    ):
        raise ValueError(
            "data_range must be finite."
        )

    if data_range <= 0:
        raise ValueError(
            "data_range must be greater than zero."
        )

    # ===================================================
    # Validate requested window size
    # ===================================================

    if not isinstance(
        window_size,
        int
    ):
        raise TypeError(
            "window_size must be an integer."
        )

    if window_size < 3:
        raise ValueError(
            "window_size must be at least 3."
        )

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be an odd integer."
        )

    # ===================================================
    # Validate sigma
    # ===================================================

    if not isinstance(
        sigma,
        (int, float)
    ):
        raise TypeError(
            "sigma must be a numeric value."
        )

    if not torch.isfinite(
        torch.tensor(
            float(sigma)
        )
    ):
        raise ValueError(
            "sigma must be finite."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than zero."
        )

    # ===================================================
    # Obtain spatial dimensions
    #
    # Tensor:
    #
    # [B, C, D, H, W]
    # ===================================================

    _, _, depth, height, width = prediction.shape

    # ===================================================
    # Determine the largest spatial dimension that is
    # guaranteed to fit in all three directions.
    # ===================================================

    minimum_dimension = min(
        depth,
        height,
        width
    )

    # ===================================================
    # A minimum 3 x 3 x 3 SSIM neighbourhood is required.
    # ===================================================

    if minimum_dimension < 3:
        raise ValueError(
            "Input volume is too small for SSIM."
            "\n"
            f"Input shape: {tuple(prediction.shape)}"
            "\n"
            "Minimum spatial dimension must be at least 3."
        )

    # ===================================================
    # Adapt the requested window size when the input
    # volume is smaller than the requested window.
    #
    # This is important for unit tests using volumes such
    # as 8 x 8 x 8.
    #
    # Example:
    #
    # requested window = 11
    # minimum dimension = 8
    #
    # effective window = 7
    #
    # The window must remain odd so that it has a central
    # voxel and symmetric padding.
    # ===================================================

    effective_window_size = min(
        window_size,
        minimum_dimension
    )

    # ---------------------------------------------------
    # Ensure that the effective window is odd.
    # ---------------------------------------------------

    if effective_window_size % 2 == 0:

        effective_window_size -= 1

    # ===================================================
    # Final safety check
    # ===================================================

    if effective_window_size < 3:
        raise ValueError(
            "Unable to construct a valid SSIM window."
            "\n"
            f"Input shape: {tuple(prediction.shape)}"
            "\n"
            f"Requested window size: {window_size}"
            "\n"
            f"Effective window size: {effective_window_size}"
        )

    # ===================================================
    # SSIM constants
    # ===================================================

    K1 = 0.01
    K2 = 0.03

    C1 = (
        K1 * float(data_range)
    ) ** 2

    C2 = (
        K2 * float(data_range)
    ) ** 2

    # ===================================================
    # Construct a 3D Gaussian window
    #
    # The effective window size is used here rather than
    # the originally requested window size.
    # ===================================================

    radius = effective_window_size // 2

    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=prediction.device,
        dtype=prediction.dtype
    )

    gaussian_1d = torch.exp(
        -(
            coordinates ** 2
        )
        /
        (
            2.0 * float(sigma) ** 2
        )
    )

    # ---------------------------------------------------
    # Normalize the 1D Gaussian so that its weights sum
    # to one.
    # ---------------------------------------------------

    gaussian_1d = (
        gaussian_1d
        /
        gaussian_1d.sum()
    )

    # ===================================================
    # Convert the 1D Gaussian into a separable 3D kernel
    #
    # Result:
    #
    # [window, window, window]
    # ===================================================

    gaussian_3d = (
        gaussian_1d[:, None, None]
        *
        gaussian_1d[None, :, None]
        *
        gaussian_1d[None, None, :]
    )

    # ===================================================
    # Add dimensions required by conv3d
    #
    # Initial shape:
    #
    # [1, 1, D, H, W]
    # ===================================================

    window = (
        gaussian_3d
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # ===================================================
    # Repeat the kernel for every seismic channel.
    #
    # groups=C means that each channel is processed
    # independently.
    # ===================================================

    channels = prediction.shape[1]

    window = window.repeat(
        channels,
        1,
        1,
        1,
        1
    )

    # ===================================================
    # Padding
    #
    # Padding is based on the EFFECTIVE window size.
    # ===================================================

    padding = radius

    # ===================================================
    # Local means
    # ===================================================

    mu_x = torch.nn.functional.conv3d(
        prediction,
        window,
        padding=padding,
        groups=channels
    )

    mu_y = torch.nn.functional.conv3d(
        target,
        window,
        padding=padding,
        groups=channels
    )

    # ===================================================
    # Squared means
    # ===================================================

    mu_x_squared = mu_x ** 2
    mu_y_squared = mu_y ** 2
    mu_xy = mu_x * mu_y

    # ===================================================
    # Local variances
    # ===================================================

    sigma_x_squared = (
        torch.nn.functional.conv3d(
            prediction ** 2,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_x_squared
    )

    sigma_y_squared = (
        torch.nn.functional.conv3d(
            target ** 2,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_y_squared
    )

    # ===================================================
    # Local covariance
    # ===================================================

    sigma_xy = (
        torch.nn.functional.conv3d(
            prediction * target,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_xy
    )

    # ===================================================
    # Numerical protection
    #
    # Floating-point arithmetic can produce extremely
    # small negative variance values.
    #
    # Variance cannot physically be negative, so clamp
    # these numerical artifacts to zero.
    # ===================================================

    sigma_x_squared = torch.clamp(
        sigma_x_squared,
        min=0.0
    )

    sigma_y_squared = torch.clamp(
        sigma_y_squared,
        min=0.0
    )

    # ===================================================
    # Luminance component
    # ===================================================

    luminance = (
        2.0 * mu_xy
        + C1
    ) / (
        mu_x_squared
        + mu_y_squared
        + C1
    )

    # ===================================================
    # Standard deviations
    # ===================================================

    sigma_x = torch.sqrt(
        sigma_x_squared
        + 1e-12
    )

    sigma_y = torch.sqrt(
        sigma_y_squared
        + 1e-12
    )

    # ===================================================
    # Contrast component
    # ===================================================

    contrast = (
        2.0
        * sigma_x
        * sigma_y
        + C2
    ) / (
        sigma_x_squared
        + sigma_y_squared
        + C2
    )

    # ===================================================
    # Structure component
    # ===================================================

    structure = (
        sigma_xy
        + C2 / 2.0
    ) / (
        sigma_x
        * sigma_y
        + C2 / 2.0
    )

    # ===================================================
    # Complete local SSIM
    # ===================================================

    score_map = (
        luminance
        * contrast
        * structure
    )

    # ===================================================
    # Average local SSIM values
    #
    # The mean is taken across:
    #
    #   1. Batch
    #   2. Channels
    #   3. Depth
    #   4. Height
    #   5. Width
    #
    # The convolution does not mix batch elements.
    # ===================================================

    score = torch.mean(
        score_map
    )

    # ===================================================
    # Final numerical validation
    # ===================================================

    if not torch.isfinite(score):

        raise FloatingPointError(
            "SSIM calculation produced a non-finite value."
        )

    # ===================================================
    # Return scalar SSIM score
    # ===================================================

    return score