import torch

from metrics.reconstruction_metrics import (
    mae,
    mse,
    rmse,
    psnr,
    snr,
    ssim
)


def create_test_volumes():

    target = torch.rand(
        1,
        1,
        64,
        64,
        64
    )

    prediction = (
        target
        + 0.05 * torch.randn_like(target)
    )

    return prediction, target


def test_reconstruction_metrics():

    prediction, target = create_test_volumes()

    print()
    print("=" * 60)
    print("TEST - RECONSTRUCTION METRICS")
    print("=" * 60)

    mae_value = mae(
        prediction,
        target
    )

    mse_value = mse(
        prediction,
        target
    )

    rmse_value = rmse(
        prediction,
        target
    )

    psnr_value = psnr(
        prediction,
        target
    )

    snr_value = snr(
        prediction,
        target
    )

    ssim_value = ssim(
        prediction,
        target
    )

    print()
    print(f"MAE  : {mae_value.item():.6f}")
    print(f"MSE  : {mse_value.item():.6f}")
    print(f"RMSE : {rmse_value.item():.6f}")
    print(f"PSNR : {psnr_value.item():.3f} dB")
    print(f"SNR  : {snr_value.item():.3f} dB")
    print(f"SSIM : {ssim_value.item():.6f}")

    # ---------------------------------------------------
    # Basic validity checks
    # ---------------------------------------------------

    assert torch.isfinite(
        mae_value
    )

    assert torch.isfinite(
        mse_value
    )

    assert torch.isfinite(
        rmse_value
    )

    assert torch.isfinite(
        psnr_value
    )

    assert torch.isfinite(
        snr_value
    )

    assert torch.isfinite(
        ssim_value
    )

    # ---------------------------------------------------
    # Mathematical consistency
    # ---------------------------------------------------

    assert mae_value >= 0

    assert mse_value >= 0

    assert rmse_value >= 0

    # RMSE must equal sqrt(MSE)
    assert torch.allclose(
        rmse_value,
        torch.sqrt(mse_value),
        atol=1e-6
    )

    print()
    print("Reconstruction Metrics Test: PASSED")