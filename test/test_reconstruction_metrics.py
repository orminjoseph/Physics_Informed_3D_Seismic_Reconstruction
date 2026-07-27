import torch

from metrics.reconstruction_metrics import (
    mae,
    mse,
    rmse,
    psnr,
    snr,
    ssim
)


def main():

    # ---------------------------------------
    # Create simple synthetic tensors
    # ---------------------------------------

    target = torch.rand(1, 1, 64, 64, 64)

    prediction = target + 0.05 * torch.randn_like(target)

    print()
    print("=" * 60)
    print("Testing Reconstruction Metrics")
    print("=" * 60)

    print()

    print(f"MAE  : {mae(prediction, target).item():.6f}")

    print(f"MSE  : {mse(prediction, target).item():.6f}")

    print(f"RMSE : {rmse(prediction, target).item():.6f}")

    print(f"PSNR : {psnr(prediction, target).item():.3f} dB")

    print(f"SNR  : {snr(prediction, target).item():.3f} dB")

    print(f"SSIM : {ssim(prediction, target).item():.6f}")

    print()

    print("Reconstruction Metrics Test: PASSED")


if __name__ == "__main__":
    main()