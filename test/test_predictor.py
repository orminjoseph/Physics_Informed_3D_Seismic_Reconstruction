"""
Test the trained predictor inference pipeline.

This test verifies that:

1. The trained checkpoint can be loaded.
2. The predictor accepts a valid 3D seismic input.
3. Reconstruction output has the expected shape.
4. Travel-time output has the expected shape.
5. Log-variance output has the expected shape.
6. Aleatoric standard deviation has the expected shape.
7. All outputs are finite.
8. Aleatoric standard deviation is non-negative.
9. The returned aleatoric standard deviation is mathematically
   consistent with the returned log variance:

       sigma_a = exp(0.5 * log_variance)
"""

import os

import torch

from utils.config import EXPERIMENT_NAME
from inference.predictor import Predictor
from models.network import Network3D


def main():
    """
    Run the predictor inference audit.
    """

    # ---------------------------------------------------------
    # 1. Define the trained checkpoint
    # ---------------------------------------------------------
    checkpoint = os.path.join(
        "outputs",
        EXPERIMENT_NAME,
        "checkpoints",
        "best_model.pth"
    )

    # ---------------------------------------------------------
    # 2. Verify that the checkpoint exists
    # ---------------------------------------------------------
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint}"
        )

    # ---------------------------------------------------------
    # 3. Create the 3D network
    # ---------------------------------------------------------
    model = Network3D()

    # ---------------------------------------------------------
    # 4. Create the predictor
    # ---------------------------------------------------------
    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device="cpu"
    )

    # ---------------------------------------------------------
    # 5. Create a synthetic seismic input
    #
    # Shape:
    # [C, D, H, W]
    #
    # This corresponds to:
    #   C = 1 seismic channel
    #   D = 64 depth/time samples
    #   H = 128
    #   W = 128
    # ---------------------------------------------------------
    torch.manual_seed(42)

    seismic_input = torch.randn(
        1,
        64,
        128,
        128
    )

    # ---------------------------------------------------------
    # 6. Run deterministic prediction
    # ---------------------------------------------------------
    reconstruction, travel_time, log_variance, aleatoric_std = (
        predictor.predict(seismic_input)
    )

    # ---------------------------------------------------------
    # 7. Define the expected output shape
    # ---------------------------------------------------------
    expected_shape = (
        1,
        1,
        64,
        128,
        128
    )

    # ---------------------------------------------------------
    # 8. Print output shapes
    # ---------------------------------------------------------
    print()
    print("Predictor inference completed.")
    print()

    print("Input shape          :", tuple(seismic_input.shape))
    print("Reconstruction shape :", tuple(reconstruction.shape))
    print("Travel-time shape    :", tuple(travel_time.shape))
    print("Log-variance shape   :", tuple(log_variance.shape))
    print("Aleatoric std shape  :", tuple(aleatoric_std.shape))

    # ---------------------------------------------------------
    # 9. Verify output shapes
    # ---------------------------------------------------------
    assert tuple(reconstruction.shape) == expected_shape, (
        f"Unexpected reconstruction shape: "
        f"{tuple(reconstruction.shape)}"
    )

    assert tuple(travel_time.shape) == expected_shape, (
        f"Unexpected travel-time shape: "
        f"{tuple(travel_time.shape)}"
    )

    assert tuple(log_variance.shape) == expected_shape, (
        f"Unexpected log-variance shape: "
        f"{tuple(log_variance.shape)}"
    )

    assert tuple(aleatoric_std.shape) == expected_shape, (
        f"Unexpected aleatoric standard deviation shape: "
        f"{tuple(aleatoric_std.shape)}"
    )

    print()
    print("Output shape checks: PASSED")

    # ---------------------------------------------------------
    # 10. Verify that all outputs contain finite values
    # ---------------------------------------------------------
    assert torch.isfinite(reconstruction).all(), (
        "Reconstruction contains NaN or Inf values."
    )

    assert torch.isfinite(travel_time).all(), (
        "Travel-time output contains NaN or Inf values."
    )

    assert torch.isfinite(log_variance).all(), (
        "Log-variance contains NaN or Inf values."
    )

    assert torch.isfinite(aleatoric_std).all(), (
        "Aleatoric standard deviation contains NaN or Inf values."
    )

    print("Finite-value checks: PASSED")

    # ---------------------------------------------------------
    # 11. Verify non-negative aleatoric standard deviation
    # ---------------------------------------------------------
    assert torch.all(aleatoric_std >= 0), (
        "Aleatoric standard deviation contains negative values."
    )

    print("Aleatoric non-negativity check: PASSED")

    # ---------------------------------------------------------
    # 12. Independently calculate aleatoric standard deviation
    #
    # The predictor should satisfy:
    #
    # sigma_a = exp(0.5 * log_variance)
    # ---------------------------------------------------------
    expected_aleatoric_std = torch.exp(
        0.5 * log_variance
    )

    # ---------------------------------------------------------
    # 13. Compare predictor output with independent calculation
    # ---------------------------------------------------------
    max_difference = torch.max(
        torch.abs(
            aleatoric_std - expected_aleatoric_std
        )
    ).item()

    print()
    print(
        "Maximum aleatoric std difference:",
        max_difference
    )

    assert torch.allclose(
        aleatoric_std,
        expected_aleatoric_std,
        rtol=1e-5,
        atol=1e-6
    ), (
        "Aleatoric standard deviation does not match "
        "exp(0.5 * log_variance)."
    )

    print("Aleatoric uncertainty equation check: PASSED")

    # ---------------------------------------------------------
    # 14. Print useful numerical diagnostics
    # ---------------------------------------------------------
    print()
    print("Reconstruction:")
    print("  min  :", reconstruction.min().item())
    print("  max  :", reconstruction.max().item())
    print("  mean :", reconstruction.mean().item())

    print()
    print("Travel time:")
    print("  min  :", travel_time.min().item())
    print("  max  :", travel_time.max().item())
    print("  mean :", travel_time.mean().item())

    print()
    print("Log variance:")
    print("  min  :", log_variance.min().item())
    print("  max  :", log_variance.max().item())
    print("  mean :", log_variance.mean().item())

    print()
    print("Aleatoric standard deviation:")
    print("  min  :", aleatoric_std.min().item())
    print("  max  :", aleatoric_std.max().item())
    print("  mean :", aleatoric_std.mean().item())

    # ---------------------------------------------------------
    # 15. Final result
    # ---------------------------------------------------------
    print()
    print("PREDICTOR INFERENCE AUDIT: PASSED")
    print()


if __name__ == "__main__":
    main()