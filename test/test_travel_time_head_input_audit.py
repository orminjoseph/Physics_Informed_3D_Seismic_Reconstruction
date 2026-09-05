"""
=========================================================
Travel-Time Head Input Audit
=========================================================

Purpose:
    Inspect the feature tensor entering the travel-time head
    before the final travel-time projection.

This is a diagnostic test only.
It does not modify production source code.

Author: Ormin Joseph
=========================================================
"""

import os
import torch

from models.network import Network3D
from utils.config import EXPERIMENT_NAME


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    "outputs",
    EXPERIMENT_NAME,
    "checkpoints",
    "best_model.pth"
)

DEVICE = torch.device("cpu")

INPUT_SHAPE = (1, 1, 64, 128, 128)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def print_statistics(name, tensor):
    """
    Print basic statistics for a tensor.
    """

    print(f"\n{name}")
    print(f"  shape : {tuple(tensor.shape)}")
    print(f"  min   : {tensor.min().item():.6e}")
    print(f"  max   : {tensor.max().item():.6e}")
    print(f"  mean  : {tensor.mean().item():.6e}")
    print(f"  std   : {tensor.std(unbiased=False).item():.6e}")
    print(f"  abs mean : {tensor.abs().mean().item():.6e}")
    print(f"  abs max  : {tensor.abs().max().item():.6e}")


# ---------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("TRAVEL-TIME HEAD INPUT AUDIT")
    print("=" * 60)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = Network3D().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Device     : {DEVICE}")

    # -----------------------------------------------------
    # Capture the input to travel_time_head
    # -----------------------------------------------------

    captured_features = {}

    def capture_input(module, inputs):
        """
        Forward pre-hook.

        Captures the tensor entering travel_time_head.
        """

        captured_features["input"] = inputs[0].detach().clone()

    hook = model.travel_time_head.register_forward_pre_hook(
        capture_input
    )

    # -----------------------------------------------------
    # Controlled input
    # -----------------------------------------------------

    torch.manual_seed(42)

    seismic_input = torch.randn(
        INPUT_SHAPE,
        device=DEVICE
    )

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    with torch.inference_mode():

        reconstruction, travel_time, log_variance = model(
            seismic_input
        )

    hook.remove()

    # -----------------------------------------------------
    # Validate capture
    # -----------------------------------------------------

    if "input" not in captured_features:
        raise RuntimeError(
            "Travel-time head input was not captured."
        )

    features = captured_features["input"]

    # -----------------------------------------------------
    # Print output statistics
    # -----------------------------------------------------

    print_statistics(
        "Features entering travel_time_head",
        features
    )

    print_statistics(
        "Travel-time output",
        travel_time
    )

    # -----------------------------------------------------
    # Head parameter statistics
    # -----------------------------------------------------

    print_statistics(
        "Travel-time head weights",
        model.travel_time_head.weight.detach()
    )

    print_statistics(
        "Travel-time head bias",
        model.travel_time_head.bias.detach()
    )

    # -----------------------------------------------------
    # Compare variation
    # -----------------------------------------------------

    feature_range = (
        features.max() - features.min()
    ).item()

    travel_time_range = (
        travel_time.max() - travel_time.min()
    ).item()

    print("\nVariation comparison")
    print(f"  feature range      : {feature_range:.6e}")
    print(f"  travel-time range  : {travel_time_range:.6e}")

    if feature_range > 0:
        print(
            "  output/input range ratio : "
            f"{travel_time_range / feature_range:.6e}"
        )

    # -----------------------------------------------------
    # Final checks
    # -----------------------------------------------------

    if not torch.isfinite(features).all():
        raise RuntimeError(
            "Non-finite values detected in travel-time features."
        )

    if not torch.isfinite(travel_time).all():
        raise RuntimeError(
            "Non-finite values detected in travel-time output."
        )

    print("\n" + "=" * 60)
    print("TRAVEL-TIME HEAD INPUT AUDIT: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()