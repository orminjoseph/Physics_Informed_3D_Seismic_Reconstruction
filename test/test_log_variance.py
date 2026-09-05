"""
============================================================
LOG VARIANCE DIAGNOSTIC TEST
============================================================

Inspect uncertainty head outputs on F3 data.

This diagnostic evaluates:
    1. Log-variance output
    2. Variance derived from log-variance
    3. Standard deviation derived from log-variance

Current Network3D outputs:
    reconstruction
    travel_time
    log_variance

Author: Ormin Joseph
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D


# ============================================================
# F3 DATASET PATH
# ============================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


# ============================================================
# TRAINED MODEL CHECKPOINT
# ============================================================

CHECKPOINT = (
    r"outputs"
    r"\synthetic_training"
    r"\checkpoints"
    r"\best_model.pth"
)


# ============================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("LOG VARIANCE DIAGNOSTIC")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD F3 DATASET
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING F3 DATA")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    # --------------------------------------------------------
    # GET FIRST SAMPLE
    #
    # F3Dataset currently returns at least four values:
    #
    #   1. corrupted input
    #   2. target
    #   3. mask
    #   4. velocity
    #
    # [:4] safely selects the four values required here.
    # --------------------------------------------------------

    corrupted, target, mask, velocity = dataset[0][:4]

    # --------------------------------------------------------
    # SELECT COMPUTATION DEVICE
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = Network3D().to(device)

    # --------------------------------------------------------
    # LOAD TRAINED CHECKPOINT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING MODEL CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # SET MODEL TO EVALUATION MODE
    # --------------------------------------------------------

    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # PREPARE INPUT
    #
    # Dataset sample:
    #     [C, D, H, W]
    #
    # Network input:
    #     [B, C, D, H, W]
    #
    # Therefore, add a batch dimension.
    # --------------------------------------------------------

    corrupted = (
        corrupted
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # MODEL INFERENCE
    #
    # Current Network3D returns THREE outputs:
    #
    #   reconstruction
    #   travel_time
    #   log_variance
    # --------------------------------------------------------

    with torch.no_grad():

        reconstruction, travel_time, log_variance = model(
            corrupted
        )

        # ----------------------------------------------------
        # LOG-VARIANCE STATISTICS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("LOG VARIANCE STATISTICS")
        print("=" * 60)

        print(
            "Mean :",
            log_variance.mean().item()
        )

        print(
            "Min  :",
            log_variance.min().item()
        )

        print(
            "Max  :",
            log_variance.max().item()
        )

        # ----------------------------------------------------
        # CONVERT LOG VARIANCE TO VARIANCE
        #
        # If:
        #
        #     log_variance = log(sigma^2)
        #
        # then:
        #
        #     variance = exp(log_variance)
        # ----------------------------------------------------

        variance = torch.exp(
            log_variance
        )

        # ----------------------------------------------------
        # VARIANCE STATISTICS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("VARIANCE STATISTICS")
        print("=" * 60)

        print(
            "Mean :",
            variance.mean().item()
        )

        print(
            "Min  :",
            variance.min().item()
        )

        print(
            "Max  :",
            variance.max().item()
        )

        # ----------------------------------------------------
        # CONVERT LOG VARIANCE TO STANDARD DEVIATION
        #
        # sigma = exp(0.5 * log_variance)
        # ----------------------------------------------------

        standard_deviation = torch.exp(
            0.5 * log_variance
        )

        # ----------------------------------------------------
        # STANDARD DEVIATION STATISTICS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("STANDARD DEVIATION STATISTICS")
        print("=" * 60)

        print(
            "Mean :",
            standard_deviation.mean().item()
        )

        print(
            "Min  :",
            standard_deviation.min().item()
        )

        print(
            "Max  :",
            standard_deviation.max().item()
        )

        # ----------------------------------------------------
        # ADDITIONAL MODEL OUTPUT SHAPE CHECK
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("OUTPUT SHAPES")
        print("=" * 60)

        print(
            "Reconstruction shape :",
            tuple(reconstruction.shape)
        )

        print(
            "Travel-time shape    :",
            tuple(travel_time.shape)
        )

        print(
            "Log-variance shape   :",
            tuple(log_variance.shape)
        )

        # ----------------------------------------------------
        # BASIC FINITE-VALUE CHECK
        #
        # This verifies that the uncertainty head has not
        # produced NaN or infinite values.
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("FINITE-VALUE CHECK")
        print("=" * 60)

        print(
            "Log variance finite      :",
            torch.isfinite(log_variance).all().item()
        )

        print(
            "Variance finite          :",
            torch.isfinite(variance).all().item()
        )

        print(
            "Standard deviation finite:",
            torch.isfinite(
                standard_deviation
            ).all().item()
        )

    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    print()
    print("=" * 60)
    print("LOG VARIANCE DIAGNOSTIC COMPLETE")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()