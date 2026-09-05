"""
============================================================
MC DROPOUT SPATIAL VALIDATION
============================================================

Checks whether predictive uncertainty is higher in missing
seismic regions than in observed regions.

Uncertainty components:
    1. Aleatoric variance:
       Mean(exp(log_variance_samples)) across MC samples.

    2. Epistemic variance:
       Population variance of MC reconstruction samples.

    3. Predictive variance:
       Aleatoric variance + Epistemic variance.

The test uses the authoritative MCDropout3D implementation
from models.mc_dropout.py.

Author: Ormin Joseph
============================================================
"""

import csv
import os

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)


# ============================================================
# CONFIGURATION
# ============================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"

PATCH_SIZE = (64, 64, 64)
STRIDE = (64, 64, 64)

MISSING_PROBABILITY = 0.30

NUM_MC_SAMPLES = 20

OUTPUT_REPORT_DIR = "outputs/reports"

CSV_FILE = os.path.join(
    OUTPUT_REPORT_DIR,
    "mc_dropout_spatial_validation.csv"
)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load the trained model checkpoint.

    Supports checkpoints containing:
        model_state_dict

    or a direct model state dictionary.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("MC DROPOUT SPATIAL VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Create the dataset
    # --------------------------------------------------------

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        missing_probability=MISSING_PROBABILITY
    )

    # --------------------------------------------------------
    # Obtain one corrupted seismic patch
    # --------------------------------------------------------

    sample = dataset[0]

    corrupted = sample[0]
    mask = sample[2]

    # --------------------------------------------------------
    # Ensure batch dimensions exist
    # --------------------------------------------------------

    if corrupted.dim() == 4:
        corrupted = corrupted.unsqueeze(0)

    if mask.dim() == 4:
        mask = mask.unsqueeze(0)

    corrupted = corrupted.to(device)
    mask = mask.to(device)

    # --------------------------------------------------------
    # Create and load the network
    # --------------------------------------------------------

    model = Network3D()

    model = load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT,
        device=device
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Create the authoritative MC Dropout engine
    # --------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=NUM_MC_SAMPLES
    )

    # --------------------------------------------------------
    # Create the predictive uncertainty estimator
    # --------------------------------------------------------

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
    )

    # ========================================================
    # RUN MC DROPOUT
    # ========================================================

    mc_results = mc_dropout.predict(
        corrupted
    )

    # --------------------------------------------------------
    # Extract MC outputs
    # --------------------------------------------------------

    reconstruction_samples = (
        mc_results["reconstruction_samples"]
    )

    reconstruction_epistemic_variance = (
        mc_results[
            "reconstruction_epistemic_variance"
        ]
    )

    log_variance_samples = (
        mc_results["log_variance_samples"]
    )

    # ========================================================
    # CALCULATE UNCERTAINTY COMPONENTS
    # ========================================================

    # --------------------------------------------------------
    # Aleatoric variance
    #
    # Correct MC aggregation:
    #
    #   mean(exp(log_variance_samples))
    #
    # NOT:
    #
    #   exp(mean(log_variance_samples))
    # --------------------------------------------------------

    aleatoric_variance = (
        uncertainty_estimator.aleatoric_variance(
            log_variance_samples
        )
    )

    # --------------------------------------------------------
    # Predictive variance
    #
    #   Predictive variance =
    #       Aleatoric variance
    #       +
    #       Epistemic variance
    # --------------------------------------------------------

    predictive_variance = (
        uncertainty_estimator.predictive_variance(
            aleatoric_variance,
            reconstruction_epistemic_variance
        )
    )

    # ========================================================
    # PREPARE MASKS
    # ========================================================

    # Remove only the batch and channel dimensions.
    # This preserves the spatial dimensions.

    uncertainty = predictive_variance[0, 0]
    mask = mask[0, 0]

    print()
    print("Predictive variance shape:", uncertainty.shape)
    print("Mask shape:", mask.shape)

    # --------------------------------------------------------
    # Identify missing and observed voxels
    # --------------------------------------------------------

    missing_mask = mask == 0
    observed_mask = mask == 1

    missing_voxel_count = (
        missing_mask.sum().item()
    )

    observed_voxel_count = (
        observed_mask.sum().item()
    )

    print()
    print("Missing voxels:", missing_voxel_count)
    print("Observed voxels:", observed_voxel_count)

    # --------------------------------------------------------
    # Validate that both regions contain voxels
    # --------------------------------------------------------

    if missing_voxel_count == 0:
        raise ValueError(
            "No missing voxels found in the selected patch."
        )

    if observed_voxel_count == 0:
        raise ValueError(
            "No observed voxels found in the selected patch."
        )

    # ========================================================
    # CALCULATE REGIONAL UNCERTAINTY
    # ========================================================

    missing_uncertainty = (
        uncertainty[missing_mask]
        .mean()
        .item()
    )

    observed_uncertainty = (
        uncertainty[observed_mask]
        .mean()
        .item()
    )

    # --------------------------------------------------------
    # Calculate the missing-to-observed ratio
    # --------------------------------------------------------

    ratio = (
        missing_uncertainty /
        (observed_uncertainty + 1e-8)
    )

    # ========================================================
    # CALCULATE REGIONAL EPISTEMIC VARIANCE
    # ========================================================

    epistemic_variance = (
        reconstruction_epistemic_variance[0, 0]
    )

    missing_epistemic_variance = (
        epistemic_variance[missing_mask]
        .mean()
        .item()
    )

    observed_epistemic_variance = (
        epistemic_variance[observed_mask]
        .mean()
        .item()
    )

    epistemic_ratio = (
        missing_epistemic_variance /
        (observed_epistemic_variance + 1e-8)
    )

    # ========================================================
    # CALCULATE REGIONAL ALEATORIC VARIANCE
    # ========================================================

    aleatoric_variance = (
        aleatoric_variance[0, 0]
    )

    missing_aleatoric_variance = (
        aleatoric_variance[missing_mask]
        .mean()
        .item()
    )

    observed_aleatoric_variance = (
        aleatoric_variance[observed_mask]
        .mean()
        .item()
    )

    aleatoric_ratio = (
        missing_aleatoric_variance /
        (observed_aleatoric_variance + 1e-8)
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("SPATIAL VALIDATION RESULTS")
    print("=" * 60)

    print()
    print("PREDICTIVE VARIANCE")

    print(
        f"Missing Region Variance : "
        f"{missing_uncertainty:.6f}"
    )

    print(
        f"Observed Region Variance: "
        f"{observed_uncertainty:.6f}"
    )

    print(
        f"Ratio (Missing/Observed): "
        f"{ratio:.4f}"
    )

    print()
    print("EPISTEMIC VARIANCE")

    print(
        f"Missing Region Variance : "
        f"{missing_epistemic_variance:.6f}"
    )

    print(
        f"Observed Region Variance: "
        f"{observed_epistemic_variance:.6f}"
    )

    print(
        f"Ratio (Missing/Observed): "
        f"{epistemic_ratio:.4f}"
    )

    print()
    print("ALEATORIC VARIANCE")

    print(
        f"Missing Region Variance : "
        f"{missing_aleatoric_variance:.6f}"
    )

    print(
        f"Observed Region Variance: "
        f"{observed_aleatoric_variance:.6f}"
    )

    print(
        f"Ratio (Missing/Observed): "
        f"{aleatoric_ratio:.4f}"
    )

    # ========================================================
    # SAVE CSV RESULTS
    # ========================================================

    os.makedirs(
        OUTPUT_REPORT_DIR,
        exist_ok=True
    )

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Missing_Probability",
            "Missing_Voxel_Count",
            "Observed_Voxel_Count",
            "Missing_Predictive_Variance",
            "Observed_Predictive_Variance",
            "Predictive_Variance_Ratio",
            "Missing_Epistemic_Variance",
            "Observed_Epistemic_Variance",
            "Epistemic_Variance_Ratio",
            "Missing_Aleatoric_Variance",
            "Observed_Aleatoric_Variance",
            "Aleatoric_Variance_Ratio"
        ])

        writer.writerow([
            MISSING_PROBABILITY,
            missing_voxel_count,
            observed_voxel_count,
            missing_uncertainty,
            observed_uncertainty,
            ratio,
            missing_epistemic_variance,
            observed_epistemic_variance,
            epistemic_ratio,
            missing_aleatoric_variance,
            observed_aleatoric_variance,
            aleatoric_ratio
        ])

    print()
    print("CSV saved to:")
    print(CSV_FILE)

    print()
    print("=" * 60)
    print("MC DROPOUT SPATIAL VALIDATION COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()