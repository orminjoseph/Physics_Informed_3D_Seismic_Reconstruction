"""
====================================================================
Missing Data Robustness Evaluation
====================================================================

Evaluates reconstruction performance as the percentage of missing
seismic data increases.

Supported dataset modes:
    - synthetic
    - F3
    - Marmousi
    - SEG/open-data
    - other modes supported by build_dataset()

The script deliberately uses build_dataset() rather than importing
a specific dataset class. This allows the evaluation to follow the
dataset selected in utils/config.py.

Evaluation:
    Missing Probability
        |
        v
    Apply random missing mask
        |
        v
    Physics-Informed 3D Network
        |
        v
    Reconstruction
        |
        v
    MAE / RMSE / PSNR / SNR / SSIM

Important:
    This experiment evaluates robustness to additional missing data.
    The original target remains unchanged.

Author: Ormin Joseph
====================================================================
"""

import os

import numpy as np
import pandas as pd
import torch

from dataset.build_dataset import build_dataset

from inference.predictor import Predictor

from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MASK_OBSERVED_VALUE,
    MASK_MISSING_VALUE
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Missing-data levels to evaluate.
MISSING_LEVELS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]

# Maximum number of samples/patches evaluated at each level.
NUM_TEST_PATCHES = 20

# Random seed makes the generated masks reproducible.
RANDOM_SEED = 42

# Device.
DEVICE = "cpu"

# Model checkpoint.
CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

# Output directory.
OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "missing_data_robustness"
)

# Output CSV.
CSV_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "missing_data_robustness.csv"
)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def evaluate_metrics(
        prediction,
        target
):
    """
    Calculate reconstruction metrics.

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed seismic volume.

    target : torch.Tensor
        Ground-truth seismic volume.

    Returns
    -------
    dict
        Dictionary containing reconstruction metrics.
    """

    return {
        "MAE": mae(
            prediction,
            target
        ).item(),

        "RMSE": rmse(
            prediction,
            target
        ).item(),

        "PSNR": psnr(
            prediction,
            target
        ).item(),

        "SNR": snr(
            prediction,
            target
        ).item(),

        "SSIM": ssim(
            prediction,
            target
        ).item()
    }


def validate_tensor(
        tensor,
        name
):
    """
    Validate that a tensor exists and contains finite values.
    """

    if not isinstance(
            tensor,
            torch.Tensor
    ):
        raise TypeError(
            f"{name} must be a torch.Tensor, "
            f"got {type(tensor)}"
        )

    if tensor.numel() == 0:
        raise ValueError(
            f"{name} is empty."
        )

    if not torch.isfinite(
            tensor
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


def prepare_batch(
        tensor,
        name
):
    """
    Convert a single seismic sample to batch format.

    Expected individual sample:
        [C, D, H, W]

    Returned batch:
        [1, C, D, H, W]
    """

    validate_tensor(
        tensor,
        name
    )

    if tensor.ndim == 4:
        return tensor.unsqueeze(0)

    if tensor.ndim == 5:
        return tensor

    raise ValueError(
        f"{name} must have 4 or 5 dimensions. "
        f"Received shape: {tuple(tensor.shape)}"
    )


def create_missing_mask(
        target,
        missing_probability,
        generator
):
    """
    Create a random missing-data mask.

    Observed voxels:
        MASK_OBSERVED_VALUE

    Missing voxels:
        MASK_MISSING_VALUE

    Parameters
    ----------
    target : torch.Tensor
        Target seismic tensor with shape [1,C,D,H,W].

    missing_probability : float
        Fraction of voxels to remove.

    generator : torch.Generator
        Reproducible random-number generator.

    Returns
    -------
    torch.Tensor
        Binary mask with the same shape as target.
    """

    if not 0.0 < missing_probability < 1.0:
        raise ValueError(
            "missing_probability must be between 0 and 1."
        )

    random_values = torch.rand(
        target.shape,
        generator=generator,
        device=target.device
    )

    mask = torch.where(
        random_values < missing_probability,
        torch.tensor(
            MASK_MISSING_VALUE,
            device=target.device,
            dtype=target.dtype
        ),
        torch.tensor(
            MASK_OBSERVED_VALUE,
            device=target.device,
            dtype=target.dtype
        )
    )

    return mask


def apply_missing_mask(
        target,
        mask
):
    """
    Apply the missing-data mask to the target seismic volume.

    Missing voxels are set to zero.

    The target itself is not modified.

    Parameters
    ----------
    target : torch.Tensor
        Ground-truth seismic volume.

    mask : torch.Tensor
        Binary observation mask.

    Returns
    -------
    torch.Tensor
        Corrupted seismic volume.
    """

    validate_tensor(
        target,
        "target"
    )

    validate_tensor(
        mask,
        "mask"
    )

    if target.shape != mask.shape:
        raise ValueError(
            "Target and mask must have identical shapes. "
            f"Target={tuple(target.shape)}, "
            f"Mask={tuple(mask.shape)}"
        )

    corrupted = (
        target * mask
    )

    return corrupted


def calculate_actual_missing_fraction(
        mask
):
    """
    Calculate the actual fraction of missing voxels.
    """

    missing = (
        mask == MASK_MISSING_VALUE
    ).float().mean()

    return missing.item()


def initialize_predictor():
    """
    Build the configured network and load the best checkpoint.
    """

    if not os.path.exists(
            CHECKPOINT
    ):
        raise FileNotFoundError(
            "Best model checkpoint was not found:\n"
            f"{CHECKPOINT}"
        )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY
    )

    predictor = Predictor(
        model=model,
        checkpoint=CHECKPOINT,
        device=DEVICE
    )

    return predictor


def get_sample_target(
        sample,
        sample_index
):
    """
    Extract the target seismic volume from a dataset sample.

    Current project convention:

        sample[0] -> input
        sample[1] -> target
        sample[2] -> mask
        sample[3] -> velocity

    For this robustness experiment we deliberately use the target
    as the clean reference and create a new missing-data mask.

    This makes the missing-data level controlled by this experiment.
    """

    if not isinstance(
            sample,
            (tuple, list)
    ):
        raise TypeError(
            f"Dataset sample {sample_index} must be a "
            f"tuple or list."
        )

    if len(sample) < 2:
        raise ValueError(
            f"Dataset sample {sample_index} does not contain "
            "the expected target tensor."
        )

    target = sample[1]

    target = prepare_batch(
        target,
        f"target[{sample_index}]"
    )

    return target


# ------------------------------------------------------------------
# Main robustness experiment
# ------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("MISSING DATA ROBUSTNESS")
    print("=" * 70)

    print(
        f"Experiment     : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode   : {DATASET_MODE}"
    )

    print(
        f"Checkpoint     : {CHECKPOINT}"
    )

    print(
        f"Device         : {DEVICE}"
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    mask_generator = torch.Generator(
        device=DEVICE
    )

    mask_generator.manual_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------------
    # Load configured dataset
    # --------------------------------------------------------------

    print()
    print("Loading configured dataset...")

    dataset = build_dataset()

    if len(dataset) == 0:
        raise ValueError(
            "The configured dataset is empty."
        )

    num_samples = min(
        NUM_TEST_PATCHES,
        len(dataset)
    )

    print(
        f"Dataset size   : {len(dataset)}"
    )

    print(
        f"Test samples   : {num_samples}"
    )

    # --------------------------------------------------------------
    # Initialize predictor
    # --------------------------------------------------------------

    predictor = initialize_predictor()

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    results = []

    # --------------------------------------------------------------
    # Test each missing-data level
    # --------------------------------------------------------------

    for missing_probability in MISSING_LEVELS:

        print()
        print("-" * 70)

        print(
            f"Testing approximately "
            f"{missing_probability * 100:.0f}% missing data"
        )

        print("-" * 70)

        mae_values = []
        rmse_values = []
        psnr_values = []
        snr_values = []
        ssim_values = []

        actual_missing_values = []

        aleatoric_std_values = []
        epistemic_std_values = []
        predictive_std_values = []

        # ----------------------------------------------------------
        # Evaluate selected samples
        # ----------------------------------------------------------

        for sample_index in range(
                num_samples
        ):

            sample = dataset[
                sample_index
            ]

            # ------------------------------------------------------
            # Get clean target
            # ------------------------------------------------------

            target = get_sample_target(
                sample,
                sample_index
            )

            # ------------------------------------------------------
            # Generate controlled missing-data mask
            # ------------------------------------------------------

            mask = create_missing_mask(
                target,
                missing_probability,
                mask_generator
            )

            # ------------------------------------------------------
            # Create corrupted input
            # ------------------------------------------------------

            corrupted = apply_missing_mask(
                target,
                mask
            )

            # ------------------------------------------------------
            # Calculate actual missing fraction
            # ------------------------------------------------------

            actual_missing = (
                calculate_actual_missing_fraction(
                    mask
                )
            )

            actual_missing_values.append(
                actual_missing
            )

            # ------------------------------------------------------
            # Run reconstruction
            # ------------------------------------------------------

            with torch.no_grad():

                (
                    reconstruction,
                    travel_time,
                    aleatoric_std,
                    epistemic_std
                ) = predictor.predict(
                    corrupted
                )

            # ------------------------------------------------------
            # Validate reconstruction
            # ------------------------------------------------------

            validate_tensor(
                reconstruction,
                "reconstruction"
            )

            # ------------------------------------------------------
            # Calculate predictive uncertainty
            # ------------------------------------------------------

            predictive_std = torch.sqrt(
                torch.clamp(
                    aleatoric_std ** 2
                    +
                    epistemic_std ** 2,
                    min=0.0
                )
            )

            # ------------------------------------------------------
            # Reconstruction metrics
            # ------------------------------------------------------

            metrics = evaluate_metrics(
                reconstruction,
                target
            )

            mae_values.append(
                metrics["MAE"]
            )

            rmse_values.append(
                metrics["RMSE"]
            )

            psnr_values.append(
                metrics["PSNR"]
            )

            snr_values.append(
                metrics["SNR"]
            )

            ssim_values.append(
                metrics["SSIM"]
            )

            # ------------------------------------------------------
            # Uncertainty statistics
            # ------------------------------------------------------

            aleatoric_std_values.append(
                aleatoric_std.mean().item()
            )

            epistemic_std_values.append(
                epistemic_std.mean().item()
            )

            predictive_std_values.append(
                predictive_std.mean().item()
            )

            print(
                f"Sample {sample_index + 1:02d}/"
                f"{num_samples:02d} | "
                f"MAE={metrics['MAE']:.4f} | "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"SSIM={metrics['SSIM']:.4f}"
            )

        # ----------------------------------------------------------
        # Aggregate results for this missing-data level
        # ----------------------------------------------------------

        result = {

            "Experiment":
                EXPERIMENT_NAME,

            "Dataset_Mode":
                DATASET_MODE,

            "Missing_Probability":
                missing_probability,

            "Actual_Missing_Fraction":
                float(
                    np.mean(
                        actual_missing_values
                    )
                ),

            "Num_Samples":
                num_samples,

            "MAE":
                float(
                    np.mean(
                        mae_values
                    )
                ),

            "RMSE":
                float(
                    np.mean(
                        rmse_values
                    )
                ),

            "PSNR":
                float(
                    np.mean(
                        psnr_values
                    )
                ),

            "SNR":
                float(
                    np.mean(
                        snr_values
                    )
                ),

            "SSIM":
                float(
                    np.mean(
                        ssim_values
                    )
                ),

            "Aleatoric_STD_Mean":
                float(
                    np.mean(
                        aleatoric_std_values
                    )
                ),

            "Epistemic_STD_Mean":
                float(
                    np.mean(
                        epistemic_std_values
                    )
                ),

            "Predictive_STD_Mean":
                float(
                    np.mean(
                        predictive_std_values
                    )
                )
        }

        results.append(
            result
        )

        print()
        print(
            f"Missing = "
            f"{missing_probability * 100:.0f}% | "
            f"MAE = "
            f"{result['MAE']:.4f} | "
            f"RMSE = "
            f"{result['RMSE']:.4f} | "
            f"PSNR = "
            f"{result['PSNR']:.4f} | "
            f"SNR = "
            f"{result['SNR']:.4f} | "
            f"SSIM = "
            f"{result['SSIM']:.4f}"
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        CSV_FILE,
        index=False
    )

    # --------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("MISSING DATA ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print("Results saved:")
    print(
        CSV_FILE
    )

    print()
    print("=" * 70)
    print("MISSING DATA ROBUSTNESS COMPLETED")
    print("=" * 70)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()