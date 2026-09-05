"""
====================================================================
Noise Robustness Evaluation
====================================================================

Evaluates the robustness of the trained seismic reconstruction model
under increasing levels of additive Gaussian noise.

Supported dataset modes:
    - synthetic
    - F3
    - Marmousi
    - SEG/open-data
    - any other mode supported by build_dataset()

Experiment:

    Configured Dataset
          |
          v
    Existing Corrupted Input
          |
          v
    Add Gaussian Noise to Observed Voxels
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
    Gaussian noise is added only to observed voxels. Missing voxels
    remain missing so that the experiment measures noise robustness
    without accidentally providing information at missing locations.

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

# Gaussian noise standard deviations.
#
# These values should be interpreted in the same amplitude scale
# used by the seismic input data.
NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20
]

# Maximum number of samples/patches evaluated.
NUM_TEST_PATCHES = 20

# Random seed for reproducibility.
RANDOM_SEED = 42

# Device used during evaluation.
DEVICE = "cpu"


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "noise_robustness"
)

CSV_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "noise_robustness.csv"
)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def validate_tensor(
        tensor,
        name
):
    """
    Validate a tensor before using it.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.

    name : str
        Descriptive name used in error messages.
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
    Convert an individual seismic volume to batch format.

    Expected individual sample:
        [C, D, H, W]

    Expected model input:
        [B, C, D, H, W]

    If the tensor is already batched, it is returned unchanged.
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


def evaluate_metrics(
        prediction,
        target
):
    """
    Calculate reconstruction metrics.
    """

    validate_tensor(
        prediction,
        "prediction"
    )

    validate_tensor(
        target,
        "target"
    )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have identical shapes. "
            f"Prediction={tuple(prediction.shape)}, "
            f"Target={tuple(target.shape)}"
        )

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


def add_gaussian_noise(
        cube,
        mask,
        noise_std
):
    """
    Add Gaussian noise to observed seismic voxels only.

    Missing voxels are deliberately left unchanged.

    Parameters
    ----------
    cube : torch.Tensor
        Corrupted seismic input.

    mask : torch.Tensor
        Observation mask.

        Observed:
            MASK_OBSERVED_VALUE

        Missing:
            MASK_MISSING_VALUE

    noise_std : float
        Standard deviation of Gaussian noise.

    Returns
    -------
    torch.Tensor
        Noisy seismic input.
    """

    validate_tensor(
        cube,
        "cube"
    )

    validate_tensor(
        mask,
        "mask"
    )

    if cube.shape != mask.shape:
        raise ValueError(
            "Cube and mask must have identical shapes. "
            f"Cube={tuple(cube.shape)}, "
            f"Mask={tuple(mask.shape)}"
        )

    if noise_std < 0.0:
        raise ValueError(
            "noise_std cannot be negative."
        )

    # No noise requested.
    if noise_std == 0.0:
        return cube.clone()

    # Generate Gaussian noise.
    noise = torch.randn_like(
        cube
    ) * noise_std

    # Add noise only where data are observed.
    observed_mask = (
        mask == MASK_OBSERVED_VALUE
    )

    noisy_cube = torch.where(
        observed_mask,
        cube + noise,
        cube
    )

    return noisy_cube


def calculate_noise_statistics(
        original,
        noisy,
        mask
):
    """
    Calculate the actual noise statistics on observed voxels.

    This is useful for verifying that the intended noise level was
    actually applied.
    """

    observed = (
        mask == MASK_OBSERVED_VALUE
    )

    if not observed.any():
        return {
            "Observed_Voxels": 0,
            "Actual_Noise_STD": 0.0
        }

    difference = (
        noisy - original
    )

    observed_noise = difference[
        observed
    ]

    return {
        "Observed_Voxels":
            int(
                observed.sum().item()
            ),

        "Actual_Noise_STD":
            observed_noise.std(
                unbiased=False
            ).item()
    }


def initialize_predictor():
    """
    Initialize Network3D and load the configured best checkpoint.
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


def extract_dataset_sample(
        sample,
        sample_index
):
    """
    Extract the components of a dataset sample.

    Current project convention:

        sample[0] -> input
        sample[1] -> target
        sample[2] -> mask
        sample[3] -> velocity

    Velocity is retained for compatibility with physics-informed
    datasets but is not required by this robustness experiment.
    """

    if not isinstance(
            sample,
            (tuple, list)
    ):
        raise TypeError(
            f"Dataset sample {sample_index} must be a "
            f"tuple or list."
        )

    if len(sample) < 3:
        raise ValueError(
            f"Dataset sample {sample_index} does not contain "
            "the expected input, target and mask."
        )

    corrupted = sample[0]
    target = sample[1]
    mask = sample[2]

    corrupted = prepare_batch(
        corrupted,
        f"input[{sample_index}]"
    )

    target = prepare_batch(
        target,
        f"target[{sample_index}]"
    )

    mask = prepare_batch(
        mask,
        f"mask[{sample_index}]"
    )

    if corrupted.shape != target.shape:
        raise ValueError(
            f"Input and target shapes differ for sample "
            f"{sample_index}: "
            f"{tuple(corrupted.shape)} vs "
            f"{tuple(target.shape)}"
        )

    if corrupted.shape != mask.shape:
        raise ValueError(
            f"Input and mask shapes differ for sample "
            f"{sample_index}: "
            f"{tuple(corrupted.shape)} vs "
            f"{tuple(mask.shape)}"
        )

    return (
        corrupted,
        target,
        mask
    )


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("NOISE ROBUSTNESS")
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

    print(
        f"Test Samples   : {NUM_TEST_PATCHES}"
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

    num_test_samples = min(
        NUM_TEST_PATCHES,
        len(dataset)
    )

    print(
        f"Dataset size   : {len(dataset)}"
    )

    print(
        f"Samples tested : {num_test_samples}"
    )

    # --------------------------------------------------------------
    # Initialize trained model
    # --------------------------------------------------------------

    predictor = initialize_predictor()

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    results = []

    # --------------------------------------------------------------
    # Test each noise level
    # --------------------------------------------------------------

    for noise_std in NOISE_LEVELS:

        print()
        print("-" * 70)

        print(
            f"Testing Gaussian noise level "
            f"σ = {noise_std:.2f}"
        )

        print("-" * 70)

        mae_values = []
        rmse_values = []
        psnr_values = []
        snr_values = []
        ssim_values = []

        aleatoric_std_values = []
        epistemic_std_values = []
        predictive_std_values = []

        actual_noise_std_values = []

        # ----------------------------------------------------------
        # Evaluate selected dataset samples
        # ----------------------------------------------------------

        for sample_index in range(
                num_test_samples
        ):

            sample = dataset[
                sample_index
            ]

            (
                corrupted,
                target,
                mask
            ) = extract_dataset_sample(
                sample,
                sample_index
            )

            # ------------------------------------------------------
            # Add Gaussian noise only to observed voxels
            # ------------------------------------------------------

            noisy_input = add_gaussian_noise(
                corrupted,
                mask,
                noise_std
            )

            # ------------------------------------------------------
            # Verify actual noise level
            # ------------------------------------------------------

            noise_statistics = (
                calculate_noise_statistics(
                    corrupted,
                    noisy_input,
                    mask
                )
            )

            if noise_std > 0.0:
                actual_noise_std_values.append(
                    noise_statistics[
                        "Actual_Noise_STD"
                    ]
                )

            # ------------------------------------------------------
            # Model inference
            # ------------------------------------------------------

            with torch.no_grad():

                (
                    reconstruction,
                    travel_time,
                    aleatoric_std,
                    epistemic_std
                ) = predictor.predict(
                    noisy_input
                )

            # ------------------------------------------------------
            # Validate outputs
            # ------------------------------------------------------

            validate_tensor(
                reconstruction,
                "reconstruction"
            )

            validate_tensor(
                aleatoric_std,
                "aleatoric_std"
            )

            validate_tensor(
                epistemic_std,
                "epistemic_std"
            )

            # ------------------------------------------------------
            # Predictive uncertainty
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
                f"{num_test_samples:02d} | "
                f"MAE={metrics['MAE']:.4f} | "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"SSIM={metrics['SSIM']:.4f}"
            )

        # ----------------------------------------------------------
        # Actual noise level
        # ----------------------------------------------------------

        if actual_noise_std_values:

            actual_noise_std = float(
                np.mean(
                    actual_noise_std_values
                )
            )

        else:

            actual_noise_std = 0.0

        # ----------------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------------

        result = {

            "Experiment":
                EXPERIMENT_NAME,

            "Dataset_Mode":
                DATASET_MODE,

            "Noise_Level":
                noise_std,

            "Actual_Noise_STD":
                actual_noise_std,

            "Num_Samples":
                num_test_samples,

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

        # ----------------------------------------------------------
        # Print summary
        # ----------------------------------------------------------

        print()
        print(
            f"Noise σ={noise_std:.2f} | "
            f"MAE={result['MAE']:.4f} | "
            f"RMSE={result['RMSE']:.4f} | "
            f"PSNR={result['PSNR']:.4f} | "
            f"SNR={result['SNR']:.4f} | "
            f"SSIM={result['SSIM']:.4f}"
        )

        print(
            f"Aleatoric STD="
            f"{result['Aleatoric_STD_Mean']:.4f} | "
            f"Epistemic STD="
            f"{result['Epistemic_STD_Mean']:.4f} | "
            f"Predictive STD="
            f"{result['Predictive_STD_Mean']:.4f}"
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
    # Display final table
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("NOISE ROBUSTNESS RESULTS")
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
    print("NOISE ROBUSTNESS COMPLETED")
    print("=" * 70)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()