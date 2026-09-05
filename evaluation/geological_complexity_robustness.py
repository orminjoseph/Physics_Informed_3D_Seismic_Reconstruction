"""
=========================================================
Geological Complexity Robustness Evaluation
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate reconstruction robustness under different
geological complexity levels.

The script is DATA-MODE AWARE.

For DATASET_MODE = "synthetic":
    The GeologicalGenerator is used to generate controlled
    geological structures of increasing complexity.

For other dataset modes:
    The configured dataset is obtained through build_dataset()
    and actual dataset patches are evaluated.

This prevents the evaluation script from being tied to
the F3 dataset or to the synthetic dataset.

Current Predictor API
---------------------
Predictor.predict() returns:

    reconstruction
    travel_time
    aleatoric_std
    epistemic_std

Predictive standard deviation is calculated as:

    predictive_variance =
        aleatoric_variance +
        epistemic_variance

    predictive_std =
        sqrt(predictive_variance)

Outputs
-------
REPORT_DIR/
    geological_complexity/
        geological_complexity_robustness.csv

Author: Ormin Joseph
=========================================================
"""

import os

import pandas as pd
import torch

from dataset.build_dataset import build_dataset
from dataset.geological_generator import GeologicalGenerator

from inference.predictor import Predictor
from models.network import Network3D

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
)


# =========================================================
# Configuration
# =========================================================

DEVICE = "cpu"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

OUTPUT_DIRECTORY = os.path.join(
    REPORT_DIR,
    "geological_complexity"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "geological_complexity_robustness.csv"
)

# Missing-data probability used when generating synthetic
# geological-complexity test cases.
MISSING_PROBABILITY = 0.30

# Number of real dataset patches to evaluate when the
# configured mode is not synthetic.
NUM_DATASET_SAMPLES = 20


# =========================================================
# Tensor Utilities
# =========================================================

def ensure_batched_tensor(tensor):
    """
    Convert a seismic tensor into:

        [B, C, D, H, W]

    Accepted input forms:

        [D, H, W]
        [C, D, H, W]
        [B, C, D, H, W]
    """

    if not isinstance(tensor, torch.Tensor):

        raise TypeError(
            "Expected torch.Tensor, received "
            f"{type(tensor)}"
        )

    if tensor.ndim == 3:

        # [D, H, W]
        return tensor.unsqueeze(0).unsqueeze(0)

    if tensor.ndim == 4:

        # [C, D, H, W]
        return tensor.unsqueeze(0)

    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        "Expected tensor with 3, 4, or 5 dimensions. "
        f"Received shape {tuple(tensor.shape)}"
    )


def prepare_mask(mask, reference):
    """
    Convert a mask into the same shape as the model output.

    Expected final shape:

        [B, C, D, H, W]
    """

    mask = ensure_batched_tensor(mask)

    reference = ensure_batched_tensor(reference)

    # If mask has one channel and reference has multiple
    # channels, expand the mask across channels.
    if (
        mask.shape[1] == 1
        and reference.shape[1] > 1
    ):

        mask = mask.expand(
            -1,
            reference.shape[1],
            -1,
            -1,
            -1
        )

    if mask.shape != reference.shape:

        raise ValueError(
            "Mask and reference tensor shapes do not match: "
            f"{tuple(mask.shape)} vs "
            f"{tuple(reference.shape)}"
        )

    return mask


def validate_finite(name, tensor):
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )


# =========================================================
# Metric Calculation
# =========================================================

def evaluate_metrics(
    prediction,
    target,
    mask
):
    """
    Calculate reconstruction error in the missing region.

    Mask convention:

        1 = observed
        0 = missing

    Therefore only voxels where mask == 0 are evaluated.
    """

    prediction = ensure_batched_tensor(
        prediction
    )

    target = ensure_batched_tensor(
        target
    )

    mask = prepare_mask(
        mask,
        prediction
    )

    # Validate dimensions.
    if prediction.shape != target.shape:

        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )

    # Validate numerical values.
    validate_finite(
        "Prediction",
        prediction
    )

    validate_finite(
        "Target",
        target
    )

    # Missing region.
    missing_region = (
        mask <= 0
    )

    number_missing = (
        missing_region.sum().item()
    )

    if number_missing == 0:

        raise ValueError(
            "The evaluation mask contains no missing voxels."
        )

    # Extract missing-region values.
    prediction_missing = prediction[
        missing_region
    ]

    target_missing = target[
        missing_region
    ]

    # Mean absolute error.
    mae_value = torch.mean(
        torch.abs(
            prediction_missing -
            target_missing
        )
    ).item()

    return {
        "MAE": mae_value,
        "Missing_Voxels": int(
            number_missing
        )
    }


# =========================================================
# Predictive Uncertainty
# =========================================================

def compute_predictive_std(
    aleatoric_std,
    epistemic_std
):
    """
    Combine aleatoric and epistemic standard deviations.

    Predictive variance is:

        Var_predictive =
            Var_aleatoric +
            Var_epistemic

    Since the Predictor returns standard deviations:

        Var_aleatoric = aleatoric_std^2
        Var_epistemic = epistemic_std^2

    Therefore:

        predictive_std =
            sqrt(
                aleatoric_std^2 +
                epistemic_std^2
            )
    """

    aleatoric_std = ensure_batched_tensor(
        aleatoric_std
    )

    epistemic_std = ensure_batched_tensor(
        epistemic_std
    )

    if aleatoric_std.shape != epistemic_std.shape:

        raise ValueError(
            "Aleatoric and epistemic uncertainty shapes "
            "do not match: "
            f"{tuple(aleatoric_std.shape)} vs "
            f"{tuple(epistemic_std.shape)}"
        )

    predictive_variance = (
        aleatoric_std ** 2
        +
        epistemic_std ** 2
    )

    predictive_std = torch.sqrt(
        torch.clamp(
            predictive_variance,
            min=0.0
        )
    )

    validate_finite(
        "Predictive uncertainty",
        predictive_std
    )

    return predictive_std


# =========================================================
# Model Construction
# =========================================================

def create_predictor():
    """
    Construct the configured Network3D model and Predictor.
    """

    if not os.path.isfile(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "Best-model checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
    )

    predictor = Predictor(
        model=model,
        checkpoint=CHECKPOINT_PATH,
        device=DEVICE
    )

    return predictor


# =========================================================
# Synthetic Geological Complexity
# =========================================================

def get_geological_structures():
    """
    Return the controlled geological structures used for
    synthetic robustness testing.

    These structures are only appropriate when evaluating
    synthetic geological complexity.
    """

    generator = GeologicalGenerator()

    structures = [
        (
            "horizontal",
            generator.generate_horizontal_layers
        ),
        (
            "dipping",
            generator.generate_dipping_layers
        ),
        (
            "faulted",
            generator.generate_faulted_layers
        ),
        (
            "folded",
            generator.generate_folded_layers
        ),
        (
            "complex",
            generator.generate_complex_structure
        ),
        (
            "highly_complex",
            generator.generate_highly_complex_structure
        ),
    ]

    return structures


def evaluate_synthetic_complexity(
    predictor
):
    """
    Evaluate the model on controlled synthetic geological
    complexity levels.
    """

    structures = (
        get_geological_structures()
    )

    results = []

    for complexity, generator_function in structures:

        print()
        print(
            f"Testing synthetic structure: "
            f"{complexity}"
        )

        # Generate target geological model.
        target = generator_function()

        target = ensure_batched_tensor(
            target
        )

        # Generate missing-data mask.
        mask = (
            torch.rand_like(target)
            > MISSING_PROBABILITY
        ).float()

        # Generate corrupted seismic input.
        corrupted = (
            target * mask
        )

        # ---------------------------------------------
        # Model prediction
        # ---------------------------------------------

        (
            reconstruction,
            travel_time,
            aleatoric_std,
            epistemic_std,
        ) = predictor.predict(
            corrupted
        )

        # ---------------------------------------------
        # Calculate predictive uncertainty
        # ---------------------------------------------

        predictive_std = (
            compute_predictive_std(
                aleatoric_std,
                epistemic_std
            )
        )

        # ---------------------------------------------
        # Calculate reconstruction metrics
        # ---------------------------------------------

        metrics = evaluate_metrics(
            reconstruction,
            target,
            mask
        )

        mean_uncertainty = (
            predictive_std.mean().item()
        )

        mean_aleatoric = (
            ensure_batched_tensor(
                aleatoric_std
            )
            .mean()
            .item()
        )

        mean_epistemic = (
            ensure_batched_tensor(
                epistemic_std
            )
            .mean()
            .item()
        )

        # ---------------------------------------------
        # Store result
        # ---------------------------------------------

        results.append(
            {
                "Experiment":
                    EXPERIMENT_NAME,

                "Dataset_Mode":
                    DATASET_MODE,

                "Evaluation_Type":
                    "Synthetic_Geological_Complexity",

                "Complexity":
                    complexity,

                "Sample_Index":
                    -1,

                "MAE":
                    metrics["MAE"],

                "Missing_Voxels":
                    metrics["Missing_Voxels"],

                "Aleatoric_STD_Mean":
                    mean_aleatoric,

                "Epistemic_STD_Mean":
                    mean_epistemic,

                "Predictive_STD_Mean":
                    mean_uncertainty,
            }
        )

        print(
            f"MAE={metrics['MAE']:.6f}, "
            f"Aleatoric={mean_aleatoric:.6f}, "
            f"Epistemic={mean_epistemic:.6f}, "
            f"Predictive={mean_uncertainty:.6f}"
        )

    return results


# =========================================================
# Non-Synthetic Dataset Evaluation
# =========================================================

def evaluate_configured_dataset(
    predictor
):
    """
    Evaluate actual patches from the configured dataset.

    This path is used for non-synthetic DATASET_MODE values.

    Examples:

        F3
        Marmousi
        SEG
        other supported dataset modes
    """

    dataset = build_dataset()

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "The configured dataset is empty."
        )

    number_to_evaluate = min(
        NUM_DATASET_SAMPLES,
        len(dataset)
    )

    results = []

    print()
    print(
        f"Dataset mode: {DATASET_MODE}"
    )

    print(
        f"Dataset length: {len(dataset)}"
    )

    print(
        f"Evaluating {number_to_evaluate} samples."
    )

    for sample_index in range(
        number_to_evaluate
    ):

        print()
        print(
            f"Evaluating sample "
            f"{sample_index + 1}/"
            f"{number_to_evaluate}"
        )

        sample = dataset[
            sample_index
        ]

        if not isinstance(
            sample,
            (tuple, list)
        ):

            raise TypeError(
                "Dataset samples must be tuples or lists."
            )

        if len(sample) < 2:

            raise ValueError(
                "Dataset sample must contain at least "
                "input and target."
            )

        # Current project convention:
        #
        # sample[0] = corrupted/input
        # sample[1] = target
        # sample[2] = mask
        # sample[3] = velocity
        corrupted = sample[0]
        target = sample[1]

        if len(sample) >= 3:

            mask = sample[2]

        else:

            raise ValueError(
                "The configured dataset does not provide "
                "a reconstruction mask."
            )

        # Convert to batch format.
        corrupted = (
            ensure_batched_tensor(
                corrupted
            )
        )

        target = (
            ensure_batched_tensor(
                target
            )
        )

        mask = (
            prepare_mask(
                mask,
                target
            )
        )

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        (
            reconstruction,
            travel_time,
            aleatoric_std,
            epistemic_std,
        ) = predictor.predict(
            corrupted
        )

        # ---------------------------------------------
        # Predictive uncertainty
        # ---------------------------------------------

        predictive_std = (
            compute_predictive_std(
                aleatoric_std,
                epistemic_std
            )
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        metrics = evaluate_metrics(
            reconstruction,
            target,
            mask
        )

        mean_aleatoric = (
            ensure_batched_tensor(
                aleatoric_std
            )
            .mean()
            .item()
        )

        mean_epistemic = (
            ensure_batched_tensor(
                epistemic_std
            )
            .mean()
            .item()
        )

        mean_predictive = (
            predictive_std.mean()
            .item()
        )

        results.append(
            {
                "Experiment":
                    EXPERIMENT_NAME,

                "Dataset_Mode":
                    DATASET_MODE,

                "Evaluation_Type":
                    "Configured_Dataset",

                "Complexity":
                    "Dataset_Sample",

                "Sample_Index":
                    sample_index,

                "MAE":
                    metrics["MAE"],

                "Missing_Voxels":
                    metrics["Missing_Voxels"],

                "Aleatoric_STD_Mean":
                    mean_aleatoric,

                "Epistemic_STD_Mean":
                    mean_epistemic,

                "Predictive_STD_Mean":
                    mean_predictive,
            }
        )

        print(
            f"MAE={metrics['MAE']:.6f}, "
            f"Aleatoric={mean_aleatoric:.6f}, "
            f"Epistemic={mean_epistemic:.6f}, "
            f"Predictive={mean_predictive:.6f}"
        )

    return results


# =========================================================
# Save Results
# =========================================================

def save_results(results):
    """
    Save robustness results to CSV.
    """

    if not results:

        raise RuntimeError(
            "No robustness results were generated."
        )

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 60)
    print("GEOLOGICAL COMPLEXITY ROBUSTNESS")
    print("=" * 60)

    print(
        f"Experiment   : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode : {DATASET_MODE}"
    )

    print(
        f"Checkpoint   : {CHECKPOINT_PATH}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = create_predictor()

    # -----------------------------------------------------
    # Select evaluation strategy
    # -----------------------------------------------------

    if DATASET_MODE.lower() == "synthetic":

        print()
        print(
            "Synthetic mode detected."
        )

        print(
            "Running controlled geological-complexity "
            "evaluation."
        )

        results = (
            evaluate_synthetic_complexity(
                predictor
            )
        )

    else:

        print()
        print(
            f"Non-synthetic mode detected: "
            f"{DATASET_MODE}"
        )

        print(
            "Running evaluation on configured "
            "dataset samples."
        )

        results = (
            evaluate_configured_dataset(
                predictor
            )
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    df = save_results(
        results
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ROBUSTNESS EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"Experiment   : {EXPERIMENT_NAME}"
    )

    print(
        f"Dataset Mode : {DATASET_MODE}"
    )

    print(
        f"Results      : {len(df)}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(df.to_string(
        index=False
    ))

    print("=" * 60)


# =========================================================
# Script Entry Point
# =========================================================

if __name__ == "__main__":

    main()