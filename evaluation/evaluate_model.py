"""
======================================================================
Model Evaluation
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This module is the orchestration layer for model evaluation.

Responsibilities
----------------
1. Build the dataset according to the active configuration.
2. Load the best checkpoint for the current experiment.
3. Adapt the project dataset output to the Evaluator interface.
4. Run the centralized Evaluator.
5. Save the complete evaluation results to CSV.

The actual metric calculations are performed by:

    evaluation.evaluator.Evaluator

which uses:

    evaluation.metrics.EvaluationMetrics

Dataset output convention
-------------------------
SyntheticSeismicDataset returns:

    input_cube
    target
    mask
    velocity_model
    mask_type
    geological_mode

Evaluator input convention
--------------------------
Each DataLoader batch must contain at least:

    input
    target
    mask

Additional metadata are preserved where available.

Checkpoint convention
---------------------
    outputs/
        <EXPERIMENT_NAME>/
            checkpoints/
                best_model.pth
                latest_checkpoint.pth

Evaluation uses:

    best_model.pth

Training resume uses:

    latest_checkpoint.pth

All experiment-dependent paths are obtained from:

    utils.config

Author: Ormin Joseph
======================================================================
"""

import os

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from models.network import Network3D

from dataset.build_dataset import build_dataset

from evaluation.evaluator import Evaluator

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
)


# ======================================================================
# EVALUATION DATASET ADAPTER
# ======================================================================

class EvaluationDatasetAdapter(Dataset):
    """
    Adapt the project's tuple-based dataset to the dictionary-based
    interface expected by Evaluator.

    Original dataset sample:

        (
            input_cube,
            target_cube,
            mask,
            velocity_model,
            mask_type,
            geological_mode
        )

    Evaluator-compatible sample:

        {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
            "velocity": velocity_model,
            "mask_type": mask_type,
            "geological_mode": geological_mode
        }

    The adapter does not modify the underlying dataset.
    It only changes the interface presented to the evaluation pipeline.
    """

    def __init__(self, dataset):

        if dataset is None:

            raise ValueError(
                "dataset cannot be None."
            )

        self.dataset = dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        if not isinstance(sample, (tuple, list)):

            raise TypeError(
                "Expected dataset sample to be a tuple or list. "
                f"Received: {type(sample)}"
            )

        if len(sample) < 3:

            raise ValueError(
                "Dataset sample must contain at least "
                "input, target and mask."
            )

        # --------------------------------------------------------------
        # Required evaluation tensors
        # --------------------------------------------------------------

        input_cube = sample[0]
        target_cube = sample[1]
        mask = sample[2]

        evaluation_sample = {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
        }

        # --------------------------------------------------------------
        # Preserve optional metadata when supplied by the dataset.
        #
        # These fields are not currently required by Evaluator, but
        # retaining them prevents useful dataset information from being
        # discarded at the evaluation boundary.
        # --------------------------------------------------------------

        if len(sample) >= 4:

            evaluation_sample["velocity"] = sample[3]

        if len(sample) >= 5:

            evaluation_sample["mask_type"] = sample[4]

        if len(sample) >= 6:

            evaluation_sample["geological_mode"] = sample[5]

        return evaluation_sample


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate(
    model_override=None,
    mc_samples=1,
    batch_size=1,
):
    """
    Evaluate the best model checkpoint for the current experiment.

    Parameters
    ----------
    model_override:
        Optional externally supplied model.

        If None, Network3D is constructed using the project's
        standard architecture configuration.

    mc_samples:
        Number of stochastic forward passes used by Evaluator.

        mc_samples = 1
            Deterministic reconstruction evaluation.

        mc_samples > 1
            MC-dropout uncertainty evaluation.

    batch_size:
        Evaluation DataLoader batch size.

    Returns
    -------
    dict
        Complete aggregated evaluation results produced by Evaluator.
    """

    # ==================================================================
    # HEADER
    # ==================================================================

    print()
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # ==================================================================
    # DEVICE
    # ==================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Experiment :", EXPERIMENT_NAME)
    print("Device     :", device)
    print("MC Samples :", mc_samples)
    print("Batch Size :", batch_size)

    # ==================================================================
    # VALIDATE EVALUATION SETTINGS
    # ==================================================================

    if mc_samples < 1:

        raise ValueError(
            "mc_samples must be at least 1."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be at least 1."
        )

    # ==================================================================
    # BUILD DATASET
    # ==================================================================

    print()
    print("-" * 80)
    print("BUILDING EVALUATION DATASET")
    print("-" * 80)

    dataset = build_dataset()

    if dataset is None:

        raise RuntimeError(
            "build_dataset() returned None."
        )

    dataset_length = len(dataset)

    print(
        "Dataset Length:",
        dataset_length
    )

    if dataset_length == 0:

        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    # ==================================================================
    # ADAPT DATASET
    # ==================================================================

    evaluation_dataset = EvaluationDatasetAdapter(
        dataset
    )

    print(
        "Dataset Adapter:",
        type(evaluation_dataset).__name__
    )

    # ==================================================================
    # BUILD DATALOADER
    # ==================================================================

    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "DataLoader Batches:",
        len(dataloader)
    )

    if len(dataloader) == 0:

        raise RuntimeError(
            "Evaluation DataLoader is empty."
        )

    # ==================================================================
    # BUILD MODEL
    # ==================================================================

    print()
    print("-" * 80)
    print("BUILDING MODEL")
    print("-" * 80)

    if model_override is None:

        model = Network3D(
            use_attention=True,
            use_residual=True,
            use_uncertainty=True
        )

    else:

        model = model_override

    # ==================================================================
    # MOVE MODEL TO DEVICE
    # ==================================================================

    model = model.to(device)

    # ==================================================================
    # CHECKPOINT
    # ==================================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print(
        "Checkpoint:",
        checkpoint
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete at least one training epoch and make sure "
            "best_model.pth exists in the checkpoint directory."
        )

    # ==================================================================
    # EVALUATOR
    # ==================================================================

    print()
    print("-" * 80)
    print("INITIALIZING EVALUATOR")
    print("-" * 80)

    evaluator = Evaluator(
        model=model,
        device=device,
        mc_samples=mc_samples
    )

    # ==================================================================
    # RUN CENTRALIZED EVALUATION
    # ==================================================================

    print()
    print("-" * 80)
    print("RUNNING CENTRALIZED MODEL EVALUATION")
    print("-" * 80)

    results = evaluator.evaluate(
        dataloader
    )

    # ==================================================================
    # VALIDATE RESULTS
    # ==================================================================

    if not isinstance(results, dict):

        raise TypeError(
            "Evaluator.evaluate() must return a dictionary. "
            f"Received: {type(results)}"
        )

    if len(results) == 0:

        raise RuntimeError(
            "Evaluator returned an empty results dictionary."
        )

    # ==================================================================
    # PRINT RESULTS
    # ==================================================================

    print()
    print("=" * 80)
    print("FINAL MODEL EVALUATION RESULTS")
    print("=" * 80)

    for key, value in results.items():

        if isinstance(value, float):

            print(
                f"{key:<35}: {value:.6f}"
            )

        else:

            print(
                f"{key:<35}: {value}"
            )

    # ==================================================================
    # CREATE REPORT DIRECTORY
    # ==================================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    output_file = os.path.join(
        REPORT_DIR,
        "evaluation_metrics.csv"
    )

    results_dataframe = pd.DataFrame(
        [results]
    )

    results_dataframe.to_csv(
        output_file,
        index=False
    )

    # ==================================================================
    # CONFIRM OUTPUT
    # ==================================================================

    print()
    print("-" * 80)
    print("EVALUATION OUTPUT")
    print("-" * 80)

    print(
        "Evaluation results saved:"
    )

    print(
        output_file
    )

    print()
    print(
        "Experiment directory:"
    )

    print(
        os.path.join(
            "outputs",
            EXPERIMENT_NAME
        )
    )

    print()
    print("=" * 80)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 80)

    return results


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    evaluate()