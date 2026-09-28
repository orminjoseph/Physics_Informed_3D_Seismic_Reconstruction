"""
======================================================================
FINAL MODEL EVALUATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Final orchestration layer for evaluating the trained model.

Responsibilities
----------------
1. Use the active global configuration.
2. Build the evaluation dataset.
3. Adapt the dataset to the centralized Evaluator interface.
4. Construct the final configured Network3D architecture.
5. Resolve the computational device from utils.config.
6. Load the trained best_model.pth using the established Predictor.
7. Run the centralized Evaluator.
8. Save the complete evaluation results to CSV.

Checkpoint convention
---------------------
Evaluation:
    outputs/<EXPERIMENT_NAME>/checkpoints/best_model.pth

Training resume:
    outputs/<EXPERIMENT_NAME>/checkpoints/latest_checkpoint.pth

Important
---------
The configuration file is the single source of truth for:

    - model architecture
    - uncertainty configuration
    - computational device
    - DataLoader configuration
    - experiment-specific output paths

The Predictor provides the established checkpoint-loading mechanism.

The Evaluator performs the actual metric calculations.

Author: Ormin Joseph
======================================================================
"""

import os

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from models.network import Network3D

from dataset.build_dataset import build_dataset

from evaluation.evaluator import Evaluator

from inference.predictor import Predictor

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MC_DROPOUT_SAMPLES,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
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

    The underlying dataset is not modified.
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
        # Preserve optional metadata
        # --------------------------------------------------------------

        if len(sample) >= 4:

            evaluation_sample["velocity"] = sample[3]

        if len(sample) >= 5:

            evaluation_sample["mask_type"] = sample[4]

        if len(sample) >= 6:

            evaluation_sample["geological_mode"] = sample[5]

        return evaluation_sample


# ======================================================================
# DEVICE RESOLUTION
# ======================================================================

def resolve_device():
    """
    Resolve the computational device according to utils.config.

    Supported modes:

        "cpu"
            Force CPU.

        "cuda"
            Force CUDA and raise an error if CUDA is unavailable.

        "auto"
            Use CUDA when available; otherwise use CPU.
    """

    configured_device = str(
        DEVICE
    ).lower()

    # --------------------------------------------------------------
    # Force CPU
    # --------------------------------------------------------------

    if configured_device == "cpu":

        return torch.device(
            "cpu"
        )

    # --------------------------------------------------------------
    # Force CUDA
    # --------------------------------------------------------------

    if configured_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' was requested, but CUDA is not "
                "available on this system."
            )

        return torch.device(
            "cuda"
        )

    # --------------------------------------------------------------
    # Automatic device selection
    # --------------------------------------------------------------

    if configured_device == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # --------------------------------------------------------------
    # Invalid configuration
    # --------------------------------------------------------------

    raise ValueError(
        "Invalid DEVICE configuration: "
        f"{DEVICE!r}. Expected 'cpu', 'cuda', or 'auto'."
    )


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate(
    model_override=None,
    mc_samples=None,
    batch_size=None,
):
    """
    Evaluate the trained best model for the active experiment.

    Parameters
    ----------
    model_override:
        Optional externally supplied model.

        If None, Network3D is constructed using the architecture
        settings in utils.config.

    mc_samples:
        Number of stochastic forward passes.

        If None:
            MC_DROPOUT_SAMPLES from utils.config is used.

        mc_samples = 1:
            Deterministic evaluation.

        mc_samples > 1:
            MC-Dropout evaluation.

    batch_size:
        Evaluation DataLoader batch size.

        If None:
            BATCH_SIZE from utils.config is used.

    Returns
    -------
    dict
        Complete aggregated evaluation results.
    """

    # ==================================================================
    # HEADER
    # ==================================================================

    print()
    print("=" * 80)
    print("FINAL MODEL EVALUATION")
    print("=" * 80)

    # ==================================================================
    # RESOLVE EVALUATION SETTINGS
    # ==================================================================

    if mc_samples is None:

        mc_samples = MC_DROPOUT_SAMPLES

    if batch_size is None:

        # Import here to keep the main configuration import organized.
        from utils.config import BATCH_SIZE

        batch_size = BATCH_SIZE

    # --------------------------------------------------------------
    # Validate MC sample count
    # --------------------------------------------------------------

    if not isinstance(
        mc_samples,
        int
    ) or mc_samples < 1:

        raise ValueError(
            "mc_samples must be a positive integer."
        )

    # --------------------------------------------------------------
    # Validate batch size
    # --------------------------------------------------------------

    if not isinstance(
        batch_size,
        int
    ) or batch_size < 1:

        raise ValueError(
            "batch_size must be a positive integer."
        )

    # ==================================================================
    # DEVICE
    # ==================================================================

    device = resolve_device()

    print()
    print("Experiment :", EXPERIMENT_NAME)
    print("Configured Device :", DEVICE)
    print("Using Device :", device)
    print("MC Samples :", mc_samples)
    print("Batch Size :", batch_size)

    # --------------------------------------------------------------
    # CUDA information
    # --------------------------------------------------------------

    if device.type == "cuda":

        print(
            "CUDA Device :",
            torch.cuda.get_device_name(
                device
            )
        )

        print(
            "CUDA Device Count :",
            torch.cuda.device_count()
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
        "Dataset Length :",
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
        "Dataset Adapter :",
        type(evaluation_dataset).__name__
    )

    # ==================================================================
    # BUILD DATALOADER
    # ==================================================================

    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        "DataLoader Batches :",
        len(dataloader)
    )

    print(
        "DataLoader Workers :",
        NUM_WORKERS
    )

    print(
        "Pin Memory :",
        PIN_MEMORY
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
    print("BUILDING FINAL NETWORK")
    print("-" * 80)

    if model_override is None:

        model = Network3D(
            use_attention=USE_ATTENTION,
            use_residual=USE_RESIDUAL,
            use_uncertainty=USE_UNCERTAINTY,
        )

    else:

        model = model_override

    # ==================================================================
    # DISPLAY ARCHITECTURE CONFIGURATION
    # ==================================================================

    print(
        "Attention Enabled :",
        USE_ATTENTION
    )

    print(
        "Residual Enabled :",
        USE_RESIDUAL
    )

    print(
        "Uncertainty Enabled :",
        USE_UNCERTAINTY
    )

    # ==================================================================
    # MOVE MODEL TO DEVICE
    # ==================================================================

    model = model.to(
        device
    )

    # ==================================================================
    # CHECKPOINT
    # ==================================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print("-" * 80)
    print("LOADING BEST MODEL CHECKPOINT")
    print("-" * 80)

    print(
        "Checkpoint :",
        checkpoint
    )

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete training and make sure best_model.pth "
            "exists in the configured checkpoint directory."
        )

    # --------------------------------------------------------------
    # Use the established Predictor checkpoint loader.
    #
    # Predictor:
    #
    #     1. loads best_model.pth
    #     2. supports the project's checkpoint formats
    #     3. loads model_state_dict
    #     4. moves the model to the selected device
    #     5. switches the model to evaluation mode
    # --------------------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    # --------------------------------------------------------------
    # Retrieve the same model instance after checkpoint loading.
    # --------------------------------------------------------------

    model = predictor.model

    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Model is in evaluation mode :",
        not model.training
    )

    # ==================================================================
    # EVALUATOR
    # ==================================================================

    print()
    print("-" * 80)
    print("INITIALIZING CENTRALIZED EVALUATOR")
    print("-" * 80)

    evaluator = Evaluator(
        model=model,
        device=device,
        mc_samples=mc_samples,
    )

    # ==================================================================
    # RUN EVALUATION
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

    if not isinstance(
        results,
        dict
    ):

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

        if isinstance(
            value,
            float
        ):

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