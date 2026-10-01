"""
=====================================================================
FINAL PhD ABLATION STUDY
=====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

Purpose
-------
This module performs a controlled component ablation study to
quantify the contribution of the major architectural components
of the proposed Physics-Informed 3D Encoder–Decoder framework.

Ablation configurations
-----------------------

1. Full_Model
   Attention       = True
   Residual        = True
   Uncertainty     = True

2. No_Attention
   Attention       = False
   Residual        = True
   Uncertainty     = True

3. No_Residual
   Attention       = True
   Residual        = False
   Uncertainty     = True

4. No_Uncertainty
   Attention       = True
   Residual        = True
   Uncertainty     = False

5. Plain_UNet
   Attention       = False
   Residual        = False
   Uncertainty     = False

Experimental controls
---------------------

All configurations:

    * use the same dataset;
    * use the same train/validation split;
    * use the same validation Sample_IDs;
    * use the same missing-data configuration;
    * use the same optimization hyperparameters;
    * use the same number of epochs;
    * are independently initialized;
    * are trained from scratch;
    * use isolated checkpoint directories;
    * are evaluated on exactly the same validation samples.

The purpose is therefore to isolate the contribution of:

    * attention;
    * residual connections;
    * predictive uncertainty.

Output
------

outputs/
    <EXPERIMENT_NAME>/
        ablation/
            Full_Model/
            No_Attention/
            No_Residual/
            No_Uncertainty/
            Plain_UNet/

        reports/
            ablation_study.csv
            ablation_summary.csv
            ablation_metadata.json

ablation_study.csv
-------------------

One row per model per validation sample.

Columns include:

    Model
    Sample_ID
    Attention
    Residual
    Uncertainty
    MAE
    RMSE
    PSNR
    SNR
    SSIM
    Checkpoint

ablation_summary.csv
--------------------

Model-level mean metrics and descriptive percentage changes
relative to the Full_Model.

Important
---------

Percentage changes are descriptive comparisons only.

They do NOT constitute statistical significance testing.

Formal statistical significance should be performed by the
dedicated statistical-significance evaluation module using the
paired per-sample observations retained in ablation_study.csv.

=====================================================================
"""

# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import json
import os
import random
from datetime import datetime
from pathlib import Path


# =====================================================================
# THIRD-PARTY LIBRARIES
# =====================================================================

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# =====================================================================
# PROJECT MODULES
# =====================================================================

from models.network import Network3D

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset

from trainer.trainer import Trainer

from inference.predictor import Predictor

from losses.total_loss import TotalLoss

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.experiment_manager import ExperimentManager

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DX,
    DY,
    DZ,
    DEVICE as CONFIG_DEVICE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    VALIDATION_SPLIT,
    SEED,
)


# =====================================================================
# 1. ABLATION CONFIGURATIONS
# =====================================================================

"""
The Full_Model configuration is derived from utils.config.py.

This prevents the ablation study from silently using a different
architecture from the actual proposed model.
"""

ABLATION_MODELS = {

    "Full_Model": {
        "use_attention": USE_ATTENTION,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Attention": {
        "use_attention": False,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Residual": {
        "use_attention": USE_ATTENTION,
        "use_residual": False,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Uncertainty": {
        "use_attention": USE_ATTENTION,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": False,
    },

    "Plain_UNet": {
        "use_attention": False,
        "use_residual": False,
        "use_uncertainty": False,
    },
}


# =====================================================================
# 2. GLOBAL REPRODUCIBILITY
# =====================================================================

def set_global_seed(seed):
    """
    Set all relevant random seeds.

    The seed is reset before each ablation configuration so that
    the experimental procedure is reproducible.

    Parameters
    ----------
    seed : int
        Reproducibility seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------
    # Deterministic CUDA behaviour
    # -------------------------------------------------------------

    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# =====================================================================
# 3. DEVICE RESOLUTION
# =====================================================================

def get_device():
    """
    Resolve the device using the centralized DEVICE configuration.

    Supported configuration:

        DEVICE = "cpu"
        DEVICE = "cuda"
        DEVICE = "auto"

    Returns
    -------
    torch.device
        Resolved computation device.
    """

    if CONFIG_DEVICE == "cpu":

        return torch.device("cpu")

    if CONFIG_DEVICE == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' is configured, but CUDA "
                "is not available on this system."
            )

        return torch.device("cuda")

    if CONFIG_DEVICE == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    raise ValueError(
        f"Unsupported DEVICE configuration: "
        f"{CONFIG_DEVICE}"
    )


# =====================================================================
# 4. DATALOADER FACTORY
# =====================================================================

def create_ablation_dataloader(
    dataset,
    shuffle=False,
):
    """
    Create the DataLoader used by the ablation study.

    DataLoader settings are taken from the central configuration
    wherever possible.

    Parameters
    ----------
    dataset : Dataset
        Dataset or Subset.

    shuffle : bool
        Whether to shuffle the dataset.

    Returns
    -------
    DataLoader
    """

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# =====================================================================
# 5. MODEL FACTORY
# =====================================================================

def build_ablation_model(
    settings,
    device,
):
    """
    Construct one independent Network3D instance.

    Parameters
    ----------
    settings : dict
        Ablation configuration.

    device : torch.device
        Computation device.

    Returns
    -------
    Network3D
        Independently initialized model.
    """

    model = Network3D(
        use_attention=settings["use_attention"],
        use_residual=settings["use_residual"],
        use_uncertainty=settings["use_uncertainty"],
    )

    model = model.to(device)

    return model


# =====================================================================
# 6. MODEL DEVICE VALIDATION
# =====================================================================

def validate_model_device(
    model,
    device,
):
    """
    Verify that every trainable parameter and registered buffer
    resides on the expected device.
    """

    expected_device = torch.device(device)

    # -----------------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------------

    for name, parameter in model.named_parameters():

        actual_device = parameter.device

        if actual_device != expected_device:

            raise RuntimeError(
                f"Model parameter '{name}' is on "
                f"{actual_device}, but expected "
                f"{expected_device}."
            )

    # -----------------------------------------------------------------
    # BUFFERS
    # -----------------------------------------------------------------

    for name, buffer in model.named_buffers():

        actual_device = buffer.device

        if actual_device != expected_device:

            raise RuntimeError(
                f"Model buffer '{name}' is on "
                f"{actual_device}, but expected "
                f"{expected_device}."
            )

    return True


# =====================================================================
# 7. STABLE SAMPLE-ID EXTRACTION
# =====================================================================

def get_sample_id(
    dataset,
    index,
):
    """
    Return a stable sample identifier.

    When validation data are represented by torch.utils.data.Subset,
    the original dataset index is retained.

    This is essential because ablation results must be paired
    sample-by-sample.
    """

    if hasattr(dataset, "indices"):

        return int(
            dataset.indices[index]
        )

    return int(index)


# =====================================================================
# 8. PREDICTION OUTPUT NORMALIZATION
# =====================================================================

def extract_reconstruction(
    prediction,
):
    """
    Extract the reconstruction from the production Predictor output.

    The evaluation code deliberately does not assume one historical
    Predictor uncertainty-return signature.

    Supported forms include:

        (reconstruction, travel_time, uncertainty)

    and

        (reconstruction, travel_time,
         aleatoric_std, epistemic_std)

    The ablation study requires only the reconstruction because
    predictive uncertainty is not one of the primary ablation metrics.

    Parameters
    ----------
    prediction : tuple
        Predictor output.

    Returns
    -------
    torch.Tensor
        Reconstruction tensor.
    """

    if not isinstance(
        prediction,
        tuple,
    ):

        raise TypeError(
            "Predictor.predict() must return a tuple."
        )

    if len(prediction) < 1:

        raise ValueError(
            "Predictor.predict() returned an empty tuple."
        )

    reconstruction = prediction[0]

    if not torch.is_tensor(reconstruction):

        raise TypeError(
            "The first Predictor output must be "
            "a torch.Tensor reconstruction."
        )

    return reconstruction


# =====================================================================
# 9. TENSOR SHAPE NORMALIZATION
# =====================================================================

def ensure_batched_tensor(
    tensor,
    name,
):
    """
    Ensure a seismic tensor has shape:

        [B, C, D, H, W]

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    name : str
        Tensor name for diagnostics.
    """

    if not torch.is_tensor(tensor):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if tensor.ndim == 4:

        tensor = tensor.unsqueeze(0)

    elif tensor.ndim != 5:

        raise ValueError(
            f"{name} must have 4 or 5 dimensions. "
            f"Received shape: {tuple(tensor.shape)}"
        )

    return tensor


# =====================================================================
# 10. FINITE METRIC VALIDATION
# =====================================================================

def validate_metric_value(
    metric_name,
    value,
    sample_id,
):
    """
    Ensure a metric is finite.

    Infinite or NaN values are not silently accepted because
    they would contaminate the paired ablation analysis.
    """

    value = float(value)

    if not np.isfinite(value):

        raise RuntimeError(
            f"Non-finite {metric_name} detected for "
            f"Sample_ID={sample_id}: {value}"
        )

    return value


# =====================================================================
# 11. PER-SAMPLE EVALUATION
# =====================================================================

def evaluate_checkpoint(
    model,
    checkpoint,
    dataset,
    device,
):
    """
    Evaluate one trained ablation model on every validation sample.

    No averaging is performed here.

    One output row is produced for every validation sample.

    This preserves the paired structure required by later statistical
    significance analysis.
    """

    if len(dataset) == 0:

        raise RuntimeError(
            "Cannot evaluate an empty validation dataset."
        )

    # -----------------------------------------------------------------
    # Move model
    # -----------------------------------------------------------------

    model = model.to(device)

    # -----------------------------------------------------------------
    # Validate model placement
    # -----------------------------------------------------------------

    validate_model_device(
        model,
        device,
    )

    # -----------------------------------------------------------------
    # Predictor
    # -----------------------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    # -----------------------------------------------------------------
    # Evaluation results
    # -----------------------------------------------------------------

    sample_results = []

    # -----------------------------------------------------------------
    # Sample-by-sample evaluation
    # -----------------------------------------------------------------

    for index in range(len(dataset)):

        sample_id = get_sample_id(
            dataset,
            index,
        )

        print(
            f"Evaluating validation sample "
            f"{index + 1}/{len(dataset)} "
            f"(Sample_ID={sample_id})"
        )

        # -------------------------------------------------------------
        # Retrieve sample
        # -------------------------------------------------------------

        sample = dataset[index]

        if len(sample) < 3:

            raise ValueError(
                "Dataset sample must contain at least "
                "(input_cube, target_cube, mask)."
            )

        input_cube = sample[0]

        target_cube = sample[1]

        mask = sample[2]

        # -------------------------------------------------------------
        # Validate tensors
        # -------------------------------------------------------------

        input_cube = ensure_batched_tensor(
            input_cube,
            "input_cube",
        )

        target_cube = ensure_batched_tensor(
            target_cube,
            "target_cube",
        )

        mask = ensure_batched_tensor(
            mask,
            "mask",
        )

        # -------------------------------------------------------------
        # Move tensors
        # -------------------------------------------------------------

        input_cube = input_cube.to(device)

        target_cube = target_cube.to(device)

        mask = mask.to(device)

        # -------------------------------------------------------------
        # Shape consistency
        # -------------------------------------------------------------

        if input_cube.shape != target_cube.shape:

            raise ValueError(
                "Input and target shapes do not match for "
                f"Sample_ID={sample_id}.\n"
                f"Input : {tuple(input_cube.shape)}\n"
                f"Target: {tuple(target_cube.shape)}"
            )

        if mask.shape != input_cube.shape:

            raise ValueError(
                "Mask and input shapes do not match for "
                f"Sample_ID={sample_id}.\n"
                f"Mask : {tuple(mask.shape)}\n"
                f"Input: {tuple(input_cube.shape)}"
            )

        # -------------------------------------------------------------
        # Prediction
        # -------------------------------------------------------------

        model.eval()

        with torch.no_grad():

            prediction = predictor.predict(
                input_cube
            )

        reconstruction = extract_reconstruction(
            prediction
        )

        reconstruction = ensure_batched_tensor(
            reconstruction,
            "reconstruction",
        )

        reconstruction = reconstruction.to(device)

        # -------------------------------------------------------------
        # Shape validation
        # -------------------------------------------------------------

        if reconstruction.shape != target_cube.shape:

            raise ValueError(
                "Reconstruction and target shapes do not match "
                f"for Sample_ID={sample_id}.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target_cube.shape)}"
            )

        # -------------------------------------------------------------
        # Numerical validation
        # -------------------------------------------------------------

        if not torch.isfinite(
            reconstruction
        ).all():

            raise RuntimeError(
                f"Reconstruction contains NaN or Inf values "
                f"for Sample_ID={sample_id}."
            )

        if not torch.isfinite(
            target_cube
        ).all():

            raise RuntimeError(
                f"Target contains NaN or Inf values "
                f"for Sample_ID={sample_id}."
            )

        # -------------------------------------------------------------
        # Reconstruction metrics
        # -------------------------------------------------------------

        sample_mae = mae(
            reconstruction,
            target_cube,
        )

        sample_rmse = rmse(
            reconstruction,
            target_cube,
        )

        sample_psnr = psnr(
            reconstruction,
            target_cube,
        )

        sample_snr = snr(
            reconstruction,
            target_cube,
        )

        sample_ssim = ssim(
            reconstruction,
            target_cube,
        )

        # -------------------------------------------------------------
        # Convert and validate
        # -------------------------------------------------------------

        sample_mae = validate_metric_value(
            "MAE",
            sample_mae.item(),
            sample_id,
        )

        sample_rmse = validate_metric_value(
            "RMSE",
            sample_rmse.item(),
            sample_id,
        )

        sample_psnr = validate_metric_value(
            "PSNR",
            sample_psnr.item(),
            sample_id,
        )

        sample_snr = validate_metric_value(
            "SNR",
            sample_snr.item(),
            sample_id,
        )

        sample_ssim = validate_metric_value(
            "SSIM",
            sample_ssim.item(),
            sample_id,
        )

        # -------------------------------------------------------------
        # Store paired observation
        # -------------------------------------------------------------

        sample_results.append(
            {
                "Sample_ID": sample_id,
                "MAE": sample_mae,
                "RMSE": sample_rmse,
                "PSNR": sample_psnr,
                "SNR": sample_snr,
                "SSIM": sample_ssim,
            }
        )

    return sample_results


# =====================================================================
# 12. TRAIN ONE ABLATION CONFIGURATION
# =====================================================================

def train_ablation_model(
    model_name,
    settings,
    train_loader,
    val_loader,
    device,
    experiment_root,
    seed,
):
    """
    Train one ablation configuration completely from scratch.

    No checkpoint is resumed.

    Each model receives its own:

        model instance
        optimizer
        Trainer
        ExperimentManager
        checkpoint directory
    """

    print()
    print("=" * 70)
    print(
        f"TRAINING ABLATION MODEL: {model_name}"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Reset reproducibility state
    # -----------------------------------------------------------------

    set_global_seed(seed)

    # -----------------------------------------------------------------
    # Display configuration
    # -----------------------------------------------------------------

    print()
    print(
        "Attention   :",
        settings["use_attention"],
    )

    print(
        "Residual    :",
        settings["use_residual"],
    )

    print(
        "Uncertainty :",
        settings["use_uncertainty"],
    )

    print(
        "Seed        :",
        seed,
    )

    print(
        "Device      :",
        device,
    )

    print(
        "Experiment  :",
        experiment_root,
    )

    # -----------------------------------------------------------------
    # Build completely new model
    # -----------------------------------------------------------------

    model = build_ablation_model(
        settings,
        device,
    )

    # -----------------------------------------------------------------
    # Validate model device
    # -----------------------------------------------------------------

    validate_model_device(
        model,
        device,
    )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------
    # Experiment manager
    # -----------------------------------------------------------------

    experiment_manager = ExperimentManager(
        root=experiment_root,
    )

    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
    )

    # -----------------------------------------------------------------
    # Train strictly from scratch
    # -----------------------------------------------------------------

    trainer.fit(
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        resume=False,
    )

    # -----------------------------------------------------------------
    # Locate best checkpoint
    # -----------------------------------------------------------------

    checkpoint = os.path.join(
        experiment_manager.checkpoints,
        "best_model.pth",
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            f"Best checkpoint was not created for "
            f"{model_name}:\n{checkpoint}"
        )

    print()
    print(
        "Best checkpoint:"
    )

    print(checkpoint)

    return checkpoint


# =====================================================================
# 13. VALIDATE ABLATION PAIRING
# =====================================================================

def validate_ablation_pairing(
    dataframe,
    expected_sample_ids,
):
    """
    Perform strict validation of the paired ablation dataset.
    """

    # -----------------------------------------------------------------
    # Expected number of rows
    # -----------------------------------------------------------------

    expected_rows = (
        len(expected_sample_ids)
        * len(ABLATION_MODELS)
    )

    if len(dataframe) != expected_rows:

        raise RuntimeError(
            "Unexpected number of ablation rows.\n"
            f"Expected: {expected_rows}\n"
            f"Received: {len(dataframe)}"
        )

    # -----------------------------------------------------------------
    # Sample IDs
    # -----------------------------------------------------------------

    expected_ids = set(
        expected_sample_ids
    )

    actual_ids = set(
        dataframe["Sample_ID"].unique()
    )

    if actual_ids != expected_ids:

        raise RuntimeError(
            "Validation Sample_ID mismatch.\n"
            f"Expected: {expected_ids}\n"
            f"Received: {actual_ids}"
        )

    # -----------------------------------------------------------------
    # Model IDs
    # -----------------------------------------------------------------

    expected_models = set(
        ABLATION_MODELS.keys()
    )

    actual_models = set(
        dataframe["Model"].unique()
    )

    if actual_models != expected_models:

        raise RuntimeError(
            "Ablation model set mismatch.\n"
            f"Expected: {expected_models}\n"
            f"Received: {actual_models}"
        )

    # -----------------------------------------------------------------
    # Duplicate Model + Sample_ID pairs
    # -----------------------------------------------------------------

    duplicates = dataframe[
        dataframe.duplicated(
            subset=[
                "Model",
                "Sample_ID",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:

        raise RuntimeError(
            "Duplicate Model + Sample_ID pairs detected:\n"
            f"{duplicates}"
        )

    # -----------------------------------------------------------------
    # Every model must contain every validation sample
    # -----------------------------------------------------------------

    for model_name in expected_models:

        model_ids = set(
            dataframe.loc[
                dataframe["Model"] == model_name,
                "Sample_ID",
            ]
        )

        if model_ids != expected_ids:

            raise RuntimeError(
                f"Sample pairing failure for {model_name}.\n"
                f"Expected: {expected_ids}\n"
                f"Received: {model_ids}"
            )

    return True


# =====================================================================
# 14. BUILD ABLATION SUMMARY
# =====================================================================

def build_ablation_summary(
    dataframe,
):
    """
    Calculate model-level means and descriptive percentage changes
    relative to the Full_Model.

    The percentage change is:

        ((Ablation - Full) / Full) * 100

    Interpretation:

        MAE/RMSE:
            positive = larger error than Full_Model
            negative = smaller error than Full_Model

        PSNR/SNR/SSIM:
            positive = larger metric than Full_Model
            negative = smaller metric than Full_Model
    """

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    # -----------------------------------------------------------------
    # Model-level mean metrics
    # -----------------------------------------------------------------

    summary = (
        dataframe
        .groupby(
            "Model",
            as_index=False,
        )[metric_columns]
        .mean()
    )

    # -----------------------------------------------------------------
    # Architecture configuration
    # -----------------------------------------------------------------

    configuration = (
        dataframe[
            [
                "Model",
                "Attention",
                "Residual",
                "Uncertainty",
            ]
        ]
        .drop_duplicates(
            subset=["Model"]
        )
    )

    summary = configuration.merge(
        summary,
        on="Model",
        how="left",
    )

    # -----------------------------------------------------------------
    # Full Model reference
    # -----------------------------------------------------------------

    full_rows = summary[
        summary["Model"] == "Full_Model"
    ]

    if full_rows.empty:

        raise RuntimeError(
            "Full_Model is missing from ablation summary."
        )

    full_model = full_rows.iloc[0]

    # -----------------------------------------------------------------
    # Percentage changes
    # -----------------------------------------------------------------

    for metric in metric_columns:

        change_column = (
            f"{metric}_Change_Percent"
        )

        reference = float(
            full_model[metric]
        )

        if np.isfinite(reference) and reference != 0.0:

            summary[change_column] = (
                (
                    summary[metric]
                    - reference
                )
                / reference
            ) * 100.0

        else:

            summary[change_column] = np.nan

    # -----------------------------------------------------------------
    # Reorder columns
    # -----------------------------------------------------------------

    summary_columns = [
        "Model",
        "Attention",
        "Residual",
        "Uncertainty",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "MAE_Change_Percent",
        "RMSE_Change_Percent",
        "PSNR_Change_Percent",
        "SNR_Change_Percent",
        "SSIM_Change_Percent",
    ]

    summary = summary[
        summary_columns
    ]

    return summary


# =====================================================================
# 15. MAIN ABLATION PROCEDURE
# =====================================================================

def run_ablation():
    """
    Execute the complete controlled ablation study.
    """

    # =================================================================
    # REPRODUCIBILITY
    # =================================================================

    set_global_seed(SEED)

    # =================================================================
    # HEADER
    # =================================================================

    print()
    print("=" * 70)
    print("FINAL PhD ABLATION STUDY")
    print("=" * 70)

    # =================================================================
    # EXPERIMENT INFORMATION
    # =================================================================

    print()
    print(
        "Experiment Name:",
        EXPERIMENT_NAME,
    )

    print(
        "Report Directory:",
        REPORT_DIR,
    )

    print(
        "Checkpoint Directory:",
        CHECKPOINT_DIR,
    )

    print(
        "Validation Split:",
        VALIDATION_SPLIT,
    )

    print(
        "Seed:",
        SEED,
    )

    # =================================================================
    # DEVICE
    # =================================================================

    device = get_device()

    print()
    print(
        "Configured Device:",
        CONFIG_DEVICE,
    )

    print(
        "Resolved Device:",
        device,
    )

    # =================================================================
    # BUILD DATASET
    # =================================================================

    print()
    print("=" * 70)
    print("BUILDING DATASET")
    print("=" * 70)

    dataset = build_dataset()

    if len(dataset) == 0:

        raise RuntimeError(
            "Ablation dataset is empty."
        )

    print()
    print(
        "Total Dataset Samples:",
        len(dataset),
    )

    # =================================================================
    # TRAIN / VALIDATION SPLIT
    # =================================================================

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    print()
    print(
        "Training Samples:",
        len(train_dataset),
    )

    print(
        "Validation Samples:",
        len(val_dataset),
    )

    # =================================================================
    # VALIDATION SAMPLE IDs
    # =================================================================

    validation_sample_ids = [
        get_sample_id(
            val_dataset,
            index,
        )
        for index in range(
            len(val_dataset)
        )
    ]

    print()
    print(
        "Validation Sample IDs:"
    )

    print(
        validation_sample_ids
    )

    # =================================================================
    # DATA LOADERS
    # =================================================================

    train_loader = create_ablation_dataloader(
        train_dataset,
        shuffle=True,
    )

    val_loader = create_ablation_dataloader(
        val_dataset,
        shuffle=False,
    )

    # =================================================================
    # RESULT STORAGE
    # =================================================================

    all_results = []

    # =================================================================
    # ABLATION LOOP
    # =================================================================

    for model_name, settings in ABLATION_MODELS.items():

        print()
        print("=" * 70)
        print(
            f"ABLATION CONFIGURATION: {model_name}"
        )
        print("=" * 70)

        # -------------------------------------------------------------
        # Independent seed
        #
        # Same seed protocol for each configuration ensures that
        # differences are attributable to the architecture rather
        # than uncontrolled random-state differences.
        # -------------------------------------------------------------

        set_global_seed(SEED)

        # -------------------------------------------------------------
        # Isolated experiment directory
        # -------------------------------------------------------------

        experiment_root = os.path.join(
            CHECKPOINT_DIR,
            "..",
            "ablation",
            model_name,
        )

        experiment_root = os.path.abspath(
            experiment_root
        )

        os.makedirs(
            experiment_root,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # Train from scratch
        # -------------------------------------------------------------

        checkpoint = train_ablation_model(
            model_name=model_name,
            settings=settings,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            experiment_root=experiment_root,
            seed=SEED,
        )

        # -------------------------------------------------------------
        # New model instance for evaluation
        # -------------------------------------------------------------

        evaluation_model = build_ablation_model(
            settings,
            device,
        )

        # -------------------------------------------------------------
        # Evaluate every validation sample
        # -------------------------------------------------------------

        sample_metrics = evaluate_checkpoint(
            model=evaluation_model,
            checkpoint=checkpoint,
            dataset=val_dataset,
            device=device,
        )

        # -------------------------------------------------------------
        # Attach configuration information
        # -------------------------------------------------------------

        for result in sample_metrics:

            result["Model"] = model_name

            result["Attention"] = (
                settings["use_attention"]
            )

            result["Residual"] = (
                settings["use_residual"]
            )

            result["Uncertainty"] = (
                settings["use_uncertainty"]
            )

            result["Checkpoint"] = (
                checkpoint
            )

            all_results.append(
                result
            )

        # -------------------------------------------------------------
        # Display model-level diagnostic means
        # -------------------------------------------------------------

        model_dataframe = pd.DataFrame(
            sample_metrics
        )

        print()
        print(
            f"{model_name} validation means"
        )

        print("-" * 50)

        for metric in [
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM",
        ]:

            print(
                f"{metric:<6}: "
                f"{model_dataframe[metric].mean():.6f}"
            )

    # =================================================================
    # COMBINE RESULTS
    # =================================================================

    dataframe = pd.DataFrame(
        all_results
    )

    # =================================================================
    # REQUIRED COLUMNS
    # =================================================================

    required_columns = [
        "Model",
        "Sample_ID",
        "Attention",
        "Residual",
        "Uncertainty",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "Checkpoint",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Ablation results are missing required columns:\n"
            f"{missing_columns}"
        )

    # =================================================================
    # VALIDATE PAIRED STRUCTURE
    # =================================================================

    validate_ablation_pairing(
        dataframe,
        validation_sample_ids,
    )

    # =================================================================
    # SORT RESULTS
    # =================================================================

    dataframe = dataframe.sort_values(
        by=[
            "Sample_ID",
            "Model",
        ]
    ).reset_index(
        drop=True
    )

    # =================================================================
    # CREATE REPORT DIRECTORY
    # =================================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # =================================================================
    # SAVE PER-SAMPLE RESULTS
    # =================================================================

    output_file = os.path.join(
        REPORT_DIR,
        "ablation_study.csv",
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    # =================================================================
    # BUILD SUMMARY
    # =================================================================

    summary_dataframe = build_ablation_summary(
        dataframe
    )

    # =================================================================
    # SAVE SUMMARY
    # =================================================================

    summary_file = os.path.join(
        REPORT_DIR,
        "ablation_summary.csv",
    )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
    )

    # =================================================================
    # SAVE METADATA
    # =================================================================

    metadata = {
        "study": "Controlled PhD Ablation Study",
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "configured_device": CONFIG_DEVICE,
        "resolved_device": str(device),
        "dataset_size": len(dataset),
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "validation_sample_ids": validation_sample_ids,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "validation_split": VALIDATION_SPLIT,
        "dx": DX,
        "dy": DY,
        "dz": DZ,
        "ablation_configurations": ABLATION_MODELS,
        "per_sample_output": output_file,
        "summary_output": summary_file,
    }

    metadata_file = os.path.join(
        REPORT_DIR,
        "ablation_metadata.json",
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # =================================================================
    # DISPLAY PER-SAMPLE RESULTS
    # =================================================================

    print()
    print("=" * 70)
    print("PER-SAMPLE ABLATION RESULTS")
    print("=" * 70)

    print()

    print(
        dataframe.to_string(
            index=False
        )
    )

    # =================================================================
    # DISPLAY SUMMARY
    # =================================================================

    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print()

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    # =================================================================
    # OUTPUT FILES
    # =================================================================

    print()
    print("=" * 70)
    print("ABLATION OUTPUT FILES")
    print("=" * 70)

    print()
    print(
        "Per-sample results:"
    )

    print(
        output_file
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Metadata:"
    )

    print(
        metadata_file
    )

    # =================================================================
    # COMPLETION
    # =================================================================

    print()
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    return (
        dataframe,
        summary_dataframe,
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    run_ablation()