"""
=========================================================
Ablation Study
=========================================================

Evaluates the contribution of the major components of the
Physics-Informed 3D Encoder-Decoder Framework.

Configurations
--------------

1. Full Model
2. No Attention
3. No Residual
4. No Uncertainty
5. Plain U-Net

Methodology
-----------

- All five configurations are trained independently
  from scratch.
- Every configuration uses the SAME dataset split.
- Every configuration is evaluated on the SAME validation
  samples.
- Each validation sample receives a Sample_ID.
- Per-sample results are retained for paired statistical
  analysis.
- Each ablation configuration has its own isolated output
  directory.
- No ablation configuration resumes from a previous
  checkpoint.

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

ablation_study.csv
-------------------

Contains one row per model per validation sample.

Columns:

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

Contains model-level mean metrics and percentage changes
relative to the Full Model.

=========================================================
"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.network import Network3D

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset

from trainer.trainer import Trainer
from inference.predictor import Predictor

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from losses.total_loss import TotalLoss

from utils.experiment_manager import ExperimentManager

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DX,
    DY,
    DZ,
)


# =========================================================
# ABLATION CONFIGURATIONS
# =========================================================

ABLATION_MODELS = {
    "Full_Model": {
        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": True,
    },

    "No_Attention": {
        "use_attention": False,
        "use_residual": True,
        "use_uncertainty": True,
    },

    "No_Residual": {
        "use_attention": True,
        "use_residual": False,
        "use_uncertainty": True,
    },

    "No_Uncertainty": {
        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": False,
    },

    "Plain_UNet": {
        "use_attention": False,
        "use_residual": False,
        "use_uncertainty": False,
    },
}


# =========================================================
# DEVICE
# =========================================================

def get_device():
    """
    Select CUDA when available; otherwise use CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =========================================================
# DATALOADER
# =========================================================

def create_ablation_dataloader(dataset, shuffle=False):
    """
    Create the DataLoader used by the ablation experiments.
    """

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# =========================================================
# MODEL BUILDER
# =========================================================

def build_model(settings, device):
    """
    Build an ablation Network3D model and explicitly move
    it to the selected device.

    Parameters
    ----------
    settings : dict
        Ablation configuration.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    Network3D
        Model placed on the selected device.
    """

    model = Network3D(
        use_attention=settings["use_attention"],
        use_residual=settings["use_residual"],
        use_uncertainty=settings["use_uncertainty"],
    )

    model = model.to(device)

    return model


# =========================================================
# VALIDATE MODEL DEVICE
# =========================================================

def validate_model_device(
    model,
    device,
):
    """
    Validate that all model parameters and buffers are on
    the requested device.

    CUDA device normalization treats:

        cuda
        cuda:0

    as equivalent when CUDA device 0 is active.
    """

    expected_device = torch.device(device)

    # -----------------------------------------------------
    # Validate model parameters
    # -----------------------------------------------------

    for name, parameter in model.named_parameters():

        actual_device = parameter.device

        if expected_device.type == "cuda":

            expected_index = (
                torch.cuda.current_device()
                if expected_device.index is None
                else expected_device.index
            )

            actual_index = (
                torch.cuda.current_device()
                if actual_device.index is None
                else actual_device.index
            )

            device_match = (
                actual_device.type == "cuda"
                and actual_index == expected_index
            )

        else:

            device_match = (
                actual_device == expected_device
            )

        if not device_match:

            raise RuntimeError(
                f"\nModel parameter '{name}' is on "
                f"{actual_device}, but the expected device "
                f"is {expected_device}."
            )

    # -----------------------------------------------------
    # Validate model buffers
    # -----------------------------------------------------

    for name, buffer in model.named_buffers():

        actual_device = buffer.device

        if expected_device.type == "cuda":

            expected_index = (
                torch.cuda.current_device()
                if expected_device.index is None
                else expected_device.index
            )

            actual_index = (
                torch.cuda.current_device()
                if actual_device.index is None
                else actual_device.index
            )

            device_match = (
                actual_device.type == "cuda"
                and actual_index == expected_index
            )

        else:

            device_match = (
                actual_device == expected_device
            )

        if not device_match:

            raise RuntimeError(
                f"\nModel buffer '{name}' is on "
                f"{actual_device}, but the expected device "
                f"is {expected_device}."
            )

    return True


# =========================================================
# GET SAMPLE ID
# =========================================================

def get_sample_id(dataset, index):
    """
    Obtain a stable Sample_ID for a validation sample.

    When split_dataset() returns a torch.utils.data.Subset,
    the original dataset index is retained.

    This is important because all five ablation models must
    be paired using the SAME validation samples.

    Parameters
    ----------
    dataset : Dataset or Subset
        Validation dataset.

    index : int
        Position within the validation dataset.

    Returns
    -------
    int
        Stable sample identifier.
    """

    # -----------------------------------------------------
    # Preserve original dataset index when available
    # -----------------------------------------------------

    if hasattr(dataset, "indices"):

        return int(
            dataset.indices[index]
        )

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    return int(index)


# =========================================================
# METRIC EVALUATION
# =========================================================

def evaluate_checkpoint(
    model,
    checkpoint,
    dataset,
    device,
):
    """
    Evaluate a trained model checkpoint over every sample
    in a validation dataset.

    IMPORTANT
    ---------

    This function intentionally returns PER-SAMPLE metrics.

    It does NOT average the validation results.

    This preserves the paired observations required for
    downstream statistical analysis.

    Parameters
    ----------
    model : Network3D
        Model architecture.

    checkpoint : str
        Path to trained checkpoint.

    dataset : Dataset
        Validation dataset.

    device : torch.device
        Evaluation device.

    Returns
    -------
    list of dict
        One dictionary per validation sample.
    """

    # -----------------------------------------------------
    # Move model to selected device
    # -----------------------------------------------------

    model = model.to(device)

    # -----------------------------------------------------
    # Validate model placement
    # -----------------------------------------------------

    validate_model_device(
        model,
        device,
    )

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    num_samples = len(dataset)

    if num_samples == 0:

        raise RuntimeError(
            "Cannot evaluate an empty validation dataset."
        )

    # -----------------------------------------------------
    # Per-sample results
    # -----------------------------------------------------

    sample_results = []

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    for index in range(num_samples):

        sample_id = get_sample_id(
            dataset,
            index,
        )

        print(
            f"Evaluating validation sample "
            f"{index + 1}/{num_samples} "
            f"(Sample_ID={sample_id})"
        )

        # -------------------------------------------------
        # Dataset sample
        # -------------------------------------------------

        sample = dataset[index]

        # -------------------------------------------------
        # Validate dataset structure
        # -------------------------------------------------

        if len(sample) < 4:

            raise ValueError(
                "Dataset sample must contain at least "
                "(input_cube, target_cube, mask, "
                "velocity_model)."
            )

        (
            input_cube,
            target_cube,
            mask,
            velocity_model,
        ) = sample[:4]

        # -------------------------------------------------
        # Move tensors to selected device
        # -------------------------------------------------

        input_cube = input_cube.to(device)

        target_cube = target_cube.to(device)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        with torch.no_grad():

            prediction = predictor.predict(
                input_cube
            )

            if not isinstance(
                prediction,
                tuple,
            ):

                raise TypeError(
                    "\nPredictor.predict() must return "
                    "a tuple."
                )

            if len(prediction) != 4:

                raise ValueError(
                    "\nUnexpected Predictor.predict() "
                    "return signature.\n"
                    f"Expected 4 values, received "
                    f"{len(prediction)}."
                )

            (
                reconstruction,
                travel_time,
                aleatoric_std,
                epistemic_std,
            ) = prediction

        # -------------------------------------------------
        # Ensure target has batch dimension
        # -------------------------------------------------

        target_batch = target_cube.unsqueeze(0)

        # -------------------------------------------------
        # Ensure prediction and target use same device
        # -------------------------------------------------

        reconstruction = reconstruction.to(device)

        target_batch = target_batch.to(device)

        # -------------------------------------------------
        # Calculate metrics
        # -------------------------------------------------

        sample_mae = mae(
            reconstruction,
            target_batch,
        )

        sample_rmse = rmse(
            reconstruction,
            target_batch,
        )

        sample_psnr = psnr(
            reconstruction,
            target_batch,
        )

        sample_snr = snr(
            reconstruction,
            target_batch,
        )

        sample_ssim = ssim(
            reconstruction,
            target_batch,
        )

        # -------------------------------------------------
        # Validate metric values
        # -------------------------------------------------

        metric_values = {
            "MAE": sample_mae,
            "RMSE": sample_rmse,
            "PSNR": sample_psnr,
            "SNR": sample_snr,
            "SSIM": sample_ssim,
        }

        for metric_name, metric_value in metric_values.items():

            if not torch.isfinite(metric_value):

                raise RuntimeError(
                    f"{metric_name} produced a "
                    f"non-finite value for "
                    f"Sample_ID={sample_id}."
                )

        # -------------------------------------------------
        # Store PER-SAMPLE results
        # -------------------------------------------------

        sample_result = {
            "Sample_ID": sample_id,
            "MAE": sample_mae.item(),
            "RMSE": sample_rmse.item(),
            "PSNR": sample_psnr.item(),
            "SNR": sample_snr.item(),
            "SSIM": sample_ssim.item(),
        }

        sample_results.append(
            sample_result
        )

    # -----------------------------------------------------
    # Return per-sample results
    # -----------------------------------------------------

    return sample_results


# =========================================================
# TRAIN ONE ABLATION MODEL
# =========================================================

def train_ablation_model(
    model_name,
    settings,
    train_loader,
    val_loader,
    device,
    experiment_root,
):
    """
    Train one ablation configuration from scratch.

    Returns
    -------
    str
        Path to the best checkpoint.
    """

    print()
    print("=" * 70)
    print(
        f"TRAINING ABLATION MODEL: {model_name}"
    )
    print("=" * 70)

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

    print()
    print(
        "Experiment Root:",
        experiment_root,
    )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model = Network3D(
        use_attention=settings["use_attention"],
        use_residual=settings["use_residual"],
        use_uncertainty=settings["use_uncertainty"],
    ).to(device)

    # =====================================================
    # VALIDATE MODEL DEVICE
    # =====================================================

    validate_model_device(
        model,
        device,
    )

    print()
    print(
        "Model Device:",
        next(model.parameters()).device,
    )

    # =====================================================
    # LOSS
    # =====================================================

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =====================================================
    # EXPERIMENT MANAGER
    # =====================================================

    experiment_manager = ExperimentManager(
        root=experiment_root,
    )

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
    )

    # =====================================================
    # TRAIN FROM SCRATCH
    # =====================================================

    trainer.fit(
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        resume=False,
    )

    # =====================================================
    # BEST CHECKPOINT
    # =====================================================

    checkpoint = os.path.join(
        experiment_manager.checkpoints,
        "best_model.pth",
    )

    if not os.path.exists(checkpoint):

        raise FileNotFoundError(
            f"\nBest checkpoint was not created for "
            f"{model_name}:\n"
            f"{checkpoint}"
        )

    print()
    print(
        "Best checkpoint:"
    )

    print(checkpoint)

    return checkpoint


# =========================================================
# MAIN ABLATION PROCEDURE
# =========================================================

def run_ablation():
    """
    Execute the complete controlled ablation study.
    """

    print()
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # =====================================================
    # EXPERIMENT INFORMATION
    # =====================================================

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME,
    )

    print(
        "Report Dir :",
        REPORT_DIR,
    )

    # =====================================================
    # DEVICE
    # =====================================================

    device = get_device()

    print()
    print(
        "Device     :",
        device,
    )

    # =====================================================
    # BUILD DATASET
    # =====================================================

    print()
    print("=" * 70)
    print("BUILDING DATASET")
    print("=" * 70)

    dataset = build_dataset()

    print()
    print(
        "Dataset Length:",
        len(dataset),
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Ablation dataset is empty."
        )

    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    print()
    print(
        "Training Samples  :",
        len(train_dataset),
    )

    print(
        "Validation Samples:",
        len(val_dataset),
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    # =====================================================
    # DISPLAY VALIDATION SAMPLE IDS
    # =====================================================

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

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = create_ablation_dataloader(
        train_dataset,
        shuffle=True,
    )

    val_loader = create_ablation_dataloader(
        val_dataset,
        shuffle=False,
    )

    # =====================================================
    # RESULTS
    # =====================================================

    all_results = []

    # =====================================================
    # LOOP THROUGH ABLATION CONFIGURATIONS
    # =====================================================

    for model_name, settings in ABLATION_MODELS.items():

        print()
        print("=" * 70)
        print(
            f"CONFIGURATION: {model_name}"
        )
        print("=" * 70)

        # =================================================
        # ISOLATED EXPERIMENT DIRECTORY
        # =================================================

        experiment_root = os.path.join(
            "outputs",
            EXPERIMENT_NAME,
            "ablation",
            model_name,
        )

        # =================================================
        # TRAIN FROM SCRATCH
        # =================================================

        checkpoint = train_ablation_model(
            model_name=model_name,
            settings=settings,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            experiment_root=experiment_root,
        )

        # =================================================
        # BUILD MODEL FOR EVALUATION
        # =================================================

        model = build_model(
            settings,
            device,
        )

        # =================================================
        # EVALUATE EVERY VALIDATION SAMPLE
        # =================================================

        sample_metrics = evaluate_checkpoint(
            model=model,
            checkpoint=checkpoint,
            dataset=val_dataset,
            device=device,
        )

        # =================================================
        # ADD CONFIGURATION INFORMATION TO EACH SAMPLE
        # =================================================

        for sample_result in sample_metrics:

            sample_result["Model"] = model_name

            sample_result["Attention"] = (
                settings["use_attention"]
            )

            sample_result["Residual"] = (
                settings["use_residual"]
            )

            sample_result["Uncertainty"] = (
                settings["use_uncertainty"]
            )

            sample_result["Checkpoint"] = (
                checkpoint
            )

            all_results.append(
                sample_result
            )

        # =================================================
        # DISPLAY MODEL-LEVEL MEANS
        # =================================================

        model_dataframe = pd.DataFrame(
            sample_metrics
        )

        print()
        print(
            f"{model_name} RESULTS"
        )

        print("-" * 50)

        print(
            f"MAE  : "
            f"{model_dataframe['MAE'].mean():.6f}"
        )

        print(
            f"RMSE : "
            f"{model_dataframe['RMSE'].mean():.6f}"
        )

        print(
            f"PSNR : "
            f"{model_dataframe['PSNR'].mean():.6f}"
        )

        print(
            f"SNR  : "
            f"{model_dataframe['SNR'].mean():.6f}"
        )

        print(
            f"SSIM : "
            f"{model_dataframe['SSIM'].mean():.6f}"
        )

    # =====================================================
    # CREATE PER-SAMPLE DATAFRAME
    # =====================================================

    dataframe = pd.DataFrame(
        all_results
    )

    # =====================================================
    # VALIDATE RESULTS
    # =====================================================

    expected_rows = (
        len(val_dataset)
        * len(ABLATION_MODELS)
    )

    if len(dataframe) != expected_rows:

        raise RuntimeError(
            "\nUnexpected number of ablation "
            "result rows.\n"
            f"Expected: {expected_rows}\n"
            f"Received: {len(dataframe)}"
        )

    # -----------------------------------------------------
    # Validate Sample_ID pairing
    # -----------------------------------------------------

    expected_sample_ids = set(
        validation_sample_ids
    )

    actual_sample_ids = set(
        dataframe["Sample_ID"].unique()
    )

    if actual_sample_ids != expected_sample_ids:

        raise RuntimeError(
            "\nValidation Sample_ID mismatch.\n"
            f"Expected: {expected_sample_ids}\n"
            f"Received: {actual_sample_ids}"
        )

    # -----------------------------------------------------
    # Validate model/sample pairing
    # -----------------------------------------------------

    duplicate_pairs = dataframe[
        dataframe.duplicated(
            subset=[
                "Model",
                "Sample_ID",
            ],
            keep=False,
        )
    ]

    if not duplicate_pairs.empty:

        raise RuntimeError(
            "\nDuplicate Model + Sample_ID "
            "pairs detected.\n"
            f"{duplicate_pairs}"
        )

    # =====================================================
    # COLUMN ORDER
    # =====================================================

    columns = [
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

    dataframe = dataframe[
        columns
    ]

    # =====================================================
    # SORT RESULTS
    # =====================================================

    dataframe = dataframe.sort_values(
        by=[
            "Sample_ID",
            "Model",
        ]
    ).reset_index(
        drop=True
    )

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # =====================================================
    # SAVE PER-SAMPLE ABLATION RESULTS
    # =====================================================

    output_file = os.path.join(
        REPORT_DIR,
        "ablation_study.csv",
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    # =====================================================
    # CREATE MODEL-LEVEL SUMMARY
    # =====================================================

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    summary_dataframe = (
        dataframe
        .groupby(
            "Model",
            as_index=False,
        )[metric_columns]
        .mean()
    )

    # =====================================================
    # ADD CONFIGURATION FLAGS
    # =====================================================

    configuration_dataframe = (
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

    summary_dataframe = (
        configuration_dataframe
        .merge(
            summary_dataframe,
            on="Model",
            how="left",
        )
    )

    # =====================================================
    # ORDER SUMMARY COLUMNS
    # =====================================================

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
    ]

    summary_dataframe = summary_dataframe[
        summary_columns
    ]

    # =====================================================
    # CALCULATE PERCENTAGE CHANGE RELATIVE TO FULL MODEL
    # =====================================================

    full_model_rows = summary_dataframe[
        summary_dataframe["Model"] == "Full_Model"
    ]

    if not full_model_rows.empty:

        full_model = (
            full_model_rows.iloc[0]
        )

        summary_rows = []

        for _, row in summary_dataframe.iterrows():

            summary_row = row.to_dict()

            # ---------------------------------------------
            # MAE percentage change
            # ---------------------------------------------

            if full_model["MAE"] != 0:

                summary_row[
                    "MAE_Change_Percent"
                ] = (
                    (
                        row["MAE"]
                        - full_model["MAE"]
                    )
                    / full_model["MAE"]
                ) * 100.0

            else:

                summary_row[
                    "MAE_Change_Percent"
                ] = 0.0

            # ---------------------------------------------
            # RMSE percentage change
            # ---------------------------------------------

            if full_model["RMSE"] != 0:

                summary_row[
                    "RMSE_Change_Percent"
                ] = (
                    (
                        row["RMSE"]
                        - full_model["RMSE"]
                    )
                    / full_model["RMSE"]
                ) * 100.0

            else:

                summary_row[
                    "RMSE_Change_Percent"
                ] = 0.0

            # ---------------------------------------------
            # PSNR percentage change
            # ---------------------------------------------

            if full_model["PSNR"] != 0:

                summary_row[
                    "PSNR_Change_Percent"
                ] = (
                    (
                        row["PSNR"]
                        - full_model["PSNR"]
                    )
                    / full_model["PSNR"]
                ) * 100.0

            else:

                summary_row[
                    "PSNR_Change_Percent"
                ] = 0.0

            # ---------------------------------------------
            # SNR percentage change
            # ---------------------------------------------

            if full_model["SNR"] != 0:

                summary_row[
                    "SNR_Change_Percent"
                ] = (
                    (
                        row["SNR"]
                        - full_model["SNR"]
                    )
                    / full_model["SNR"]
                ) * 100.0

            else:

                summary_row[
                    "SNR_Change_Percent"
                ] = 0.0

            # ---------------------------------------------
            # SSIM percentage change
            # ---------------------------------------------

            if full_model["SSIM"] != 0:

                summary_row[
                    "SSIM_Change_Percent"
                ] = (
                    (
                        row["SSIM"]
                        - full_model["SSIM"]
                    )
                    / full_model["SSIM"]
                ) * 100.0

            else:

                summary_row[
                    "SSIM_Change_Percent"
                ] = 0.0

            summary_rows.append(
                summary_row
            )

        summary_dataframe = pd.DataFrame(
            summary_rows
        )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary_file = os.path.join(
        REPORT_DIR,
        "ablation_summary.csv",
    )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
    )

    # =====================================================
    # DISPLAY FINAL PER-SAMPLE TABLE
    # =====================================================

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

    # =====================================================
    # DISPLAY SUMMARY
    # =====================================================

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

    # =====================================================
    # DISPLAY OUTPUT FILES
    # =====================================================

    print()
    print(
        "Results saved:"
    )

    print(
        output_file
    )

    print()
    print(
        "Summary saved:"
    )

    print(
        summary_file
    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print()
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    return dataframe, summary_dataframe


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_ablation()