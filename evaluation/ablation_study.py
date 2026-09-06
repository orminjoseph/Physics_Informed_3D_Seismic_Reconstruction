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

- The Full Model uses the already-trained best checkpoint.
- Every ablation configuration is trained independently
  from scratch.
- Each ablation configuration has its own isolated output
  directory.
- No ablation configuration resumes from a previous
  checkpoint.

Output
------

outputs/
    <EXPERIMENT_NAME>/
        ablation/
            No_Attention/
            No_Residual/
            No_Uncertainty/
            Plain_UNet/

        reports/
            ablation_study.csv
            ablation_summary.csv

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
# MODEL DEVICE VALIDATION
# =========================================================

def validate_model_device(model, device):
    """
    Verify that the model parameters are located on the
    requested device.

    This provides an early and clear diagnostic instead of
    allowing a later CUDA/CPU mismatch to occur.
    """

    for name, parameter in model.named_parameters():
        if parameter.device != device:
            raise RuntimeError(
                f"Model parameter '{name}' is on "
                f"{parameter.device}, but the expected device "
                f"is {device}."
            )


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
    Evaluate a trained model checkpoint over a dataset.

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
    dict
        Average MAE, RMSE, PSNR, SNR and SSIM.
    """

    # -----------------------------------------------------
    # Explicitly move model to device
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
    # Metric accumulators
    # -----------------------------------------------------

    total_mae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_snr = 0.0
    total_ssim = 0.0

    num_samples = len(dataset)

    if num_samples == 0:
        raise RuntimeError(
            "Cannot evaluate an empty validation dataset."
        )

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    for index in range(num_samples):

        print(
            f"Evaluating sample "
            f"{index + 1}/{num_samples}"
        )

        # -------------------------------------------------
        # Load sample
        # -------------------------------------------------

        (
            input_cube,
            target_cube,
            mask,
            velocity_model,
        ) = dataset[index]

        # -------------------------------------------------
        # Move input to selected device
        # -------------------------------------------------

        input_cube = input_cube.to(device)

        target_cube = target_cube.to(device)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        with torch.no_grad():

            reconstruction, uncertainty = (
                predictor.predict(
                    input_cube
                )
            )

        # -------------------------------------------------
        # Ensure target has batch dimension
        # -------------------------------------------------

        target_batch = target_cube.unsqueeze(0)

        # -------------------------------------------------
        # Ensure prediction and target are on same device
        # -------------------------------------------------

        reconstruction = reconstruction.to(device)

        target_batch = target_batch.to(device)

        # -------------------------------------------------
        # Metric calculation
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
                    f"non-finite value for sample "
                    f"{index + 1}."
                )

        # -------------------------------------------------
        # Accumulate metrics
        # -------------------------------------------------

        total_mae += sample_mae.item()

        total_rmse += sample_rmse.item()

        total_psnr += sample_psnr.item()

        total_snr += sample_snr.item()

        total_ssim += sample_ssim.item()

    # -----------------------------------------------------
    # Return average metrics
    # -----------------------------------------------------

    return {
        "MAE": total_mae / num_samples,
        "RMSE": total_rmse / num_samples,
        "PSNR": total_psnr / num_samples,
        "SNR": total_snr / num_samples,
        "SSIM": total_ssim / num_samples,
    }


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
    Execute the complete ablation study.
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

    results = []

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
        # FULL MODEL
        # =================================================

        if model_name == "Full_Model":

            checkpoint = os.path.join(
                "outputs",
                EXPERIMENT_NAME,
                "checkpoints",
                "best_model.pth",
            )

            if not os.path.exists(checkpoint):

                raise FileNotFoundError(
                    "Full Model checkpoint not found:\n"
                    f"{checkpoint}"
                )

            print()
            print(
                "Using existing Full Model checkpoint:"
            )

            print(checkpoint)

            # ---------------------------------------------
            # Build Full Model explicitly on device
            # ---------------------------------------------

            model = build_model(
                settings,
                device,
            )

            # ---------------------------------------------
            # Evaluate
            # ---------------------------------------------

            metrics = evaluate_checkpoint(
                model=model,
                checkpoint=checkpoint,
                dataset=val_dataset,
                device=device,
            )

        # =================================================
        # ABLATION MODELS
        # =================================================

        else:

            experiment_root = os.path.join(
                "outputs",
                EXPERIMENT_NAME,
                "ablation",
                model_name,
            )

            # ---------------------------------------------
            # Train from scratch
            # ---------------------------------------------

            checkpoint = train_ablation_model(
                model_name=model_name,
                settings=settings,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                experiment_root=experiment_root,
            )

            # ---------------------------------------------
            # Build the trained architecture
            # explicitly on the selected device
            # ---------------------------------------------

            model = build_model(
                settings,
                device,
            )

            # ---------------------------------------------
            # Evaluate
            # ---------------------------------------------

            metrics = evaluate_checkpoint(
                model=model,
                checkpoint=checkpoint,
                dataset=val_dataset,
                device=device,
            )

        # =================================================
        # ADD CONFIGURATION INFORMATION
        # =================================================

        metrics["Model"] = model_name

        metrics["Attention"] = (
            settings["use_attention"]
        )

        metrics["Residual"] = (
            settings["use_residual"]
        )

        metrics["Uncertainty"] = (
            settings["use_uncertainty"]
        )

        # =================================================
        # ADD RESULTS
        # =================================================

        results.append(metrics)

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        print()
        print(
            f"{model_name} RESULTS"
        )

        print("-" * 50)

        print(
            f"MAE  : {metrics['MAE']:.6f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.6f}"
        )

        print(
            f"PSNR : {metrics['PSNR']:.6f}"
        )

        print(
            f"SNR  : {metrics['SNR']:.6f}"
        )

        print(
            f"SSIM : {metrics['SSIM']:.6f}"
        )

    # =====================================================
    # CREATE RESULTS DATAFRAME
    # =====================================================

    dataframe = pd.DataFrame(
        results
    )

    # =====================================================
    # COLUMN ORDER
    # =====================================================

    columns = [
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

    dataframe = dataframe[
        columns
    ]

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # =====================================================
    # SAVE ABLATION RESULTS
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
    # CREATE SUMMARY
    # =====================================================

    summary_file = os.path.join(
        REPORT_DIR,
        "ablation_summary.csv",
    )

    summary_dataframe = dataframe.copy()

    # -----------------------------------------------------
    # Calculate percentage change relative to Full Model
    # -----------------------------------------------------

    full_model_rows = dataframe[
        dataframe["Model"] == "Full_Model"
    ]

    if not full_model_rows.empty:

        full_model = full_model_rows.iloc[0]

        summary_rows = []

        for _, row in dataframe.iterrows():

            summary_row = row.to_dict()

            # ---------------------------------------------
            # MAE percentage change
            # ---------------------------------------------

            if full_model["MAE"] != 0:

                summary_row["MAE_Change_Percent"] = (
                    (
                        row["MAE"]
                        - full_model["MAE"]
                    )
                    / full_model["MAE"]
                ) * 100.0

            else:

                summary_row["MAE_Change_Percent"] = 0.0

            # ---------------------------------------------
            # RMSE percentage change
            # ---------------------------------------------

            if full_model["RMSE"] != 0:

                summary_row["RMSE_Change_Percent"] = (
                    (
                        row["RMSE"]
                        - full_model["RMSE"]
                    )
                    / full_model["RMSE"]
                ) * 100.0

            else:

                summary_row["RMSE_Change_Percent"] = 0.0

            # ---------------------------------------------
            # PSNR percentage change
            # ---------------------------------------------

            if full_model["PSNR"] != 0:

                summary_row["PSNR_Change_Percent"] = (
                    (
                        row["PSNR"]
                        - full_model["PSNR"]
                    )
                    / full_model["PSNR"]
                ) * 100.0

            else:

                summary_row["PSNR_Change_Percent"] = 0.0

            # ---------------------------------------------
            # SNR percentage change
            # ---------------------------------------------

            if full_model["SNR"] != 0:

                summary_row["SNR_Change_Percent"] = (
                    (
                        row["SNR"]
                        - full_model["SNR"]
                    )
                    / full_model["SNR"]
                ) * 100.0

            else:

                summary_row["SNR_Change_Percent"] = 0.0

            # ---------------------------------------------
            # SSIM percentage change
            # ---------------------------------------------

            if full_model["SSIM"] != 0:

                summary_row["SSIM_Change_Percent"] = (
                    (
                        row["SSIM"]
                        - full_model["SSIM"]
                    )
                    / full_model["SSIM"]
                ) * 100.0

            else:

                summary_row["SSIM_Change_Percent"] = 0.0

            summary_rows.append(
                summary_row
            )

        summary_dataframe = pd.DataFrame(
            summary_rows
        )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
    )

    # =====================================================
    # DISPLAY FINAL TABLE
    # =====================================================

    print()
    print("=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    print()

    print(
        dataframe.to_string(
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

    return dataframe


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_ablation()