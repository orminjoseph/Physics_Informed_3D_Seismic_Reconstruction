"""
=========================================================
Ablation Study
=========================================================

Evaluates the contribution of the major components of the
Physics-Informed 3D Encoder-Decoder Framework.

Configurations:

1. Full Model
2. No Attention
3. No Residual
4. No Uncertainty
5. Plain U-Net

Experimental design:

    - The Full Model uses the already-trained
      synthetic_training best checkpoint.
    - Each ablation model is trained independently
      from scratch.
    - Each ablation model has an isolated experiment
      directory.
    - No ablation model resumes from a previous checkpoint.
    - The same dataset split is used for all configurations.
    - All models are evaluated on the SAME validation samples.
    - Metrics are saved PER VALIDATION SAMPLE.
    - Sample_ID is retained so that paired statistical
      significance testing can be performed later.

Output:

    outputs/
        <EXPERIMENT_NAME>/
            ablation/
                No_Attention/
                    checkpoints/
                    logs/
                    reports/
                    plots/
                    reconstructions/
                    tensorboard/
                    training_progress/

                No_Residual/
                    ...

                No_Uncertainty/
                    ...

                Plain_UNet/
                    ...

            reports/
                ablation_study.csv
                ablation_summary.csv

The file:

    ablation_study.csv

contains one row per model per validation sample.

This is required for the paired statistical significance
analysis in statistical_significance.py.

Example:

    Model,Sample_ID,Attention,Residual,Uncertainty,MAE,RMSE,PSNR,SNR,SSIM
    Full_Model,0,True,True,True,...
    No_Attention,0,False,True,True,...
    No_Residual,0,True,False,True,...

Author: Ormin Joseph
=========================================================
"""

import os

import numpy as np
import pandas as pd
import torch

from models.network import Network3D

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from trainer.trainer import Trainer

from inference.predictor import Predictor

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from losses.total_loss import TotalLoss

from utils.config import (
    EXPERIMENT_NAME,
    OUTPUT_ROOT,
    REPORT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DX,
    DY,
    DZ
)


# =========================================================
# ABLATION CONFIGURATIONS
# =========================================================

ABLATION_MODELS = {

    "Full_Model": {

        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": True

    },

    "No_Attention": {

        "use_attention": False,
        "use_residual": True,
        "use_uncertainty": True

    },

    "No_Residual": {

        "use_attention": True,
        "use_residual": False,
        "use_uncertainty": True

    },

    "No_Uncertainty": {

        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": False

    },

    "Plain_UNet": {

        "use_attention": False,
        "use_residual": False,
        "use_uncertainty": False

    }

}


# =========================================================
# ABLATION ROOT DIRECTORY
# =========================================================

ABLATION_ROOT = os.path.join(
    OUTPUT_ROOT,
    EXPERIMENT_NAME,
    "ablation"
)


# =========================================================
# FULL MODEL CHECKPOINT
# =========================================================

FULL_MODEL_CHECKPOINT = os.path.join(
    OUTPUT_ROOT,
    EXPERIMENT_NAME,
    "checkpoints",
    "best_model.pth"
)


# =========================================================
# ISOLATED ABLATION EXPERIMENT MANAGER
# =========================================================

class AblationExperimentManager:
    """
    Experiment manager for one ablation configuration.

    Every ablation configuration receives its own isolated
    directory structure.

    The manager deliberately does not modify the global
    EXPERIMENT_NAME in utils.config.
    """

    def __init__(self, root):

        self.root = root

        # -------------------------------------------------
        # Checkpoints
        # -------------------------------------------------

        self.checkpoints = os.path.join(
            self.root,
            "checkpoints"
        )

        # -------------------------------------------------
        # Logs
        # -------------------------------------------------

        self.logs = os.path.join(
            self.root,
            "logs"
        )

        # -------------------------------------------------
        # Reports
        # -------------------------------------------------

        self.reports = os.path.join(
            self.root,
            "reports"
        )

        # -------------------------------------------------
        # Plots
        # -------------------------------------------------

        self.plots = os.path.join(
            self.root,
            "plots"
        )

        # -------------------------------------------------
        # Reconstructions
        # -------------------------------------------------

        self.reconstructions = os.path.join(
            self.root,
            "reconstructions"
        )

        # -------------------------------------------------
        # TensorBoard
        # -------------------------------------------------

        self.tensorboard = os.path.join(
            self.root,
            "tensorboard"
        )

        # -------------------------------------------------
        # Training progress
        # -------------------------------------------------

        self.training_progress = os.path.join(
            self.root,
            "training_progress"
        )

        # -------------------------------------------------
        # Global checkpoint reference
        #
        # Retained for compatibility with the existing
        # Trainer / ExperimentManager interface.
        # -------------------------------------------------

        self.global_checkpoints = self.checkpoints

        # -------------------------------------------------
        # Create directories
        # -------------------------------------------------

        self.create()

    # =====================================================
    # CREATE DIRECTORIES
    # =====================================================

    def create(self):

        directories = [

            self.root,

            self.checkpoints,

            self.logs,

            self.reports,

            self.plots,

            self.reconstructions,

            self.tensorboard,

            self.training_progress

        ]

        for directory in directories:

            os.makedirs(
                directory,
                exist_ok=True
            )


# =========================================================
# VALIDATE TENSOR
# =========================================================

def validate_tensor(
        tensor,
        name
):
    """
    Validate tensor type, shape and finite values.
    """

    if not isinstance(
        tensor,
        torch.Tensor
    ):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if tensor.numel() == 0:

        raise ValueError(
            f"{name} is empty."
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            f"{name} contains non-finite values."
        )


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

def prepare_input_tensor(
        input_cube
):
    """
    Convert a dataset input cube into the model's expected
    [B, C, D, H, W] format.
    """

    validate_tensor(
        input_cube,
        "input_cube"
    )

    # Dataset convention:
    #
    # [C, D, H, W]
    #
    # Add batch dimension.

    if input_cube.ndim == 4:

        input_batch = input_cube.unsqueeze(0)

    elif input_cube.ndim == 5:

        input_batch = input_cube

    else:

        raise ValueError(
            "input_cube must have shape "
            "[C,D,H,W] or [B,C,D,H,W]. "
            f"Received: {tuple(input_cube.shape)}"
        )

    return input_batch


# =========================================================
# PREPARE TARGET TENSOR
# =========================================================

def prepare_target_tensor(
        target_cube
):
    """
    Convert target cube into [B,C,D,H,W] format.
    """

    validate_tensor(
        target_cube,
        "target_cube"
    )

    if target_cube.ndim == 4:

        target_batch = target_cube.unsqueeze(0)

    elif target_cube.ndim == 5:

        target_batch = target_cube

    else:

        raise ValueError(
            "target_cube must have shape "
            "[C,D,H,W] or [B,C,D,H,W]. "
            f"Received: {tuple(target_cube.shape)}"
        )

    return target_batch


# =========================================================
# EVALUATE ONE CHECKPOINT
# =========================================================

def evaluate_checkpoint(
        model,
        checkpoint,
        dataset,
        device
):
    """
    Evaluate a trained checkpoint over every sample in the
    supplied validation dataset.

    IMPORTANT:

        Metrics are returned PER SAMPLE.

    This is necessary for paired statistical testing.

    Returns
    -------
    list of dict

        One dictionary per validation sample.
    """

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint}"
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(

        model=model,

        checkpoint=checkpoint,

        device=device

    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    sample_results = []

    # -----------------------------------------------------
    # Evaluate every validation sample
    # -----------------------------------------------------

    for index in range(
        len(dataset)
    ):

        print(
            f"Evaluating sample "
            f"{index + 1}/{len(dataset)}"
        )

        # -------------------------------------------------
        # Dataset convention
        # -------------------------------------------------

        sample = dataset[index]

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
            velocity_model
        ) = sample[:4]

        # -------------------------------------------------
        # Validate dataset tensors
        # -------------------------------------------------

        validate_tensor(
            input_cube,
            "input_cube"
        )

        validate_tensor(
            target_cube,
            "target_cube"
        )

        validate_tensor(
            mask,
            "mask"
        )

        validate_tensor(
            velocity_model,
            "velocity_model"
        )

        # -------------------------------------------------
        # Prepare tensors
        # -------------------------------------------------

        input_batch = prepare_input_tensor(
            input_cube
        )

        target_batch = prepare_target_tensor(
            target_cube
        )

        # -------------------------------------------------
        # Check input/target dimensions
        # -------------------------------------------------

        if input_batch.shape != target_batch.shape:

            raise ValueError(
                "\nInput and target shapes do not match.\n"
                f"Input : {tuple(input_batch.shape)}\n"
                f"Target: {tuple(target_batch.shape)}"
            )

        # -------------------------------------------------
        # Move input to device
        # -------------------------------------------------

        input_batch = input_batch.to(
            device
        )

        target_batch = target_batch.to(
            device
        )

        # -------------------------------------------------
        # Prediction
        #
        # Current Predictor API returns:
        #
        # reconstruction,
        # travel_time,
        # aleatoric_std,
        # epistemic_std
        # -------------------------------------------------

        prediction = predictor.predict(
            input_batch
        )

        if not isinstance(
            prediction,
            tuple
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
            epistemic_std
        ) = prediction

        # -------------------------------------------------
        # Validate reconstruction
        # -------------------------------------------------

        validate_tensor(
            reconstruction,
            "reconstruction"
        )

        # -------------------------------------------------
        # Ensure reconstruction shape matches target
        # -------------------------------------------------

        if reconstruction.shape != target_batch.shape:

            raise ValueError(
                "\nReconstruction and target shapes "
                "do not match.\n"
                f"Reconstruction: "
                f"{tuple(reconstruction.shape)}\n"
                f"Target: "
                f"{tuple(target_batch.shape)}"
            )

        # -------------------------------------------------
        # Calculate metrics for THIS sample only
        # -------------------------------------------------

        sample_mae = mae(
            reconstruction,
            target_batch
        ).item()

        sample_rmse = rmse(
            reconstruction,
            target_batch
        ).item()

        sample_psnr = psnr(
            reconstruction,
            target_batch
        ).item()

        sample_snr = snr(
            reconstruction,
            target_batch
        ).item()

        sample_ssim = ssim(
            reconstruction,
            target_batch
        ).item()

        # -------------------------------------------------
        # Validate metric values
        # -------------------------------------------------

        metric_values = {

            "MAE": sample_mae,

            "RMSE": sample_rmse,

            "PSNR": sample_psnr,

            "SNR": sample_snr,

            "SSIM": sample_ssim

        }

        for metric_name, value in metric_values.items():

            if not np.isfinite(value):

                raise ValueError(
                    f"\nNon-finite {metric_name} "
                    f"for validation sample {index}."
                )

        # -------------------------------------------------
        # Store per-sample result
        # -------------------------------------------------

        sample_results.append({

            "Sample_ID":
                index,

            "MAE":
                sample_mae,

            "RMSE":
                sample_rmse,

            "PSNR":
                sample_psnr,

            "SNR":
                sample_snr,

            "SSIM":
                sample_ssim

        })

    return sample_results


# =========================================================
# TRAIN ONE ABLATION MODEL
# =========================================================

def train_ablation_model(
        model_name,
        settings,
        train_loader,
        val_loader,
        device
):
    """
    Train one ablation configuration from scratch.

    No checkpoint is loaded and resume=False is explicitly
    passed to the Trainer.
    """

    print()
    print("=" * 70)
    print(
        f"TRAINING ABLATION MODEL: {model_name}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Architecture configuration
    # -----------------------------------------------------

    print()
    print(
        "Attention   :",
        settings["use_attention"]
    )

    print(
        "Residual    :",
        settings["use_residual"]
    )

    print(
        "Uncertainty :",
        settings["use_uncertainty"]
    )

    # -----------------------------------------------------
    # Isolated experiment directory
    # -----------------------------------------------------

    experiment_root = os.path.join(

        ABLATION_ROOT,

        model_name

    )

    print()
    print(
        "Experiment Root:",
        experiment_root
    )

    # -----------------------------------------------------
    # Experiment manager
    # -----------------------------------------------------

    experiment_manager = (
        AblationExperimentManager(
            root=experiment_root
        )
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = Network3D(

        use_attention=settings[
            "use_attention"
        ],

        use_residual=settings[
            "use_residual"
        ],

        use_uncertainty=settings[
            "use_uncertainty"
        ]

    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = TotalLoss(

        dx=DX,

        dy=DY,

        dz=DZ

    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(

        model=model,

        criterion=criterion,

        optimizer=optimizer,

        device=device,

        experiment_manager=experiment_manager

    )

    # -----------------------------------------------------
    # Train from scratch
    # -----------------------------------------------------

    print()
    print(
        "Training from scratch."
    )

    trainer.fit(

        train_loader,

        val_loader,

        epochs=NUM_EPOCHS,

        resume=False

    )

    # -----------------------------------------------------
    # Best checkpoint
    # -----------------------------------------------------

    checkpoint = os.path.join(

        experiment_manager.checkpoints,

        "best_model.pth"

    )

    if not os.path.isfile(
        checkpoint
    ):

        raise FileNotFoundError(

            f"\nBest checkpoint was not created "
            f"for {model_name}:\n"
            f"{checkpoint}"

        )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        checkpoint
    )

    return checkpoint


# =========================================================
# BUILD MODEL FROM SETTINGS
# =========================================================

def build_model(
        settings
):
    """
    Construct a Network3D using the supplied ablation
    configuration.
    """

    return Network3D(

        use_attention=settings[
            "use_attention"
        ],

        use_residual=settings[
            "use_residual"
        ],

        use_uncertainty=settings[
            "use_uncertainty"
        ]

    )


# =========================================================
# MAIN ABLATION PROCEDURE
# =========================================================

def run_ablation():

    print()
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME
    )

    print(
        "Ablation Root:",
        ABLATION_ROOT
    )

    print(
        "Report Dir :",
        REPORT_DIR
    )

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )

    print()
    print(
        "Device     :",
        device
    )

    # =====================================================
    # VERIFY FULL MODEL CHECKPOINT
    # =====================================================

    print()
    print("=" * 70)
    print("VERIFYING FULL MODEL CHECKPOINT")
    print("=" * 70)

    print()
    print(
        "Full Model Checkpoint:"
    )

    print(
        FULL_MODEL_CHECKPOINT
    )

    if not os.path.isfile(
        FULL_MODEL_CHECKPOINT
    ):

        raise FileNotFoundError(

            "\nFull Model checkpoint not found:\n"
            f"{FULL_MODEL_CHECKPOINT}\n\n"
            "The existing synthetic_training experiment "
            "must contain best_model.pth before the "
            "ablation study can be performed."

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
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Ablation dataset is empty."
        )

    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================

    print()
    print("=" * 70)
    print("CREATING TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    print()
    print(
        "Training Samples  :",
        len(train_dataset)
    )

    print(
        "Validation Samples:",
        len(val_dataset)
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

    train_loader = create_dataloader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0

    )

    val_loader = create_dataloader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )

    # =====================================================
    # RESULTS
    # =====================================================

    all_results = []

    # =====================================================
    # LOOP THROUGH CONFIGURATIONS
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

            print()
            print(
                "Using existing Full Model checkpoint."
            )

            checkpoint = FULL_MODEL_CHECKPOINT

        # =================================================
        # ABLATION MODELS
        # =================================================

        else:

            checkpoint = train_ablation_model(

                model_name=model_name,

                settings=settings,

                train_loader=train_loader,

                val_loader=val_loader,

                device=device

            )

        # =================================================
        # BUILD MODEL
        # =================================================

        model = build_model(
            settings
        )

        # =================================================
        # EVALUATE MODEL
        # =================================================

        print()
        print(
            f"EVALUATING: {model_name}"
        )

        sample_metrics = evaluate_checkpoint(

            model=model,

            checkpoint=checkpoint,

            dataset=val_dataset,

            device=device

        )

        # =================================================
        # ADD MODEL CONFIGURATION
        # =================================================

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

            result["Checkpoint"] = checkpoint

            all_results.append(
                result
            )

        # =================================================
        # DISPLAY PER-MODEL SUMMARY
        # =================================================

        model_dataframe = pd.DataFrame(
            sample_metrics
        )

        print()
        print(
            f"{model_name} MEAN VALIDATION RESULTS"
        )

        print(
            "-" * 60
        )

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
    # REQUIRED COLUMN ORDER
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

        "Checkpoint"

    ]

    dataframe = dataframe[
        columns
    ]

    # =====================================================
    # VALIDATE PAIRING STRUCTURE
    # =====================================================

    print()
    print("=" * 70)
    print("VALIDATING ABLATION PAIRING STRUCTURE")
    print("=" * 70)

    # -----------------------------------------------------
    # Expected number of models
    # -----------------------------------------------------

    expected_models = len(
        ABLATION_MODELS
    )

    actual_models = dataframe[
        "Model"
    ].nunique()

    if actual_models != expected_models:

        raise RuntimeError(

            "\nUnexpected number of models in "
            "ablation results.\n"
            f"Expected: {expected_models}\n"
            f"Found   : {actual_models}"

        )

    # -----------------------------------------------------
    # Expected validation sample IDs
    # -----------------------------------------------------

    expected_sample_ids = set(
        range(
            len(val_dataset)
        )
    )

    # -----------------------------------------------------
    # Validate every model has every validation sample
    # -----------------------------------------------------

    for model_name in ABLATION_MODELS:

        model_data = dataframe[
            dataframe["Model"] == model_name
        ]

        actual_sample_ids = set(
            model_data[
                "Sample_ID"
            ].tolist()
        )

        if actual_sample_ids != expected_sample_ids:

            missing = (
                expected_sample_ids
                - actual_sample_ids
            )

            extra = (
                actual_sample_ids
                - expected_sample_ids
            )

            raise RuntimeError(

                f"\nValidation sample mismatch "
                f"for {model_name}.\n"
                f"Missing Sample_IDs: "
                f"{sorted(missing)}\n"
                f"Unexpected Sample_IDs: "
                f"{sorted(extra)}"

            )

        # -------------------------------------------------
        # Check duplicate model/sample combinations
        # -------------------------------------------------

        duplicates = model_data[
            model_data[
                "Sample_ID"
            ].duplicated(
                keep=False
            )
        ]

        if len(duplicates) > 0:

            raise RuntimeError(

                f"\nDuplicate Sample_ID detected "
                f"for model {model_name}."

            )

    # =====================================================
    # CREATE AGGREGATE SUMMARY
    # =====================================================

    summary = (
        dataframe
        .groupby(
            [
                "Model",
                "Attention",
                "Residual",
                "Uncertainty"
            ],
            as_index=False
        )
        .agg({

            "MAE": "mean",

            "RMSE": "mean",

            "PSNR": "mean",

            "SNR": "mean",

            "SSIM": "mean"

        })
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Per-sample results
    # -----------------------------------------------------

    output_file = os.path.join(

        REPORT_DIR,

        "ablation_study.csv"

    )

    dataframe.to_csv(

        output_file,

        index=False

    )

    # -----------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------

    summary_file = os.path.join(

        REPORT_DIR,

        "ablation_summary.csv"

    )

    summary.to_csv(

        summary_file,

        index=False

    )

    # =====================================================
    # DISPLAY FINAL PER-SAMPLE TABLE
    # =====================================================

    print()
    print("=" * 70)
    print("ABLATION STUDY PER-SAMPLE RESULTS")
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
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    # =====================================================
    # OUTPUT INFORMATION
    # =====================================================

    print()
    print(
        "Per-sample results saved:"
    )

    print(
        output_file
    )

    print()
    print(
        "Aggregate summary saved:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Validation samples per model:",
        len(val_dataset)
    )

    print(
        "Total result rows:",
        len(dataframe)
    )

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