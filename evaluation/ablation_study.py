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

Important:

    - The Full Model uses the already-trained
      synthetic_training checkpoint.
    - Each ablation model is trained independently
      from scratch.
    - Each ablation model has its own isolated
      checkpoint, log and TensorBoard directories.
    - No ablation model resumes from a previous checkpoint.
    - The same dataset split is used for all configurations.
    - Validation data is used for final ablation comparison.

Output:

    outputs/
        synthetic_training/
            ablation/
                No_Attention/
                    checkpoints/
                    logs/
                    tensorboard/
                    ...
                No_Residual/
                    checkpoints/
                    logs/
                    tensorboard/
                    ...
                No_Uncertainty/
                    checkpoints/
                    logs/
                    tensorboard/
                    ...
                Plain_UNet/
                    checkpoints/
                    logs/
                    tensorboard/
                    ...

            reports/
                ablation_study.csv

=========================================================
"""

import os

import torch
import pandas as pd

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
# ISOLATED ABLATION EXPERIMENT MANAGER
# =========================================================

class AblationExperimentManager:
    """
    Experiment manager specifically for one ablation model.

    This class follows the same directory attributes used by
    the existing Trainer and ExperimentManager.

    It does not modify utils.config.EXPERIMENT_NAME.
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
# METRIC EVALUATION
# =========================================================

def evaluate_checkpoint(
        model,
        checkpoint,
        dataset,
        device
):
    """
    Evaluate a trained checkpoint over the supplied dataset.

    Returns
    -------
    dict
        MAE, RMSE, PSNR, SNR and SSIM.
    """

    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = Predictor(

        model=model,

        checkpoint=checkpoint,

        device=device

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
            "Evaluation dataset is empty."
        )

    # -----------------------------------------------------
    # Evaluate every sample
    # -----------------------------------------------------

    for index in range(num_samples):

        print(
            f"Evaluating sample "
            f"{index + 1}/{num_samples}"
        )

        # -------------------------------------------------
        # Dataset convention
        # -------------------------------------------------

        (
            input_cube,
            target_cube,
            mask,
            velocity_model
        ) = dataset[index]

        # -------------------------------------------------
        # Prediction
        #
        # Predictor convention:
        #
        # reconstruction
        # travel_time
        # uncertainty
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            uncertainty
        ) = predictor.predict(
            input_cube
        )

        # -------------------------------------------------
        # Target batch dimension
        # -------------------------------------------------

        target_batch = (
            target_cube.unsqueeze(0)
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        total_mae += mae(
            reconstruction,
            target_batch
        ).item()

        total_rmse += rmse(
            reconstruction,
            target_batch
        ).item()

        total_psnr += psnr(
            reconstruction,
            target_batch
        ).item()

        total_snr += snr(
            reconstruction,
            target_batch
        ).item()

        total_ssim += ssim(
            reconstruction,
            target_batch
        ).item()

    # =====================================================
    # Average metrics
    # =====================================================

    return {

        "MAE":
            total_mae / num_samples,

        "RMSE":
            total_rmse / num_samples,

        "PSNR":
            total_psnr / num_samples,

        "SNR":
            total_snr / num_samples,

        "SSIM":
            total_ssim / num_samples

    }


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

    Each model receives an isolated experiment directory.
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

        "outputs",

        EXPERIMENT_NAME,

        "ablation",

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

    if not os.path.exists(checkpoint):

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

    results = []

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

            checkpoint = os.path.join(

                "outputs",

                EXPERIMENT_NAME,

                "checkpoints",

                "best_model.pth"

            )

            if not os.path.exists(checkpoint):

                raise FileNotFoundError(

                    "\nFull Model checkpoint not found:\n"
                    f"{checkpoint}\n\n"
                    "The existing synthetic_training "
                    "experiment must contain "
                    "best_model.pth before the ablation "
                    "study can be performed."

                )

            print()
            print(
                "Using existing Full Model checkpoint:"
            )

            print(
                checkpoint
            )

            # -------------------------------------------------
            # Build Full Model architecture
            # -------------------------------------------------

            model = Network3D(

                use_attention=True,

                use_residual=True,

                use_uncertainty=True

            )

            # -------------------------------------------------
            # Evaluate existing Full Model
            # -------------------------------------------------

            metrics = evaluate_checkpoint(

                model=model,

                checkpoint=checkpoint,

                dataset=val_dataset,

                device=device

            )

        # =================================================
        # ABLATION MODELS
        # =================================================

        else:

            # -------------------------------------------------
            # Train ablation model
            # -------------------------------------------------

            checkpoint = train_ablation_model(

                model_name=model_name,

                settings=settings,

                train_loader=train_loader,

                val_loader=val_loader,

                device=device

            )

            # -------------------------------------------------
            # Build same architecture for evaluation
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Evaluate ablation checkpoint
            # -------------------------------------------------

            metrics = evaluate_checkpoint(

                model=model,

                checkpoint=checkpoint,

                dataset=val_dataset,

                device=device

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

        results.append(
            metrics
        )

        # =================================================
        # DISPLAY MODEL RESULTS
        # =================================================

        print()
        print(
            f"{model_name} RESULTS"
        )

        print(
            "-" * 50
        )

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
    # CREATE DATAFRAME
    # =====================================================

    dataframe = pd.DataFrame(
        results
    )

    # =====================================================
    # ORDER COLUMNS
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

        "SSIM"

    ]

    dataframe = dataframe[
        columns
    ]

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    os.makedirs(

        REPORT_DIR,

        exist_ok=True

    )

    output_file = os.path.join(

        REPORT_DIR,

        "ablation_study.csv"

    )

    dataframe.to_csv(

        output_file,

        index=False

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

    print()
    print(
        "Results saved:"
    )

    print(
        output_file
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