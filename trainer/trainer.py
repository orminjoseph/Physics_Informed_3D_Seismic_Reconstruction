"""
=========================================================
Trainer
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""
import os

import torch

import time

import csv

from utils.experiment_manager import ExperimentManager

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from metrics.reconstruction_metrics import (
    mae,
    mse,
    rmse,
    psnr,
    snr,
    ssim
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEBUG_VALIDATION = False

class Trainer:
    """
    Trainer class.
    """

    def __init__(

        self,

        model,

        criterion,

        optimizer,

        device

    ):
        # ------------------------------------------
        # TensorBoard
        # ------------------------------------------


        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.experiment = ExperimentManager()

        self.writer = SummaryWriter(

            log_dir=self.experiment.tensorboard

        )

        self.model.to(device)

        self.checkpoint_directory = self.experiment.checkpoints


        # ------------------------------------------
        # Experiment Logger
        # ------------------------------------------

        self.log_file = os.path.join(

            self.experiment.logs,

            "training_history.csv"

        )

        if not os.path.exists(self.log_file):
            with open(

                    self.log_file,

                    "w",

                    newline=""

            ) as file:
                writer = csv.writer(file)

                writer.writerow([

                    "Epoch",

                    "Train_Total",

                    "Validation_Total",

                    "Train_MAE",

                    "Validation_MAE",

                    "RMSE",

                    "PSNR",

                    "SNR",

                    "SSIM",

                    "Learning_Rate"

                ])

        # ------------------------------------------
        # Early Stopping
        # ------------------------------------------

        self.best_validation_loss = float("inf")

        self.patience = 10

        self.wait = 0

        # ------------------------------------------
        # Learning Rate Scheduler
        # ------------------------------------------

        self.scheduler = ReduceLROnPlateau(

            self.optimizer,

            mode="min",

            factor=0.5,

            patience=5,

        )


        self.best_epoch = 0

        self.best_metrics = {

            "MAE": 0.0,

            "RMSE": 0.0,

            "PSNR": 0.0,

            "SNR": 0.0,

            "SSIM": 0.0

        }

    def train_epoch(

            self,

            dataloader

    ):
        # ----------------------------------------
        # Set model to training mode
        # ----------------------------------------

        self.model.train()

        # ----------------------------------------
        # Running loss accumulators
        # ----------------------------------------

        running_total = 0.0
        running_mae = 0.0
        running_physics = 0.0
        running_uncertainty = 0.0
        running_ssim = 0.0

        # ----------------------------------------
        # Iterate over batches
        # ----------------------------------------

        for batch_index, batch in enumerate(dataloader):

            ground_truth = batch["ground_truth"].to(self.device)

            corrupted = batch["corrupted"].to(self.device)

            mask = batch["mask"].to(self.device)


            # Temporary normalized velocity model
            # (will later come from the dataset)

            velocity_model = torch.ones_like(
                ground_truth
            )
            self.optimizer.zero_grad()
            reconstruction, log_variance = self.model(
                corrupted
            )
            losses = self.criterion(

                reconstruction,

                ground_truth,

                velocity_model,

                log_variance

            )
            loss = losses["total"]

            loss.backward()

            self.optimizer.step()

            running_total += losses["total"].item()

            running_mae += losses["mae"].item()

            running_physics += losses["physics"].item()

            running_uncertainty += losses["uncertainty"].item()

            running_ssim += losses["ssim"].item()

        num_batches = len(dataloader)
        return {

            "total": running_total / num_batches,

            "mae": running_mae / num_batches,

            "physics": running_physics / num_batches,

            "uncertainty": running_uncertainty / num_batches,

            "ssim": running_ssim / num_batches

        }

    def validate_epoch(

            self,

            dataloader

    ):
        self.model.eval()

        running_total = 0.0
        running_mae = 0.0
        running_physics = 0.0
        running_uncertainty = 0.0
        running_ssim = 0.0

        running_metric_mae = 0.0
        running_metric_rmse = 0.0
        running_metric_psnr = 0.0
        running_metric_snr = 0.0
        running_metric_ssim = 0.0

        with torch.no_grad():

            for batch_index, batch in enumerate(dataloader):

                ground_truth = batch["ground_truth"].to(self.device)

                corrupted = batch["corrupted"].to(self.device)

                mask = batch["mask"].to(self.device)

                velocity_model = torch.ones_like(
                    ground_truth
                )

                reconstruction, log_variance = self.model(
                    corrupted
                )

                # Print only occasionally
                if DEBUG_VALIDATION and batch_index % 20 == 0:
                    print()

                    print("Validation Batch Statistics")
                    print("---------------------------")

                    print(
                        f"Prediction : "
                        f"min={reconstruction.min().item():.4f}, "
                        f"max={reconstruction.max().item():.4f}, "
                        f"mean={reconstruction.mean().item():.4f}"
                    )

                    print(
                        f"Ground Truth : "
                        f"min={ground_truth.min().item():.4f}, "
                        f"max={ground_truth.max().item():.4f}, "
                        f"mean={ground_truth.mean().item():.4f}"
                    )

                    print(
                        f"Log Variance : "
                        f"min={log_variance.min().item():.4f}, "
                        f"max={log_variance.max().item():.4f}, "
                        f"mean={log_variance.mean().item():.4f}"
                    )

                # Compute losses for EVERY batch
                losses = self.criterion(

                    reconstruction,

                    ground_truth,

                    velocity_model,

                    log_variance

                )
                if batch_index == 0:
                    uncertainty = torch.exp(
                        0.5 * log_variance
                    )

                    self.save_validation_visualization(

                        corrupted,

                        ground_truth,

                        reconstruction,

                        uncertainty,

                        self.current_epoch

                    )

                metric_mae = mae(reconstruction, ground_truth)

                metric_rmse = rmse(reconstruction, ground_truth)

                metric_psnr = psnr(reconstruction, ground_truth)

                metric_snr = snr(reconstruction, ground_truth)

                metric_ssim = ssim(reconstruction, ground_truth)

                running_total += losses["total"].item()

                running_mae += losses["mae"].item()

                running_physics += losses["physics"].item()

                running_uncertainty += losses["uncertainty"].item()

                running_ssim += losses["ssim"].item()

                running_metric_mae += metric_mae.item()

                running_metric_rmse += metric_rmse.item()

                running_metric_psnr += metric_psnr.item()

                running_metric_snr += metric_snr.item()

                running_metric_ssim += metric_ssim.item()
            # ----------------------------------------
            # Average validation losses
            # ----------------------------------------

            num_batches = len(dataloader)

            return {

                "total": running_total / num_batches,

                "mae": running_mae / num_batches,

                "physics": running_physics / num_batches,

                "uncertainty": running_uncertainty / num_batches,

                "ssim": running_ssim / num_batches,

                "metric_mae": running_metric_mae / num_batches,

                "metric_rmse": running_metric_rmse / num_batches,

                "metric_psnr": running_metric_psnr / num_batches,

                "metric_snr": running_metric_snr / num_batches,

                "metric_ssim": running_metric_ssim / num_batches
            }

    def save_validation_visualization(

            self,

            corrupted,

            ground_truth,

            reconstruction,

            uncertainty,

            epoch

    ):
        """
        Save validation comparison figure.
        """

        import os

        os.makedirs(
            "outputs/training_progress",
            exist_ok=True
        )

        corrupted = corrupted[0, 0].cpu().numpy()

        ground_truth = ground_truth[0, 0].cpu().numpy()

        reconstruction = reconstruction[0, 0].detach().cpu().numpy()

        uncertainty = uncertainty[0, 0].detach().cpu().numpy()

        # Middle slice

        middle = corrupted.shape[0] // 2

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 8)
        )

        axes[0, 0].imshow(
            corrupted[middle],
            cmap="gray"
        )
        axes[0, 0].set_title("Corrupted")

        axes[0, 1].imshow(
            ground_truth[middle],
            cmap="gray"
        )
        axes[0, 1].set_title("Ground Truth")

        axes[1, 0].imshow(
            reconstruction[middle],
            cmap="gray"
        )
        axes[1, 0].set_title("Reconstruction")

        axes[1, 1].imshow(
            uncertainty[middle],
            cmap="hot"
        )
        axes[1, 1].set_title("Uncertainty")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(

            f"outputs/training_progress/"
            f"Epoch_{epoch + 1:03d}.png"

        )

        plt.close()

    def save_checkpoint(

            self,

            epoch,

            loss

    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "scheduler_state_dict": self.scheduler.state_dict(),

            "best_validation_loss": self.best_validation_loss,

            "loss": loss

        }


        # ----------------------------------------
        # Save latest checkpoint
        # ----------------------------------------

        torch.save(

            checkpoint,

            os.path.join(

                self.checkpoint_directory,

                "latest_checkpoint.pth"

            )


        )

        # ----------------------------------------
        # Save periodic checkpoint
        # ----------------------------------------

        if epoch % 10 == 0:
            torch.save(

                checkpoint,

                os.path.join(

                    self.checkpoint_directory,

                    f"epoch_{epoch:04d}.pth"

                )

            )

            print(

                f"Archived checkpoint saved (Epoch {epoch})"

            )

        print(

            f"Latest checkpoint saved (Epoch {epoch})"

        )

        # ----------------------------------------
        # Save best model
        # ----------------------------------------

        if loss < self.best_validation_loss:
            self.best_validation_loss = loss

            torch.save(

                checkpoint,

                os.path.join(

                    self.checkpoint_directory,

                    "best_model.pth"

                )

            )

            print(

                f"New best model saved! Loss = {loss:.6f}"

            )

    def load_checkpoint(

            self,

            checkpoint_path

    ):

        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        self.optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        if "scheduler_state_dict" in checkpoint:

            self.scheduler.load_state_dict(

                checkpoint["scheduler_state_dict"]

            )

        else:

            print("Warning: Scheduler state not found in checkpoint.")

        # ------------------------------------------
        # Best validation loss
        # ------------------------------------------

        if "best_validation_loss" in checkpoint:

            self.best_validation_loss = checkpoint["best_validation_loss"]

        else:

            self.best_validation_loss = checkpoint["loss"]

        # ------------------------------------------
        # Best epoch
        # ------------------------------------------

        if "best_epoch" in checkpoint:

            self.best_epoch = checkpoint["best_epoch"]

        else:

            self.best_epoch = checkpoint["epoch"]

        # ------------------------------------------
        # Best metrics
        # ------------------------------------------

        if "best_metrics" in checkpoint:

            self.best_metrics = checkpoint["best_metrics"]

        else:

            self.best_metrics = {

                "MAE": 0.0,

                "RMSE": 0.0,

                "PSNR": 0.0,

                "SNR": 0.0,

                "SSIM": 0.0

            }

        start_epoch = checkpoint["epoch"]

        print()

        print("=" * 60)

        print("Checkpoint loaded successfully.")

        print(f"Resuming from Epoch {start_epoch}")

        print(f"Best Epoch            : {self.best_epoch}")

        print(f"Best Validation Loss  : {self.best_validation_loss:.6f}")

        print("=" * 60)

        return start_epoch

    def fit(

            self,

            train_dataloader,

            validation_dataloader,

            epochs,

            resume=True

    ):

        start_time = time.time()

        start_epoch = 0

        latest_checkpoint = "checkpoints/latest_checkpoint.pth"

        if resume and os.path.exists(latest_checkpoint):
            start_epoch = self.load_checkpoint(
                latest_checkpoint
            )

        for epoch in range(start_epoch, epochs):

            train_losses = self.train_epoch(

                train_dataloader

            )

            self.current_epoch = epoch

            validation_losses = self.validate_epoch(

                validation_dataloader

            )

            self.scheduler.step(
                validation_losses["total"]
            )

            self.writer.add_scalar(

                "Loss/Train_Total",

                train_losses["total"],

                epoch

            )

            self.writer.add_scalar(

                "Loss/Validation_Total",

                validation_losses["total"],

                epoch

            )

            self.writer.add_scalar(
                "Metrics/Validation_MAE",
                validation_losses["metric_mae"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_RMSE",
                validation_losses["metric_rmse"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_PSNR",
                validation_losses["metric_psnr"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_SNR",
                validation_losses["metric_snr"],
                epoch
            )

            self.writer.add_scalar(
                "Metrics/Validation_SSIM",
                validation_losses["metric_ssim"],
                epoch
            )
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar(

                "Learning_Rate",

                current_lr,

                epoch

            )

            print()

            print("=" * 60)

            print(f"Epoch {epoch + 1}/{epochs}")

            print("=" * 60)

            print("Training")

            print("-" * 30)


            print(f"Total Loss       : {train_losses['total']:.6f}")

            print(f"MAE Loss         : {train_losses['mae']:.6f}")

            print(f"Physics Loss     : {train_losses['physics']:.6f}")

            print(f"Uncertainty Loss : {train_losses['uncertainty']:.6f}")

            print(f"SSIM Loss        : {train_losses['ssim']:.6f}")

            print()


            print("Validation")
            print("-" * 30)

            print(f"Total Loss       : {validation_losses['total']:.6f}")

            print(f"MAE Loss         : {validation_losses['mae']:.6f}")

            print(f"Physics Loss     : {validation_losses['physics']:.6f}")

            print(f"Uncertainty Loss : {validation_losses['uncertainty']:.6f}")

            print(f"SSIM Loss        : {validation_losses['ssim']:.6f}")

            print()

            print("Reconstruction Metrics")
            print("-" * 30)

            print(f"MAE   : {validation_losses['metric_mae']:.6f}")

            print(f"RMSE  : {validation_losses['metric_rmse']:.6f}")

            print(f"PSNR  : {validation_losses['metric_psnr']:.3f} dB")

            print(f"SNR   : {validation_losses['metric_snr']:.3f} dB")

            print(f"SSIM  : {validation_losses['metric_ssim']:.6f}")

            print(f"Learning Rate  : {current_lr:.8f}")

            # ------------------------------------------
            # Save experiment history
            # ------------------------------------------

            with open(
                    self.log_file,
                    "a",
                    newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    epoch + 1,

                    train_losses["total"],

                    validation_losses["total"],

                    train_losses["mae"],

                    validation_losses["mae"],

                    validation_losses["metric_rmse"],

                    validation_losses["metric_psnr"],

                    validation_losses["metric_snr"],

                    validation_losses["metric_ssim"],

                    current_lr

                ])


            self.save_checkpoint(

                epoch=epoch + 1,

                loss=validation_losses["total"]

            )

            # ------------------------------------------
            # Early Stopping
            # ------------------------------------------

            if validation_losses["total"] < self.best_validation_loss:

                self.best_validation_loss = validation_losses["total"]

                self.best_epoch = epoch + 1

                self.best_metrics = {

                    "MAE": validation_losses["metric_mae"],

                    "RMSE": validation_losses["metric_rmse"],

                    "PSNR": validation_losses["metric_psnr"],

                    "SNR": validation_losses["metric_snr"],

                    "SSIM": validation_losses["metric_ssim"]

                }

                self.wait = 0

            else:

                self.wait += 1

                print(
                    f"Early Stopping Counter: "
                    f"{self.wait}/{self.patience}"
                )

                if self.wait >= self.patience:
                    print()

                    print("=" * 60)

                    print("Early stopping triggered.")

                    print("=" * 60)

                    break

        training_time = time.time() - start_time

        self.save_training_summary(

            best_epoch=self.best_epoch,

            best_validation_loss=self.best_validation_loss,

            best_metrics=self.best_metrics,

            training_time=training_time

        )

        self.writer.close()



    def save_training_summary(

            self,

            best_epoch,

            best_validation_loss,

            best_metrics,

            training_time

    ):
        report_file = os.path.join(

            self.experiment.reports,

            "training_summary.txt"

        )

        with open(report_file, "w") as file:

            file.write("Training Summary\n")

            file.write("=" * 40 + "\n\n")

            file.write(f"Best Epoch            : {best_epoch}\n")
            file.write(f"Best Validation Loss  : {best_validation_loss:.6f}\n\n")
            file.write(f"Best MAE              : {best_metrics['MAE']:.6f}\n")
            file.write(f"Best RMSE             : {best_metrics['RMSE']:.6f}\n")
            file.write(f"Best PSNR             : {best_metrics['PSNR']:.3f} dB\n")
            file.write(f"Best SNR              : {best_metrics['SNR']:.3f} dB\n")
            file.write(f"Best SSIM             : {best_metrics['SSIM']:.6f}\n\n")

            file.write(f"Training Time         : {training_time:.2f} seconds\n")







