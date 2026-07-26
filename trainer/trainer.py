"""
=========================================================
Trainer
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""
import os

import torch

from torch.utils.tensorboard import SummaryWriter

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

        self.writer = SummaryWriter(
            log_dir="runs/physics_informed_seismic"
        )

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.model.to(device)

        self.checkpoint_directory = "checkpoints"

        self.best_loss = float("inf")

        os.makedirs(

            self.checkpoint_directory,

            exist_ok=True

        )
        self.best_loss = float("inf")

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

                running_total += losses["total"].item()

                running_mae += losses["mae"].item()

                running_physics += losses["physics"].item()

                running_uncertainty += losses["uncertainty"].item()

                running_ssim += losses["ssim"].item()
            # ----------------------------------------
            # Average validation losses
            # ----------------------------------------

            num_batches = len(dataloader)

            return {

                "total": running_total / num_batches,

                "mae": running_mae / num_batches,

                "physics": running_physics / num_batches,

                "uncertainty": running_uncertainty / num_batches,

                "ssim": running_ssim / num_batches

            }



    def save_checkpoint(

            self,

            epoch,

            loss

    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

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

        print(

            f"Latest checkpoint saved (Epoch {epoch})"

        )

        # ----------------------------------------
        # Save best model
        # ----------------------------------------

        if loss < self.best_loss:
            self.best_loss = loss

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

        self.best_loss = checkpoint["loss"]

        start_epoch = checkpoint["epoch"]

        print(

            f"Checkpoint loaded successfully."

        )

        print(

            f"Resuming from Epoch {start_epoch}"

        )

        print(

            f"Best Loss = {self.best_loss:.6f}"

        )

        return start_epoch

    def fit(

            self,

            train_dataloader,

            validation_dataloader,

            epochs

    ):

        for epoch in range(epochs):
            train_losses = self.train_epoch(

                train_dataloader

            )
            validation_losses = self.validate_epoch(

                validation_dataloader

            )
            self.writer.add_scalar(

                "Loss/Train_Total",

                train_losses["total"],

                epoch

            )
            self.writer.add_scalar(
                "Loss/Train_MAE",
                train_losses["mae"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_Physics",
                train_losses["physics"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_Uncertainty",
                train_losses["uncertainty"],
                epoch
            )

            self.writer.add_scalar(
                "Loss/Train_SSIM",
                train_losses["ssim"],
                epoch
            )

            self.writer.add_scalar(

                "Loss/Validation_Total",

                validation_losses["total"],

                epoch

            )
            self.writer.add_scalar(

                "Loss/Validation_MAE",

                validation_losses["mae"],

                epoch

            )

            self.writer.add_scalar(

                "Loss/Validation_Physics",

                validation_losses["physics"],

                epoch

            )

            self.writer.add_scalar(

                "Loss/Validation_Uncertainty",

                validation_losses["uncertainty"],

                epoch

            )

            self.writer.add_scalar(

                "Loss/Validation_SSIM",

                validation_losses["ssim"],

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


            self.save_checkpoint(

                epoch=epoch + 1,

                loss=validation_losses["total"]

            )
        self.writer.close()





