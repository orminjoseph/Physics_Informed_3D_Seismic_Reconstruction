"""
=========================================================
Trainer
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
=========================================================
"""
import os

import torch


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

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = device

        self.model.to(device)

        self.checkpoint_directory = "checkpoints"

        os.makedirs(

            self.checkpoint_directory,

            exist_ok=True

        )

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

        for batch in dataloader:
            ground_truth, corrupted, mask = batch

            ground_truth = ground_truth.to(self.device)

            corrupted = corrupted.to(self.device)

            mask = mask.to(self.device)


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

        torch.save(

            checkpoint,

            os.path.join(

                self.checkpoint_directory,

                f"checkpoint_epoch_{epoch}.pth"

            )

        )

        print(

            f"Checkpoint saved: Epoch {epoch}"

        )



    def fit(

            self,

            dataloader,

            epochs

    ):
        for epoch in range(epochs):
            train_losses = self.train_epoch(
                dataloader
            )
            print()

            print("=" * 60)

            print(f"Epoch {epoch + 1}/{epochs}")

            print("=" * 60)

            print(f"Total Loss       : {train_losses['total']:.6f}")

            print(f"MAE Loss         : {train_losses['mae']:.6f}")

            print(f"Physics Loss     : {train_losses['physics']:.6f}")

            print(f"Uncertainty Loss : {train_losses['uncertainty']:.6f}")

            print(f"SSIM Loss        : {train_losses['ssim']:.6f}")

            self.save_checkpoint(

                epoch=epoch + 1,

                loss=train_losses["total"]

            )





