"""
=========================================================
trainer
=========================================================

Handles training for the Physics-Informed 3D U-Net.

Author: Ormin Joseph
=========================================================
"""
import os
import torch
import time

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        device
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        # --------------------------------------------------
        # Training State
        # --------------------------------------------------

        self.best_validation_loss = float("inf")

        self.best_epoch = 0

        self.current_epoch = 0

        self.early_stop_counter = 0


        self.patience = 10
        self.history = {
            "mae": [],
            "physics": [],
            "uncertainty": [],
            "ssim": [],
            "total": []
        }

        self.validation_history = {
            "mae": [],
            "physics": [],
            "uncertainty": [],
            "ssim": [],
            "total": []
        }

    def train_step(
            self,
            input_cube,
            target_cube,
            velocity_model
    ):
        """
        Perform one optimization step.
        """

        self.model.train()

        input_cube = input_cube.to(self.device)

        target_cube = target_cube.to(self.device)

        velocity_model = velocity_model.to(self.device)

        self.optimizer.zero_grad()

        reconstruction, log_variance = self.model(
            input_cube
        )

        total_loss, losses = self.loss_function(
            reconstruction=reconstruction,
            target=target_cube,
            log_variance=log_variance,
            velocity_model=velocity_model
        )

        total_loss.backward()

        self.optimizer.step()

        return losses


    def train_epoch(
        self,
        dataloader
    ):
        """
        Train the model for one complete epoch.
        """

        epoch_losses = {
            "mae": 0.0,
            "physics": 0.0,
            "uncertainty": 0.0,
            "ssim": 0.0,
            "total": 0.0
        }

        num_batches = len(dataloader)

        for input_cube, target_cube, _, velocity_model in dataloader:

            losses = self.train_step(
                input_cube,
                target_cube,
                velocity_model
            )


            for key in epoch_losses:
                epoch_losses[key] += losses[key]

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Save losses into training history
        for key in self.history:
            self.history[key].append(
                epoch_losses[key]
            )

        return epoch_losses

    def validate_epoch(
            self,
            dataloader
    ):
        """
        Validate the model for one complete epoch.
        """

        self.model.eval()

        epoch_losses = {
            "mae": 0.0,
            "physics": 0.0,
            "uncertainty": 0.0,
            "ssim": 0.0,
            "total": 0.0
        }

        num_batches = len(dataloader)

        with torch.no_grad():

            for input_cube, target_cube, _, velocity_model in dataloader:

                input_cube = input_cube.to(self.device)

                target_cube = target_cube.to(self.device)

                velocity_model = velocity_model.to(self.device)

                reconstruction, log_variance = self.model(
                    input_cube
                )

                total_loss, losses = self.loss_function(
                    reconstruction=reconstruction,
                    target=target_cube,
                    log_variance=log_variance,
                    velocity_model=velocity_model
                )

                for key in epoch_losses:
                    epoch_losses[key] += losses[key]


        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Save validation losses
        for key in self.validation_history:
            self.validation_history[key].append(
                epoch_losses[key]
            )

        return epoch_losses

    def fit(
            self,
            train_loader,
            validation_loader,
            epochs
    ):
        """
        Train the model for multiple epochs.
        """

        training_start = time.time()



        for epoch in range(epochs):

            self.current_epoch = epoch + 1
            print("\n" + "=" * 60)

            print(
                f"Epoch {self.current_epoch}/{epochs}"
            )


            print("=" * 60)

            train_losses = self.train_epoch(
                train_loader
            )
            validation_losses = self.validate_epoch(
                validation_loader
            )
            print("\nTraining Losses")

            for name, value in train_losses.items():
                print(
                    f"{name:<15}: {value:.6f}"
                )
            print("\nValidation Losses")

            for name, value in validation_losses.items():
                print(
                    f"{name:<15}: {value:.6f}"
                )
            improved = False

            if validation_losses["total"] < self.best_validation_loss:
                improved = True

                self.best_validation_loss = validation_losses["total"]

                self.best_epoch = self.current_epoch

                self.save_best_model()

            if self.early_stop(improved):
                break

            self.save_checkpoint(
                f"checkpoints/checkpoint_epoch_{self.current_epoch}.pth",
                self.current_epoch
            )


        training_time = time.time() - training_start

        print("\nTraining completed.")

        print(f"Total Training Time : {training_time:.2f} seconds")


    def save_checkpoint(
            self,
            filename,
            epoch
    ):

        """
        Save model checkpoint.
        """

        # Create the directory if it does not exist
        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "history":
                self.history,

            "validation_history":
                self.validation_history
        }

        torch.save(
            checkpoint,
            filename
        )

        print(
            f"Checkpoint saved: {filename}"
        )

    def save_best_model(self):
        """
        Save the current model as the best model.
        """

        self.save_checkpoint(
            "checkpoints/best_model.pth",
            self.current_epoch
        )

        print(
            f"New Best Model "
            f"(Epoch {self.current_epoch})"
        )

    def early_stop(
            self,
            improved
    ):
        """
        Determine whether training should stop early.
        """

        if improved:
            self.early_stop_counter = 0

            return False

        self.early_stop_counter += 1

        print(
            f"No validation improvement "
            f"({self.early_stop_counter}/"
            f"{self.patience})"
        )

        if self.early_stop_counter >= self.patience:
            print("\n" + "=" * 60)

            print("Early stopping activated.")

            print(
                f"Best model was obtained at "
                f"Epoch {self.best_epoch}"
            )

            print("=" * 60)

            return True

        return False


    def load_checkpoint(
            self,
            filename
    ):
        """
        Load a previously saved checkpoint.
        """

        checkpoint = torch.load(
            filename,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.history = checkpoint["history"]

        self.validation_history = checkpoint[
            "validation_history"
        ]
        epoch = checkpoint["epoch"]

        print(
            f"Checkpoint loaded from Epoch {epoch}: {filename}"
        )

        return epoch
