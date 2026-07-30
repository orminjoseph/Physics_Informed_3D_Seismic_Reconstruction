"""
=========================================================
Monte Carlo Dropout Predictor
=========================================================

Performs uncertainty-aware seismic reconstruction.

Author: Ormin Joseph
=========================================================
"""

import torch


class MCDropoutPredictor:

    def __init__(
            self,
            model,
            checkpoint,
            device,
            num_samples=20
    ):

        self.device = device

        self.model = model.to(device)

        checkpoint_data = torch.load(
            checkpoint,
            map_location=device
        )

        self.model.load_state_dict(
            checkpoint_data["model_state_dict"]
        )

        self.num_samples = num_samples

    def enable_dropout(self):

        for module in self.model.modules():

            if module.__class__.__name__.startswith(
                    "Dropout"
            ):
                module.train()

    def predict(
            self,
            corrupted_cube
    ):

        if corrupted_cube.dim() == 4:

            corrupted_cube = corrupted_cube.unsqueeze(0)

        corrupted_cube = corrupted_cube.to(
            self.device
        )

        predictions = []

        self.model.eval()

        self.enable_dropout()

        with torch.no_grad():

            for _ in range(
                    self.num_samples
            ):

                reconstruction, _ = self.model(
                    corrupted_cube
                )

                predictions.append(
                    reconstruction
                )

        predictions = torch.stack(
            predictions,
            dim=0
        )

        mean_prediction = predictions.mean(
            dim=0
        )

        epistemic_uncertainty = predictions.std(
            dim=0
        )

        return (
            mean_prediction.cpu(),
            epistemic_uncertainty.cpu()
        )