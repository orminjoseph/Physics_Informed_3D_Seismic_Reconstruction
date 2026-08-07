"""
=========================================================
Deep Ensemble Predictor
=========================================================

Uses multiple trained models to estimate
epistemic uncertainty.

Author: Ormin Joseph
=========================================================
"""

import torch


class EnsemblePredictor:

    def __init__(
            self,
            model_class,
            checkpoint_list,
            device
    ):

        self.device = device
        self.models = []

        for checkpoint in checkpoint_list:

            model = model_class().to(device)

            checkpoint_data = torch.load(
                checkpoint,
                map_location=device
            )

            model.load_state_dict(
                checkpoint_data["model_state_dict"]
            )

            model.eval()

            self.models.append(model)

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

        with torch.no_grad():

            for model in self.models:

                reconstruction, _ = model(
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

        ensemble_uncertainty = predictions.std(
            dim=0
        )

        return (
            mean_prediction.cpu(),
            ensemble_uncertainty.cpu()
        )