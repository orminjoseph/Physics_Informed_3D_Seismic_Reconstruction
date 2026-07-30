"""
=========================================================
Model Evaluator
=========================================================

Evaluates a trained model on an entire dataset.

Author: Ormin Joseph
=========================================================
"""

import torch

from evaluation.metrics import EvaluationMetrics


class Evaluator:

    def __init__(self, model, device):

        self.model = model.to(device)

        self.device = device

    def evaluate(
        self,
        dataloader
    ):

        self.model.eval()

        results = {

            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "relative_error": 0.0,
            "snr": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "uncertainty": 0.0

        }

        num_batches = len(dataloader)

        with torch.no_grad():

            for batch in dataloader:
                input_cube = batch["input"].unsqueeze(1).to(
                    self.device
                )

                target_cube = batch["target"].unsqueeze(1).to(
                    self.device
                )

                mask = batch["mask"].unsqueeze(1).to(
                    self.device
                )

                prediction, log_variance = self.model(
                    input_cube
                )

                results["mae"] += EvaluationMetrics.mae(
                    prediction,
                    target_cube
                ).item()

                results["mse"] += EvaluationMetrics.mse(
                    prediction,
                    target_cube
                ).item()

                results["rmse"] += EvaluationMetrics.rmse(
                    prediction,
                    target_cube
                ).item()

                results["relative_error"] += (
                    EvaluationMetrics.relative_error(
                        prediction,
                        target_cube
                    ).item()
                )

                results["snr"] += EvaluationMetrics.snr(
                    prediction,
                    target_cube
                ).item()

                results["psnr"] += EvaluationMetrics.psnr(
                    prediction,
                    target_cube
                ).item()

                results["ssim"] += EvaluationMetrics.ssim(
                    prediction,
                    target_cube
                ).item()

                results["uncertainty"] += (
                    EvaluationMetrics.uncertainty(
                        log_variance
                    ).item()
                )

        for key in results:

            results[key] /= num_batches

        return results