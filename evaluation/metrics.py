"""
=========================================================
Evaluation Metrics
=========================================================

Performance metrics for seismic reconstruction.

Author: Ormin Joseph
=========================================================
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch.nn.functional as F
class EvaluationMetrics:
    """
    Computes reconstruction quality metrics.
    """

    @staticmethod
    def mae(prediction, target):
        """
        Mean Absolute Error
        """
        return torch.mean(
            torch.abs(prediction - target)
        )

    @staticmethod
    def mse(prediction, target):
        """
        Mean Squared Error
        """
        return torch.mean(
            (prediction - target) ** 2
        )

    @staticmethod
    def rmse(prediction, target):
        """
        Root Mean Squared Error
        """
        return torch.sqrt(
            EvaluationMetrics.mse(
                prediction,
                target
            )
        )

    @staticmethod
    def relative_error(prediction, target):
        """
        Relative L2 Error
        """
        numerator = torch.norm(
            prediction - target
        )

        denominator = torch.norm(
            target
        ) + 1e-8

        return numerator / denominator

    @staticmethod
    def snr(prediction, target):
        """
        Signal-to-Noise Ratio (dB)
        """

        signal = torch.sum(
            target ** 2
        )

        noise = torch.sum(
            (target - prediction) ** 2
        ) + 1e-8

        return 10 * torch.log10(
            signal / noise
        )

    @staticmethod
    def psnr(prediction, target):
        """
        Peak Signal-to-Noise Ratio
        """

        mse = EvaluationMetrics.mse(
            prediction,
            target
        )

        max_value = torch.max(target)

        return 20 * torch.log10(
            max_value
        ) - 10 * torch.log10(
            mse + 1e-8
        )

    @staticmethod
    def ssim(prediction, target):
        """
        Structural Similarity Index
        """

        return ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

    @staticmethod
    def uncertainty(log_variance):
        """
        Average predictive uncertainty.
        """

        variance = torch.exp(
            log_variance
        )

        return torch.mean(
            variance
        )