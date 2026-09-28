"""
======================================================================
Evaluation Metrics
======================================================================

Performance metrics for seismic data reconstruction.

This module provides standardized quantitative metrics for evaluating
reconstructed seismic volumes against their corresponding reference
(target) volumes.

Metrics
-------
Global reconstruction metrics
    1. MAE
    2. MSE
    3. RMSE
    4. Relative L2 Error
    5. SNR
    6. PSNR
    7. SSIM

Regional reconstruction metrics
    8. Missing-region MAE
    9. Missing-region RMSE
   10. Observed-region MAE
   11. Observed-region RMSE

Uncertainty metrics
   12. Aleatoric Variance
   13. Epistemic Variance
   14. Predictive Variance
   15. Predictive Standard Deviation

Methodological conventions
--------------------------
Seismic amplitudes are assumed to be normalized to:

    [-1, 1]

Therefore:

    data_range = 2.0

The PSNR implementation uses this fixed range rather than deriving
the peak value independently from each target volume. This ensures
that PSNR values remain comparable across reconstruction methods,
samples, experiments, and ablation studies.

SSIM also uses the same fixed data range.

Mask convention
---------------
    1 = observed
    0 = missing

Uncertainty convention
----------------------
The network predicts:

    log_variance = log(sigma^2)

Therefore:

    aleatoric_variance = exp(log_variance)

For MC-Dropout predictions:

    epistemic_variance
        = Var(reconstruction_samples)

Predictive variance is defined as:

    predictive_variance
        = aleatoric_variance + epistemic_variance

Predictive standard deviation is:

    predictive_std
        = sqrt(predictive_variance)

Author: Ormin Joseph
======================================================================
"""

import torch

from pytorch_msssim import ssim


class EvaluationMetrics:
    """
    Collection of standardized evaluation metrics for seismic
    reconstruction.

    All methods return PyTorch tensors so that they can be used
    consistently during evaluation and reporting.
    """

    # ==============================================================
    # STANDARDIZED SEISMIC AMPLITUDE RANGE
    # ==============================================================

    # The project uses normalized seismic amplitudes in [-1, 1].
    #
    # Therefore:
    #
    #     DATA_RANGE = 1 - (-1)
    #                = 2
    #
    # DATA_RANGE is deliberately defined here because it is not
    # currently present in utils.config.py.
    DATA_RANGE = 2.0

    # Small numerical constant used to prevent division by zero
    # and logarithm-of-zero operations.
    EPSILON = 1.0e-8

    # ==============================================================
    # INTERNAL MASK VALIDATION
    # ==============================================================

    @staticmethod
    def _validate_mask(
        mask,
        reference
    ):
        """
        Validate a reconstruction evaluation mask.

        Mask convention:

            1 = observed
            0 = missing

        Parameters
        ----------
        mask : torch.Tensor
            Binary observation mask.

        reference : torch.Tensor
            Tensor whose shape must match the mask.

        Raises
        ------
        TypeError
            If mask is not a torch.Tensor.

        ValueError
            If mask shape differs from reference or mask is not binary.
        """

        if not isinstance(
            mask,
            torch.Tensor
        ):
            raise TypeError(
                "mask must be a torch.Tensor."
            )

        if mask.shape != reference.shape:
            raise ValueError(
                "Mask shape must match the reference tensor shape. "
                f"Mask: {tuple(mask.shape)}, "
                f"Reference: {tuple(reference.shape)}"
            )

        if not torch.isfinite(mask).all():
            raise ValueError(
                "Mask contains NaN or Inf values."
            )

        if not torch.all(
            (mask == 0) | (mask == 1)
        ):
            raise ValueError(
                "Mask must contain only 0 and 1."
            )

    # ==============================================================
    # MAE
    # ==============================================================

    @staticmethod
    def mae(
        prediction,
        target
    ):
        """
        Mean Absolute Error (MAE).

        Lower values indicate smaller reconstruction error.
        """

        return torch.mean(
            torch.abs(
                prediction - target
            )
        )

    # ==============================================================
    # MSE
    # ==============================================================

    @staticmethod
    def mse(
        prediction,
        target
    ):
        """
        Mean Squared Error (MSE).

        MSE penalizes larger reconstruction errors more strongly
        because the residual is squared.
        """

        return torch.mean(
            (
                prediction - target
            ) ** 2
        )

    # ==============================================================
    # RMSE
    # ==============================================================

    @staticmethod
    def rmse(
        prediction,
        target
    ):
        """
        Root Mean Squared Error (RMSE).

        RMSE is expressed in the same amplitude units as the
        seismic data.
        """

        return torch.sqrt(
            EvaluationMetrics.mse(
                prediction,
                target
            )
            +
            EvaluationMetrics.EPSILON
        )

    # ==============================================================
    # RELATIVE L2 ERROR
    # ==============================================================

    @staticmethod
    def relative_error(
        prediction,
        target
    ):
        """
        Relative L2 Reconstruction Error.

        Defined as:

            ||prediction - target||_2
            --------------------------------
                    ||target||_2

        Lower values indicate smaller relative reconstruction error.
        """

        numerator = torch.norm(
            prediction - target
        )

        denominator = (
            torch.norm(target)
            +
            EvaluationMetrics.EPSILON
        )

        return (
            numerator
            /
            denominator
        )

    # ==============================================================
    # SNR
    # ==============================================================

    @staticmethod
    def snr(
        prediction,
        target
    ):
        """
        Signal-to-Noise Ratio (SNR).

        Defined as:

            SNR = 10 log10(
                ||target||² /
                ||target - prediction||²
            )

        Result is expressed in decibels (dB).

        Higher values indicate better reconstruction quality.
        """

        signal_power = torch.sum(
            target ** 2
        )

        reconstruction_error_power = torch.sum(
            (
                target - prediction
            ) ** 2
        )

        return 10.0 * torch.log10(
            (
                signal_power
                +
                EvaluationMetrics.EPSILON
            )
            /
            (
                reconstruction_error_power
                +
                EvaluationMetrics.EPSILON
            )
        )

    # ==============================================================
    # PSNR
    # ==============================================================

    @staticmethod
    def psnr(
        prediction,
        target,
        data_range=None
    ):
        """
        Peak Signal-to-Noise Ratio (PSNR).

        Defined as:

            PSNR = 10 log10(
                data_range² / MSE
            )

        The project-standard data range is:

            data_range = 2.0

        corresponding to normalized seismic amplitudes in [-1, 1].

        A fixed range is used so PSNR remains comparable across
        methods, samples, experiments, and ablation studies.
        """

        if data_range is None:
            data_range = (
                EvaluationMetrics.DATA_RANGE
            )

        if data_range <= 0:
            raise ValueError(
                "data_range must be greater than zero."
            )

        mse_value = (
            EvaluationMetrics.mse(
                prediction,
                target
            )
        )

        return 10.0 * torch.log10(
            (
                data_range ** 2
            )
            /
            (
                mse_value
                +
                EvaluationMetrics.EPSILON
            )
        )

    # ==============================================================
    # SSIM
    # ==============================================================

    @staticmethod
    def ssim(
        prediction,
        target,
        data_range=None
    ):
        """
        Structural Similarity Index (SSIM).

        The project-standard data range is 2.0 because seismic
        amplitudes are normalized to [-1, 1].
        """

        if data_range is None:
            data_range = (
                EvaluationMetrics.DATA_RANGE
            )

        if data_range <= 0:
            raise ValueError(
                "data_range must be greater than zero."
            )

        return ssim(
            prediction,
            target,
            data_range=data_range,
            size_average=True
        )

    # ==============================================================
    # MISSING-REGION MAE
    # ==============================================================

    @staticmethod
    def missing_mae(
        prediction,
        target,
        mask
    ):
        """
        Mean Absolute Error calculated only within missing voxels.

        Mask convention:

            mask = 0 -> missing

        This metric is particularly important for seismic
        reconstruction because global error can be influenced
        substantially by already-observed samples.
        """

        EvaluationMetrics._validate_mask(
            mask,
            target
        )

        missing_region = (
            mask == 0
        )

        if not torch.any(
            missing_region
        ):
            return torch.zeros(
                (),
                dtype=prediction.dtype,
                device=prediction.device
            )

        absolute_error = torch.abs(
            prediction - target
        )

        return torch.mean(
            absolute_error[
                missing_region
            ]
        )

    # ==============================================================
    # MISSING-REGION RMSE
    # ==============================================================

    @staticmethod
    def missing_rmse(
        prediction,
        target,
        mask
    ):
        """
        Root Mean Squared Error calculated only within missing
        voxels.

        Mask convention:

            mask = 0 -> missing
        """

        EvaluationMetrics._validate_mask(
            mask,
            target
        )

        missing_region = (
            mask == 0
        )

        if not torch.any(
            missing_region
        ):
            return torch.zeros(
                (),
                dtype=prediction.dtype,
                device=prediction.device
            )

        squared_error = (
            prediction - target
        ) ** 2

        return torch.sqrt(
            torch.mean(
                squared_error[
                    missing_region
                ]
            )
            +
            EvaluationMetrics.EPSILON
        )

    # ==============================================================
    # OBSERVED-REGION MAE
    # ==============================================================

    @staticmethod
    def observed_mae(
        prediction,
        target,
        mask
    ):
        """
        Mean Absolute Error calculated only within observed voxels.

        Mask convention:

            mask = 1 -> observed
        """

        EvaluationMetrics._validate_mask(
            mask,
            target
        )

        observed_region = (
            mask == 1
        )

        if not torch.any(
            observed_region
        ):
            return torch.zeros(
                (),
                dtype=prediction.dtype,
                device=prediction.device
            )

        absolute_error = torch.abs(
            prediction - target
        )

        return torch.mean(
            absolute_error[
                observed_region
            ]
        )

    # ==============================================================
    # OBSERVED-REGION RMSE
    # ==============================================================

    @staticmethod
    def observed_rmse(
        prediction,
        target,
        mask
    ):
        """
        Root Mean Squared Error calculated only within observed
        voxels.

        Mask convention:

            mask = 1 -> observed
        """

        EvaluationMetrics._validate_mask(
            mask,
            target
        )

        observed_region = (
            mask == 1
        )

        if not torch.any(
            observed_region
        ):
            return torch.zeros(
                (),
                dtype=prediction.dtype,
                device=prediction.device
            )

        squared_error = (
            prediction - target
        ) ** 2

        return torch.sqrt(
            torch.mean(
                squared_error[
                    observed_region
                ]
            )
            +
            EvaluationMetrics.EPSILON
        )

    # ==============================================================
    # ALEATORIC UNCERTAINTY
    # ==============================================================

    @staticmethod
    def uncertainty(
        log_variance
    ):
        """
        Estimate mean aleatoric variance from log variance.

        The network predicts:

            log_variance = log(sigma²)

        Therefore:

            variance = exp(log_variance)

        The returned value is the spatial mean variance.

        This method is retained for backward compatibility with
        earlier evaluation code.
        """

        variance = torch.exp(
            log_variance
        )

        return torch.mean(
            variance
        )

    # ==============================================================
    # ALEATORIC VARIANCE
    # ==============================================================

    @staticmethod
    def aleatoric_variance(
        log_variance
    ):
        """
        Convert predicted log variance to aleatoric variance.

        Defined as:

            sigma²_aleatoric = exp(log_variance)

        Returns the full spatial variance tensor rather than its
        spatial mean.
        """

        variance = torch.exp(
            log_variance
        )

        if torch.any(
            variance < 0
        ):
            raise RuntimeError(
                "Aleatoric variance contains negative values."
            )

        return variance

    # ==============================================================
    # EPISTEMIC VARIANCE
    # ==============================================================

    @staticmethod
    def epistemic_variance(
        reconstruction_samples
    ):
        """
        Estimate epistemic variance from MC-Dropout reconstruction
        samples.

        Parameters
        ----------
        reconstruction_samples : torch.Tensor
            MC reconstruction samples with shape:

                [N, B, C, D, H, W]

            where N is the number of stochastic forward passes.

        Returns
        -------
        torch.Tensor
            Spatial epistemic variance with shape:

                [B, C, D, H, W]
        """

        if not isinstance(
            reconstruction_samples,
            torch.Tensor
        ):
            raise TypeError(
                "reconstruction_samples must be a torch.Tensor."
            )

        if reconstruction_samples.ndim < 2:
            raise ValueError(
                "reconstruction_samples must contain an MC "
                "sample dimension."
            )

        if reconstruction_samples.shape[0] < 2:
            return torch.zeros_like(
                reconstruction_samples[0]
            )

        if not torch.isfinite(
            reconstruction_samples
        ).all():

            raise RuntimeError(
                "reconstruction_samples contains NaN or Inf."
            )

        return torch.var(
            reconstruction_samples,
            dim=0,
            unbiased=False
        )

    # ==============================================================
    # PREDICTIVE VARIANCE
    # ==============================================================

    @staticmethod
    def predictive_variance(
        aleatoric_variance,
        epistemic_variance
    ):
        """
        Compute total predictive variance.

        Defined as:

            predictive_variance
                = aleatoric_variance
                + epistemic_variance

        Both components must be non-negative.
        """

        if not isinstance(
            aleatoric_variance,
            torch.Tensor
        ):
            raise TypeError(
                "aleatoric_variance must be a torch.Tensor."
            )

        if not isinstance(
            epistemic_variance,
            torch.Tensor
        ):
            raise TypeError(
                "epistemic_variance must be a torch.Tensor."
            )

        if (
            aleatoric_variance.shape
            !=
            epistemic_variance.shape
        ):
            raise ValueError(
                "Aleatoric and epistemic variance tensors must "
                "have identical shapes. "
                f"Aleatoric: "
                f"{tuple(aleatoric_variance.shape)}, "
                f"Epistemic: "
                f"{tuple(epistemic_variance.shape)}"
            )

        if torch.any(
            aleatoric_variance < 0
        ):
            raise ValueError(
                "Aleatoric variance cannot contain negative values."
            )

        if torch.any(
            epistemic_variance < 0
        ):
            raise ValueError(
                "Epistemic variance cannot contain negative values."
            )

        predictive_variance = (
            aleatoric_variance
            +
            epistemic_variance
        )

        if not torch.isfinite(
            predictive_variance
        ).all():

            raise RuntimeError(
                "Predictive variance contains NaN or Inf."
            )

        return predictive_variance

    # ==============================================================
    # PREDICTIVE STANDARD DEVIATION
    # ==============================================================

    @staticmethod
    def predictive_std(
        predictive_variance
    ):
        """
        Compute predictive standard deviation.

        Defined as:

            predictive_std
                = sqrt(predictive_variance)
        """

        if not isinstance(
            predictive_variance,
            torch.Tensor
        ):
            raise TypeError(
                "predictive_variance must be a torch.Tensor."
            )

        if torch.any(
            predictive_variance < 0
        ):
            raise ValueError(
                "Predictive variance cannot contain negative values."
            )

        predictive_std = torch.sqrt(
            predictive_variance
            +
            EvaluationMetrics.EPSILON
        )

        if not torch.isfinite(
            predictive_std
        ).all():

            raise RuntimeError(
                "Predictive standard deviation contains NaN or Inf."
            )

        return predictive_std