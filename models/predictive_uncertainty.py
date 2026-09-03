"""
=========================================================
Predictive Uncertainty Estimator
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Combines:

    1. Aleatoric uncertainty
       obtained from the heteroscedastic network output

    2. Epistemic uncertainty
       obtained from Monte Carlo Dropout

to estimate total predictive uncertainty.

Uncertainty decomposition
-------------------------

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

where:

    sigma_aleatoric^2
        = exp(log_variance)

and:

    sigma_epistemic^2
        = variance of MC Dropout predictions.

The estimator operates voxel-wise on 3D seismic volumes.

Tensor convention
-----------------

    [B, C, D, H, W]

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class PredictiveUncertaintyEstimator(nn.Module):
    """
    Estimate total predictive uncertainty by combining
    aleatoric and epistemic uncertainty.

    Aleatoric uncertainty
    --------------------

    The network predicts:

        log(sigma^2)

    Therefore:

        sigma^2 = exp(log(sigma^2))

    Epistemic uncertainty
    ---------------------

    Epistemic variance is obtained from multiple stochastic
    forward passes using MC Dropout.

    Predictive uncertainty
    ----------------------

        sigma_predictive^2
            =
        sigma_aleatoric^2
            +
        sigma_epistemic^2
    """

    def __init__(
        self,
        min_log_variance=-10.0,
        max_log_variance=10.0,
        eps=1.0e-8
    ):
        super().__init__()

        # =================================================
        # VALIDATE CONFIGURATION
        # =================================================

        if min_log_variance >= max_log_variance:
            raise ValueError(
                "min_log_variance must be less than "
                "max_log_variance."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be greater than zero."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.min_log_variance = float(
            min_log_variance
        )

        self.max_log_variance = float(
            max_log_variance
        )

        self.eps = float(eps)

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _validate_tensor(
        tensor,
        name
    ):
        """
        Validate a 3D uncertainty tensor.

        Required shape:

            [B, C, D, H, W]
        """

        if not isinstance(
            tensor,
            torch.Tensor
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B,C,D,H,W]. "
                f"Received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # ALEATORIC VARIANCE
    # =====================================================

    def aleatoric_variance(
        self,
        log_variance
    ):
        """
        Convert predicted logarithmic variance into
        aleatoric variance.

        Given:

            log_variance = log(sigma^2)

        calculate:

            sigma_aleatoric^2
                =
            exp(log_variance)

        Returns
        -------

        aleatoric_variance : torch.Tensor

            Shape:

                [B,C,D,H,W]
        """

        self._validate_tensor(
            log_variance,
            "log_variance"
        )

        # -------------------------------------------------
        # Prevent numerically extreme exponential values.
        # -------------------------------------------------

        bounded_log_variance = torch.clamp(
            log_variance,
            min=self.min_log_variance,
            max=self.max_log_variance
        )

        # -------------------------------------------------
        # Convert log variance to variance.
        # -------------------------------------------------

        variance = torch.exp(
            bounded_log_variance
        )

        # -------------------------------------------------
        # Ensure strictly positive variance.
        # -------------------------------------------------

        variance = torch.clamp(
            variance,
            min=self.eps
        )

        return variance

    # =====================================================
    # EPISTEMIC VARIANCE
    # =====================================================

    def epistemic_variance(
            self,
            mc_predictions
    ):
        """
        Calculate epistemic variance from MC Dropout
        predictions.

        Expected input shape:

            [N, B, C, D, H, W]

        where:

            N = number of stochastic MC samples.

        The variance is calculated across the MC-sample
        dimension.

        For this project, the MC predictions are treated
        as the Monte-Carlo samples used to approximate the
        predictive distribution. Therefore, population
        variance (unbiased=False) is used.

        Returns
        -------

        variance : torch.Tensor

            Shape:

                [B,C,D,H,W]
        """

        # =================================================
        # VALIDATE INPUT TYPE
        # =================================================

        if not isinstance(
                mc_predictions,
                torch.Tensor
        ):
            raise TypeError(
                "mc_predictions must be a torch.Tensor."
            )

        # =================================================
        # VALIDATE INPUT DIMENSIONS
        # =================================================

        if mc_predictions.ndim != 6:
            raise ValueError(
                "mc_predictions must have shape "
                "[N,B,C,D,H,W]. "
                f"Received "
                f"{tuple(mc_predictions.shape)}."
            )

        # =================================================
        # VALIDATE NUMBER OF MC SAMPLES
        # =================================================

        if mc_predictions.shape[0] < 2:
            raise ValueError(
                "At least two MC predictions are required "
                "to estimate epistemic variance."
            )

        # =================================================
        # VALIDATE NUMERICAL VALUES
        # =================================================

        if not torch.isfinite(
                mc_predictions
        ).all():
            raise ValueError(
                "mc_predictions contains NaN or Inf values."
            )

        # =================================================
        # MC-DROPOUT EPISTEMIC VARIANCE
        # =================================================
        #
        # The first dimension contains the stochastic
        # MC-Dropout predictions:
        #
        #     [N,B,C,D,H,W]
        #
        # Variance is therefore calculated over dimension 0.
        #
        # unbiased=False is used because these MC predictions
        # constitute the Monte-Carlo sample set used to
        # approximate the predictive distribution.
        # =================================================

        variance = torch.var(
            mc_predictions,
            dim=0,
            unbiased=False
        )

        # =================================================
        # NUMERICAL STABILITY
        # =================================================
        #
        # A zero epistemic variance is mathematically valid
        # when all stochastic predictions are identical.
        #
        # A small positive floor is retained to avoid numerical
        # problems in subsequent uncertainty calculations.
        # =================================================

        variance = torch.clamp(
            variance,
            min=self.eps
        )

        return variance

    # =====================================================
    # PREDICTIVE VARIANCE
    # =====================================================

    def predictive_variance(
        self,
        log_variance,
        mc_predictions
    ):
        """
        Calculate total predictive variance.

        Decomposition:

            sigma_predictive^2
                =
            sigma_aleatoric^2
                +
            sigma_epistemic^2
        """

        aleatoric = self.aleatoric_variance(
            log_variance
        )

        epistemic = self.epistemic_variance(
            mc_predictions
        )

        if aleatoric.shape != epistemic.shape:
            raise ValueError(
                "Aleatoric and epistemic variance "
                "must have identical shapes. "
                f"Aleatoric: {tuple(aleatoric.shape)}, "
                f"Epistemic: {tuple(epistemic.shape)}."
            )

        predictive = (
            aleatoric
            +
            epistemic
        )

        return predictive

    # =====================================================
    # STANDARD DEVIATIONS
    # =====================================================

    @staticmethod
    def standard_deviation(
        variance
    ):
        """
        Convert variance into standard deviation.
        """

        if not isinstance(
            variance,
            torch.Tensor
        ):
            raise TypeError(
                "variance must be a torch.Tensor."
            )

        if not torch.isfinite(
            variance
        ).all():
            raise ValueError(
                "variance contains NaN or Inf values."
            )

        variance = torch.clamp(
            variance,
            min=0.0
        )

        return torch.sqrt(
            variance
        )

    # =====================================================
    # COMPLETE ESTIMATION
    # =====================================================

    def forward(
        self,
        log_variance,
        mc_predictions
    ):
        """
        Calculate the complete predictive uncertainty
        decomposition.

        Parameters
        ----------

        log_variance:
            Network-predicted logarithmic variance.

            Shape:

                [B,C,D,H,W]

        mc_predictions:
            MC Dropout reconstruction samples.

            Shape:

                [N,B,C,D,H,W]

        Returns
        -------

        dict containing:

            aleatoric_variance
            epistemic_variance
            predictive_variance

            aleatoric_std
            epistemic_std
            predictive_std
        """

        # =================================================
        # ALEATORIC
        # =================================================

        aleatoric = self.aleatoric_variance(
            log_variance
        )

        # =================================================
        # EPISTEMIC
        # =================================================

        epistemic = self.epistemic_variance(
            mc_predictions
        )

        # =================================================
        # PREDICTIVE
        # =================================================

        predictive = (
            aleatoric
            +
            epistemic
        )

        # =================================================
        # STANDARD DEVIATIONS
        # =================================================

        aleatoric_std = self.standard_deviation(
            aleatoric
        )

        epistemic_std = self.standard_deviation(
            epistemic
        )

        predictive_std = self.standard_deviation(
            predictive
        )

        # =================================================
        # RETURN
        # =================================================

        return {
            "aleatoric_variance": aleatoric,

            "epistemic_variance": epistemic,

            "predictive_variance": predictive,

            "aleatoric_std": aleatoric_std,

            "epistemic_std": epistemic_std,

            "predictive_std": predictive_std
        }