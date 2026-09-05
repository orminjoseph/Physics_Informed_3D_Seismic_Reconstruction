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

For each voxel:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

Aleatoric uncertainty
---------------------

The network predicts:

    s = log(sigma_a^2)

Therefore, for one stochastic forward pass:

    sigma_a^2 = exp(s)

When MC Dropout is used, each stochastic forward pass
produces its own conditional aleatoric variance:

    sigma_a^(2,n) = exp(s_n)

The final aleatoric variance is the mean of these
conditional variances:

    sigma_a^2
        =
    (1/N) sum_n exp(s_n)

IMPORTANT
---------

The following is NOT used:

    exp(mean(s_n))

because:

    E[exp(s)] != exp(E[s])

Epistemic uncertainty
---------------------

MC Dropout produces stochastic reconstruction predictions:

    y_hat^(1), ..., y_hat^(N)

The epistemic variance is:

    sigma_e^2
        =
    (1/N) sum_n
        (y_hat^(n) - y_hat_mean)^2

where:

    y_hat_mean
        =
    (1/N) sum_n y_hat^(n)

Predictive uncertainty
----------------------

The total predictive variance is:

    sigma_predictive^2
        =
    sigma_aleatoric^2
        +
    sigma_epistemic^2

and:

    sigma_predictive
        =
    sqrt(sigma_predictive^2)

Tensor conventions
------------------

Single deterministic tensor:

    [B, C, D, H, W]

MC Dropout tensor:

    [N, B, C, D, H, W]

where:

    N = number of MC Dropout samples
    B = batch size
    C = seismic channels
    D = depth
    H = inline/crossline spatial dimension
    W = inline/crossline spatial dimension

Author:
Ormin Joseph
=========================================================
"""

import torch
import torch.nn as nn


class PredictiveUncertaintyEstimator(nn.Module):
    """
    Estimate aleatoric, epistemic, and total predictive
    uncertainty for 3D seismic reconstruction.

    The estimator does NOT perform MC Dropout itself.

    MC Dropout sampling is handled by:

        models/mc_dropout.py

    This class receives the resulting stochastic predictions
    and calculates the uncertainty decomposition.

    Core decomposition:

        predictive_variance
            =
        aleatoric_variance
            +
        epistemic_variance
    """

    def __init__(
        self,
        min_log_variance=-10.0,
        max_log_variance=10.0,
        eps=1.0e-8
    ):
        """
        Parameters
        ----------
        min_log_variance : float
            Lower numerical bound applied to log variance.

        max_log_variance : float
            Upper numerical bound applied to log variance.

        eps : float
            Small positive numerical stability constant.
        """

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
    # SINGLE-TENSOR VALIDATION
    # =====================================================

    @staticmethod
    def _validate_tensor(
        tensor,
        name
    ):
        """
        Validate a standard 3D seismic tensor.

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

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # MC-TENSOR VALIDATION
    # =====================================================

    @staticmethod
    def _validate_mc_tensor(
        tensor,
        name
    ):
        """
        Validate an MC Dropout tensor.

        Required shape:

            [N, B, C, D, H, W]

        where N is the number of stochastic MC samples.
        """

        if not isinstance(
            tensor,
            torch.Tensor
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.ndim != 6:
            raise ValueError(
                f"{name} must have shape "
                "[N,B,C,D,H,W]. "
                f"Received {tuple(tensor.shape)}."
            )

        if tensor.shape[0] < 2:
            raise ValueError(
                f"{name} requires at least two MC "
                "samples to estimate epistemic variance."
            )

        if not torch.isfinite(
            tensor
        ).all():
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
        Convert logarithmic variance into aleatoric variance.

        Supports either:

            [B,C,D,H,W]

        or:

            [N,B,C,D,H,W]

        For a single tensor:

            sigma_a^2 = exp(log_variance)

        For MC samples:

            sigma_a^2
                =
            mean(exp(log_variance_n), dim=0)

        IMPORTANT:

        The MC formulation averages the conditional variances,
        NOT the logarithmic variances.
        """

        # =================================================
        # SINGLE DETERMINISTIC LOG-VARIANCE
        # =================================================

        if log_variance.ndim == 5:

            self._validate_tensor(
                log_variance,
                "log_variance"
            )

            bounded_log_variance = torch.clamp(
                log_variance,
                min=self.min_log_variance,
                max=self.max_log_variance
            )

            variance = torch.exp(
                bounded_log_variance
            )

        # =================================================
        # MC LOG-VARIANCE SAMPLES
        # =================================================

        elif log_variance.ndim == 6:

            self._validate_mc_tensor(
                log_variance,
                "log_variance"
            )

            bounded_log_variance = torch.clamp(
                log_variance,
                min=self.min_log_variance,
                max=self.max_log_variance
            )

            # ---------------------------------------------
            # Convert each stochastic log-variance sample
            # into its corresponding conditional variance.
            # ---------------------------------------------

            conditional_variances = torch.exp(
                bounded_log_variance
            )

            # ---------------------------------------------
            # Average conditional variances across MC
            # samples.
            #
            # This is:
            #
            # E[sigma_a^2 | model sample]
            #
            # rather than:
            #
            # exp(E[log sigma_a^2])
            # ---------------------------------------------

            variance = torch.mean(
                conditional_variances,
                dim=0
            )

        else:

            raise ValueError(
                "log_variance must have shape "
                "[B,C,D,H,W] or [N,B,C,D,H,W]. "
                f"Received {tuple(log_variance.shape)}."
            )

        # =================================================
        # NUMERICAL STABILITY
        # =================================================

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
        reconstruction predictions.

        Expected input:

            [N,B,C,D,H,W]

        The variance is calculated across the MC dimension.

        Population variance is used:

            unbiased=False

        because the MC predictions are treated as samples
        from the approximate predictive distribution.

        IMPORTANT:

        This function must receive reconstruction predictions,
        NOT log-variance predictions.

        Variance of log-variance is not part of the final
        epistemic reconstruction uncertainty.
        """

        # =================================================
        # VALIDATE MC PREDICTIONS
        # =================================================

        self._validate_mc_tensor(
            mc_predictions,
            "mc_predictions"
        )

        # =================================================
        # EPISTEMIC VARIANCE
        # =================================================
        #
        # Shape:
        #
        # [N,B,C,D,H,W]
        #
        # ->
        #
        # [B,C,D,H,W]
        #
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
        # Do not force zero epistemic uncertainty to become
        # mathematically positive.
        #
        # Zero variance is valid when all MC predictions
        # are identical.
        #
        # Therefore the variance is clamped only at zero.
        # =================================================

        variance = torch.clamp(
            variance,
            min=0.0
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

        Parameters
        ----------

        log_variance:
            Either:

                [B,C,D,H,W]

            or:

                [N,B,C,D,H,W]

            When MC log-variance samples are supplied,
            aleatoric variance is averaged correctly across
            MC samples.

        mc_predictions:
            MC Dropout reconstruction predictions:

                [N,B,C,D,H,W]

        Returns
        -------

        predictive_variance:
            [B,C,D,H,W]

        Formula:

            predictive variance
                =
            aleatoric variance
                +
            epistemic variance
        """

        # =================================================
        # CALCULATE ALEATORIC VARIANCE
        # =================================================

        aleatoric = self.aleatoric_variance(
            log_variance
        )

        # =================================================
        # CALCULATE EPISTEMIC VARIANCE
        # =================================================

        epistemic = self.epistemic_variance(
            mc_predictions
        )

        # =================================================
        # SHAPE COMPATIBILITY
        # =================================================

        if aleatoric.shape != epistemic.shape:
            raise ValueError(
                "Aleatoric and epistemic variance must have "
                "identical shapes. "
                f"Aleatoric: {tuple(aleatoric.shape)}, "
                f"Epistemic: {tuple(epistemic.shape)}."
            )

        # =================================================
        # TOTAL PREDICTIVE VARIANCE
        # =================================================

        predictive = (
            aleatoric
            +
            epistemic
        )

        # =================================================
        # NUMERICAL SAFETY
        # =================================================

        predictive = torch.clamp(
            predictive,
            min=self.eps
        )

        return predictive

    # =====================================================
    # STANDARD DEVIATION
    # =====================================================

    @staticmethod
    def standard_deviation(
        variance
    ):
        """
        Convert variance into standard deviation.

        Input:

            [B,C,D,H,W]

        Output:

            [B,C,D,H,W]
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

        # -------------------------------------------------
        # Variance cannot physically be negative.
        # -------------------------------------------------

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
        Calculate the complete uncertainty decomposition.

        Parameters
        ----------

        log_variance:
            Network-predicted logarithmic variance.

            Preferred MC form:

                [N,B,C,D,H,W]

            A deterministic form:

                [B,C,D,H,W]

            is also accepted.

        mc_predictions:
            MC Dropout reconstruction predictions.

            Required shape:

                [N,B,C,D,H,W]

        Returns
        -------

        Dictionary containing:

            aleatoric_variance
            epistemic_variance
            predictive_variance

            aleatoric_std
            epistemic_std
            predictive_std

        IMPORTANT:

        The returned epistemic variance is based exclusively
        on stochastic reconstruction predictions.

        Variance of the predicted log-variance is deliberately
        excluded from the predictive decomposition.
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
        # VERIFY OUTPUT COMPATIBILITY
        # =================================================

        if aleatoric.shape != epistemic.shape:
            raise ValueError(
                "Aleatoric and epistemic variance must have "
                "identical shapes. "
                f"Aleatoric: {tuple(aleatoric.shape)}, "
                f"Epistemic: {tuple(epistemic.shape)}."
            )

        # =================================================
        # PREDICTIVE VARIANCE
        # =================================================

        predictive = (
            aleatoric
            +
            epistemic
        )

        predictive = torch.clamp(
            predictive,
            min=self.eps
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
        # RETURN COMPLETE DECOMPOSITION
        # =================================================

        return {
            "aleatoric_variance": aleatoric,

            "epistemic_variance": epistemic,

            "predictive_variance": predictive,

            "aleatoric_std": aleatoric_std,

            "epistemic_std": epistemic_std,

            "predictive_std": predictive_std
        }