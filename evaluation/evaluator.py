"""
======================================================================
Model Evaluator
======================================================================

PhD-standard evaluator for the Physics-Informed 3D Encoder-Decoder
Framework with Predictive Uncertainty for Seismic Data Reconstruction.

The evaluator:

    1. Evaluates the model over an entire DataLoader.
    2. Computes global reconstruction metrics.
    3. Computes missing-region reconstruction metrics.
    4. Computes observed-region reconstruction metrics.
    5. Evaluates observed-data preservation.
    6. Computes aleatoric uncertainty.
    7. Optionally computes epistemic uncertainty from MC predictions.
    8. Computes total predictive uncertainty.
    9. Aggregates results using sample-weighted statistics.
   10. Performs numerical validation throughout evaluation.

Tensor convention:

    Input:
        [B, C, D, H, W]

    Target:
        [B, C, D, H, W]

    Mask:
        [B, C, D, H, W]

Mask convention:

    1 = observed
    0 = missing

Model output convention:

    reconstruction
    log_variance

For MC uncertainty estimation, the model may be evaluated repeatedly
with stochastic dropout enabled.

MC-Dropout policy:

    The complete model remains in evaluation mode while only dropout
    modules are activated for stochastic forward passes.

    This prevents other training-dependent layers, such as BatchNorm,
    from changing behavior during uncertainty estimation.

Author: Ormin Joseph
======================================================================
"""

import torch

from evaluation.metrics import EvaluationMetrics


# =====================================================================
# MODEL EVALUATOR
# =====================================================================

class Evaluator:
    """
    Evaluate a trained seismic reconstruction model.
    """

    # -----------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------

    def __init__(
        self,
        model,
        device,
        mc_samples=1,
    ):
        """
        Parameters
        ----------
        model:
            Trained reconstruction model.

        device:
            torch.device used for evaluation.

        mc_samples:
            Number of stochastic forward passes used for uncertainty
            estimation.

            mc_samples = 1:
                Deterministic evaluation.

            mc_samples > 1:
                MC-Dropout uncertainty estimation.
        """

        if not isinstance(
            device,
            torch.device,
        ):

            raise TypeError(
                "device must be an instance of torch.device."
            )

        if mc_samples < 1:

            raise ValueError(
                "mc_samples must be at least 1."
            )

        if model is None:

            raise ValueError(
                "model cannot be None."
            )

        self.device = device

        self.model = (
            model.to(self.device)
        )

        self.mc_samples = int(
            mc_samples
        )

    # =================================================================
    # PUBLIC EVALUATION METHOD
    # =================================================================

    def evaluate(
        self,
        dataloader,
    ):
        """
        Evaluate the model over the complete DataLoader.

        Parameters
        ----------
        dataloader:
            PyTorch DataLoader containing evaluation samples.

            Each batch must contain at least:

                input
                target
                mask

        Returns
        -------
        dict
            Aggregated evaluation metrics.
        """

        if dataloader is None:

            raise ValueError(
                "dataloader cannot be None."
            )

        if len(dataloader) == 0:

            raise ValueError(
                "Evaluation DataLoader is empty."
            )

        # -------------------------------------------------------------
        # Complete model enters evaluation mode.
        #
        # MC-Dropout mode is handled separately inside _predict().
        # -------------------------------------------------------------

        self.model.eval()

        # -------------------------------------------------------------
        # Accumulators
        #
        # Metrics are accumulated using batch-size weighting rather
        # than assigning identical weight to every batch.
        # -------------------------------------------------------------

        accumulators = {

            # ---------------------------------------------------------
            # Global reconstruction metrics
            # ---------------------------------------------------------

            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "relative_error": 0.0,
            "snr": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,

            # ---------------------------------------------------------
            # Missing-region reconstruction metrics
            # ---------------------------------------------------------

            "missing_mae": 0.0,
            "missing_rmse": 0.0,

            # ---------------------------------------------------------
            # Observed-region reconstruction metrics
            # ---------------------------------------------------------

            "observed_mae": 0.0,
            "observed_rmse": 0.0,

            # ---------------------------------------------------------
            # Observed-data preservation
            # ---------------------------------------------------------

            "observed_preservation_error": 0.0,

            # ---------------------------------------------------------
            # Predictive uncertainty
            # ---------------------------------------------------------

            "aleatoric_variance": 0.0,
            "epistemic_variance": 0.0,
            "predictive_variance": 0.0,
            "predictive_std": 0.0,

            # ---------------------------------------------------------
            # Missing-region uncertainty
            # ---------------------------------------------------------

            "missing_aleatoric_variance": 0.0,
            "missing_epistemic_variance": 0.0,
            "missing_predictive_variance": 0.0,
            "missing_predictive_std": 0.0,
        }

        total_samples = 0

        total_missing_voxels = 0

        total_observed_voxels = 0

        # =============================================================
        # EVALUATION LOOP
        # =============================================================

        with torch.no_grad():

            for batch_index, batch in enumerate(
                dataloader
            ):

                # -----------------------------------------------------
                # Validate batch structure
                # -----------------------------------------------------

                if not isinstance(
                    batch,
                    dict,
                ):

                    raise TypeError(
                        "Each evaluation batch must be a dictionary. "
                        f"Received: {type(batch)}"
                    )

                required_keys = {
                    "input",
                    "target",
                    "mask",
                }

                missing_keys = (
                    required_keys
                    -
                    set(batch.keys())
                )

                if missing_keys:

                    raise KeyError(
                        "Evaluation batch is missing required keys: "
                        f"{missing_keys}"
                    )

                # -----------------------------------------------------
                # Prepare tensors
                # -----------------------------------------------------

                input_cube = (
                    batch["input"]
                    .to(self.device)
                )

                target_cube = (
                    batch["target"]
                    .to(self.device)
                )

                mask = (
                    batch["mask"]
                    .to(self.device)
                )

                # -----------------------------------------------------
                # Ensure channel dimension
                #
                # Dataset convention:
                #
                #     [B, D, H, W]
                #
                # Network convention:
                #
                #     [B, C, D, H, W]
                # -----------------------------------------------------

                if input_cube.ndim == 4:

                    input_cube = (
                        input_cube.unsqueeze(1)
                    )

                if target_cube.ndim == 4:

                    target_cube = (
                        target_cube.unsqueeze(1)
                    )

                if mask.ndim == 4:

                    mask = (
                        mask.unsqueeze(1)
                    )

                # -----------------------------------------------------
                # Validate dimensions
                # -----------------------------------------------------

                if input_cube.ndim != 5:

                    raise ValueError(
                        "Input tensor must have shape "
                        "[B, C, D, H, W]. "
                        f"Received: "
                        f"{tuple(input_cube.shape)}"
                    )

                if target_cube.shape != input_cube.shape:

                    raise ValueError(
                        "Input and target shapes differ. "
                        f"Input: "
                        f"{tuple(input_cube.shape)}, "
                        f"Target: "
                        f"{tuple(target_cube.shape)}"
                    )

                if mask.shape != input_cube.shape:

                    raise ValueError(
                        "Input and mask shapes differ. "
                        f"Input: "
                        f"{tuple(input_cube.shape)}, "
                        f"Mask: "
                        f"{tuple(mask.shape)}"
                    )

                # -----------------------------------------------------
                # Validate finite values
                # -----------------------------------------------------

                self._validate_finite(
                    input_cube,
                    "input_cube",
                )

                self._validate_finite(
                    target_cube,
                    "target_cube",
                )

                self._validate_finite(
                    mask,
                    "mask",
                )

                # -----------------------------------------------------
                # Validate mask
                # -----------------------------------------------------

                if not torch.all(
                    (mask == 0)
                    |
                    (mask == 1)
                ):

                    raise ValueError(
                        "Evaluation mask must contain only 0 and 1."
                    )

                # =====================================================
                # MODEL PREDICTION
                # =====================================================

                (
                    reconstruction,
                    log_variance,
                    reconstruction_samples,
                    aleatoric_variance,
                ) = self._predict(
                    input_cube
                )

                # -----------------------------------------------------
                # Validate reconstruction
                # -----------------------------------------------------

                self._validate_finite(
                    reconstruction,
                    "reconstruction",
                )

                self._validate_finite(
                    log_variance,
                    "log_variance",
                )

                self._validate_finite(
                    aleatoric_variance,
                    "aleatoric_variance",
                )

                if reconstruction.shape != target_cube.shape:

                    raise ValueError(
                        "Model reconstruction shape does not match "
                        "target shape. "
                        f"Reconstruction: "
                        f"{tuple(reconstruction.shape)}, "
                        f"Target: "
                        f"{tuple(target_cube.shape)}"
                    )

                # =====================================================
                # DATA-CONSISTENCY PROJECTION
                # =====================================================

                # -----------------------------------------------------
                # Observed samples are preserved exactly.
                #
                # mask = 1:
                #
                #     reconstruction = input
                #
                # mask = 0:
                #
                #     reconstruction = model prediction
                # -----------------------------------------------------

                reconstruction = (
                    reconstruction
                    *
                    (1.0 - mask)
                    +
                    input_cube
                    *
                    mask
                )

                # -----------------------------------------------------
                # Validate projected reconstruction.
                # -----------------------------------------------------

                self._validate_finite(
                    reconstruction,
                    "projected reconstruction",
                )

                # =====================================================
                # OBSERVED-DATA PRESERVATION
                # =====================================================

                observed_difference = torch.abs(
                    reconstruction
                    -
                    input_cube
                )

                observed_difference = (
                    observed_difference[
                        mask == 1
                    ]
                )

                if observed_difference.numel() > 0:

                    observed_preservation_error = float(
                        observed_difference
                        .max()
                        .item()
                    )

                else:

                    observed_preservation_error = 0.0

                # =====================================================
                # COMMON METRICS
                # =====================================================

                batch_metrics = {

                    "mae":
                        EvaluationMetrics.mae(
                            reconstruction,
                            target_cube,
                        ),

                    "mse":
                        EvaluationMetrics.mse(
                            reconstruction,
                            target_cube,
                        ),

                    "rmse":
                        EvaluationMetrics.rmse(
                            reconstruction,
                            target_cube,
                        ),

                    "relative_error":
                        EvaluationMetrics.relative_error(
                            reconstruction,
                            target_cube,
                        ),

                    "snr":
                        EvaluationMetrics.snr(
                            reconstruction,
                            target_cube,
                        ),

                    "psnr":
                        EvaluationMetrics.psnr(
                            reconstruction,
                            target_cube,
                        ),

                    "ssim":
                        EvaluationMetrics.ssim(
                            reconstruction,
                            target_cube,
                        ),

                    "missing_mae":
                        EvaluationMetrics.missing_mae(
                            reconstruction,
                            target_cube,
                            mask,
                        ),

                    "missing_rmse":
                        EvaluationMetrics.missing_rmse(
                            reconstruction,
                            target_cube,
                            mask,
                        ),

                    "observed_mae":
                        EvaluationMetrics.observed_mae(
                            reconstruction,
                            target_cube,
                            mask,
                        ),

                    "observed_rmse":
                        EvaluationMetrics.observed_rmse(
                            reconstruction,
                            target_cube,
                            mask,
                        ),
                }

                # =====================================================
                # EPISTEMIC UNCERTAINTY
                # =====================================================

                if reconstruction_samples is not None:

                    epistemic_variance = (
                        EvaluationMetrics.epistemic_variance(
                            reconstruction_samples
                        )
                    )

                else:

                    epistemic_variance = torch.zeros_like(
                        aleatoric_variance
                    )

                # -----------------------------------------------------
                # Validate epistemic variance.
                # -----------------------------------------------------

                self._validate_finite(
                    epistemic_variance,
                    "epistemic_variance",
                )

                # =====================================================
                # PREDICTIVE UNCERTAINTY
                # =====================================================

                predictive_variance = (
                    EvaluationMetrics.predictive_variance(
                        aleatoric_variance,
                        epistemic_variance,
                    )
                )

                self._validate_finite(
                    predictive_variance,
                    "predictive_variance",
                )

                # -----------------------------------------------------
                # Predictive variance should never be negative.
                # -----------------------------------------------------

                if torch.any(
                    predictive_variance < 0
                ):

                    raise RuntimeError(
                        "Predictive variance contains negative values."
                    )

                predictive_std = (
                    EvaluationMetrics.predictive_std(
                        predictive_variance
                    )
                )

                self._validate_finite(
                    predictive_std,
                    "predictive_std",
                )

                # =====================================================
                # MISSING-REGION UNCERTAINTY
                # =====================================================

                missing_mask = (
                    mask == 0
                )

                if torch.any(
                    missing_mask
                ):

                    missing_aleatoric = (
                        aleatoric_variance[
                            missing_mask
                        ].mean()
                    )

                    missing_epistemic = (
                        epistemic_variance[
                            missing_mask
                        ].mean()
                    )

                    missing_predictive = (
                        predictive_variance[
                            missing_mask
                        ].mean()
                    )

                    missing_std = (
                        predictive_std[
                            missing_mask
                        ].mean()
                    )

                else:

                    missing_aleatoric = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                    missing_epistemic = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                    missing_predictive = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                    missing_std = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                # =====================================================
                # SAMPLE WEIGHT
                # =====================================================

                batch_size = (
                    input_cube.shape[0]
                )

                total_samples += (
                    batch_size
                )

                # -----------------------------------------------------
                # Count missing and observed voxels.
                # -----------------------------------------------------

                batch_missing_voxels = int(
                    torch.sum(
                        mask == 0
                    ).item()
                )

                batch_observed_voxels = int(
                    torch.sum(
                        mask == 1
                    ).item()
                )

                total_missing_voxels += (
                    batch_missing_voxels
                )

                total_observed_voxels += (
                    batch_observed_voxels
                )

                # =====================================================
                # ACCUMULATE GLOBAL AND REGIONAL METRICS
                # =====================================================

                for key, value in (
                    batch_metrics.items()
                ):

                    accumulators[key] += (
                        float(
                            value.item()
                        )
                        *
                        batch_size
                    )

                # =====================================================
                # ACCUMULATE UNCERTAINTY
                # =====================================================

                accumulators[
                    "aleatoric_variance"
                ] += float(
                    aleatoric_variance.mean().item()
                ) * batch_size

                accumulators[
                    "epistemic_variance"
                ] += float(
                    epistemic_variance.mean().item()
                ) * batch_size

                accumulators[
                    "predictive_variance"
                ] += float(
                    predictive_variance.mean().item()
                ) * batch_size

                accumulators[
                    "predictive_std"
                ] += float(
                    predictive_std.mean().item()
                ) * batch_size

                # -----------------------------------------------------
                # Missing-region uncertainty.
                # -----------------------------------------------------

                accumulators[
                    "missing_aleatoric_variance"
                ] += float(
                    missing_aleatoric.item()
                ) * batch_size

                accumulators[
                    "missing_epistemic_variance"
                ] += float(
                    missing_epistemic.item()
                ) * batch_size

                accumulators[
                    "missing_predictive_variance"
                ] += float(
                    missing_predictive.item()
                ) * batch_size

                accumulators[
                    "missing_predictive_std"
                ] += float(
                    missing_std.item()
                ) * batch_size

                # -----------------------------------------------------
                # Observed-data preservation.
                # -----------------------------------------------------

                accumulators[
                    "observed_preservation_error"
                ] += (
                    observed_preservation_error
                    *
                    batch_size
                )

        # =============================================================
        # FINAL AGGREGATION
        # =============================================================

        if total_samples == 0:

            raise RuntimeError(
                "No samples were evaluated."
            )

        results = {}

        for key, value in (
            accumulators.items()
        ):

            results[key] = (
                value
                /
                total_samples
            )

        # =============================================================
        # DATASET STATISTICS
        # =============================================================

        total_voxels = (
            total_missing_voxels
            +
            total_observed_voxels
        )

        if total_voxels > 0:

            measured_missing_rate = (
                total_missing_voxels
                /
                total_voxels
            )

        else:

            measured_missing_rate = 0.0

        results[
            "num_samples"
        ] = int(
            total_samples
        )

        results[
            "total_missing_voxels"
        ] = int(
            total_missing_voxels
        )

        results[
            "total_observed_voxels"
        ] = int(
            total_observed_voxels
        )

        results[
            "measured_missing_rate"
        ] = float(
            measured_missing_rate
        )

        results[
            "mc_samples"
        ] = int(
            self.mc_samples
        )

        # =============================================================
        # FINAL NUMERICAL VALIDATION
        # =============================================================

        for key, value in (
            results.items()
        ):

            if isinstance(
                value,
                float,
            ):

                if not torch.isfinite(
                    torch.tensor(value)
                ):

                    raise RuntimeError(
                        "Non-finite evaluation result detected: "
                        f"{key}={value}"
                    )

        return results

    # =================================================================
    # MC-DROPOUT MODE CONTROL
    # =================================================================

    @staticmethod
    def _enable_mc_dropout(
        model,
    ):
        """
        Enable only dropout modules for MC-Dropout inference.

        The complete model is first placed in evaluation mode.

        Only dropout modules are subsequently returned to training mode.

        This ensures that stochasticity is introduced specifically by
        dropout rather than by the entire network.
        """

        # -------------------------------------------------------------
        # Put the complete network in evaluation mode.
        # -------------------------------------------------------------

        model.eval()

        # -------------------------------------------------------------
        # Activate only dropout layers.
        # -------------------------------------------------------------

        for module in model.modules():

            if isinstance(
                module,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                    torch.nn.AlphaDropout,
                    torch.nn.FeatureAlphaDropout,
                )
            ):

                module.train()

    # =================================================================
    # PREDICTION
    # =================================================================

    def _predict(
        self,
        input_cube,
    ):
        """
        Perform deterministic or controlled MC-Dropout prediction.

        Returns
        -------
        reconstruction:
            Mean reconstruction.

        log_variance:
            Mean log-variance representation.

        reconstruction_samples:
            MC reconstruction samples, or None when deterministic
            evaluation is requested.

        aleatoric_variance:
            Predicted aleatoric variance.

            For deterministic evaluation:

                exp(log_variance)

            For MC evaluation:

                mean(exp(log_variance_samples))

            The MC case is aggregated in variance space to avoid
            replacing E[variance] with exp(E[log_variance]).
        """

        # =============================================================
        # DETERMINISTIC EVALUATION
        # =============================================================

        if self.mc_samples == 1:

            # ---------------------------------------------------------
            # Keep the complete model in evaluation mode.
            # ---------------------------------------------------------

            self.model.eval()

            output = self.model(
                input_cube
            )

            (
                reconstruction,
                log_variance,
            ) = self._extract_model_output(
                output
            )

            # ---------------------------------------------------------
            # Convert predicted log-variance to variance.
            # ---------------------------------------------------------

            aleatoric_variance = torch.exp(
                log_variance
            )

            # ---------------------------------------------------------
            # Validate numerical stability.
            # ---------------------------------------------------------

            self._validate_finite(
                aleatoric_variance,
                "aleatoric_variance",
            )

            return (
                reconstruction,
                log_variance,
                None,
                aleatoric_variance,
            )

        # =============================================================
        # MC-DROPOUT EVALUATION
        # =============================================================

        # -------------------------------------------------------------
        # Preserve the model's original top-level training state.
        # -------------------------------------------------------------

        was_training = (
            self.model.training
        )

        # -------------------------------------------------------------
        # Store the original training/evaluation state of every module.
        #
        # This allows us to restore mixed module states accurately after
        # MC inference.
        # -------------------------------------------------------------

        original_module_states = {
            module: module.training
            for module in self.model.modules()
        }

        # -------------------------------------------------------------
        # Enable controlled MC-Dropout.
        # -------------------------------------------------------------

        self._enable_mc_dropout(
            self.model
        )

        reconstruction_samples = []

        log_variance_samples = []

        try:

            # =========================================================
            # STOCHASTIC FORWARD PASSES
            # =========================================================

            for _ in range(
                self.mc_samples
            ):

                output = self.model(
                    input_cube
                )

                (
                    reconstruction,
                    log_variance,
                ) = self._extract_model_output(
                    output
                )

                # -----------------------------------------------------
                # Validate each stochastic prediction immediately.
                # -----------------------------------------------------

                self._validate_finite(
                    reconstruction,
                    "MC reconstruction",
                )

                self._validate_finite(
                    log_variance,
                    "MC log_variance",
                )

                reconstruction_samples.append(
                    reconstruction
                )

                log_variance_samples.append(
                    log_variance
                )

        finally:

            # ---------------------------------------------------------
            # Restore every module to its original training/evaluation
            # state.
            # ---------------------------------------------------------

            for module, training_state in (
                original_module_states.items()
            ):

                module.train(
                    training_state
                )

            # ---------------------------------------------------------
            # Explicitly restore the original top-level state.
            # ---------------------------------------------------------

            self.model.train(
                was_training
            )

        # =============================================================
        # STACK MC SAMPLES
        # =============================================================

        reconstruction_samples = (
            torch.stack(
                reconstruction_samples,
                dim=0,
            )
        )

        log_variance_samples = (
            torch.stack(
                log_variance_samples,
                dim=0,
            )
        )

        # -------------------------------------------------------------
        # Validate stacked samples.
        # -------------------------------------------------------------

        self._validate_finite(
            reconstruction_samples,
            "reconstruction_samples",
        )

        self._validate_finite(
            log_variance_samples,
            "log_variance_samples",
        )

        # =============================================================
        # MEAN RECONSTRUCTION
        # =============================================================

        reconstruction = (
            reconstruction_samples.mean(
                dim=0
            )
        )

        # =============================================================
        # MEAN LOG-VARIANCE
        # =============================================================

        log_variance = (
            log_variance_samples.mean(
                dim=0
            )
        )

        # =============================================================
        # ALEATORIC VARIANCE
        # =============================================================

        # -------------------------------------------------------------
        # Each MC prediction supplies its own heteroscedastic variance:
        #
        #     variance_i = exp(log_variance_i)
        #
        # The MC aggregate is:
        #
        #     E[variance]
        #
        # rather than:
        #
        #     exp(E[log_variance])
        #
        # These two expressions are not generally identical.
        # -------------------------------------------------------------

        aleatoric_variance_samples = torch.exp(
            log_variance_samples
        )

        aleatoric_variance = (
            aleatoric_variance_samples.mean(
                dim=0
            )
        )

        # =============================================================
        # FINAL VALIDATION
        # =============================================================

        self._validate_finite(
            reconstruction,
            "MC mean reconstruction",
        )

        self._validate_finite(
            log_variance,
            "MC mean log_variance",
        )

        self._validate_finite(
            aleatoric_variance,
            "MC aleatoric_variance",
        )

        # -------------------------------------------------------------
        # Variance must not be negative.
        # -------------------------------------------------------------

        if torch.any(
            aleatoric_variance < 0
        ):

            raise RuntimeError(
                "Aleatoric variance contains negative values."
            )

        # =============================================================
        # RETURN
        # =============================================================

        return (
            reconstruction,
            log_variance,
            reconstruction_samples,
            aleatoric_variance,
        )

    # =================================================================
    # MODEL OUTPUT HANDLING
    # =================================================================

    @staticmethod
    def _extract_model_output(
        output,
    ):
        """
        Extract reconstruction and log-variance from model output.

        Supported conventions:

            (reconstruction, log_variance)

        or:

            {
                "reconstruction": ...,
                "log_variance": ...
            }

        If the model returns additional outputs, such as travel time,
        they are ignored by this evaluator because reconstruction and
        uncertainty are the quantities required here.
        """

        # -------------------------------------------------------------
        # Tuple/list model output
        # -------------------------------------------------------------

        if isinstance(
            output,
            (tuple, list)
        ):

            if len(output) < 2:

                raise ValueError(
                    "Model tuple/list output must contain at least "
                    "reconstruction and log_variance."
                )

            reconstruction = output[0]

            # ---------------------------------------------------------
            # IMPORTANT:
            #
            # Your current Network3D / Predictor convention has:
            #
            #     reconstruction
            #     travel_time
            #     log_variance
            #
            # Therefore, when three or more outputs are returned,
            # log_variance is output[2].
            #
            # For a two-output model:
            #
            #     reconstruction
            #     log_variance
            #
            # log_variance is output[1].
            # ---------------------------------------------------------

            if len(output) >= 3:

                log_variance = output[2]

            else:

                log_variance = output[1]

        # -------------------------------------------------------------
        # Dictionary model output
        # -------------------------------------------------------------

        elif isinstance(
            output,
            dict
        ):

            if "reconstruction" not in output:

                raise KeyError(
                    "Model output dictionary does not contain "
                    "'reconstruction'."
                )

            if "log_variance" not in output:

                raise KeyError(
                    "Model output dictionary does not contain "
                    "'log_variance'."
                )

            reconstruction = (
                output["reconstruction"]
            )

            log_variance = (
                output["log_variance"]
            )

        # -------------------------------------------------------------
        # Unsupported model output
        # -------------------------------------------------------------

        else:

            raise TypeError(
                "Unsupported model output type: "
                f"{type(output)}"
            )

        # -------------------------------------------------------------
        # Validate output objects.
        # -------------------------------------------------------------

        if not isinstance(
            reconstruction,
            torch.Tensor,
        ):

            raise TypeError(
                "Model reconstruction output must be a torch.Tensor. "
                f"Received: {type(reconstruction)}"
            )

        if not isinstance(
            log_variance,
            torch.Tensor,
        ):

            raise TypeError(
                "Model log_variance output must be a torch.Tensor. "
                f"Received: {type(log_variance)}"
            )

        return (
            reconstruction,
            log_variance,
        )

    # =================================================================
    # FINITE-VALUE VALIDATION
    # =================================================================

    @staticmethod
    def _validate_finite(
        tensor,
        name,
    ):
        """
        Ensure a tensor contains only finite values.
        """

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise TypeError(
                f"{name} must be a torch.Tensor. "
                f"Received: {type(tensor)}"
            )

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )