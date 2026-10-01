"""
======================================================================
Model Evaluator
======================================================================

PhD-standard evaluator for the Physics-Informed 3D Encoder-Decoder
Framework with Predictive Uncertainty for Seismic Data Reconstruction.

This evaluator is responsible for:

    1. Evaluating the trained model over a complete DataLoader.
    2. Computing global reconstruction metrics.
    3. Computing missing-region reconstruction metrics.
    4. Computing observed-region reconstruction metrics.
    5. Evaluating exact observed-data preservation.
    6. Computing aleatoric uncertainty.
    7. Computing epistemic uncertainty using MC-Dropout.
    8. Computing predictive uncertainty.
    9. Computing missing-region uncertainty.
   10. Aggregating metrics over the evaluation dataset.
   11. Performing numerical and structural validation.

IMPORTANT METRICS ARCHITECTURE
--------------------------------

All reconstruction and uncertainty metric mathematics are delegated
to the canonical project-wide implementation:

    metrics/reconstruction_metrics.py

This evaluator does NOT maintain a second implementation of:

    MAE
    MSE
    RMSE
    Relative L2 Error
    PSNR
    SNR
    SSIM
    Missing MAE
    Missing RMSE
    Observed MAE
    Observed RMSE
    Aleatoric Variance
    Epistemic Variance
    Predictive Variance
    Predictive Standard Deviation

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

    Common three-output model:

        reconstruction
        travel_time
        log_variance

    Two-output model:

        reconstruction
        log_variance

For MC-Dropout:

    The complete model remains in evaluation mode while only dropout
    modules are activated for stochastic forward passes.

    This prevents BatchNorm and other training-dependent modules from
    changing behavior during uncertainty estimation.

Author: Ormin Joseph
======================================================================
"""

from __future__ import annotations

import torch

from metrics.reconstruction_metrics import (
    calculate_reconstruction_metrics,
    calculate_uncertainty_metrics,
)


# =====================================================================
# MODEL EVALUATOR
# =====================================================================

class Evaluator:
    """
    Evaluate a trained seismic reconstruction model.

    Metric calculations are delegated to:

        metrics.reconstruction_metrics
    """

    # =================================================================
    # CONSTRUCTOR
    # =================================================================

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
            Trained seismic reconstruction model.

        device:
            torch.device used for evaluation.

        mc_samples:
            Number of stochastic forward passes for MC-Dropout.

            mc_samples = 1:
                Deterministic evaluation.

            mc_samples > 1:
                MC-Dropout uncertainty estimation.
        """

        if model is None:

            raise ValueError(
                "model cannot be None."
            )

        if not isinstance(
            device,
            torch.device,
        ):

            raise TypeError(
                "device must be an instance of torch.device."
            )

        if int(mc_samples) < 1:

            raise ValueError(
                "mc_samples must be at least 1."
            )

        self.device = device

        self.model = model.to(
            self.device
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

            Each batch must contain:

                input
                target
                mask

        Returns
        -------
        dict
            Aggregated evaluation metrics.
        """

        # -------------------------------------------------------------
        # Validate DataLoader.
        # -------------------------------------------------------------

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
        # MC-Dropout is controlled separately inside _predict().
        # -------------------------------------------------------------

        self.model.eval()

        # =============================================================
        # ACCUMULATORS
        # =============================================================

        accumulators = {

            # ---------------------------------------------------------
            # Global reconstruction metrics
            # ---------------------------------------------------------

            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "relative_l2": 0.0,
            "relative_error": 0.0,
            "psnr": 0.0,
            "snr": 0.0,
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
            # Global uncertainty
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

                # =====================================================
                # VALIDATE BATCH
                # =====================================================

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

                # =====================================================
                # MOVE DATA TO DEVICE
                # =====================================================

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

                # =====================================================
                # ENSURE CHANNEL DIMENSION
                # =====================================================

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

                # =====================================================
                # VALIDATE SHAPES
                # =====================================================

                if input_cube.ndim != 5:

                    raise ValueError(
                        "Input tensor must have shape "
                        "[B, C, D, H, W]. "
                        f"Received: {tuple(input_cube.shape)}"
                    )

                if target_cube.shape != input_cube.shape:

                    raise ValueError(
                        "Input and target shapes differ. "
                        f"Input: {tuple(input_cube.shape)}, "
                        f"Target: {tuple(target_cube.shape)}"
                    )

                if mask.shape != input_cube.shape:

                    raise ValueError(
                        "Input and mask shapes differ. "
                        f"Input: {tuple(input_cube.shape)}, "
                        f"Mask: {tuple(mask.shape)}"
                    )

                # =====================================================
                # VALIDATE NUMERICAL VALUES
                # =====================================================

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

                # =====================================================
                # VALIDATE MASK
                # =====================================================

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

                # =====================================================
                # VALIDATE MODEL OUTPUT
                # =====================================================

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

                # Observed samples are preserved exactly.
                #
                # mask = 1:
                #
                #     reconstruction = input
                #
                # mask = 0:
                #
                #     reconstruction = model prediction

                reconstruction = (
                    reconstruction
                    *
                    (1.0 - mask)
                    +
                    input_cube
                    *
                    mask
                )

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
                        observed_difference.max().item()
                    )

                else:

                    observed_preservation_error = 0.0

                # =====================================================
                # CANONICAL RECONSTRUCTION METRICS
                # =====================================================

                reconstruction_metrics = (
                    calculate_reconstruction_metrics(
                        prediction=reconstruction,
                        target=target_cube,
                        mask=mask,
                    )
                )

                # -----------------------------------------------------
                # Validate returned metrics.
                # -----------------------------------------------------

                self._validate_metric_dictionary(
                    reconstruction_metrics,
                    "reconstruction metrics",
                )

                # =====================================================
                # CANONICAL UNCERTAINTY METRICS
                # =====================================================

                if reconstruction_samples is not None:

                    uncertainty_metrics = (
                        calculate_uncertainty_metrics(
                            log_variance=log_variance,
                            reconstruction_samples=
                                reconstruction_samples,
                        )
                    )

                else:

                    # -------------------------------------------------
                    # Deterministic evaluation:
                    #
                    # No stochastic MC samples means epistemic
                    # uncertainty is zero.
                    # -------------------------------------------------

                    epistemic_variance = torch.zeros_like(
                        aleatoric_variance
                    )

                    predictive_variance = (
                        aleatoric_variance
                        +
                        epistemic_variance
                    )

                    predictive_std = torch.sqrt(
                        predictive_variance
                    )

                    uncertainty_metrics = {
                        "aleatoric_variance":
                            aleatoric_variance,

                        "epistemic_variance":
                            epistemic_variance,

                        "predictive_variance":
                            predictive_variance,

                        "predictive_std":
                            predictive_std,
                    }

                # =====================================================
                # VALIDATE UNCERTAINTY
                # =====================================================

                self._validate_uncertainty_dictionary(
                    uncertainty_metrics
                )

                # -----------------------------------------------------
                # Extract uncertainty tensors.
                # -----------------------------------------------------

                aleatoric_variance = (
                    uncertainty_metrics[
                        "aleatoric_variance"
                    ]
                )

                epistemic_variance = (
                    uncertainty_metrics[
                        "epistemic_variance"
                    ]
                )

                predictive_variance = (
                    uncertainty_metrics[
                        "predictive_variance"
                    ]
                )

                predictive_std = (
                    uncertainty_metrics[
                        "predictive_std"
                    ]
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
                # BATCH SIZE
                # =====================================================

                batch_size = (
                    input_cube.shape[0]
                )

                total_samples += (
                    batch_size
                )

                # =====================================================
                # VOXEL COUNTS
                # =====================================================

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
                # ACCUMULATE RECONSTRUCTION METRICS
                # =====================================================

                for key in (
                    "mae",
                    "mse",
                    "rmse",
                    "relative_l2",
                    "psnr",
                    "snr",
                    "ssim",
                    "missing_mae",
                    "missing_rmse",
                    "observed_mae",
                    "observed_rmse",
                ):

                    value = reconstruction_metrics[
                        key
                    ]

                    accumulators[key] += (
                        float(
                            value.item()
                        )
                        *
                        batch_size
                    )

                # -----------------------------------------------------
                # Backward-compatible relative_error alias.
                # -----------------------------------------------------

                accumulators[
                    "relative_error"
                ] += (
                    float(
                        reconstruction_metrics[
                            "relative_l2"
                        ].item()
                    )
                    *
                    batch_size
                )

                # =====================================================
                # ACCUMULATE OBSERVED PRESERVATION
                # =====================================================

                accumulators[
                    "observed_preservation_error"
                ] += (
                    observed_preservation_error
                    *
                    batch_size
                )

                # =====================================================
                # ACCUMULATE GLOBAL UNCERTAINTY
                # =====================================================

                accumulators[
                    "aleatoric_variance"
                ] += (
                    float(
                        aleatoric_variance.mean().item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "epistemic_variance"
                ] += (
                    float(
                        epistemic_variance.mean().item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "predictive_variance"
                ] += (
                    float(
                        predictive_variance.mean().item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "predictive_std"
                ] += (
                    float(
                        predictive_std.mean().item()
                    )
                    *
                    batch_size
                )

                # =====================================================
                # ACCUMULATE MISSING-REGION UNCERTAINTY
                # =====================================================

                accumulators[
                    "missing_aleatoric_variance"
                ] += (
                    float(
                        missing_aleatoric.item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "missing_epistemic_variance"
                ] += (
                    float(
                        missing_epistemic.item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "missing_predictive_variance"
                ] += (
                    float(
                        missing_predictive.item()
                    )
                    *
                    batch_size
                )

                accumulators[
                    "missing_predictive_std"
                ] += (
                    float(
                        missing_std.item()
                    )
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
    # MC-DROPOUT CONTROL
    # =================================================================

    @staticmethod
    def _enable_mc_dropout(
        model,
    ):
        """
        Put the complete model in evaluation mode and activate only
        dropout modules.

        This keeps BatchNorm and other training-dependent layers in
        evaluation mode while allowing MC-Dropout stochasticity.
        """

        model.eval()

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
        Perform deterministic or MC-Dropout prediction.

        Returns
        -------
        reconstruction:
            Mean reconstruction.

        log_variance:
            Mean predicted log-variance.

        reconstruction_samples:
            MC reconstruction samples when mc_samples > 1.

        aleatoric_variance:
            Predicted aleatoric variance.
        """

        # =============================================================
        # DETERMINISTIC EVALUATION
        # =============================================================

        if self.mc_samples == 1:

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
            # Convert log variance to variance.
            # ---------------------------------------------------------

            safe_log_variance = torch.clamp(
                log_variance,
                min=-30.0,
                max=30.0,
            )

            aleatoric_variance = torch.exp(
                safe_log_variance
            )

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

            return (
                reconstruction,
                log_variance,
                None,
                aleatoric_variance,
            )

        # =============================================================
        # MC-DROPOUT EVALUATION
        # =============================================================

        original_module_states = {
            module: module.training
            for module in self.model.modules()
        }

        self._enable_mc_dropout(
            self.model
        )

        reconstruction_samples = []

        log_variance_samples = []

        try:

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
            # Restore every module's original state.
            # ---------------------------------------------------------

            for module, training_state in (
                original_module_states.items()
            ):

                module.train(
                    training_state
                )

        # =============================================================
        # STACK MC SAMPLES
        # =============================================================

        reconstruction_samples = torch.stack(
            reconstruction_samples,
            dim=0,
        )

        log_variance_samples = torch.stack(
            log_variance_samples,
            dim=0,
        )

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
        # MEAN LOG VARIANCE
        # =============================================================

        log_variance = (
            log_variance_samples.mean(
                dim=0
            )
        )

        # =============================================================
        # ALEATORIC VARIANCE
        # =============================================================

        # Each stochastic forward pass has its own predicted
        # heteroscedastic variance.
        #
        # Therefore:
        #
        #     E[variance]
        #
        # is calculated rather than:
        #
        #     exp(E[log_variance])

        safe_log_variance_samples = torch.clamp(
            log_variance_samples,
            min=-30.0,
            max=30.0,
        )

        aleatoric_variance_samples = torch.exp(
            safe_log_variance_samples
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

        if torch.any(
            aleatoric_variance < 0
        ):

            raise RuntimeError(
                "Aleatoric variance contains negative values."
            )

        return (
            reconstruction,
            log_variance,
            reconstruction_samples,
            aleatoric_variance,
        )

    # =================================================================
    # MODEL OUTPUT EXTRACTION
    # =================================================================

    @staticmethod
    def _extract_model_output(
        output,
    ):
        """
        Extract reconstruction and log-variance from model output.

        Supported model outputs:

        1. Two-output tuple/list:

            reconstruction
            log_variance

        2. Three-output tuple/list:

            reconstruction
            travel_time
            log_variance

        3. Dictionary:

            {
                "reconstruction": ...,
                "log_variance": ...
            }

        Travel time is intentionally not used as a reconstruction
        metric in this evaluator. It remains available to separate
        physics-informed diagnostic workflows.
        """

        # =============================================================
        # TUPLE / LIST OUTPUT
        # =============================================================

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

            if len(output) >= 3:

                # -----------------------------------------------------
                # Current three-output convention:
                #
                #     reconstruction
                #     travel_time
                #     log_variance
                # -----------------------------------------------------

                log_variance = output[2]

            else:

                # -----------------------------------------------------
                # Two-output convention:
                #
                #     reconstruction
                #     log_variance
                # -----------------------------------------------------

                log_variance = output[1]

        # =============================================================
        # DICTIONARY OUTPUT
        # =============================================================

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

        else:

            raise TypeError(
                "Unsupported model output type: "
                f"{type(output)}"
            )

        # =============================================================
        # OUTPUT TYPE VALIDATION
        # =============================================================

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
    # METRIC DICTIONARY VALIDATION
    # =================================================================

    @staticmethod
    def _validate_metric_dictionary(
        metrics,
        name,
    ):
        """
        Validate a dictionary of scalar metric tensors.
        """

        if not isinstance(
            metrics,
            dict,
        ):

            raise TypeError(
                f"{name} must be a dictionary."
            )

        for key, value in metrics.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):

                raise TypeError(
                    f"{name}['{key}'] must be a torch.Tensor."
                )

            if value.numel() != 1:

                raise ValueError(
                    f"{name}['{key}'] must be scalar. "
                    f"Shape: {tuple(value.shape)}"
                )

            if not torch.isfinite(
                value
            ).all():

                raise RuntimeError(
                    f"{name}['{key}'] contains NaN or Inf."
                )

    # =================================================================
    # UNCERTAINTY DICTIONARY VALIDATION
    # =================================================================

    @staticmethod
    def _validate_uncertainty_dictionary(
        uncertainty_metrics,
    ):
        """
        Validate uncertainty tensors returned by the canonical
        uncertainty metric implementation.
        """

        required_keys = {
            "aleatoric_variance",
            "epistemic_variance",
            "predictive_variance",
            "predictive_std",
        }

        missing_keys = (
            required_keys
            -
            set(uncertainty_metrics.keys())
        )

        if missing_keys:

            raise KeyError(
                "Uncertainty metric dictionary is missing keys: "
                f"{missing_keys}"
            )

        for key in required_keys:

            value = uncertainty_metrics[
                key
            ]

            if not isinstance(
                value,
                torch.Tensor,
            ):

                raise TypeError(
                    f"Uncertainty metric '{key}' must be a tensor."
                )

            if not torch.isfinite(
                value
            ).all():

                raise RuntimeError(
                    f"Uncertainty metric '{key}' contains "
                    "NaN or Inf values."
                )

            if torch.any(
                value < 0
            ):

                raise RuntimeError(
                    f"Uncertainty metric '{key}' contains "
                    "negative values."
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
        Ensure that a tensor contains only finite values.
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