"""
=========================================================
Volume Reconstructor
=========================================================

Reconstructs a complete 3D seismic volume from overlapping
3D patches using MC Dropout uncertainty estimation.

Uncertainty decomposition:

    Aleatoric Variance
        = mean(exp(log_variance_samples))

    Epistemic Variance
        = Var(reconstruction_samples)

    Predictive Variance
        = Aleatoric Variance + Epistemic Variance

    Predictive Standard Deviation
        = sqrt(Predictive Variance)

Author: Ormin Joseph
=========================================================
"""

import numpy as np
import torch

from dataset.patch_extractor import PatchExtractor
from models.predictive_uncertainty import PredictiveUncertaintyEstimator


class VolumeReconstructor:
    """
    Reconstruct a complete 3D seismic volume from patches.
    """

    def __init__(self, predictor, patch_size, stride):
        """
        Parameters
        ----------
        predictor : object
            MC Dropout predictor.

        patch_size : tuple
            Patch dimensions:
            (depth, height, width).

        stride : tuple
            Patch stride:
            (depth, height, width).
        """

        self.predictor = predictor
        self.patch_size = tuple(patch_size)
        self.stride = tuple(stride)

        # Authoritative predictive uncertainty estimator.
        self.uncertainty_estimator = PredictiveUncertaintyEstimator()

        # Validate patch size.
        if len(self.patch_size) != 3:
            raise ValueError(
                "patch_size must contain exactly three values."
            )

        # Validate stride.
        if len(self.stride) != 3:
            raise ValueError(
                "stride must contain exactly three values."
            )

        # All patch dimensions must be positive.
        if any(size <= 0 for size in self.patch_size):
            raise ValueError(
                "All patch_size values must be positive."
            )

        # All stride dimensions must be positive.
        if any(step <= 0 for step in self.stride):
            raise ValueError(
                "All stride values must be positive."
            )

    def reconstruct(self, volume):
        """
        Reconstruct a complete 3D seismic volume.

        Parameters
        ----------
        volume : numpy.ndarray
            Input seismic volume with shape:

                (depth, height, width)

        Returns
        -------
        reconstructed_volume : numpy.ndarray
            Reconstructed seismic volume.

        predictive_std_volume : numpy.ndarray
            Predictive uncertainty represented as
            standard deviation.
        """

        # -----------------------------------------------------
        # 1. Validate input volume
        # -----------------------------------------------------

        if not isinstance(volume, np.ndarray):
            raise TypeError(
                "volume must be a NumPy array."
            )

        if volume.ndim != 3:
            raise ValueError(
                "volume must have shape "
                "(depth, height, width)."
            )

        if not np.isfinite(volume).all():
            raise ValueError(
                "volume contains NaN or infinite values."
            )

        depth, height, width = volume.shape

        patch_depth, patch_height, patch_width = self.patch_size

        # -----------------------------------------------------
        # 2. Validate patch dimensions
        # -----------------------------------------------------

        if patch_depth > depth:
            raise ValueError(
                "Patch depth cannot be larger than "
                "the input volume depth."
            )

        if patch_height > height:
            raise ValueError(
                "Patch height cannot be larger than "
                "the input volume height."
            )

        if patch_width > width:
            raise ValueError(
                "Patch width cannot be larger than "
                "the input volume width."
            )

        # -----------------------------------------------------
        # 3. Extract patches
        # -----------------------------------------------------

        extractor = PatchExtractor(
            patch_size=self.patch_size,
            stride=self.stride
        )

        patches = extractor.extract(volume)

        if len(patches) == 0:
            raise RuntimeError(
                "PatchExtractor returned no patches."
            )

        # -----------------------------------------------------
        # 4. Create accumulation arrays
        # -----------------------------------------------------

        reconstructed_volume = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        reconstruction_counter = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        predictive_variance_volume = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        uncertainty_counter = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        # -----------------------------------------------------
        # 5. Process every seismic patch
        # -----------------------------------------------------

        for patch, z, y, x in patches:

            # Input patch:
            #
            #     (D, H, W)
            #
            # Network input:
            #
            #     (B, C, D, H, W)

            patch_tensor = torch.from_numpy(
                patch
            ).float()

            patch_tensor = patch_tensor.unsqueeze(0)
            patch_tensor = patch_tensor.unsqueeze(0)

            # -------------------------------------------------
            # 6. Run MC Dropout prediction
            # -------------------------------------------------

            prediction = self.predictor.predict(
                patch_tensor
            )

            if not isinstance(prediction, dict):
                raise TypeError(
                    "predictor.predict() must return "
                    "a dictionary."
                )

            required_keys = {
                "reconstruction_samples",
                "log_variance_samples"
            }

            missing_keys = required_keys.difference(
                prediction.keys()
            )

            if missing_keys:
                raise KeyError(
                    "Missing required MC Dropout outputs: "
                    f"{missing_keys}"
                )

            reconstruction_samples = prediction[
                "reconstruction_samples"
            ]

            log_variance_samples = prediction[
                "log_variance_samples"
            ]

            # -------------------------------------------------
            # 7. Calculate aleatoric variance
            # -------------------------------------------------
            #
            # sigma_a^2 =
            #
            #     mean(exp(log_variance_samples))
            #
            # The estimator handles the MC dimension.
            # -------------------------------------------------

            aleatoric_variance = (
                self.uncertainty_estimator.aleatoric_variance(
                    log_variance_samples
                )
            )

            # -------------------------------------------------
            # 8. Calculate epistemic variance
            # -------------------------------------------------
            #
            # sigma_e^2 =
            #
            #     Var(reconstruction_samples)
            #
            # IMPORTANT:
            # Epistemic uncertainty is calculated from
            # reconstruction predictions, NOT log variance.
            # -------------------------------------------------

            epistemic_variance = (
                self.uncertainty_estimator.epistemic_variance(
                    reconstruction_samples
                )
            )

            # -------------------------------------------------
            # 9. Calculate predictive variance
            # -------------------------------------------------
            #
            # sigma_pred^2 =
            #
            #     sigma_a^2 + sigma_e^2
            #
            # The finalized estimator expects:
            #
            #     log_variance
            #     reconstruction_samples
            # -------------------------------------------------

            predictive_variance = (
                self.uncertainty_estimator.predictive_variance(
                    log_variance_samples,
                    reconstruction_samples
                )
            )

            # -------------------------------------------------
            # 10. Verify variance decomposition
            # -------------------------------------------------

            decomposition_error = torch.max(
                torch.abs(
                    predictive_variance
                    - (
                        aleatoric_variance
                        + epistemic_variance
                    )
                )
            )

            if not torch.isfinite(
                decomposition_error
            ):
                raise RuntimeError(
                    "Predictive variance decomposition "
                    "produced a non-finite value."
                )

            if decomposition_error.item() > 1e-5:
                raise RuntimeError(
                    "Predictive variance decomposition "
                    "check failed. "
                    f"Maximum difference: "
                    f"{decomposition_error.item()}"
                )

            # -------------------------------------------------
            # 11. Calculate mean reconstruction
            # -------------------------------------------------

            reconstruction_mean = (
                reconstruction_samples.mean(dim=0)
            )

            # -------------------------------------------------
            # 12. Convert reconstruction to NumPy
            # -------------------------------------------------

            reconstructed_patch = (
                reconstruction_mean
                .detach()
                .cpu()
                .numpy()
            )

            predictive_variance_patch = (
                predictive_variance
                .detach()
                .cpu()
                .numpy()
            )

            # -------------------------------------------------
            # 13. Validate tensor dimensions
            # -------------------------------------------------

            expected_dimensions = 5

            if reconstructed_patch.ndim != expected_dimensions:
                raise ValueError(
                    "Reconstruction must have shape "
                    "(B, C, D, H, W). "
                    f"Received: {reconstructed_patch.shape}"
                )

            if predictive_variance_patch.ndim != expected_dimensions:
                raise ValueError(
                    "Predictive variance must have shape "
                    "(B, C, D, H, W). "
                    f"Received: "
                    f"{predictive_variance_patch.shape}"
                )

            # -------------------------------------------------
            # 14. Remove batch and channel dimensions
            # -------------------------------------------------

            reconstructed_patch = reconstructed_patch[0, 0]

            predictive_variance_patch = (
                predictive_variance_patch[0, 0]
            )

            # -------------------------------------------------
            # 15. Validate numerical values
            # -------------------------------------------------

            if not np.isfinite(
                reconstructed_patch
            ).all():
                raise ValueError(
                    "Reconstructed patch contains "
                    "NaN or infinite values."
                )

            if not np.isfinite(
                predictive_variance_patch
            ).all():
                raise ValueError(
                    "Predictive variance patch contains "
                    "NaN or infinite values."
                )

            # Numerical protection against tiny negative
            # floating-point values.

            predictive_variance_patch = np.maximum(
                predictive_variance_patch,
                0.0
            )

            # -------------------------------------------------
            # 16. Determine patch dimensions
            # -------------------------------------------------

            pd, ph, pw = reconstructed_patch.shape

            # -------------------------------------------------
            # 17. Accumulate reconstruction
            # -------------------------------------------------

            reconstructed_volume[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += reconstructed_patch

            reconstruction_counter[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += 1.0

            # -------------------------------------------------
            # 18. Accumulate predictive variance
            # -------------------------------------------------

            predictive_variance_volume[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += predictive_variance_patch

            uncertainty_counter[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += 1.0

        # -----------------------------------------------------
        # 19. Check complete volume coverage
        # -----------------------------------------------------

        if np.any(reconstruction_counter == 0):

            uncovered = np.sum(
                reconstruction_counter == 0
            )

            raise RuntimeError(
                "Patch extraction did not cover the "
                f"entire volume. Uncovered voxels: "
                f"{uncovered}"
            )

        if np.any(uncertainty_counter == 0):

            uncovered = np.sum(
                uncertainty_counter == 0
            )

            raise RuntimeError(
                "Patch extraction did not cover the "
                f"entire uncertainty volume. "
                f"Uncovered voxels: {uncovered}"
            )

        # -----------------------------------------------------
        # 20. Average overlapping reconstruction patches
        # -----------------------------------------------------

        reconstructed_volume /= (
            reconstruction_counter
        )

        # -----------------------------------------------------
        # 21. Average overlapping predictive variances
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # Average variance first.
        # Convert to standard deviation only afterward.
        # -----------------------------------------------------

        predictive_variance_volume /= (
            uncertainty_counter
        )

        predictive_variance_volume = np.maximum(
            predictive_variance_volume,
            0.0
        )

        # -----------------------------------------------------
        # 22. Convert predictive variance to std
        # -----------------------------------------------------

        predictive_std_volume = np.sqrt(
            predictive_variance_volume
        ).astype(np.float32)

        # -----------------------------------------------------
        # 23. Final numerical validation
        # -----------------------------------------------------

        if not np.isfinite(
            reconstructed_volume
        ).all():
            raise RuntimeError(
                "Final reconstructed volume contains "
                "NaN or infinite values."
            )

        if not np.isfinite(
            predictive_std_volume
        ).all():
            raise RuntimeError(
                "Final predictive uncertainty volume "
                "contains NaN or infinite values."
            )

        # -----------------------------------------------------
        # 24. Return results
        # -----------------------------------------------------

        return (
            reconstructed_volume.astype(np.float32),
            predictive_std_volume
        )