"""
=========================================================
Dictionary Learning Baseline
=========================================================

Dictionary-learning-based reconstruction for 3-D seismic
data with missing samples.

Purpose
-------
This module provides a controlled dictionary-learning
baseline for seismic data reconstruction.

Input convention
----------------
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

where:
    C = number of seismic channels
    D = depth/time samples
    H = crossline direction
    W = inline direction

Mask convention
---------------
    1.0 = observed sample
    0.0 = missing sample

Method
------
1. Extract 3-D patches from sufficiently observed regions.
2. Learn a compact dictionary from those patches.
3. Represent patches using sparse coefficients.
4. Reconstruct the missing samples from the sparse
   dictionary representation.
5. Aggregate overlapping reconstructed patches.
6. Restore all observed samples exactly.

Important
---------
The dictionary is learned only from the supplied corrupted
input. The target/ground-truth volume must NEVER be passed
to this function.

Author: Ormin Joseph
=========================================================
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# Optional scikit-learn import
# ---------------------------------------------------------------------

try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.feature_extraction.image import extract_patches_2d
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for the Dictionary Learning "
        "baseline. Install it with:\n\n"
        "python -m pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------

DEFAULT_PATCH_SIZE = (8, 8, 8)

DEFAULT_N_COMPONENTS = 64

DEFAULT_ALPHA = 1.0

DEFAULT_MAX_ITER = 20

DEFAULT_BATCH_SIZE = 64

DEFAULT_MAX_TRAINING_PATCHES = 2000

DEFAULT_MIN_OBSERVED_FRACTION = 0.80

DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Validate the corrupted seismic cube and observation mask.
    """

    # The corrupted volume must be a PyTorch tensor.
    if not isinstance(corrupted_cube, torch.Tensor):
        raise TypeError(
            "corrupted_cube must be a torch.Tensor."
        )

    # The mask must also be a PyTorch tensor.
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask must be a torch.Tensor."
        )

    # Both tensors must have four dimensions:
    # (C, D, H, W).
    if corrupted_cube.ndim != 4:
        raise ValueError(
            "corrupted_cube must have shape (C, D, H, W). "
            f"Received: {tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape (C, D, H, W). "
            f"Received: {tuple(mask.shape)}"
        )

    # The two tensors must have identical shapes.
    if corrupted_cube.shape != mask.shape:
        raise ValueError(
            "corrupted_cube and mask must have identical shapes. "
            f"Received {tuple(corrupted_cube.shape)} and "
            f"{tuple(mask.shape)}."
        )

    # Both tensors must contain finite values.
    if not torch.isfinite(corrupted_cube).all():
        raise ValueError(
            "corrupted_cube contains NaN or infinite values."
        )

    if not torch.isfinite(mask).all():
        raise ValueError(
            "mask contains NaN or infinite values."
        )

    # The mask must contain only 0 and 1.
    unique_mask = torch.unique(mask)

    if not torch.all(
        (unique_mask == 0) | (unique_mask == 1)
    ):
        raise ValueError(
            "mask must contain only 0.0 and 1.0. "
            f"Found values: {unique_mask.tolist()}"
        )


# ---------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------

def _extract_3d_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    min_observed_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 3-D patches and identify sufficiently observed patches.

    Parameters
    ----------
    volume:
        3-D seismic volume with shape (D, H, W).

    mask:
        Corresponding binary mask.

    patch_size:
        Tuple (pd, ph, pw).

    min_observed_fraction:
        Minimum fraction of observed samples required for a
        patch to be used for dictionary learning.
    """

    pd, ph, pw = patch_size

    depth, height, width = volume.shape

    # Make sure the patch fits inside the volume.
    if (
        pd > depth
        or ph > height
        or pw > width
    ):
        raise ValueError(
            "Patch size must not exceed the input volume size. "
            f"Volume: {volume.shape}, "
            f"Patch: {patch_size}"
        )

    patches = []

    valid_patches = []

    # Slide the patch through the 3-D volume.
    for d in range(depth - pd + 1):

        for h in range(height - ph + 1):

            for w in range(width - pw + 1):

                # Extract the current seismic patch.
                patch = volume[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                # Extract the corresponding mask patch.
                mask_patch = mask[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                # Calculate the fraction of observed samples.
                observed_fraction = float(
                    np.mean(mask_patch)
                )

                # Store every sufficiently observed patch.
                if observed_fraction >= min_observed_fraction:

                    patches.append(
                        patch.reshape(-1)
                    )

                    valid_patches.append(
                        mask_patch.reshape(-1)
                    )

    # No suitable patches means dictionary learning cannot proceed.
    if len(patches) == 0:
        return (
            np.empty(
                (0, pd * ph * pw),
                dtype=np.float32,
            ),
            np.empty(
                (0, pd * ph * pw),
                dtype=np.float32,
            ),
        )

    return (
        np.asarray(
            patches,
            dtype=np.float32,
        ),
        np.asarray(
            valid_patches,
            dtype=np.float32,
        ),
    )


# ---------------------------------------------------------------------
# Random patch selection
# ---------------------------------------------------------------------

def _select_training_patches(
    patches: np.ndarray,
    max_training_patches: int,
    random_state: int,
) -> np.ndarray:
    """
    Randomly select a bounded number of training patches.

    Limiting the number of patches keeps the baseline
    computationally practical for the 750-case benchmark.
    """

    # If the number of patches is already small,
    # no subsampling is necessary.
    if len(patches) <= max_training_patches:
        return patches

    # Create a local reproducible random generator.
    rng = np.random.default_rng(random_state)

    # Randomly select patch indices without replacement.
    indices = rng.choice(
        len(patches),
        size=max_training_patches,
        replace=False,
    )

    return patches[indices]


# ---------------------------------------------------------------------
# Dictionary learning
# ---------------------------------------------------------------------

def _learn_dictionary(
    training_patches: np.ndarray,
    n_components: int,
    alpha: float,
    max_iter: int,
    batch_size: int,
    random_state: int,
) -> MiniBatchDictionaryLearning:
    """
    Learn a compact dictionary from seismic patches.
    """

    # The dictionary-learning model learns atoms that represent
    # the dominant structures in the supplied seismic patches.
    dictionary_model = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=random_state,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=8,
        verbose=False,
    )

    # Fit the dictionary using ONLY patches extracted from
    # the corrupted input volume.
    dictionary_model.fit(training_patches)

    return dictionary_model


# ---------------------------------------------------------------------
# Patch reconstruction
# ---------------------------------------------------------------------

def _reconstruct_channel(
    volume: np.ndarray,
    mask: np.ndarray,
    dictionary_model: MiniBatchDictionaryLearning,
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct one seismic channel using the learned dictionary.
    """

    pd, ph, pw = patch_size

    depth, height, width = volume.shape

    # Accumulator for reconstructed patch values.
    reconstruction_sum = np.zeros_like(
        volume,
        dtype=np.float32,
    )

    # Accumulator for the number of contributions at every voxel.
    reconstruction_count = np.zeros_like(
        volume,
        dtype=np.float32,
    )

    # Extract sparse coefficients for all sliding patches.
    patches = []

    locations = []

    for d in range(depth - pd + 1):

        for h in range(height - ph + 1):

            for w in range(width - pw + 1):

                # Extract the current patch.
                patch = volume[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                # Store its flattened representation.
                patches.append(
                    patch.reshape(-1)
                )

                # Remember where this patch came from.
                locations.append(
                    (d, h, w)
                )

    # Convert all patches to a matrix.
    patches_array = np.asarray(
        patches,
        dtype=np.float32,
    )

    # Dictionary learning requires finite values.
    if not np.isfinite(patches_array).all():
        raise RuntimeError(
            "Non-finite values detected in reconstruction patches."
        )

    # Transform every patch into sparse dictionary coefficients.
    sparse_codes = dictionary_model.transform(
        patches_array
    )

    # Reconstruct all patches from the learned dictionary.
    reconstructed_patches = (
        sparse_codes
        @ dictionary_model.components_
    )

    # Add reconstructed patches back into the volume.
    for index, (d, h, w) in enumerate(locations):

        reconstructed_patch = reconstructed_patches[
            index
        ].reshape(
            pd,
            ph,
            pw,
        )

        reconstruction_sum[
            d:d + pd,
            h:h + ph,
            w:w + pw,
        ] += reconstructed_patch

        reconstruction_count[
            d:d + pd,
            h:h + ph,
            w:w + pw,
        ] += 1.0

    # Avoid division by zero.
    valid = reconstruction_count > 0

    reconstruction = np.zeros_like(
        reconstruction_sum,
        dtype=np.float32,
    )

    reconstruction[valid] = (
        reconstruction_sum[valid]
        / reconstruction_count[valid]
    )

    # For any boundary location that did not receive a patch,
    # retain the original input value.
    reconstruction[~valid] = volume[~valid]

    # -----------------------------------------------------------------
    # Exact data consistency
    # -----------------------------------------------------------------

    # Restore every observed seismic sample exactly.
    reconstruction[mask == 1] = volume[mask == 1]

    return reconstruction


# ---------------------------------------------------------------------
# Public reconstruction function
# ---------------------------------------------------------------------

def dictionary_learning_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    patch_size: Tuple[int, int, int] = DEFAULT_PATCH_SIZE,
    n_components: int = DEFAULT_N_COMPONENTS,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_training_patches: int = DEFAULT_MAX_TRAINING_PATCHES,
    min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> torch.Tensor:
    """
    Reconstruct a 3-D seismic volume using dictionary learning.

    Parameters
    ----------
    corrupted_cube:
        Incomplete seismic tensor with shape (C, D, H, W).

    mask:
        Binary observation mask with:
            1 = observed
            0 = missing

    patch_size:
        3-D dictionary-learning patch size.

    n_components:
        Number of dictionary atoms.

    alpha:
        Sparse coding regularization parameter.

    max_iter:
        Number of dictionary-learning iterations.

    batch_size:
        Mini-batch size.

    max_training_patches:
        Maximum number of patches used for dictionary learning.

    min_observed_fraction:
        Minimum observed fraction required for a patch to be
        included in dictionary training.

    random_state:
        Reproducibility seed.

    Returns
    -------
    torch.Tensor
        Reconstructed cube with the same shape, dtype and
        device as the input.
    """

    # -------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------

    _validate_inputs(
        corrupted_cube,
        mask,
    )

    # -------------------------------------------------------------
    # Validate parameters
    # -------------------------------------------------------------

    if len(patch_size) != 3:
        raise ValueError(
            "patch_size must contain exactly three integers."
        )

    if any(
        int(value) <= 0
        for value in patch_size
    ):
        raise ValueError(
            "All patch dimensions must be positive."
        )

    if n_components <= 0:
        raise ValueError(
            "n_components must be positive."
        )

    if max_iter <= 0:
        raise ValueError(
            "max_iter must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if max_training_patches <= 0:
        raise ValueError(
            "max_training_patches must be positive."
        )

    if not (
        0.0
        < min_observed_fraction
        <= 1.0
    ):
        raise ValueError(
            "min_observed_fraction must be in (0, 1]."
        )

    # -------------------------------------------------------------
    # Preserve original tensor properties
    # -------------------------------------------------------------

    original_device = corrupted_cube.device

    original_dtype = corrupted_cube.dtype

    # Convert to CPU float32 for scikit-learn.
    corrupted_numpy = (
        corrupted_cube.detach()
        .cpu()
        .float()
        .numpy()
    )

    mask_numpy = (
        mask.detach()
        .cpu()
        .float()
        .numpy()
    )

    channels = corrupted_numpy.shape[0]

    reconstructed_channels = []

    # -------------------------------------------------------------
    # Process every seismic channel independently
    # -------------------------------------------------------------

    for channel_index in range(channels):

        # Extract one seismic channel.
        channel = corrupted_numpy[
            channel_index
        ]

        # Extract the corresponding observation mask.
        channel_mask = mask_numpy[
            channel_index
        ]

        # ---------------------------------------------------------
        # Extract sufficiently observed training patches
        # ---------------------------------------------------------

        training_patches, _ = _extract_3d_patches(
            volume=channel,
            mask=channel_mask,
            patch_size=patch_size,
            min_observed_fraction=min_observed_fraction,
        )

        # ---------------------------------------------------------
        # Fallback if no sufficiently observed patches exist
        # ---------------------------------------------------------

        if len(training_patches) == 0:

            # No reliable training patches are available.
            #
            # In this exceptional case we return the zero-filled
            # input while preserving observed samples exactly.
            #
            # The controlled benchmark will therefore expose this
            # situation through its reconstruction metrics.
            reconstruction = channel.copy()

            reconstruction[channel_mask == 0] = 0.0

            reconstructed_channels.append(
                reconstruction
            )

            continue

        # ---------------------------------------------------------
        # Limit the number of training patches
        # ---------------------------------------------------------

        training_patches = _select_training_patches(
            patches=training_patches,
            max_training_patches=max_training_patches,
            random_state=random_state,
        )

        # ---------------------------------------------------------
        # Check dictionary dimensionality
        # ---------------------------------------------------------

        patch_dimension = int(
            np.prod(patch_size)
        )

        if training_patches.shape[1] != patch_dimension:
            raise RuntimeError(
                "Unexpected training-patch dimension: "
                f"{training_patches.shape[1]}. "
                f"Expected: {patch_dimension}."
            )

        # ---------------------------------------------------------
        # Learn dictionary
        # ---------------------------------------------------------

        dictionary_model = _learn_dictionary(
            training_patches=training_patches,
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state,
        )

        # ---------------------------------------------------------
        # Reconstruct the channel
        # ---------------------------------------------------------

        reconstruction = _reconstruct_channel(
            volume=channel,
            mask=channel_mask,
            dictionary_model=dictionary_model,
            patch_size=patch_size,
        )

        # ---------------------------------------------------------
        # Final finite-value check
        # ---------------------------------------------------------

        if not np.isfinite(reconstruction).all():
            raise RuntimeError(
                "Dictionary-learning reconstruction produced "
                "non-finite values."
            )

        # ---------------------------------------------------------
        # Final exact observed-data enforcement
        # ---------------------------------------------------------

        reconstruction[channel_mask == 1] = (
            channel[channel_mask == 1]
        )

        reconstructed_channels.append(
            reconstruction
        )

    # -------------------------------------------------------------
    # Reassemble channels
    # -------------------------------------------------------------

    reconstructed_numpy = np.stack(
        reconstructed_channels,
        axis=0,
    )

    # -------------------------------------------------------------
    # Convert back to PyTorch
    # -------------------------------------------------------------

    reconstructed_tensor = torch.from_numpy(
        reconstructed_numpy
    )

    # Restore original dtype and device.
    reconstructed_tensor = (
        reconstructed_tensor
        .to(
            device=original_device,
            dtype=original_dtype,
        )
    )

    # -------------------------------------------------------------
    # Final shape validation
    # -------------------------------------------------------------

    if reconstructed_tensor.shape != corrupted_cube.shape:
        raise RuntimeError(
            "Dictionary-learning reconstruction changed "
            "the input shape. "
            f"Input: {tuple(corrupted_cube.shape)}, "
            f"Output: {tuple(reconstructed_tensor.shape)}."
        )

    # -------------------------------------------------------------
    # Final observed-data validation
    # -------------------------------------------------------------

    observed_difference = torch.max(
        torch.abs(
            reconstructed_tensor[mask == 1]
            - corrupted_cube[mask == 1]
        )
    )

    if observed_difference.item() != 0.0:
        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum difference: "
            f"{observed_difference.item():.6e}"
        )

    return reconstructed_tensor


# ---------------------------------------------------------------------
# Public aliases
# ---------------------------------------------------------------------

__all__ = [
    "dictionary_learning_reconstruction",
]