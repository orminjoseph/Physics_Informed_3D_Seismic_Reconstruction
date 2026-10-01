"""
=================================================================
Dictionary Learning Controlled-Matrix Baseline
=================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Baseline method:
    Dictionary Learning

Methodological basis:
    Sparse representation of 3-D seismic patches using a learned
    dictionary and sparse coefficients.

Input convention:
    corrupted_cube : (C, D, H, W)
    mask           : (C, D, H, W)

Mask convention:
    1 -> observed
    0 -> missing

Controlled experimental design:
    Geological modes:
        horizontal
        dipping
        faulted
        folded
        complex
        highly_complex

    Missing-data mechanisms:
        random_voxels
        missing_traces
        missing_inlines
        missing_crosslines
        missing_blocks

    Missing rates:
        10%, 20%, 30%, 40%, 50%

    Seeds:
        42, 43, 44, 45, 46

    Total:
        6 × 5 × 5 × 5 = 750 cases

Smoke test:
    CONTROLLED_MATRIX_CASE_LIMIT = 10

Final experiment:
    CONTROLLED_MATRIX_CASE_LIMIT = None

Important methodological rule:
    The dictionary is learned ONLY from the corrupted input.

    Ground-truth/target data are NEVER supplied to the
    dictionary-learning algorithm.

Observed seismic samples are restored exactly after
reconstruction.

Author: Ormin Joseph
=================================================================
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ==============================================================
# SCIKIT-LEARN
# ==============================================================

try:

    from sklearn.decomposition import (
        MiniBatchDictionaryLearning,
    )

except ImportError as exc:

    raise ImportError(
        "scikit-learn is required for the Dictionary "
        "Learning baseline.\n\n"
        "Install it with:\n"
        "python -m pip install scikit-learn"
    ) from exc


# ==============================================================
# CENTRAL CONFIGURATION
# ==============================================================

from utils.config import (
    BASELINE_NUM_SAMPLES,
    BASELINE_CUBE_SIZE,
    OBSERVED_PRESERVATION_TOLERANCE,
    REPORT_DIR,
    CONTROLLED_MATRIX_CASE_LIMIT,
)


# ==============================================================
# CONTROLLED EXPERIMENT FACTORS
# ==============================================================

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
]


MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


# ==============================================================
# DICTIONARY LEARNING PARAMETERS
# ==============================================================

DEFAULT_PATCH_SIZE = (
    8,
    8,
    8,
)

DEFAULT_N_COMPONENTS = 64

DEFAULT_ALPHA = 1.0

DEFAULT_MAX_ITER = 20

DEFAULT_BATCH_SIZE = 64

DEFAULT_MAX_TRAINING_PATCHES = 2000

DEFAULT_MIN_OBSERVED_FRACTION = 0.80

DEFAULT_TRANSFORM_NONZERO_COEFFICIENTS = 8

DEFAULT_RANDOM_STATE = 42


# ==============================================================
# REPRODUCIBILITY
# ==============================================================

def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==============================================================
# FINITE-VALUE VALIDATION
# ==============================================================

def check_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.isfinite(
        tensor
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# ==============================================================
# INPUT VALIDATION
# ==============================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Validate corrupted seismic cube and observation mask.
    """

    if not isinstance(
        corrupted_cube,
        torch.Tensor,
    ):

        raise TypeError(
            "corrupted_cube must be a torch.Tensor."
        )

    if not isinstance(
        mask,
        torch.Tensor,
    ):

        raise TypeError(
            "mask must be a torch.Tensor."
        )

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received {tuple(corrupted_cube.shape)}."
        )

    if mask.ndim != 4:

        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received {tuple(mask.shape)}."
        )

    if corrupted_cube.shape != mask.shape:

        raise ValueError(
            "corrupted_cube and mask must have "
            "identical shapes.\n"
            f"Cube: {tuple(corrupted_cube.shape)}\n"
            f"Mask: {tuple(mask.shape)}"
        )

    check_finite(
        corrupted_cube,
        "corrupted_cube",
    )

    check_finite(
        mask,
        "mask",
    )

    unique_mask = torch.unique(
        mask
    )

    if not torch.all(
        (unique_mask == 0)
        |
        (unique_mask == 1)
    ):

        raise ValueError(
            "mask must contain only 0 and 1. "
            f"Found: {unique_mask.tolist()}"
        )

    if not torch.any(
        mask == 1
    ):

        raise ValueError(
            "mask contains no observed samples."
        )


# ==============================================================
# PARAMETER VALIDATION
# ==============================================================

def _validate_parameters(
    patch_size: Tuple[int, int, int],
    n_components: int,
    alpha: float,
    max_iter: int,
    batch_size: int,
    max_training_patches: int,
    min_observed_fraction: float,
) -> None:
    """
    Validate dictionary-learning parameters.
    """

    if len(patch_size) != 3:

        raise ValueError(
            "patch_size must contain exactly three dimensions."
        )

    if any(
        int(value) <= 0
        for value in patch_size
    ):

        raise ValueError(
            "All patch dimensions must be positive."
        )

    if n_components < 1:

        raise ValueError(
            "n_components must be >= 1."
        )

    if alpha < 0:

        raise ValueError(
            "alpha must be >= 0."
        )

    if max_iter < 1:

        raise ValueError(
            "max_iter must be >= 1."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be >= 1."
        )

    if max_training_patches < 1:

        raise ValueError(
            "max_training_patches must be >= 1."
        )

    if not (
        0.0
        <
        min_observed_fraction
        <=
        1.0
    ):

        raise ValueError(
            "min_observed_fraction must satisfy "
            "0 < value <= 1."
        )


# ==============================================================
# 3-D PATCH EXTRACTION
# ==============================================================

def _extract_3d_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    min_observed_fraction: float,
) -> np.ndarray:
    """
    Extract sufficiently observed 3-D seismic patches.

    Only patches satisfying the minimum observed-fraction
    criterion are returned.

    Ground truth is not used.
    """

    pd, ph, pw = patch_size

    depth, height, width = (
        volume.shape
    )

    if (
        pd > depth
        or ph > height
        or pw > width
    ):

        raise ValueError(
            "Patch size exceeds the input volume.\n"
            f"Volume: {volume.shape}\n"
            f"Patch: {patch_size}"
        )

    patches = []

    for d in range(
        depth - pd + 1
    ):

        for h in range(
            height - ph + 1
        ):

            for w in range(
                width - pw + 1
            ):

                patch = volume[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                mask_patch = mask[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                observed_fraction = float(
                    np.mean(
                        mask_patch
                    )
                )

                if (
                    observed_fraction
                    >=
                    min_observed_fraction
                ):

                    patches.append(
                        patch.reshape(-1)
                    )

    if len(patches) == 0:

        return np.empty(
            (
                0,
                pd * ph * pw,
            ),
            dtype=np.float32,
        )

    return np.asarray(
        patches,
        dtype=np.float32,
    )


# ==============================================================
# TRAINING-PATCH SUBSAMPLING
# ==============================================================

def _select_training_patches(
    patches: np.ndarray,
    max_training_patches: int,
    random_state: int,
) -> np.ndarray:
    """
    Limit dictionary-learning patches while maintaining
    reproducibility.
    """

    if len(patches) <= max_training_patches:

        return patches

    rng = np.random.default_rng(
        random_state
    )

    indices = rng.choice(
        len(patches),
        size=max_training_patches,
        replace=False,
    )

    return patches[
        indices
    ]


# ==============================================================
# DICTIONARY LEARNING
# ==============================================================

def _learn_dictionary(
    training_patches: np.ndarray,
    n_components: int,
    alpha: float,
    max_iter: int,
    batch_size: int,
    transform_nonzero_coefficients: int,
    random_state: int,
) -> MiniBatchDictionaryLearning:
    """
    Learn a sparse dictionary from corrupted seismic patches.

    The dictionary is learned exclusively from the corrupted
    input data.
    """

    number_of_training_patches = (
        training_patches.shape[0]
    )

    if number_of_training_patches < 1:

        raise ValueError(
            "No training patches were supplied."
        )

    # ----------------------------------------------------------
    # A dictionary cannot contain more useful atoms than
    # available training examples.
    # ----------------------------------------------------------

    effective_components = min(
        n_components,
        number_of_training_patches,
    )

    # ----------------------------------------------------------
    # The number of requested non-zero coefficients must not
    # exceed the number of dictionary atoms.
    # ----------------------------------------------------------

    effective_nonzero = min(
        transform_nonzero_coefficients,
        effective_components,
    )

    # ----------------------------------------------------------
    # MiniBatchDictionaryLearning requires a valid batch size.
    # ----------------------------------------------------------

    effective_batch_size = min(
        batch_size,
        number_of_training_patches,
    )

    dictionary_model = (
        MiniBatchDictionaryLearning(
            n_components=effective_components,
            alpha=alpha,
            max_iter=max_iter,
            batch_size=effective_batch_size,
            random_state=random_state,
            fit_algorithm="lars",
            transform_algorithm="omp",
            transform_n_nonzero_coefs=effective_nonzero,
            verbose=False,
        )
    )

    dictionary_model.fit(
        training_patches
    )

    return dictionary_model


# ==============================================================
# PATCH RECONSTRUCTION
# ==============================================================

def _reconstruct_channel(
    volume: np.ndarray,
    mask: np.ndarray,
    dictionary_model: MiniBatchDictionaryLearning,
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct one seismic channel from dictionary atoms.

    All sliding patches from the corrupted input are transformed
    into sparse dictionary coefficients.

    Overlapping reconstructed patches are averaged.
    """

    pd, ph, pw = patch_size

    depth, height, width = (
        volume.shape
    )

    reconstruction_sum = np.zeros(
        volume.shape,
        dtype=np.float32,
    )

    reconstruction_count = np.zeros(
        volume.shape,
        dtype=np.float32,
    )

    patches = []

    locations = []

    # ----------------------------------------------------------
    # Extract every patch from the corrupted volume.
    # ----------------------------------------------------------

    for d in range(
        depth - pd + 1
    ):

        for h in range(
            height - ph + 1
        ):

            for w in range(
                width - pw + 1
            ):

                patch = volume[
                    d:d + pd,
                    h:h + ph,
                    w:w + pw,
                ]

                patches.append(
                    patch.reshape(-1)
                )

                locations.append(
                    (
                        d,
                        h,
                        w,
                    )
                )

    if len(patches) == 0:

        raise RuntimeError(
            "No reconstruction patches could be extracted."
        )

    patches_array = np.asarray(
        patches,
        dtype=np.float32,
    )

    if not np.isfinite(
        patches_array
    ).all():

        raise RuntimeError(
            "Reconstruction patches contain "
            "NaN or Inf values."
        )

    # ----------------------------------------------------------
    # Sparse dictionary encoding.
    # ----------------------------------------------------------

    sparse_codes = (
        dictionary_model.transform(
            patches_array
        )
    )

    if not np.isfinite(
        sparse_codes
    ).all():

        raise RuntimeError(
            "Dictionary transform produced "
            "NaN or Inf values."
        )

    # ----------------------------------------------------------
    # Reconstruct patches from dictionary atoms.
    # ----------------------------------------------------------

    reconstructed_patches = (
        sparse_codes
        @
        dictionary_model.components_
    )

    if not np.isfinite(
        reconstructed_patches
    ).all():

        raise RuntimeError(
            "Dictionary reconstruction produced "
            "NaN or Inf values."
        )

    # ----------------------------------------------------------
    # Aggregate overlapping patches.
    # ----------------------------------------------------------

    for index, (
        d,
        h,
        w,
    ) in enumerate(
        locations
    ):

        reconstructed_patch = (
            reconstructed_patches[
                index
            ]
            .reshape(
                pd,
                ph,
                pw,
            )
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

    # ----------------------------------------------------------
    # Average overlapping patch contributions.
    # ----------------------------------------------------------

    valid = (
        reconstruction_count > 0
    )

    reconstruction = np.zeros(
        volume.shape,
        dtype=np.float32,
    )

    reconstruction[
        valid
    ] = (
        reconstruction_sum[
            valid
        ]
        /
        reconstruction_count[
            valid
        ]
    )

    # ----------------------------------------------------------
    # Retain original values where no patch contributed.
    # ----------------------------------------------------------

    reconstruction[
        ~valid
    ] = volume[
        ~valid
    ]

    # ----------------------------------------------------------
    # Exact observed-data consistency.
    # ----------------------------------------------------------

    reconstruction[
        mask == 1
    ] = volume[
        mask == 1
    ]

    return reconstruction


# ==============================================================
# PUBLIC DICTIONARY LEARNING RECONSTRUCTION
# ==============================================================

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
    transform_nonzero_coefficients: int = (
        DEFAULT_TRANSFORM_NONZERO_COEFFICIENTS
    ),
    random_state: int = DEFAULT_RANDOM_STATE,
) -> torch.Tensor:
    """
    Reconstruct a 3-D seismic cube using dictionary learning.

    Ground-truth data are never supplied to the algorithm.

    Parameters
    ----------
    corrupted_cube:
        Incomplete seismic cube with shape (C, D, H, W).

    mask:
        Binary observation mask.

        1 = observed
        0 = missing

    patch_size:
        3-D patch dimensions.

    n_components:
        Maximum number of dictionary atoms.

    alpha:
        Sparse coding regularization parameter.

    max_iter:
        Maximum dictionary-learning iterations.

    batch_size:
        Mini-batch size.

    max_training_patches:
        Maximum number of training patches.

    min_observed_fraction:
        Minimum observed fraction required for training patches.

    transform_nonzero_coefficients:
        Maximum number of non-zero coefficients used by OMP.

    random_state:
        Reproducibility seed.

    Returns
    -------
    torch.Tensor
        Reconstructed cube with the same shape, dtype and
        device as the input.
    """

    # ----------------------------------------------------------
    # Validate input.
    # ----------------------------------------------------------

    _validate_inputs(
        corrupted_cube,
        mask,
    )

    # ----------------------------------------------------------
    # Validate parameters.
    # ----------------------------------------------------------

    _validate_parameters(
        patch_size=patch_size,
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        batch_size=batch_size,
        max_training_patches=max_training_patches,
        min_observed_fraction=min_observed_fraction,
    )

    if transform_nonzero_coefficients < 1:

        raise ValueError(
            "transform_nonzero_coefficients must be >= 1."
        )

    # ----------------------------------------------------------
    # Preserve original tensor properties.
    # ----------------------------------------------------------

    original_device = (
        corrupted_cube.device
    )

    original_dtype = (
        corrupted_cube.dtype
    )

    # ----------------------------------------------------------
    # Convert to CPU float32 for scikit-learn.
    # ----------------------------------------------------------

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

    reconstructed_channels = []

    number_of_channels = (
        corrupted_numpy.shape[0]
    )

    # ==========================================================
    # PROCESS CHANNELS
    # ==========================================================

    for channel_index in range(
        number_of_channels
    ):

        channel = (
            corrupted_numpy[
                channel_index
            ]
        )

        channel_mask = (
            mask_numpy[
                channel_index
            ]
        )

        # ------------------------------------------------------
        # Learn dictionary from sufficiently observed patches.
        # ------------------------------------------------------

        training_patches = (
            _extract_3d_patches(
                volume=channel,
                mask=channel_mask,
                patch_size=patch_size,
                min_observed_fraction=(
                    min_observed_fraction
                ),
            )
        )

        # ------------------------------------------------------
        # No suitable patches.
        # ------------------------------------------------------

        if len(
            training_patches
        ) == 0:

            # --------------------------------------------------
            # This is a legitimate failure/fallback condition.
            #
            # Observed data are retained exactly.
            # Missing samples remain at their corrupted-input
            # initialization, which is normally zero in the
            # supplied synthetic dataset.
            # --------------------------------------------------

            reconstruction = (
                channel.copy()
            )

            reconstruction[
                channel_mask == 0
            ] = 0.0

            reconstructed_channels.append(
                reconstruction
            )

            continue

        # ------------------------------------------------------
        # Bound the number of training patches.
        # ------------------------------------------------------

        training_patches = (
            _select_training_patches(
                patches=training_patches,
                max_training_patches=(
                    max_training_patches
                ),
                random_state=random_state,
            )
        )

        # ------------------------------------------------------
        # Learn dictionary.
        # ------------------------------------------------------

        dictionary_model = (
            _learn_dictionary(
                training_patches=training_patches,
                n_components=n_components,
                alpha=alpha,
                max_iter=max_iter,
                batch_size=batch_size,
                transform_nonzero_coefficients=(
                    transform_nonzero_coefficients
                ),
                random_state=random_state,
            )
        )

        # ------------------------------------------------------
        # Reconstruct channel.
        # ------------------------------------------------------

        reconstruction = (
            _reconstruct_channel(
                volume=channel,
                mask=channel_mask,
                dictionary_model=dictionary_model,
                patch_size=patch_size,
            )
        )

        # ------------------------------------------------------
        # Numerical validation.
        # ------------------------------------------------------

        if not np.isfinite(
            reconstruction
        ).all():

            raise RuntimeError(
                "Dictionary-learning reconstruction "
                "contains NaN or Inf values."
            )

        # ------------------------------------------------------
        # Final observed-data restoration.
        # ------------------------------------------------------

        reconstruction[
            channel_mask == 1
        ] = channel[
            channel_mask == 1
        ]

        reconstructed_channels.append(
            reconstruction
        )

    # ==========================================================
    # REASSEMBLE CHANNELS
    # ==========================================================

    reconstructed_numpy = np.stack(
        reconstructed_channels,
        axis=0,
    )

    if not np.isfinite(
        reconstructed_numpy
    ).all():

        raise RuntimeError(
            "Final dictionary-learning reconstruction "
            "contains NaN or Inf values."
        )

    # ==========================================================
    # CONVERT BACK TO PYTORCH
    # ==========================================================

    reconstructed = torch.from_numpy(
        reconstructed_numpy
    )

    reconstructed = reconstructed.to(
        device=original_device,
        dtype=original_dtype,
    )

    # ==========================================================
    # SHAPE VALIDATION
    # ==========================================================

    if reconstructed.shape != (
        corrupted_cube.shape
    ):

        raise RuntimeError(
            "Dictionary-learning reconstruction changed "
            "the input shape.\n"
            f"Input: {tuple(corrupted_cube.shape)}\n"
            f"Output: {tuple(reconstructed.shape)}"
        )

    # ==========================================================
    # EXACT OBSERVED-DATA VALIDATION
    # ==========================================================

    observed_values = (
        mask == 1
    )

    if torch.any(
        observed_values
    ):

        observed_difference = torch.max(
            torch.abs(
                reconstructed[
                    observed_values
                ]
                -
                corrupted_cube[
                    observed_values
                ]
            )
        )

        if (
            observed_difference.item()
            >
            OBSERVED_PRESERVATION_TOLERANCE
        ):

            raise RuntimeError(
                "Dictionary-learning reconstruction failed "
                "to preserve observed samples.\n"
                f"Maximum difference: "
                f"{observed_difference.item():.6e}"
            )

    return reconstructed


# ==============================================================
# METRICS
# ==============================================================

def _mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(
        torch.mean(
            torch.abs(
                prediction - target
            )
        ).item()
    )


def _rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(
        torch.sqrt(
            torch.mean(
                (
                    prediction - target
                ) ** 2
            )
        ).item()
    )


def _psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> float:

    mse = torch.mean(
        (
            prediction - target
        ) ** 2
    ).item()

    if mse <= 0.0:

        return float("inf")

    return float(
        20.0
        *
        math.log10(
            data_range
        )
        -
        10.0
        *
        math.log10(
            mse
        )
    )


def _snr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    signal_power = torch.mean(
        target ** 2
    ).item()

    noise_power = torch.mean(
        (
            target - prediction
        ) ** 2
    ).item()

    if noise_power <= 0.0:

        return float("inf")

    if signal_power <= 0.0:

        return float("-inf")

    return float(
        10.0
        *
        math.log10(
            signal_power
            /
            noise_power
        )
    )


def _ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """
    Global SSIM-style metric consistent with the existing
    controlled baseline framework.
    """

    x = prediction.float()
    y = target.float()

    mean_x = torch.mean(
        x
    )

    mean_y = torch.mean(
        y
    )

    variance_x = torch.var(
        x,
        unbiased=False,
    )

    variance_y = torch.var(
        y,
        unbiased=False,
    )

    covariance = torch.mean(
        (
            x - mean_x
        )
        *
        (
            y - mean_y
        )
    )

    c1 = (
        0.01
        *
        data_range
    ) ** 2

    c2 = (
        0.03
        *
        data_range
    ) ** 2

    numerator = (
        (
            2.0
            *
            mean_x
            *
            mean_y
            +
            c1
        )
        *
        (
            2.0
            *
            covariance
            +
            c2
        )
    )

    denominator = (
        (
            mean_x ** 2
            +
            mean_y ** 2
            +
            c1
        )
        *
        (
            variance_x
            +
            variance_y
            +
            c2
        )
    )

    if denominator.item() == 0.0:

        return 1.0

    return float(
        (
            numerator
            /
            denominator
        ).item()
    )


def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate global and missing-region metrics.
    """

    reconstruction = (
        reconstruction.float()
    )

    target = (
        target.float()
    )

    mask = (
        mask.float()
    )

    metrics = {
        "MAE": _mae(
            reconstruction,
            target,
        ),
        "RMSE": _rmse(
            reconstruction,
            target,
        ),
        "PSNR": _psnr(
            reconstruction,
            target,
        ),
        "SNR": _snr(
            reconstruction,
            target,
        ),
        "SSIM": _ssim(
            reconstruction,
            target,
        ),
    }

    missing = (
        mask == 0
    )

    if torch.any(
        missing
    ):

        missing_prediction = (
            reconstruction[
                missing
            ]
        )

        missing_target = (
            target[
                missing
            ]
        )

        metrics[
            "Missing_MAE"
        ] = float(
            torch.mean(
                torch.abs(
                    missing_prediction
                    -
                    missing_target
                )
            ).item()
        )

        metrics[
            "Missing_RMSE"
        ] = float(
            torch.sqrt(
                torch.mean(
                    (
                        missing_prediction
                        -
                        missing_target
                    ) ** 2
                )
            ).item()
        )

    else:

        metrics[
            "Missing_MAE"
        ] = 0.0

        metrics[
            "Missing_RMSE"
        ] = 0.0

    return metrics


# ==============================================================
# CONTROLLED CASE GENERATION
# ==============================================================

def build_controlled_cases() -> List[dict]:
    """
    Construct the complete 750-case factorial design.

    6 geological modes
    × 5 missing mechanisms
    × 5 missing rates
    × 5 seeds
    = 750 cases
    """

    cases = []

    case_id = 1

    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    cases.append(
                        {
                            "Case_ID": case_id,
                            "geological_mode": (
                                geological_mode
                            ),
                            "mask_mode": (
                                mask_mode
                            ),
                            "requested_missing_rate": (
                                missing_rate
                            ),
                            "seed": seed,
                        }
                    )

                    case_id += 1

    return cases


# ==============================================================
# CONTROLLED CASE SELECTION
# ==============================================================

def select_cases_for_execution(
    cases: List[dict],
    case_limit,
) -> List[dict]:
    """
    Select the requested number of cases.

    None means all 750 cases.
    """

    if case_limit is None:

        return cases

    if not isinstance(
        case_limit,
        int,
    ):

        raise TypeError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "an integer or None."
        )

    if case_limit < 1:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            ">= 1 or None."
        )

    return cases[
        :case_limit
    ]


# ==============================================================
# SINGLE CONTROLLED EXPERIMENT
# ==============================================================

def run_single_experiment(
    case: dict,
) -> dict:
    """
    Execute one controlled dictionary-learning experiment.
    """

    from datasets.synthetic_dataset import (
        SyntheticSeismicDataset,
    )

    # ----------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------

    set_seed(
        case["seed"]
    )

    # ----------------------------------------------------------
    # Build the controlled synthetic dataset.
    # ----------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=BASELINE_NUM_SAMPLES,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=(
            case[
                "requested_missing_rate"
            ]
        ),
        geological_mode=(
            case[
                "geological_mode"
            ]
        ),
        mask_mode=(
            case[
                "mask_mode"
            ]
        ),
        seed=case[
            "seed"
        ],
    )

    (
        corrupted_cube,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    # Velocity is not required by dictionary learning.
    del velocity

    # ----------------------------------------------------------
    # Validate generated data.
    # ----------------------------------------------------------

    if corrupted_cube.shape != target.shape:

        raise RuntimeError(
            "Corrupted cube and target have different shapes."
        )

    if corrupted_cube.shape != mask.shape:

        raise RuntimeError(
            "Corrupted cube and mask have different shapes."
        )

    check_finite(
        corrupted_cube,
        "corrupted_cube",
    )

    check_finite(
        target,
        "target",
    )

    check_finite(
        mask,
        "mask",
    )

    # ----------------------------------------------------------
    # Actual missing rate.
    # ----------------------------------------------------------

    actual_missing_rate = float(
        torch.mean(
            (
                mask == 0
            ).float()
        ).item()
    )

    # ----------------------------------------------------------
    # Run dictionary-learning reconstruction.
    # ----------------------------------------------------------

    start_time = (
        time.perf_counter()
    )

    reconstruction = (
        dictionary_learning_reconstruction(
            corrupted_cube=corrupted_cube,
            mask=mask,
        )
    )

    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )

    check_finite(
        reconstruction,
        "reconstruction",
    )

    # ----------------------------------------------------------
    # Observed-data preservation.
    # ----------------------------------------------------------

    observed = (
        mask == 1
    )

    if torch.any(
        observed
    ):

        observed_preservation_error = float(
            torch.max(
                torch.abs(
                    reconstruction[
                        observed
                    ]
                    -
                    corrupted_cube[
                        observed
                    ]
                )
            ).item()
        )

    else:

        observed_preservation_error = 0.0

    if (
        observed_preservation_error
        >
        OBSERVED_PRESERVATION_TOLERANCE
    ):

        raise RuntimeError(
            "Observed-data preservation tolerance exceeded.\n"
            f"Maximum difference: "
            f"{observed_preservation_error:.6e}"
        )

    # ----------------------------------------------------------
    # Calculate metrics.
    # ----------------------------------------------------------

    metrics = calculate_metrics(
        reconstruction=reconstruction,
        target=target,
        mask=mask,
    )

    # ----------------------------------------------------------
    # Store results.
    # ----------------------------------------------------------

    return {
        "Case_ID": case[
            "Case_ID"
        ],
        "Method": (
            "Dictionary_Learning"
        ),
        "geological_mode": (
            actual_geological_mode
        ),
        "mask_mode": (
            actual_mask_mode
        ),
        "requested_missing_rate": (
            case[
                "requested_missing_rate"
            ]
        ),
        "actual_missing_rate": (
            actual_missing_rate
        ),
        "seed": case[
            "seed"
        ],
        "Num_Samples": (
            BASELINE_NUM_SAMPLES
        ),
        "Cube_D": (
            BASELINE_CUBE_SIZE[0]
        ),
        "Cube_H": (
            BASELINE_CUBE_SIZE[1]
        ),
        "Cube_W": (
            BASELINE_CUBE_SIZE[2]
        ),
        "MAE": metrics[
            "MAE"
        ],
        "RMSE": metrics[
            "RMSE"
        ],
        "PSNR": metrics[
            "PSNR"
        ],
        "SNR": metrics[
            "SNR"
        ],
        "SSIM": metrics[
            "SSIM"
        ],
        "Missing_MAE": metrics[
            "Missing_MAE"
        ],
        "Missing_RMSE": metrics[
            "Missing_RMSE"
        ],
        "Runtime_Seconds": (
            runtime_seconds
        ),
        "Observed_Preservation_Error": (
            observed_preservation_error
        ),
        "Patch_Size": (
            str(DEFAULT_PATCH_SIZE)
        ),
        "N_Components": (
            DEFAULT_N_COMPONENTS
        ),
        "Alpha": (
            DEFAULT_ALPHA
        ),
        "Max_Iterations": (
            DEFAULT_MAX_ITER
        ),
        "Batch_Size": (
            DEFAULT_BATCH_SIZE
        ),
        "Max_Training_Patches": (
            DEFAULT_MAX_TRAINING_PATCHES
        ),
        "Min_Observed_Fraction": (
            DEFAULT_MIN_OBSERVED_FRACTION
        ),
        "Transform_Nonzero_Coefficients": (
            DEFAULT_TRANSFORM_NONZERO_COEFFICIENTS
        ),
        "Status": "PASS",
    }


# ==============================================================
# SUMMARY STATISTICS
# ==============================================================

def calculate_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate grouped summary statistics.

    Groups:
        geological mode
        mask mode
        requested missing rate
    """

    group_columns = [
        "geological_mode",
        "mask_mode",
        "requested_missing_rate",
    ]

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "Missing_MAE",
        "Missing_RMSE",
        "Runtime_Seconds",
        "Observed_Preservation_Error",
    ]

    summary = (
        dataframe
        .groupby(
            group_columns,
            dropna=False,
        )[metric_columns]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        "_".join(
            column
        ).strip("_")
        if isinstance(
            column,
            tuple,
        )
        else column
        for column in summary.columns
    ]

    return summary


# ==============================================================
# WRITE RESULTS
# ==============================================================

def write_results(
    results: List[dict],
    report_directory: Path,
) -> None:
    """
    Write raw controlled-matrix results and grouped summary.
    """

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    )

    raw_path = (
        report_directory
        /
        "dictionary_learning_controlled_matrix.csv"
    )

    summary_path = (
        report_directory
        /
        "dictionary_learning_controlled_matrix_summary.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False,
    )

    summary = calculate_summary(
        dataframe
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nRaw results saved to:"
    )

    print(
        raw_path
    )

    print(
        "\nSummary results saved to:"
    )

    print(
        summary_path
    )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:
    """
    Execute the controlled dictionary-learning experiment.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DICTIONARY LEARNING CONTROLLED MATRIX"
    )

    print(
        "=" * 70
    )