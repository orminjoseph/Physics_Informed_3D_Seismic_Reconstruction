"""
======================================================================
DATASET BUILDER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Construct the seismic dataset selected by the global configuration.

Supported dataset modes
-----------------------

1. synthetic
2. f3

The active dataset mode is controlled by:

    DATASET_MODE

in:

    utils/config.py

The builder does not perform the train/validation split.

Dataset workflow
----------------

    Global Configuration
            |
            v
    build_dataset()
            |
            v
    Complete Dataset
            |
            v
    split_dataset()
            |
            +------------------+
            |                  |
            v                  v
       Training Set       Validation Set

Tensor convention
-----------------

Individual dataset samples are expected to follow:

    [C, D, H, W]

where:

    C = channel
    D = depth
    H = crossline
    W = inline

The DataLoader subsequently creates:

    [B, C, D, H, W]

where:

    B = batch size

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import os


# =====================================================================
# PROJECT CONFIGURATION
# =====================================================================

from utils.config import (
    DATASET_MODE,

    # ---------------------------------------------------------------
    # Synthetic dataset configuration
    # ---------------------------------------------------------------

    SYNTHETIC_NUM_SAMPLES,
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,

    # ---------------------------------------------------------------
    # F3 dataset configuration
    # ---------------------------------------------------------------

    F3_PATH,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY
)


# =====================================================================
# SYNTHETIC DATASET
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)


# =====================================================================
# DATASET BUILDER
# =====================================================================

def build_dataset():
    """
    Construct the complete dataset selected by DATASET_MODE.

    Returns
    -------
    torch.utils.data.Dataset
        Complete seismic dataset before train/validation splitting.

    Raises
    ------
    ValueError
        If DATASET_MODE is not supported.

    FileNotFoundError
        If DATASET_MODE is 'f3' and the configured SEG-Y file
        does not exist.

    Notes
    -----
    This function intentionally does not perform dataset splitting.

    The returned dataset should subsequently be passed to:

        split_dataset(dataset)

    so that training and validation data remain explicitly separated.
    """

    # =================================================================
    # DISPLAY DATASET CONFIGURATION
    # =================================================================

    print()
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    print(
        f"Dataset mode: {DATASET_MODE}"
    )

    # -----------------------------------------------------------------
    # Normalize mode string.
    # -----------------------------------------------------------------

    if not isinstance(
        DATASET_MODE,
        str
    ):

        raise TypeError(
            "DATASET_MODE must be a string."
        )

    mode = DATASET_MODE.strip().lower()

    # =================================================================
    # VALIDATE DATASET MODE
    # =================================================================

    supported_modes = {
        "synthetic",
        "f3"
    }

    if mode not in supported_modes:

        raise ValueError(
            f"Unsupported DATASET_MODE: '{DATASET_MODE}'.\n"
            f"Supported modes are: "
            f"{sorted(supported_modes)}"
        )

    # =================================================================
    # 1. SYNTHETIC DATASET
    # =================================================================

    if mode == "synthetic":

        # -------------------------------------------------------------
        # Validate synthetic configuration.
        # -------------------------------------------------------------

        if SYNTHETIC_NUM_SAMPLES < 1:

            raise ValueError(
                "SYNTHETIC_NUM_SAMPLES must be at least 1."
            )

        if (
            not isinstance(
                SYNTHETIC_PATCH_SIZE,
                (tuple, list)
            )
            or
            len(SYNTHETIC_PATCH_SIZE) != 3
        ):

            raise ValueError(
                "SYNTHETIC_PATCH_SIZE must contain exactly "
                "three dimensions: "
                "(Depth, Crossline, Inline)."
            )

        if any(
            dimension <= 0
            for dimension in SYNTHETIC_PATCH_SIZE
        ):

            raise ValueError(
                "All dimensions in SYNTHETIC_PATCH_SIZE "
                "must be greater than zero."
            )

        if not (
            0.0
            <=
            SYNTHETIC_MISSING_PROBABILITY
            <=
            1.0
        ):

            raise ValueError(
                "SYNTHETIC_MISSING_PROBABILITY must be "
                "between 0 and 1."
            )

        # -------------------------------------------------------------
        # Display synthetic configuration.
        # -------------------------------------------------------------

        print()
        print("Synthetic dataset configuration")
        print("-" * 60)

        print(
            f"Number of samples       : "
            f"{SYNTHETIC_NUM_SAMPLES}"
        )

        print(
            f"Patch size              : "
            f"{SYNTHETIC_PATCH_SIZE}"
        )

        print(
            f"Missing probability     : "
            f"{SYNTHETIC_MISSING_PROBABILITY}"
        )

        # -------------------------------------------------------------
        # Construct synthetic dataset.
        # -------------------------------------------------------------

        dataset = SyntheticSeismicDataset(

            num_samples=
            SYNTHETIC_NUM_SAMPLES,

            cube_size=
            tuple(
                SYNTHETIC_PATCH_SIZE
            ),

            missing_probability=
            SYNTHETIC_MISSING_PROBABILITY
        )

        # -------------------------------------------------------------
        # Validate resulting dataset.
        # -------------------------------------------------------------

        if len(dataset) == 0:

            raise RuntimeError(
                "Synthetic dataset was created but contains "
                "zero samples."
            )

        print()
        print(
            f"Synthetic dataset created successfully."
        )

        print(
            f"Total samples: {len(dataset)}"
        )

        print("=" * 60)
        print()

        return dataset

    # =================================================================
    # 2. F3 DATASET
    # =================================================================

    if mode == "f3":

        # -------------------------------------------------------------
        # Validate F3 SEG-Y path.
        # -------------------------------------------------------------

        if not isinstance(
            F3_PATH,
            str
        ):

            raise TypeError(
                "F3_PATH must be a string."
            )

        if not os.path.isfile(
            F3_PATH
        ):

            raise FileNotFoundError(
                "F3 SEG-Y dataset was not found.\n"
                f"Configured path:\n"
                f"{F3_PATH}"
            )

        # -------------------------------------------------------------
        # Validate F3 patch size.
        # -------------------------------------------------------------

        if (
            not isinstance(
                F3_PATCH_SIZE,
                (tuple, list)
            )
            or
            len(F3_PATCH_SIZE) != 3
        ):

            raise ValueError(
                "F3_PATCH_SIZE must contain exactly "
                "three dimensions: "
                "(Depth, Crossline, Inline)."
            )

        if any(
            dimension <= 0
            for dimension in F3_PATCH_SIZE
        ):

            raise ValueError(
                "All dimensions in F3_PATCH_SIZE "
                "must be greater than zero."
            )

        # -------------------------------------------------------------
        # Validate F3 stride.
        # -------------------------------------------------------------

        if (
            not isinstance(
                F3_STRIDE,
                (tuple, list)
            )
            or
            len(F3_STRIDE) != 3
        ):

            raise ValueError(
                "F3_STRIDE must contain exactly "
                "three dimensions: "
                "(Depth, Crossline, Inline)."
            )

        if any(
            stride <= 0
            for stride in F3_STRIDE
        ):

            raise ValueError(
                "All dimensions in F3_STRIDE "
                "must be greater than zero."
            )

        # -------------------------------------------------------------
        # Validate missing-data probability.
        # -------------------------------------------------------------

        if not (
            0.0
            <=
            F3_MISSING_PROBABILITY
            <=
            1.0
        ):

            raise ValueError(
                "F3_MISSING_PROBABILITY must be "
                "between 0 and 1."
            )

        # -------------------------------------------------------------
        # Display F3 configuration.
        # -------------------------------------------------------------

        print()
        print("F3 dataset configuration")
        print("-" * 60)

        print(
            f"SEG-Y path              : "
            f"{F3_PATH}"
        )

        print(
            f"Patch size              : "
            f"{F3_PATCH_SIZE}"
        )

        print(
            f"Stride                  : "
            f"{F3_STRIDE}"
        )

        print(
            f"Missing probability     : "
            f"{F3_MISSING_PROBABILITY}"
        )

        # -------------------------------------------------------------
        # Import F3 dataset only when required.
        # -------------------------------------------------------------
        #
        # This avoids requiring F3-specific dependencies when
        # operating in synthetic mode.

        from dataset.f3_dataset import (
            F3Dataset
        )

        # -------------------------------------------------------------
        # Construct F3 dataset.
        # -------------------------------------------------------------

        dataset = F3Dataset(

            segy_path=
            F3_PATH,

            patch_size=
            tuple(
                F3_PATCH_SIZE
            ),

            stride=
            tuple(
                F3_STRIDE
            ),

            missing_probability=
            F3_MISSING_PROBABILITY
        )

        # -------------------------------------------------------------
        # Validate resulting dataset.
        # -------------------------------------------------------------

        if len(dataset) == 0:

            raise RuntimeError(
                "F3 dataset was created but contains "
                "zero patches."
            )

        print()
        print(
            "F3 dataset created successfully."
        )

        print(
            f"Total patches: {len(dataset)}"
        )

        print("=" * 60)
        print()

        return dataset

    # =================================================================
    # SAFETY FALLBACK
    # =================================================================

    raise RuntimeError(
        f"Dataset mode '{mode}' reached an unexpected "
        "execution path."
    )


# =====================================================================
# END OF MODULE
# =====================================================================
