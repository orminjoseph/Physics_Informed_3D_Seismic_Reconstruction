"""
=========================================================
Dataset Builder
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

Supported dataset modes
-----------------------

1. synthetic
2. f3
Configuration
-------------

General configuration:
    utils/config.py

Training configuration:
    utils/config.py

Tensor convention:
    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# GENERAL CONFIGURATION
# =========================================================

from utils.config import (
    DATASET_MODE,

    # Synthetic dataset
    SYNTHETIC_NUM_SAMPLES,
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,

    # F3 dataset
    F3_PATH,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY,
)



# =========================================================
# DATASET CLASSES
# =========================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from dataset.f3_dataset import (
    F3Dataset
)


# =========================================================
# DATASET BUILDER
# =========================================================

def build_dataset():
    """
    Construct the dataset selected by DATASET_MODE.

    Returns
    -------
    Dataset
        Selected seismic dataset.
    """

    print()
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    print(
        f"Dataset Mode: {DATASET_MODE}"
    )

    mode = DATASET_MODE.lower()


    # =====================================================
    # 1. SYNTHETIC DATASET
    # =====================================================

    if mode == "synthetic":

        dataset = SyntheticSeismicDataset(

            num_samples=
            SYNTHETIC_NUM_SAMPLES,

            cube_size=
            SYNTHETIC_PATCH_SIZE,

            missing_probability=
            SYNTHETIC_MISSING_PROBABILITY
        )

        return dataset


    # =====================================================
    # 2. F3 DATASET
    # =====================================================

    elif mode == "f3":

        dataset = F3Dataset(

            segy_path=
            F3_PATH,

            patch_size=
            F3_PATCH_SIZE,

            stride=
            F3_STRIDE,

            missing_probability=
            F3_MISSING_PROBABILITY
        )

        return dataset

    # =====================================================
    # UNKNOWN DATASET MODE
    # =====================================================

    else:

        raise ValueError(
            f"Unknown DATASET_MODE: {DATASET_MODE}. "
            f"Expected one of: "
            f"synthetic, f3."
        )