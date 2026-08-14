"""
Dataset Builder

Switch between:
1. Synthetic dataset
2. Full F3 dataset

using DATASET_MODE in config.py
"""

from utils.config import *

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from dataset.f3_dataset import (
    F3Dataset
)
from dataset.marmousi2_patch_dataset import (
    Marmousi2PatchDataset
)

def build_dataset():

    print()
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    print(
        f"Dataset Mode: {DATASET_MODE}"
    )

    # ----------------------------------
    # Synthetic Dataset
    # ----------------------------------

    if DATASET_MODE.lower() == "synthetic":

        dataset = SyntheticSeismicDataset(

            num_samples=
            SYNTHETIC_NUM_SAMPLES,

            cube_size=
            SYNTHETIC_PATCH_SIZE,

            missing_probability=
            SYNTHETIC_MISSING_PROBABILITY
        )

        return dataset

    # ----------------------------------
    # F3 Dataset
    # ----------------------------------

    elif DATASET_MODE.lower() == "f3":

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

    # ----------------------------------
    # Marmousi2 Dataset
    # ----------------------------------

    elif DATASET_MODE.lower() == "marmousi2":

        dataset = Marmousi2PatchDataset(

            segy_path=
            MARMOUSI2_PATH,

            patch_size=
            MARMOUSI2_PATCH_SIZE,

            missing_rate=
            MARMOUSI2_MISSING_PROBABILITY,

            mask_type=
            MARMOUSI2_MASK_TYPE
        )

        return dataset

    else:

        raise ValueError(
            f"Unknown DATASET_MODE: {DATASET_MODE}"
        )