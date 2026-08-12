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

    else:

        raise ValueError(
            f"Unknown DATASET_MODE: {DATASET_MODE}"
        )