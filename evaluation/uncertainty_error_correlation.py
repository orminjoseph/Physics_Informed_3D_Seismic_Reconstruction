import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from dataset.f3_dataset import F3Dataset

from inference.predictor import Predictor

from models.network import Network3D


F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = (
    r"outputs"
    r"\current_experiment"
    r"\checkpoints"
    r"\best_model.pth"
)


def main():

    print()
    print("=" * 60)
    print("UNCERTAINTY-ERROR CORRELATION")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    all_errors = []
    all_uncertainties = []

    NUM_PATCHES = 20

    for patch_index in range(
        min(
            NUM_PATCHES,
            len(dataset)
        )
    ):

        corrupted, target, mask, velocity = (
            dataset[patch_index]
        )

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        error = abs(
            reconstruction.squeeze()
            .detach()
            .cpu()
            .numpy()
            -
            target.squeeze()
            .cpu()
            .numpy()
        )

        uncertainty_map = (
            uncertainty.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        all_errors.extend(
            error.flatten()
        )

        all_uncertainties.extend(
            uncertainty_map.flatten()
        )

    pearson_corr, pearson_p = pearsonr(
        all_errors,
        all_uncertainties
    )

    spearman_corr, spearman_p = spearmanr(
        all_errors,
        all_uncertainties
    )

    results = pd.DataFrame({
        "Metric": [
            "Pearson",
            "Spearman"
        ],

        "Correlation": [
            pearson_corr,
            spearman_corr
        ],

        "P_Value": [
            pearson_p,
            spearman_p
        ]
    })

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "uncertainty_error_correlation.csv"
    )

    results.to_csv(
        csv_file,
        index=False
    )

    print()
    print(results)

    print()
    print("Saved:")
    print(csv_file)


if __name__ == "__main__":
    main()