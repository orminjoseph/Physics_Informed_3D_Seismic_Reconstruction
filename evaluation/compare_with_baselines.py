"""
=========================================================
Baseline Comparison
=========================================================

Compare:

1. Nearest Neighbor
2. Linear Interpolation
3. Physics-Informed Network

Author: Ormin Joseph
=========================================================
"""

import csv
import os

import numpy as np
from dataset.f3_dataset import F3Dataset

from evaluation.baseline_nearest_neighbor import (
    nearest_neighbor_reconstruction
)

from evaluation.baseline_linear_interpolation import (
    linear_interpolation_reconstruction
)

from inference.predictor import Predictor

from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

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
    r"\latest_checkpoint.pth"
)


def evaluate(prediction, target):

    return [

        mae(prediction, target).item(),

        rmse(prediction, target).item(),

        psnr(prediction, target).item(),

        snr(prediction, target).item(),

        ssim(prediction, target).item()

    ]


def main():

    print()
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    NUM_TEST_PATCHES = 20

    nn_all = []
    linear_all = []
    network_all = []

    # ----------------------------------
    # Physics-Informed Network
    # ----------------------------------

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    for patch_index in range(

            min(
                NUM_TEST_PATCHES,
                len(dataset)
            )
    ):
        print(
            f"Processing patch "
            f"{patch_index + 1}/"
            f"{NUM_TEST_PATCHES}"
        )

        corrupted, target, mask, velocity = dataset[
            patch_index
        ]

        target_batch = target.unsqueeze(0)

        # ------------------------------
        # Nearest Neighbor
        # ------------------------------

        nn_prediction = (
            nearest_neighbor_reconstruction(
                corrupted,
                mask
            )
        )

        nn_prediction = nn_prediction.unsqueeze(0)

        nn_metrics = evaluate(
            nn_prediction,
            target_batch
        )

        nn_all.append(
            nn_metrics
        )

        # ------------------------------
        # Linear Interpolation
        # ------------------------------

        linear_prediction = (
            linear_interpolation_reconstruction(
                corrupted,
                mask
            )
        )

        linear_prediction = linear_prediction.unsqueeze(0)

        linear_metrics = evaluate(
            linear_prediction,
            target_batch
        )

        linear_all.append(
            linear_metrics
        )

        # ------------------------------
        # Network
        # ------------------------------

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        network_metrics = evaluate(
            reconstruction,
            target_batch
        )

        network_all.append(
            network_metrics
        )

    nn_metrics = np.mean(
        nn_all,
        axis=0
    ).tolist()

    linear_metrics = np.mean(
        linear_all,
        axis=0
    ).tolist()

    network_metrics = np.mean(
        network_all,
        axis=0
    ).tolist()

    # ----------------------------------
    # Save Results
    # ----------------------------------

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "baseline_comparison.csv"
    )

    with open(
        csv_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "Method",

            "MAE",

            "RMSE",

            "PSNR",

            "SNR",

            "SSIM"

        ])

        writer.writerow(
            ["Nearest_Neighbor"] +
            nn_metrics
        )

        writer.writerow(
            ["Linear_Interpolation"] +
            linear_metrics
        )

        writer.writerow(
            ["Physics_Informed_Network"] +
            network_metrics
        )

        writer.writerow(
            ["Num_Patches", NUM_TEST_PATCHES]
        )

    print()
    print("Results saved:")
    print(csv_file)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Average over {NUM_TEST_PATCHES} patches"
    )
    print()

    print(
        "Nearest Neighbor:",
        nn_metrics
    )

    print(
        "Linear Interpolation:",
        linear_metrics
    )

    print(
        "Physics-Informed Network:",
        network_metrics
    )


if __name__ == "__main__":
    main()