
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print("UNCERTAINTY CALIBRATION")
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

        print(
            f"Processing patch {patch_index}"
        )

        corrupted, target, mask, velocity = (
            dataset[patch_index]
        )

        reconstruction, uncertainty = (
            predictor.predict(
                corrupted
            )
        )

        error = np.abs(

            reconstruction.squeeze()
            .detach()
            .cpu()
            .numpy()

            -

            target.squeeze()
            .cpu()
            .numpy()
        )

        uncertainty_map = np.exp(

            np.clip(

                uncertainty.squeeze()
                .detach()
                .cpu()
                .numpy(),

                -10,

                10
            )
        )

        all_errors.extend(
            error.flatten()
        )

        all_uncertainties.extend(
            uncertainty_map.flatten()
        )

    # ---------------------------------------------------
    # Convert to numpy arrays
    # ---------------------------------------------------

    all_errors = np.array(
        all_errors
    )

    all_uncertainties = np.array(
        all_uncertainties
    )

    # ---------------------------------------------------
    # Normalize both to [0,1]
    # ---------------------------------------------------

    all_errors = (
        all_errors
        - all_errors.min()
    ) / (
        all_errors.max()
        - all_errors.min()
        + 1e-8
    )

    all_uncertainties = (
        all_uncertainties
        - all_uncertainties.min()
    ) / (
        all_uncertainties.max()
        - all_uncertainties.min()
        + 1e-8
    )

    # ---------------------------------------------------
    # Calibration bins
    # ---------------------------------------------------

    num_bins = 10

    bins = np.linspace(
        0,
        1,
        num_bins + 1
    )

    mean_uncertainty = []
    mean_error = []

    for i in range(num_bins):

        indices = (

            (all_uncertainties >= bins[i])

            &

            (all_uncertainties < bins[i + 1])
        )

        if indices.sum() == 0:
            continue

        mean_uncertainty.append(

            all_uncertainties[indices]
            .mean()
        )

        mean_error.append(

            all_errors[indices]
            .mean()
        )

    mean_uncertainty = np.array(
        mean_uncertainty
    )

    mean_error = np.array(
        mean_error
    )

    # ---------------------------------------------------
    # Expected Calibration Error
    # ---------------------------------------------------

    ece = np.mean(

        np.abs(

            mean_uncertainty

            -

            mean_error
        )
    )

    # ---------------------------------------------------
    # Save calibration table
    # ---------------------------------------------------

    calibration = pd.DataFrame({

        "Mean_Uncertainty":
            mean_uncertainty,

        "Mean_Error":
            mean_error
    })

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "uncertainty_calibration.csv"
    )

    calibration.to_csv(
        csv_file,
        index=False
    )

    # ---------------------------------------------------
    # Reliability diagram
    # ---------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        mean_uncertainty,
        mean_error,
        marker="o",
        linewidth=2,
        label="Calibration Curve"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=2,
        label="Perfect Calibration"
    )

    plt.xlabel(
        "Predicted Uncertainty"
    )

    plt.ylabel(
        "Observed Error"
    )

    plt.title(
        f"ECE = {ece:.4f}"
    )

    plt.legend()

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    figure_file = (
        "outputs/figures/"
        "uncertainty_calibration.png"
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------
    # Print results
    # ---------------------------------------------------

    print()

    print(
        f"ECE = {ece:.6f}"
    )

    print()

    print(
        calibration
    )

    print()

    print("Saved:")
    print(csv_file)
    print(figure_file)


if __name__ == "__main__":
    main()