"""
============================================================
MC DROPOUT SPATIAL VALIDATION
============================================================

Checks whether uncertainty is higher in
missing regions than known regions.
============================================================
"""

import csv
import os
import torch

from dataset.f3_dataset import F3Dataset

from models.network import Network3D

from inference.mc_dropout_predictor import (
    MCDropoutPredictor
)

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = "checkpoints/best_model.pth"


def main():

    print("=" * 60)
    print("MC DROPOUT SPATIAL VALIDATION")
    print("=" * 60)

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = MCDropoutPredictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device=device,
        num_samples=20
    )

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    uncertainty = uncertainty.squeeze()
    mask = mask.squeeze()

    print()
    print("Uncertainty shape:", uncertainty.shape)
    print("Mask shape:", mask.shape)

    missing_mask = (mask == 0)
    known_mask = (mask == 1)

    print("Missing voxels:", missing_mask.sum().item())
    print("Known voxels:", known_mask.sum().item())

    missing_uncertainty = (
        uncertainty[missing_mask]
        .mean()
        .item()
    )

    known_uncertainty = (
        uncertainty[known_mask]
        .mean()
        .item()
    )

    ratio = (
            missing_uncertainty /
            (known_uncertainty + 1e-8)
    )

    print()
    print("=" * 60)
    print("SPATIAL VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Missing Region Uncertainty : "
        f"{missing_uncertainty:.6f}"
    )

    print(
        f"Known Region Uncertainty   : "
        f"{known_uncertainty:.6f}"
    )

    print(
        f"Ratio (Missing/Known)      : "
        f"{ratio:.4f}"
    )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    with open(
        "outputs/reports/"
        "mc_dropout_spatial_validation.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Missing_Uncertainty",
            "Known_Uncertainty",
            "Ratio"
        ])

        writer.writerow([
            missing_uncertainty,
            known_uncertainty,
            ratio
        ])

    print()
    print(
        "CSV saved to:"
    )

    print(
        "outputs/reports/"
        "mc_dropout_spatial_validation.csv"
    )


if __name__ == "__main__":
    main()