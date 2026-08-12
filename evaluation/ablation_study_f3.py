import os
import numpy as np
import pandas as pd

from dataset.f3_dataset import F3Dataset

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

from utils.config import DATASET_MODE

CHECKPOINT = os.path.join(
    "outputs",
    DATASET_MODE,
    "checkpoints",
    "best_model.pth"
)


def evaluate_metrics(
        prediction,
        target
):

    return {

        "MAE": mae(
            prediction,
            target
        ).item(),

        "RMSE": rmse(
            prediction,
            target
        ).item(),

        "PSNR": psnr(
            prediction,
            target
        ).item(),

        "SNR": snr(
            prediction,
            target
        ).item(),

        "SSIM": ssim(
            prediction,
            target
        ).item()
    }


def evaluate_variant(
        variant_name,
        use_residual,
        use_attention,
        use_uncertainty,
        dataset
):

    print()
    print(f"Evaluating: {variant_name}")

    predictor = Predictor(

        model=Network3D(

            use_residual=use_residual,

            use_attention=use_attention,

            use_uncertainty=use_uncertainty
        ),

        checkpoint=CHECKPOINT,

        device="cpu"
    )

    mae_values = []
    rmse_values = []
    psnr_values = []
    snr_values = []
    ssim_values = []

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

        metrics = evaluate_metrics(

            reconstruction,

            target.unsqueeze(0)
        )

        mae_values.append(
            metrics["MAE"]
        )

        rmse_values.append(
            metrics["RMSE"]
        )

        psnr_values.append(
            metrics["PSNR"]
        )

        snr_values.append(
            metrics["SNR"]
        )

        ssim_values.append(
            metrics["SSIM"]
        )

    return {

        "Model": variant_name,

        "MAE": np.mean(mae_values),

        "RMSE": np.mean(rmse_values),

        "PSNR": np.mean(psnr_values),

        "SNR": np.mean(snr_values),

        "SSIM": np.mean(ssim_values)
    }


def main():

    print()
    print("=" * 60)
    print("ARCHITECTURE ABLATION STUDY")
    print("=" * 60)

    dataset = F3Dataset(

        segy_path=F3_PATH,

        patch_size=(64, 64, 64),

        stride=(64, 64, 64),

        missing_probability=0.30
    )

    results = []

    variants = [

        (
            "Full_Model",
            True,
            True,
            True
        ),

        (
            "No_Attention",
            True,
            False,
            True
        ),

        (
            "No_Residual",
            False,
            True,
            True
        ),

        (
            "No_Uncertainty",
            True,
            True,
            False
        ),

        (
            "Plain_UNet",
            False,
            False,
            False
        )
    ]

    for (
        name,
        residual,
        attention,
        uncertainty
    ) in variants:

        result = evaluate_variant(

            name,

            residual,

            attention,

            uncertainty,

            dataset
        )

        results.append(result)

    results = pd.DataFrame(results)

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "ablation_study.csv"
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