"""
=========================================================
Model Evaluation
=========================================================

Physics-Informed 3D Encoder-Decoder Framework

Loads the best model checkpoint for the CURRENT EXPERIMENT
and computes:

    MAE
    RMSE
    PSNR
    SNR
    SSIM

Dataset output convention:

    input_cube
    target_cube
    mask
    velocity_model

Predictor output convention:

    reconstruction
    travel_time
    uncertainty

All experiment outputs are controlled by:

    EXPERIMENT_NAME

Therefore:

    outputs/
        <EXPERIMENT_NAME>/
            checkpoints/
            reports/
            figures/
            ...

=========================================================
"""

import os

import torch
import pandas as pd

from models.network import Network3D

from inference.predictor import Predictor

from dataset.build_dataset import build_dataset

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR
)


def evaluate(model_override=None):

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Experiment :", EXPERIMENT_NAME)
    print("Device     :", device)

    # =====================================================
    # BUILD DATASET
    # =====================================================

    dataset = build_dataset()

    print(
        "Dataset Length:",
        len(dataset)
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    if model_override is None:

        model = Network3D()

    else:

        model = model_override

    # =====================================================
    # MOVE MODEL TO DEVICE
    # =====================================================

    model = model.to(device)

    # =====================================================
    # CHECKPOINT
    # =====================================================

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    print()
    print(
        "Checkpoint:",
        checkpoint
    )

    if not os.path.exists(checkpoint):

        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete at least one training epoch "
            "and make sure best_model.pth exists."
        )

    # =====================================================
    # PREDICTOR
    # =====================================================

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device
    )

    # =====================================================
    # METRIC ACCUMULATORS
    # =====================================================

    total_mae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_snr = 0.0
    total_ssim = 0.0

    num_samples = len(dataset)

    # =====================================================
    # EVALUATE EVERY SAMPLE
    # =====================================================

    for i in range(num_samples):

        print(
            f"Evaluating sample "
            f"{i + 1}/{num_samples}"
        )

        # -------------------------------------------------
        # DATASET OUTPUT
        # -------------------------------------------------

        (
            input_cube,
            target_cube,
            mask,
            velocity_model
        ) = dataset[i]

        # -------------------------------------------------
        # PREDICTION
        #
        # Predictor.predict() returns:
        #
        # reconstruction
        # travel_time
        # uncertainty
        # -------------------------------------------------

        (
            reconstruction,
            travel_time,
            uncertainty
        ) = predictor.predict(
            input_cube
        )

        # -------------------------------------------------
        # TARGET BATCH DIMENSION
        # -------------------------------------------------

        target_batch = (
            target_cube.unsqueeze(0)
        )

        # -------------------------------------------------
        # COMPUTE METRICS
        # -------------------------------------------------

        sample_mae = mae(
            reconstruction,
            target_batch
        ).item()

        sample_rmse = rmse(
            reconstruction,
            target_batch
        ).item()

        sample_psnr = psnr(
            reconstruction,
            target_batch
        ).item()

        sample_snr = snr(
            reconstruction,
            target_batch
        ).item()

        sample_ssim = ssim(
            reconstruction,
            target_batch
        ).item()

        # -------------------------------------------------
        # ACCUMULATE RESULTS
        # -------------------------------------------------

        total_mae += sample_mae
        total_rmse += sample_rmse
        total_psnr += sample_psnr
        total_snr += sample_snr
        total_ssim += sample_ssim

    # =====================================================
    # AVERAGE RESULTS
    # =====================================================

    results = {

        "MAE":
            total_mae / num_samples,

        "RMSE":
            total_rmse / num_samples,

        "PSNR":
            total_psnr / num_samples,

        "SNR":
            total_snr / num_samples,

        "SSIM":
            total_ssim / num_samples
    }

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    for key, value in results.items():

        print(
            f"{key:<10}: {value:.6f}"
        )

    # =====================================================
    # CREATE REPORT DIRECTORY
    # =====================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_file = os.path.join(
        REPORT_DIR,
        "evaluation_metrics.csv"
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        output_file,
        index=False
    )

    # =====================================================
    # CONFIRM OUTPUT LOCATION
    # =====================================================

    print()
    print(
        "Evaluation results saved:"
    )

    print(
        output_file
    )

    print()
    print(
        "Experiment directory:"
    )

    print(
        os.path.join(
            "outputs",
            EXPERIMENT_NAME
        )
    )

    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate()