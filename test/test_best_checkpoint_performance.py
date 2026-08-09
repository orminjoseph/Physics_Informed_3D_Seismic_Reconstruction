"""
=========================================================
Checkpoint Comparison
=========================================================

Compare:
1. best_model.pth
2. latest_checkpoint.pth

Author: Ormin Joseph
=========================================================
"""

import os
import torch

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

BEST_CHECKPOINT = (
    r"outputs"
    r"\experiment_20260731_040411"
    r"\checkpoints"
    r"\best_model.pth"
)

LATEST_CHECKPOINT = (
    r"outputs"
    r"\experiment_20260731_040411"
    r"\checkpoints"
    r"\latest_checkpoint.pth"
)


def evaluate_checkpoint(checkpoint_path):

    predictor = Predictor(
        model=Network3D(),
        checkpoint=checkpoint_path,
        device="cpu"
    )

    dataset = F3Dataset(
        segy_path=F3_PATH,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    )

    corrupted, target, mask, velocity = dataset[0]

    target_batch = target.unsqueeze(0)

    reconstruction, uncertainty = predictor.predict(
        corrupted
    )

    return {
        "MAE": mae(reconstruction, target_batch).item(),
        "RMSE": rmse(reconstruction, target_batch).item(),
        "PSNR": psnr(reconstruction, target_batch).item(),
        "SNR": snr(reconstruction, target_batch).item(),
        "SSIM": ssim(reconstruction, target_batch).item()
    }


print()
print("=" * 60)
print("CHECKPOINT COMPARISON")
print("=" * 60)

best_metrics = evaluate_checkpoint(
    BEST_CHECKPOINT
)

latest_metrics = evaluate_checkpoint(
    LATEST_CHECKPOINT
)

print()
print("BEST MODEL")
print(best_metrics)

print()
print("LATEST CHECKPOINT")
print(latest_metrics)