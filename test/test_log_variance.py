"""
============================================================
LOG VARIANCE DIAGNOSTIC TEST
============================================================

Inspect uncertainty head outputs on F3 data.

Author: Ormin Joseph
============================================================
"""

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D


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
    print("LOG VARIANCE DIAGNOSTIC")
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

    model = Network3D().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        corrupted = corrupted.unsqueeze(0).to(device)

        reconstruction, log_variance = model(
            corrupted
        )

        print()
        print("=" * 60)
        print("LOG VARIANCE STATISTICS")
        print("=" * 60)

        print(
            "Mean :",
            log_variance.mean().item()
        )

        print(
            "Min  :",
            log_variance.min().item()
        )

        print(
            "Max  :",
            log_variance.max().item()
        )

        variance = torch.exp(
            log_variance
        )

        print()
        print("=" * 60)
        print("VARIANCE STATISTICS")
        print("=" * 60)

        print(
            "Mean :",
            variance.mean().item()
        )

        print(
            "Min  :",
            variance.min().item()
        )

        print(
            "Max  :",
            variance.max().item()
        )


if __name__ == "__main__":
    main()