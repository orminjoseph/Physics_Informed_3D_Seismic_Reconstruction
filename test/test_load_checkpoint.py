"""
============================================================
Testing Checkpoint Loading
============================================================
"""

import torch

from models.network import Network3D
from losses.total_loss import TotalLoss
from trainer.trainer import Trainer


def test_load_checkpoint():

    print()
    print("=" * 60)
    print("Testing Checkpoint Loading")
    print("=" * 60)

    device = torch.device("cpu")

    model = Network3D()

    criterion = TotalLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=1e-3

    )

    trainer = Trainer(

        model,

        criterion,

        optimizer,

        device

    )

    epoch = trainer.load_checkpoint(

        "checkpoints/latest_checkpoint.pth"

    )

    print()

    print(f"Loaded Epoch : {epoch}")

    print()

    print("Checkpoint Loading Test: PASSED")


def main():

    test_load_checkpoint()


if __name__ == "__main__":

    main()