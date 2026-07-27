import torch

from models.network import Network3D
from trainer.trainer import Trainer
from losses.total_loss import TotalLoss


def main():

    model = Network3D()

    criterion = TotalLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu"
    )

    epoch = trainer.load_checkpoint(
        "checkpoints/latest_checkpoint.pth"
    )

    print()

    print("=" * 60)
    print("Resume Training Test")
    print("=" * 60)

    print(f"Loaded Epoch            : {epoch}")
    print(f"Best Validation Loss    : {trainer.best_validation_loss:.6f}")

    print()
    print("Resume checkpoint loaded successfully.")


if __name__ == "__main__":
    main()