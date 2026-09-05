import torch

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D

from losses.total_loss import TotalLoss

from trainer.trainer import Trainer

from utils.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DX,
    DY,
    DZ
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = build_dataset()

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    criterion = TotalLoss(
        dx=DX,
        dy=DY,
        dz=DZ
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    trainer.fit(
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        resume=False
    )


if __name__ == "__main__":

    main()
