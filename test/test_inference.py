import torch

from models.network import Network3D
from inference.predictor import Predictor
from dataset.seismic_dataset import SeismicDataset


def main():
    model = Network3D()

    predictor = Predictor(

        model=model,

        checkpoint="checkpoints/best_model.pth",

        device="cpu"

    )

    dataset = SeismicDataset()

    sample = dataset[0]

    corrupted = sample["corrupted"]

    reconstruction, uncertainty = predictor.predict(corrupted)

    print("\nPrediction completed.\n")

    print("Input Shape:")
    print(corrupted.shape)

    print("Reconstruction Shape:")
    print(reconstruction.shape)

    print("Uncertainty Shape:")
    print(uncertainty.shape)


if __name__ == "__main__":
    main()