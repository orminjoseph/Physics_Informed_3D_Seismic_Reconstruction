"""
Test the trained predictor.
"""

from inference.predictor import Predictor


def main():
    from models.network import Network3D

    model = Network3D()

    predictor = Predictor(

        model=model,

        checkpoint="checkpoints/best_model.pth",

        device="cpu"

    )


    print()

    print("Predictor loaded successfully.")

    print()

    print("Model is ready for inference.")


if __name__ == "__main__":

    main()