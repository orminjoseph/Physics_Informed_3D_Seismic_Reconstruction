import torch

from models.network import Network3D
from utils.helpers import get_device

from evaluation.evaluator import Evaluator

from dataset.geological_generator import GeologicalGenerator
from dataset.seismic_dataset import SeismicDataset
from dataset.dataloader import create_dataloader


def main():

    device = get_device()

    checkpoint_path = (
        "outputs/"
        "experiment_20260729_201904/"
        "checkpoints/"
        "best_model.pth"
    )

    model = Network3D().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("=" * 60)
    print("Model Loaded Successfully")
    print("=" * 60)

    generator = GeologicalGenerator()

    cube = generator.generate(
        dipping=True
    )

    cube = cube.squeeze(0).numpy()

    dataset = SeismicDataset(
        cube=cube
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    evaluator = Evaluator(
        model=model,
        device=device
    )

    batch = next(iter(dataloader))


    results = evaluator.evaluate(
        dataloader
    )

    print()
    print("=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)

    output_file = "outputs/BASLINE_V1.txt"

    with open(output_file, "w") as f:

        f.write(
            "Baseline Physics-Informed 3D Seismic Reconstruction\n"
        )

        f.write("=" * 50 + "\n\n")

        for key, value in results.items():
            f.write(
                f"{key.upper():15s}: {value:.6f}\n"
            )

    print()
    print(f"Results saved to {output_file}")

    for key, value in results.items():
        print(
            f"{key.upper():15}: {value:.6f}"
        )


if __name__ == "__main__":
    main()