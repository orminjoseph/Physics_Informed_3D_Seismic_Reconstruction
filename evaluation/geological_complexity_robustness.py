import os
import pandas as pd
import torch

from inference.predictor import Predictor
from models.network import Network3D

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from dataset.geological_generator import GeologicalGenerator

CHECKPOINT = (
    r"outputs"
    r"\current_experiment"
    r"\checkpoints"
    r"\best_model.pth"
)


def evaluate_metrics(
        prediction,
        target,
        mask
):
    if mask.dim() == 4:
        mask = mask.unsqueeze(0)

    missing_region = (
            mask == 0
    )

    prediction_missing = prediction[
        missing_region
    ]

    target_missing = target[
        missing_region
    ]

    mae_value = torch.mean(
        torch.abs(
            prediction_missing
            - target_missing
        )
    ).item()

    return {
        "MAE": mae_value
    }

def main():
    print()
    print("=" * 60)
    print("GEOLOGICAL COMPLEXITY ROBUSTNESS")
    print("=" * 60)

    predictor = Predictor(
        model=Network3D(),
        checkpoint=CHECKPOINT,
        device="cpu"
    )

    generator = GeologicalGenerator()

    complexities = [
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex"
    ]

    results = []

    for complexity in complexities:

        print()
        print(f"Testing: {complexity}")

        if complexity == "horizontal":
            target = generator.generate_horizontal_layers()

        elif complexity == "dipping":
            target = generator.generate_dipping_layers()

        elif complexity == "faulted":
            target = generator.generate_faulted_layers()

        elif complexity == "folded":
            target = generator.generate_folded_layers()

        elif complexity == "highly_complex":

            target = (
                generator.generate_highly_complex_structure()
            )

        else:
            target = generator.generate_complex_structure()

        mask = (
                torch.rand_like(target) > 0.30
        ).float()

        corrupted = target * mask

        reconstruction, uncertainty = (
            predictor.predict(corrupted)
        )

        target_batch = target.unsqueeze(0)

        metrics = evaluate_metrics(
            reconstruction,
            target_batch,
            mask
        )

        mean_uncertainty = (
            uncertainty.mean().item()
        )

        results.append({
            "Complexity": complexity,
            "MAE": metrics["MAE"],
            "Mean_Uncertainty": mean_uncertainty
        })

        print(
            f"MAE={metrics['MAE']:.4f}, "
            f"UNC={mean_uncertainty:.4f}"
        )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    csv_file = (
        "outputs/reports/"
        "geological_complexity_robustness.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(
        csv_file,
        index=False
    )

    print()
    print("Saved:")
    print(csv_file)

    print()
    print(df)

if __name__ == "__main__":
    main()