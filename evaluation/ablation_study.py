"""
=========================================================
Ablation Study
=========================================================

Evaluates:

1. Full Model
2. No Attention
3. No Residual
4. No Uncertainty
5. Plain U-Net

=========================================================
"""

import os

import pandas as pd

from models.network import Network3D

from evaluation.evaluate_model import evaluate

from utils.config import DATASET_MODE


ABLATION_MODELS = {

    "Full_Model": {

        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": True

    },

    "No_Attention": {

        "use_attention": False,
        "use_residual": True,
        "use_uncertainty": True

    },

    "No_Residual": {

        "use_attention": True,
        "use_residual": False,
        "use_uncertainty": True

    },

    "No_Uncertainty": {

        "use_attention": True,
        "use_residual": True,
        "use_uncertainty": False

    },

    "Plain_UNet": {

        "use_attention": False,
        "use_residual": False,
        "use_uncertainty": False

    }

}


def run_ablation():

    results = []

    print()
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    for model_name, settings in ABLATION_MODELS.items():

        print()
        print(f"Evaluating: {model_name}")

        model = Network3D(

            use_attention=settings["use_attention"],
            use_residual=settings["use_residual"],
            use_uncertainty=settings["use_uncertainty"]

        )

        metrics = evaluate(
            model_override=model
        )

        metrics["Model"] = model_name

        results.append(metrics)

    dataframe = pd.DataFrame(results)

    output_dir = os.path.join(
        "outputs",
        DATASET_MODE,
        "reports"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    dataframe.to_csv(

        os.path.join(
            output_dir,
            "ablation_study.csv"
        ),

        index=False

    )

    print()
    print(dataframe)

    return dataframe


if __name__ == "__main__":

    run_ablation()