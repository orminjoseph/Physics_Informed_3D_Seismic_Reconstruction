"""
MASTER PROJECT PIPELINE

Physics-Informed 3D Seismic Reconstruction

Author: Ormin Joseph
"""

import time

from train.train_model import main as train_model


def run_project():

    start_time = time.time()

    print()
    print("=" * 80)
    print("PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION")
    print("=" * 80)

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------

    print()
    print("STEP 1 : TRAINING")

    train_model()

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------

    print()
    print("STEP 2 : FULL EVALUATION")

    from evaluation.run_full_evaluation import (
        run_full_evaluation
    )

    run_full_evaluation()

    # -------------------------------------------------
    # FINISHED
    # -------------------------------------------------

    total_time = (
        time.time() - start_time
    ) / 3600

    print()
    print("=" * 80)
    print("PROJECT COMPLETE")
    print("=" * 80)

    print(
        f"Total Runtime : "
        f"{total_time:.2f} hours"
    )


if __name__ == "__main__":

    run_project()