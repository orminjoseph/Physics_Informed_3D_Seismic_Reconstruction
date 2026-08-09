"""
=========================================================
MULTI-RUN TRAINING FOR DEEP ENSEMBLES
=========================================================
"""

import random
import numpy as np
import torch
import subprocess

NUM_RUNS = 5
BASE_SEED = 42


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    print("=" * 60)
    print("DEEP ENSEMBLE TRAINING")
    print("=" * 60)

    for run_id in range(1, NUM_RUNS + 1):

        seed = BASE_SEED + run_id

        print()
        print(f"Starting Ensemble Run {run_id}")
        print(f"Seed = {seed}")

        set_seed(seed)

        subprocess.run(
            [
                "python",
                "-m",
                "train.train_f3"
            ],
            check=True
        )

        print(f"Run {run_id} completed")


if __name__ == "__main__":
    main()