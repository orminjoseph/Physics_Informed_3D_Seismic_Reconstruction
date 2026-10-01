"""
=========================================================
MASTER PROJECT PIPELINE
=========================================================

Physics-Informed 3D Seismic Reconstruction

Master pipeline for:

    1. Model training
    2. Full evaluation

Training can be enabled or disabled from:

    utils/config.py

Configuration:

    RUN_TRAINING = True
        -> Train the model, then run evaluation.

    RUN_TRAINING = False
        -> Skip training and use the existing trained
           checkpoint for evaluation.

The evaluation pipeline is handled by:

    evaluation/run_full_evaluation.py

Author: Ormin Joseph
=========================================================
"""

import os
import time


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# CONFIGURATION
# =========================================================

from utils.config import (
    RUN_TRAINING,
    CHECKPOINT_DIR,
)


# =========================================================
# TRAINING IMPORT
# =========================================================

from train.train_model import (
    main as train_model
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_checkpoint_path():
    """
    Return the expected best-model checkpoint path.

    The path is derived from CHECKPOINT_DIR in config.py.

    Returns
    -------
    str
        Full path to best_model.pth.
    """

    return os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )


def verify_existing_checkpoint():
    """
    Verify that an existing trained checkpoint is available.

    This is required when RUN_TRAINING=False.
    """

    checkpoint_path = get_checkpoint_path()

    print()
    print(
        "Checking existing model checkpoint..."
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nRUN_TRAINING is set to False, but "
            "the trained model checkpoint was not found.\n\n"
            f"Expected checkpoint:\n"
            f"{checkpoint_path}\n\n"
            "Either:\n"
            "1. Set RUN_TRAINING = True in "
            "utils/config.py and train the model, or\n"
            "2. Place the existing best_model.pth "
            "checkpoint in the configured checkpoint directory."
        )

    if os.path.getsize(
        checkpoint_path
    ) == 0:

        raise RuntimeError(
            "\nThe existing model checkpoint is empty:\n"
            f"{checkpoint_path}"
        )

    print(
        "[OK] Existing trained checkpoint found."
    )

    return checkpoint_path


# =========================================================
# MASTER PIPELINE
# =========================================================

def run_project():

    start_time = time.time()

    print()
    print("=" * 80)
    print(
        "PHYSICS-INFORMED 3D SEISMIC "
        "RECONSTRUCTION"
    )
    print("=" * 80)

    print()
    print(
        f"Training Control : "
        f"RUN_TRAINING = {RUN_TRAINING}"
    )

    print(
        f"Checkpoint Dir   : "
        f"{CHECKPOINT_DIR}"
    )

    # =====================================================
    # STEP 1 : TRAINING
    # =====================================================

    if RUN_TRAINING:

        print()
        print("=" * 80)
        print("STEP 1 : TRAINING")
        print("=" * 80)

        train_model()

        print()
        print(
            "[SUCCESS] Training completed."
        )

        # -------------------------------------------------
        # Verify that training produced the checkpoint.
        # -------------------------------------------------

        checkpoint_path = get_checkpoint_path()

        if not os.path.isfile(
            checkpoint_path
        ):

            raise FileNotFoundError(
                "\nTraining completed, but the expected "
                "best-model checkpoint was not found.\n\n"
                f"Expected checkpoint:\n"
                f"{checkpoint_path}"
            )

        print(
            f"Best checkpoint:\n"
            f"{checkpoint_path}"
        )

    # =====================================================
    # STEP 1 : SKIP TRAINING
    # =====================================================

    else:

        print()
        print("=" * 80)
        print("STEP 1 : TRAINING SKIPPED")
        print("=" * 80)

        print()
        print(
            "RUN_TRAINING = False"
        )

        print(
            "The existing trained model will be used."
        )

        verify_existing_checkpoint()

    # =====================================================
    # STEP 2 : FULL EVALUATION
    # =====================================================

    print()
    print("=" * 80)
    print("STEP 2 : FULL EVALUATION")
    print("=" * 80)

    from evaluation.run_full_evaluation import (
        run_full_evaluation
    )

    run_full_evaluation()

    # =====================================================
    # FINISHED
    # =====================================================

    total_time = (
        time.time() - start_time
    ) / 3600.0

    print()
    print("=" * 80)
    print("PROJECT COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Total Runtime : "
        f"{total_time:.2f} hours"
    )

    print()
    print(
        f"Training Run  : "
        f"{RUN_TRAINING}"
    )

    print(
        f"Checkpoint    : "
        f"{get_checkpoint_path()}"
    )

    print()
    print("=" * 80)


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_project()