"""
=========================================================
Test Evaluator
=========================================================

Tests the model evaluation pipeline.

Author: Ormin Joseph
=========================================================
"""

import test.setup_path

from test.test_factory import (
    create_model,
    create_dataloader
)

from evaluation.evaluator import Evaluator

from utils.config import DEVICE


# -------------------------------------------------------
# Create model
# -------------------------------------------------------

model = create_model()

# -------------------------------------------------------
# Create DataLoader
# -------------------------------------------------------

train_loader, validation_loader = create_dataloader()

# -------------------------------------------------------
# Create evaluator
# -------------------------------------------------------

evaluator = Evaluator(
    model=model,
    device=DEVICE
)

# -------------------------------------------------------
# Evaluate model
# -------------------------------------------------------

results = evaluator.evaluate(
    validation_loader
)

# -------------------------------------------------------
# Display evaluation results
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("Model Evaluation")
print("=" * 60)

for key, value in results.items():

    print(
        f"{key:<20}: {value:.6f}"
    )

print("=" * 60)