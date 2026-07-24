import test.setup_path

from test.test_factory import (
    create_trainer
)

# -------------------------------------------------------
# Create Trainer
# -------------------------------------------------------

trainer = create_trainer()

# -------------------------------------------------------
# Load checkpoint
# -------------------------------------------------------

trainer.load_checkpoint(
    "checkpoints/checkpoint_epoch_3.pth"
)

# -------------------------------------------------------
# Display history
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("Loaded Training History")
print("=" * 60)

for key, values in trainer.history.items():

    print(
        f"{key:<15}: {values}"
    )

print("=" * 60)