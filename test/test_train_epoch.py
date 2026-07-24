import test.setup_path

from test.test_factory import (
    create_trainer,
    create_dataloader
)

loader = create_dataloader()
trainer = create_trainer()
# -------------------------------------------------------
# Train one epoch
# -------------------------------------------------------

losses = trainer.train_epoch(loader)

# -------------------------------------------------------
# Display results
# -------------------------------------------------------

print("=" * 60)

for name, value in losses.items():
    print(f"{name:<15}: {value:.6f}")

print("=" * 60)
print("\nTraining History")

print("=" * 60)

for key, values in trainer.history.items():
    print(f"{key:<15}: {values}")

print("=" * 60)