import test.setup_path

from test.test_factory import (
    create_trainer,
    create_dataloader
)

# -------------------------------------------------------
# Create Trainer and DataLoader
# -------------------------------------------------------

trainer = create_trainer()

train_loader, validation_loader = create_dataloader()

# -------------------------------------------------------
# Train for multiple epochs
# -------------------------------------------------------

trainer.fit(
    train_loader,
    validation_loader,
    epochs=3
)


# -------------------------------------------------------
# Display training history
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("Training History")
print("=" * 60)

for key, values in trainer.history.items():
    print(f"{key:<15}: {values}")

print("=" * 60)

print("\n")
print("=" * 60)
print("Validation History")
print("=" * 60)

for key, values in trainer.validation_history.items():
    print(f"{key:<15}: {values}")

print("=" * 60)