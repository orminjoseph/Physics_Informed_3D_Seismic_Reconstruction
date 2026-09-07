import test.setup_path

from test.test_factory import (
    create_model,
    create_trainer,
    create_dataloader
)

print("=" * 60)

model = create_model()
print("✓ Model created successfully")

trainer = create_trainer()
print("✓ Trainer created successfully")

train_loader, validation_loader = create_dataloader()

print("✓ Training DataLoader created successfully")

print("✓ Validation DataLoader created successfully")

print(f"Training batches   : {len(train_loader)}")

print(f"Validation batches : {len(validation_loader)}")

print("=" * 60)