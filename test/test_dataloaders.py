from dataset.build_dataloaders import (
    build_dataloaders
)

print()
print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)

train_loader, validation_loader = (
    build_dataloaders()
)

print()
print("Train Batches:", len(train_loader))
print("Validation Batches:", len(validation_loader))
print()

batch = next(iter(train_loader))

inputs, targets, masks, velocities = batch

print("Input:", inputs.shape)
print("Target:", targets.shape)
print("Mask:", masks.shape)
print("Velocity:", velocities.shape)
print()