from dataset.build_dataset import build_dataset
from dataset.dataloader import create_dataloader

dataset = build_dataset()

loader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=False
)

batch = next(iter(loader))

inputs, targets, mask, velocity = batch

print("Inputs :", inputs.shape)
print("Targets:", targets.shape)
print("Mask   :", mask.shape)
print("Velocity:", velocity.shape)