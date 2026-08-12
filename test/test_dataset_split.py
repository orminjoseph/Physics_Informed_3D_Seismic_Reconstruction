from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset

dataset = build_dataset()

train_dataset, validation_dataset = (
    split_dataset(dataset)
)

print()
print("Total:", len(dataset))
print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print()