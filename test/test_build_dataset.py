# test/test_build_dataset.py

from dataset.build_dataset import build_dataset

dataset = build_dataset()

print()
print("=" * 60)
print("DATASET FACTORY TEST")
print("=" * 60)

print()
print("Dataset Type:")
print(type(dataset))

print()
print("Dataset Size:")
print(len(dataset))

sample = dataset[0]

print()
print("Input Shape:")
print(sample["input"].shape)

print()
print("Target Shape:")
print(sample["target"].shape)

print()
print("Mask Shape:")
print(sample["mask"].shape)