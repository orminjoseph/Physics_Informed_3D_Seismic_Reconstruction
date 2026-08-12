# test/test_build_dataset.py

from dataset.build_dataset import build_dataset

dataset = build_dataset()

print()
print("Dataset Type:")
print(type(dataset))

print()
print("Dataset Size:")
print(len(dataset))