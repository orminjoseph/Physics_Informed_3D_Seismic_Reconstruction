# test/test_marmousi_dataset_output.py

from dataset.build_dataset import build_dataset

dataset = build_dataset()

sample = dataset[0]

print("Tuple Length:", len(sample))

for i, item in enumerate(sample):
    print(
        f"Item {i} Shape:",
        item.shape
    )