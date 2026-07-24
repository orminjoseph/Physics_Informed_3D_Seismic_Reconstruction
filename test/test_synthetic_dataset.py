from datasets.synthetic_dataset import SyntheticSeismicDataset

dataset = SyntheticSeismicDataset(
    num_samples=5
)

print("=" * 60)

print("Dataset Size:", len(dataset))

print("=" * 60)

input_cube, target, mask = dataset[0]

print("Input Shape :", input_cube.shape)
print("Target Shape:", target.shape)
print("Mask Shape  :", mask.shape)

print("=" * 60)

print("Missing Voxels:",
      (mask == 0).sum().item())