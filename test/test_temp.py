from dataset.seismic_dataset import SeismicDataset

dataset = SeismicDataset("datasets")

sample = dataset[0]

ground_truth = sample["ground_truth"]
corrupted = sample["corrupted"]
mask = sample["mask"]

print("\nGround Truth")
print("Shape :", ground_truth.shape)
print("Min   :", ground_truth.min().item())
print("Max   :", ground_truth.max().item())
print("Mean  :", ground_truth.mean().item())

print("\nCorrupted")
print("Shape :", corrupted.shape)
print("Min   :", corrupted.min().item())
print("Max   :", corrupted.max().item())
print("Mean  :", corrupted.mean().item())

print("\nMask")
print("Shape :", mask.shape)
print("Min   :", mask.min().item())
print("Max   :", mask.max().item())