from dataset.seismic_dataset import SeismicDataset

dataset = SeismicDataset("datasets")

sample = dataset[0]

print(sample[0].shape)
print(sample[1].shape)
print(sample[2].shape)