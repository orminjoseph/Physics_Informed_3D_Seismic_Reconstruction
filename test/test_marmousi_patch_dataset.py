from dataset.marmousi2_patch_dataset import Marmousi2PatchDataset

dataset = Marmousi2PatchDataset(
    segy_path=r"C:\Users\ormin\Desktop\SEG_FILES\elastic-marmousi-model\elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy\MODEL_P-WAVE_VELOCITY_1.25m.segy"
)

sample = dataset[0]

print("Dataset Size:", len(dataset))
print("Tuple Length:", len(sample))

for i, item in enumerate(sample):
    print(f"Item {i} Shape:", item.shape)