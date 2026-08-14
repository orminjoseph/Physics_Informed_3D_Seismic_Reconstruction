from dataset.marmousi2_dataset import Marmousi2Dataset

dataset = Marmousi2Dataset(
    segy_path=r"C:\Users\ormin\Desktop\SEG_FILES\elastic-marmousi-model\elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy\MODEL_P-WAVE_VELOCITY_1.25m.segy"
)

sample = dataset[0]

print("Input Shape :", sample["input"].shape)
print("Target Shape:", sample["target"].shape)
print("Mask Shape  :", sample["mask"].shape)