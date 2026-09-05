import torch

from dataset.f3_dataset import F3Dataset
from inference.predictor import Predictor
from models.network import Network3D

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0][:4]

predictor = Predictor(
    model=Network3D(),
    checkpoint=(
    r"outputs"
    r"\synthetic_training"
    r"\checkpoints"
    r"\best_model.pth"),
    device="cpu"
)

reconstruction, travel_time, uncertainty = predictor.predict(
    corrupted
)

print()
print("=" * 60)
print("UNCERTAINTY STATISTICS")
print("=" * 60)

print("Mean :", uncertainty.mean().item())
print("Std  :", uncertainty.std().item())
print("Min  :", uncertainty.min().item())
print("Max  :", uncertainty.max().item())