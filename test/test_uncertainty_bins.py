import torch
import numpy as np

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
    patch_size=(64,64,64),
    stride=(64,64,64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0]

predictor = Predictor(
    model=Network3D(),
    checkpoint="checkpoints/best_model.pth",
    device="cpu"
)

reconstruction, uncertainty = predictor.predict(
    corrupted
)

error = torch.abs(
    reconstruction.squeeze(0) - target
)

error = error.numpy().flatten()
uncertainty = uncertainty.squeeze(0).numpy().flatten()

q25 = np.percentile(uncertainty,25)
q75 = np.percentile(uncertainty,75)

low_error = error[uncertainty <= q25]
high_error = error[uncertainty >= q75]

print()
print("="*60)
print("UNCERTAINTY BIN ANALYSIS")
print("="*60)

print("Mean Error (Low Uncertainty):",
      low_error.mean())

print("Mean Error (High Uncertainty):",
      high_error.mean())