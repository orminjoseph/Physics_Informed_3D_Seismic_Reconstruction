"""
Linear Interpolation Baseline Test
"""

from dataset.f3_dataset import F3Dataset

from evaluation.baseline_linear_interpolation import (
    linear_interpolation_reconstruction
)

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

corrupted, target, mask, velocity = dataset[0]

reconstruction = (
    linear_interpolation_reconstruction(
        corrupted,
        mask
    )
)

print()
print("=" * 60)
print("LINEAR INTERPOLATION TEST")
print("=" * 60)

print("Input Shape :", corrupted.shape)
print("Output Shape:", reconstruction.shape)