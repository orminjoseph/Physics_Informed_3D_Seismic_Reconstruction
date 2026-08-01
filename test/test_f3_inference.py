import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D
from inference.predictor import Predictor
from evaluation.metrics import EvaluationMetrics

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

CHECKPOINT = (
    r"outputs"
    r"\experiment_20260731_040411"
    r"\checkpoints"
    r"\best_model.pth"
)

print()
print("=" * 60)
print("F3 INFERENCE TEST")
print("=" * 60)

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64,64,64),
    stride=(64,64,64),
    missing_probability=0.30
)

corrupted, target, mask, velocity = dataset[0]

device = torch.device("cpu")

model = Network3D(
    in_channels=1,
    out_channels=1
)

predictor = Predictor(
    model=model,
    checkpoint=CHECKPOINT,
    device=device
)

reconstruction, uncertainty = predictor.predict(
    corrupted.unsqueeze(0)
)

print()
print("=" * 60)
print("DATA RANGE CHECK")
print("=" * 60)

print("Target Min :", target.min().item())
print("Target Max :", target.max().item())

print("Prediction Min :", reconstruction.min().item())
print("Prediction Max :", reconstruction.max().item())

target_batch = target.unsqueeze(0)

metric_mae = EvaluationMetrics.mae(
    reconstruction,
    target_batch
)

metric_rmse = EvaluationMetrics.rmse(
    reconstruction,
    target_batch
)

metric_psnr = EvaluationMetrics.psnr(
    reconstruction,
    target_batch
)

metric_snr = EvaluationMetrics.snr(
    reconstruction,
    target_batch
)

metric_ssim = EvaluationMetrics.ssim(
    reconstruction,
    target_batch
)

mean_uncertainty = torch.mean(
    uncertainty
)

print()
print("Input Shape       :", corrupted.shape)
print("Target Shape      :", target.shape)
print("Reconstruction    :", reconstruction.shape)
print("Uncertainty Shape :", uncertainty.shape)

print()
print("=" * 60)
print("F3 RECONSTRUCTION METRICS")
print("=" * 60)

print(f"MAE  : {metric_mae.item():.6f}")
print(f"RMSE : {metric_rmse.item():.6f}")
print(f"PSNR : {metric_psnr.item():.3f} dB")
print(f"SNR  : {metric_snr.item():.3f} dB")
print(f"SSIM : {metric_ssim.item():.6f}")

print()
print(f"Mean Uncertainty : {mean_uncertainty.item():.6f}")

print()
print("Inference Test: PASSED")