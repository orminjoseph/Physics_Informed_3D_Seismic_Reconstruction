"""
=========================================================
Test Evaluation Metrics
=========================================================

Tests all evaluation metrics.

Author: Ormin Joseph
=========================================================
"""

import test.setup_path

import torch

from evaluation.metrics import EvaluationMetrics


# -------------------------------------------------------
# Create synthetic tensors
# -------------------------------------------------------

target = torch.rand(
    1,
    1,
    16,
    32,
    32
)

prediction = target + 0.05 * torch.randn_like(target)

log_variance = torch.zeros_like(target)


# -------------------------------------------------------
# Compute metrics
# -------------------------------------------------------

mae = EvaluationMetrics.mae(
    prediction,
    target
)

mse = EvaluationMetrics.mse(
    prediction,
    target
)

rmse = EvaluationMetrics.rmse(
    prediction,
    target
)

relative_error = EvaluationMetrics.relative_error(
    prediction,
    target
)

snr = EvaluationMetrics.snr(
    prediction,
    target
)

psnr = EvaluationMetrics.psnr(
    prediction,
    target
)

uncertainty = EvaluationMetrics.uncertainty(
    log_variance
)


# -------------------------------------------------------
# Display results
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("Evaluation Metrics")
print("=" * 60)

print(f"MAE              : {mae:.6f}")
print(f"MSE              : {mse:.6f}")
print(f"RMSE             : {rmse:.6f}")
print(f"Relative Error   : {relative_error:.6f}")
print(f"SNR (dB)         : {snr:.6f}")
print(f"PSNR (dB)        : {psnr:.6f}")
print(f"Uncertainty      : {uncertainty:.6f}")

print("=" * 60)