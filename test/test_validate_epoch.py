"""
TEST - VALIDATE ONE EPOCH

Tests the validate_epoch() method of the Trainer.

The test verifies that:

1. A synthetic validation dataset can be created.
2. A validation DataLoader can be created.
3. The Trainer initializes correctly.
4. One validation epoch executes successfully.
5. All losses are finite.
6. All reconstruction metrics are finite.
Author: Ormin Joseph

"""


import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from dataset.synthetic_dataset import SyntheticSeismicDataset
from models.network import Network3D
from losses.total_loss import TotalLoss

# ============================================================

# TEST

# ============================================================

def test_validate_epoch():
    print()
    print("=" * 60)
    print("TEST - VALIDATE ONE EPOCH")
    print("=" * 60)

# ========================================================
# DEVICE
# ========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(f"Device: {device}")

# ========================================================
# VALIDATION DATASET
# ========================================================

validation_dataset = SyntheticSeismicDataset(
    num_samples=1,
    cube_size=(64, 128, 128),
    missing_probability=0.3
)

# ========================================================
# VALIDATION DATALOADER
# ========================================================

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False
)

print()
print(
    f"Validation batches : "
    f"{len(validation_loader)}"
)

# ========================================================
# MODEL
# ========================================================

model = Network3D()

# ========================================================
# LOSS FUNCTION
# ========================================================

criterion = TotalLoss(
    dx=1.0,
    dy=1.0,
    dz=1.0
)

# ========================================================
# OPTIMIZER
# ========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ========================================================
# TRAINER
# ========================================================

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

print(
    "Trainer initialization: OK"
)

# ========================================================
# VALIDATE ONE EPOCH
# ========================================================

validation_results = (
    trainer.validate_epoch(
        validation_loader
    )
)

# ========================================================
# EXPECTED OUTPUTS
# ========================================================

expected_keys = [

    "total",
    "mae",
    "physics",
    "uncertainty",
    "ssim",

    "metric_mae",
    "metric_rmse",
    "metric_psnr",
    "metric_snr",
    "metric_ssim"
]

# ========================================================
# VALIDATE OUTPUT KEYS
# ========================================================

for key in expected_keys:

    assert (
        key in validation_results
    ), (
        f"Missing validation result: {key}"
    )

print()
print(
    "Validation result keys: OK"
)

# ========================================================
# VALIDATE FINITE VALUES
# ========================================================

print()
print("VALIDATION RESULTS")
print("-" * 40)

for key in expected_keys:

    value = validation_results[key]

    print(
        f"{key:<20}: "
        f"{value:.6f}"
    )

    assert torch.isfinite(
        torch.tensor(value)
    ), (
        f"Non-finite value detected "
        f"for {key}: {value}"
    )

print()
print(
    "All validation losses and metrics "
    "are finite: OK"
)

# ========================================================
# BASIC LOSS VALIDATION
# ========================================================

assert (
    validation_results["total"] >= 0
), (
    "Validation total loss is negative."
)

print(
    "Total validation loss check: OK"
)

# ========================================================
# TEST PASSED
# ========================================================

print()
print("=" * 60)
print("VALIDATE EPOCH TEST: PASSED")
print("=" * 60)