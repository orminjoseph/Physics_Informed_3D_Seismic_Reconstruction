# =========================================================
# PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION
# GLOBAL CONFIGURATION FILE
# =========================================================

"""
Global configuration for the complete
Physics-Informed 3D Seismic Reconstruction project.

Research framework
------------------

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data
Reconstruction in Complex Geological Settings.

This file centralizes all major project parameters.

Configuration categories
------------------------

1. Dataset configuration
2. Synthetic dataset configuration
3. F3 dataset configuration
4. Training configuration
5. Validation configuration
6. Checkpoint configuration
7. Early-stopping configuration
8. Model configuration
9. Physics configuration
10. Composite-loss weights
11. Physics-loss weights
12. Seismic physical sampling
13. Velocity-model configuration
14. Device configuration
15. DataLoader configuration
16. Output configuration
17. Reproducibility configuration

Tensor convention
-----------------

All seismic volumes use:

    [B, C, D, H, W]

where:

    B = batch size
    C = channel
    D = depth
    H = crossline
    W = inline

Important physics convention
----------------------------

The Eikonal equation is

    |∇T|² = 1 / V²

where:

    T = seismic travel time [s]
    V = P-wave velocity [m/s]

The velocity model used by the physics loss must therefore
represent physically meaningful P-wave velocity.

Author: Ormin Joseph
"""


# =========================================================
# 1. DATASET MODE
# =========================================================

# Active dataset used by the training pipeline.

DATASET_MODE = "synthetic"

# Available alternatives:
#
# DATASET_MODE = "synthetic"
# DATASET_MODE = "f3"


# =========================================================
# 2. EXPERIMENT NAME
# =========================================================

EXPERIMENT_NAME = "synthetic_training"

# Examples:
#
# "synthetic_pretraining"
# "f3_finetuning"
# "ablation_study"
# "uncertainty_analysis"


# =========================================================
# 3. SYNTHETIC DATASET CONFIGURATION
# =========================================================

# Number of synthetic seismic volumes.

SYNTHETIC_NUM_SAMPLES = 10


# Synthetic seismic cube dimensions:
#
#     [Depth, Crossline, Inline]

SYNTHETIC_PATCH_SIZE = (
    64,
    128,
    128
)


# Probability that a seismic sample/voxel is removed
# during synthetic missing-data generation.

SYNTHETIC_MISSING_PROBABILITY = 0.30


# =========================================================
# 4. F3 DATASET CONFIGURATION
# =========================================================

# Path to the F3 Netherlands SEG-Y seismic dataset.

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


# F3 patch dimensions:
#
#     [Depth, Crossline, Inline]

F3_PATCH_SIZE = (
    64,
    64,
    64
)


# Patch extraction stride:
#
#     [Depth, Crossline, Inline]

F3_STRIDE = (
    64,
    64,
    64
)


# Missing-data probability for F3 experiments.

F3_MISSING_PROBABILITY = 0.30


# =========================================================
# 5. TRAINING CONFIGURATION
# =========================================================

# Batch size.
#
# A value of 1 is appropriate for the current 3D CPU
# development pipeline because 3D seismic tensors consume
# substantial memory.

BATCH_SIZE = 1


# Number of training epochs.

NUM_EPOCHS = 5


# Initial learning rate for Adam.

LEARNING_RATE = 1e-4


# L2 regularization coefficient.

WEIGHT_DECAY = 1e-5


# =========================================================
# 6. VALIDATION CONFIGURATION
# =========================================================

# Fraction of available data reserved for validation.

VALIDATION_SPLIT = 0.10


# =========================================================
# 7. CHECKPOINT CONFIGURATION
# =========================================================

# Save a periodic epoch checkpoint every N epochs.

SAVE_EVERY = 5


# =========================================================
# 8. EARLY STOPPING CONFIGURATION
# =========================================================

# Number of consecutive validation epochs without
# improvement before training is stopped.

PATIENCE = 10


# =========================================================
# 9. MODEL CONFIGURATION
# =========================================================

# Enable attention gates in the decoder.

USE_ATTENTION = True


# Enable residual blocks.

USE_RESIDUAL = True


# Enable predictive uncertainty estimation.

USE_UNCERTAINTY = True

# Number of stochastic forward passes for
# Monte Carlo Dropout epistemic uncertainty.
MC_DROPOUT_SAMPLES = 20
# =========================================================
# 10. PHYSICS-INFORMED CONFIGURATION
# =========================================================

# Enable the physics-informed loss.

USE_PHYSICS_LOSS = True


# =========================================================
# 11. PHYSICS SUPERVISION OPTIONS
# =========================================================

# Source coordinates are currently NOT available in the
# present dataset.
#
# Therefore, source-condition supervision must remain
# disabled until valid source coordinates are supplied.

USE_SOURCE_LOSS = False


# Supervised travel-time targets are currently NOT available
# in the present dataset.
#
# Therefore, supervised travel-time loss remains disabled.

USE_TRAVEL_TIME_LOSS = False


# The Eikonal equation itself remains active.

USE_EIKONAL_LOSS = True


# =========================================================
# 12. COMPOSITE LOSS WEIGHTS
# =========================================================

"""
Global composite loss:

    L_total =
        λ_mae L_mae
        +
        λ_physics L_physics
        +
        λ_uncertainty L_uncertainty
        +
        λ_ssim L_ssim
"""

LOSS_WEIGHTS = {

    # Reconstruction fidelity.
    "mae": 1.0,

    # Physics-informed constraint.
    "physics": 0.10,

    # Predictive uncertainty.
    "uncertainty": 0.01,

    # Structural similarity.
    "ssim": 0.10
}


# =========================================================
# 13. PHYSICS-LOSS COMPONENT WEIGHTS
# =========================================================

"""
Physics loss:

    L_physics =
        λ_eikonal L_eikonal
        +
        λ_source L_source
        +
        λ_travel_time L_travel_time

At the present stage:

    L_source = 0
    L_travel_time = 0

because the current dataset does not provide valid
source coordinates or independent travel-time targets.

The Eikonal loss remains active.
"""

PHYSICS_LOSS_WEIGHTS = {

    # Eikonal equation residual.
    "eikonal": 1.0,

    # Source condition:
    #
    #     T(x_s, y_s, z_s) = 0
    #
    # Currently disabled by USE_SOURCE_LOSS.

    "source": 1.0,

    # Supervised travel-time loss.
    #
    # Currently disabled by USE_TRAVEL_TIME_LOSS.

    "travel_time": 1.0
}


# =========================================================
# 14. PHYSICAL SEISMIC SAMPLING
# =========================================================

"""
Physical spatial sampling.

These values determine the physical distance represented
by one voxel in each spatial direction.

Units:

    DX = metres
    DY = metres
    DZ = metres
"""

# Inline spacing [m].

DX = 1.0


# Crossline spacing [m].

DY = 1.0


# Depth spacing [m].

DZ = 1.0


# =========================================================
# 15A. TRAVEL-TIME CONFIGURATION
# =========================================================

"""
Travel-time scaling.

The network predicts a positive dimensionless travel-time
field which is converted to physical seconds before the
Eikonal equation is evaluated.

This prevents the neural network from initially producing
travel-time values on an uncontrolled physical scale.
"""

TRAVEL_TIME_SCALE = 0.1
# =========================================================
# 15. VELOCITY MODEL CONFIGURATION
# =========================================================

"""
P-wave velocity configuration.

The Eikonal equation requires physical velocity:

    V [m/s]

If the dataset stores physical velocity values directly,
set:

    VELOCITY_NORMALIZED = False

If the dataset stores normalized velocity values in [0,1],
set:

    VELOCITY_NORMALIZED = True

and define the corresponding physical minimum and maximum.

The physics loss will then convert normalized velocity
back into physical velocity before evaluating:

    1 / V²
"""


# Current assumption:
#
# The velocity model supplied to the physics loss is
# physically meaningful P-wave velocity.

VELOCITY_NORMALIZED = False


# Physical minimum P-wave velocity [m/s].
#
# These values are configuration placeholders and should
# correspond to the actual velocity range used by the
# dataset.

VELOCITY_MIN = 1500.0


# Physical maximum P-wave velocity [m/s].

VELOCITY_MAX = 5000.0

# =========================================================
# 16. SEISMIC AMPLITUDE NORMALIZATION
# =========================================================

"""
Seismic amplitude normalization.

The current reconstruction and SSIM implementations assume
the seismic amplitudes are represented approximately in:

    [-1, 1]

Therefore:

    data_range = 2.0

for SSIM.

This configuration makes that assumption explicit.
"""

SEISMIC_AMPLITUDE_MIN = -1.0

SEISMIC_AMPLITUDE_MAX = 1.0

SEISMIC_DATA_RANGE = (
    SEISMIC_AMPLITUDE_MAX
    - SEISMIC_AMPLITUDE_MIN
)


# =========================================================
# 17. MASK CONFIGURATION
# =========================================================

"""
Missing-data mask convention.

IMPORTANT:

    1 = observed/available voxel
    0 = missing voxel

This convention will be used consistently by the
dataset and reconstruction loss.

If the dataset uses the opposite convention, this value
must be changed before training.
"""

MASK_OBSERVED_VALUE = 1.0

MASK_MISSING_VALUE = 0.0


# =========================================================
# 18. DATALOADER CONFIGURATION
# =========================================================

# Number of worker processes used by DataLoader.

# Zero is safest for the current Windows development
# environment and simplifies debugging.

NUM_WORKERS = 0


# Pin memory when using CUDA.
#
# The Trainer can use this setting when GPU training
# becomes available.

PIN_MEMORY = False


# Persistent workers are disabled because NUM_WORKERS = 0.

PERSISTENT_WORKERS = False


# =========================================================
# 19. MIXED-PRECISION CONFIGURATION
# =========================================================

"""
Automatic mixed precision.

Disabled during the current CPU development stage.

When CUDA training is confirmed, this can be enabled.
"""

USE_AMP = False


# =========================================================
# 20. DEVICE CONFIGURATION
# =========================================================

"""
Current validated development device.

CPU is intentionally retained until CUDA-enabled PyTorch
is confirmed on the target machine.
"""

DEVICE = "cpu"

# Later:

# DEVICE = "cuda"


# =========================================================
# 21. OUTPUT CONFIGURATION
# =========================================================

# Root output directory.

OUTPUT_ROOT = "outputs"


# Experiment-specific checkpoint directory.

CHECKPOINT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"checkpoints"
)


# Experiment-specific figure directory.

FIGURE_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"figures"
)


# Experiment-specific report directory.

REPORT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"reports"
)


# =========================================================
# 22. REPRODUCIBILITY
# =========================================================

# Global random seed.

SEED = 42


# =========================================================
# 23. CONFIGURATION VALIDATION
# =========================================================

def validate_config():
    """
    Validate the global configuration before training.

    This function prevents inconsistent configuration
    values from silently entering the training pipeline.
    """

    # -----------------------------------------------------
    # Dataset mode
    # -----------------------------------------------------

    if DATASET_MODE not in {
        "synthetic",
        "f3"
    }:

        raise ValueError(
            "DATASET_MODE must be either "
            "'synthetic' or 'f3'."
        )

    # -----------------------------------------------------
    # Batch size
    # -----------------------------------------------------

    if BATCH_SIZE < 1:

        raise ValueError(
            "BATCH_SIZE must be at least 1."
        )

    # -----------------------------------------------------
    # Epochs
    # -----------------------------------------------------

    if NUM_EPOCHS < 1:

        raise ValueError(
            "NUM_EPOCHS must be at least 1."
        )

    # -----------------------------------------------------
    # Learning rate
    # -----------------------------------------------------

    if LEARNING_RATE <= 0:

        raise ValueError(
            "LEARNING_RATE must be greater than zero."
        )

    # -----------------------------------------------------
    # Weight decay
    # -----------------------------------------------------

    if WEIGHT_DECAY < 0:

        raise ValueError(
            "WEIGHT_DECAY cannot be negative."
        )

    # -----------------------------------------------------
    # Validation split
    # -----------------------------------------------------

    if not 0.0 < VALIDATION_SPLIT < 1.0:

        raise ValueError(
            "VALIDATION_SPLIT must be between 0 and 1."
        )

    # -----------------------------------------------------
    # Missing probability
    # -----------------------------------------------------

    if not 0.0 <= SYNTHETIC_MISSING_PROBABILITY <= 1.0:

        raise ValueError(
            "SYNTHETIC_MISSING_PROBABILITY must be "
            "between 0 and 1."
        )

    if not 0.0 <= F3_MISSING_PROBABILITY <= 1.0:

        raise ValueError(
            "F3_MISSING_PROBABILITY must be "
            "between 0 and 1."
        )

    # -----------------------------------------------------
    # Spatial sampling
    # -----------------------------------------------------

    if DX <= 0 or DY <= 0 or DZ <= 0:

        raise ValueError(
            "DX, DY and DZ must all be greater than zero."
        )

    # -----------------------------------------------------
    # Velocity range
    # -----------------------------------------------------

    if VELOCITY_MIN <= 0:

        raise ValueError(
            "VELOCITY_MIN must be greater than zero."
        )

    if VELOCITY_MAX <= VELOCITY_MIN:

        raise ValueError(
            "VELOCITY_MAX must be greater than "
            "VELOCITY_MIN."
        )

    # -----------------------------------------------------
    # Loss weights
    # -----------------------------------------------------

    for name, value in LOSS_WEIGHTS.items():

        if value < 0:

            raise ValueError(
                f"LOSS_WEIGHTS['{name}'] "
                "cannot be negative."
            )

    # -----------------------------------------------------
    # Physics loss weights
    # -----------------------------------------------------

    for name, value in PHYSICS_LOSS_WEIGHTS.items():

        if value < 0:

            raise ValueError(
                f"PHYSICS_LOSS_WEIGHTS['{name}'] "
                "cannot be negative."
            )

    # -----------------------------------------------------
    # Physics-loss consistency
    # -----------------------------------------------------

    if not USE_PHYSICS_LOSS:

        if USE_EIKONAL_LOSS:
            raise ValueError(
                "USE_EIKONAL_LOSS cannot be True when "
                "USE_PHYSICS_LOSS is False."
            )

        if USE_SOURCE_LOSS:
            raise ValueError(
                "USE_SOURCE_LOSS cannot be True when "
                "USE_PHYSICS_LOSS is False."
            )

        if USE_TRAVEL_TIME_LOSS:
            raise ValueError(
                "USE_TRAVEL_TIME_LOSS cannot be True "
                "when USE_PHYSICS_LOSS is False."
            )

    # -----------------------------------------------------
    # Current dataset limitation
    # -----------------------------------------------------

    if USE_SOURCE_LOSS:

        raise ValueError(
            "USE_SOURCE_LOSS is currently True, but "
            "the present dataset does not provide valid "
            "source coordinates."
        )

    if USE_TRAVEL_TIME_LOSS:

        raise ValueError(
            "USE_TRAVEL_TIME_LOSS is currently True, "
            "but the present dataset does not provide "
            "independent travel-time targets."
        )

    # -----------------------------------------------------
    # Mask convention
    # -----------------------------------------------------

    if MASK_OBSERVED_VALUE == MASK_MISSING_VALUE:

        raise ValueError(
            "MASK_OBSERVED_VALUE and MASK_MISSING_VALUE "
            "must be different."
        )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    if DEVICE not in {
        "cpu",
        "cuda"
    }:

        raise ValueError(
            "DEVICE must be either 'cpu' or 'cuda'."
        )

    # -----------------------------------------------------
    # AMP
    # -----------------------------------------------------

    if USE_AMP and DEVICE != "cuda":

        raise ValueError(
            "USE_AMP=True requires DEVICE='cuda'."
        )

    # -----------------------------------------------------
    # Configuration successfully validated
    # -----------------------------------------------------

    return True


# =========================================================
# 24. RUN CONFIGURATION VALIDATION
# =========================================================

validate_config()