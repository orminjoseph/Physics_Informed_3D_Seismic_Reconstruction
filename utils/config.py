# =====================================================
# PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION
# GLOBAL CONFIGURATION FILE
# =====================================================

# =====================================================
# DATASET MODE
# =====================================================

#DATASET_MODE = "synthetic"
# DATASET_MODE = "f3"
DATASET_MODE = "marmousi2"

# =====================================================
# EXPERIMENT NAME
# =====================================================

#EXPERIMENT_NAME = ("synthetic_pretraining")

EXPERIMENT_NAME = "marmousi2_training"
#EXPERIMENT_NAME = "f3_training"

# Examples:
# "synthetic_pretraining"
# "f3_finetuning"
# "ablation_study"
# "uncertainty_analysis"

# =====================================================
# SYNTHETIC DATASET
# =====================================================

SYNTHETIC_NUM_SAMPLES = 100

SYNTHETIC_PATCH_SIZE = (
    64,
    128,
    128
)

SYNTHETIC_MISSING_PROBABILITY = 0.30

# =====================================================
# F3 DATASET
# =====================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

F3_PATCH_SIZE = (
    64,
    64,
    64
)

F3_STRIDE = (
    64,
    64,
    64
)

F3_MISSING_PROBABILITY = 0.30

# ==================================================
# MARMOUSI2 DATASET
# ==================================================

MARMOUSI2_PATH = (
    r"C:\Users\ormin\Desktop\SEG_FILES"
    r"\elastic-marmousi-model"
    r"\elastic-marmousi-model"
    r"\model"
    r"\MODEL_P-WAVE_VELOCITY_1.25m.segy"
    r"\MODEL_P-WAVE_VELOCITY_1.25m.segy"
)

MARMOUSI2_PATCH_SIZE = (128, 128)

MARMOUSI2_MISSING_PROBABILITY = 0.30

MARMOUSI2_MASK_TYPE = "random_trace"

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# =====================================================
# MODEL OPTIONS
# =====================================================

USE_ATTENTION = True

USE_RESIDUAL = True

USE_UNCERTAINTY = True

# =====================================================
# PHYSICS-INFORMED SETTINGS
# =====================================================

USE_PHYSICS_LOSS = True

PHYSICS_WEIGHT = 0.10

# =====================================================
# LOSS WEIGHTS
# =====================================================

LOSS_WEIGHTS = {

    "mae": 1.0,

    "physics": 0.10,

    "uncertainty": 0.01,

    "ssim": 0.10
}

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda"
# DEVICE = "cpu"

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

OUTPUT_ROOT = "outputs"

CHECKPOINT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"checkpoints"
)

FIGURE_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"figures"
)

REPORT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"reports"
)

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42