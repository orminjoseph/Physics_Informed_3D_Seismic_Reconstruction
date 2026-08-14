"""
Training Configuration
"""

# ======================================
# Training
# ======================================

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# ======================================
# Validation
# ======================================

VALIDATION_SPLIT = 0.10

# ======================================
# Checkpointing
# ======================================

SAVE_EVERY = 5

# ======================================
# Early Stopping
# ======================================

PATIENCE = 10

# ======================================
# Device
# ======================================

DEVICE = "cpu"

# ======================================
# Experiment
# ======================================

#EXPERIMENT_NAME = "synthetic_training"
EXPERIMENT_NAME = "marmousi2_training"
#EXPERIMENT_NAME = "f3_training"

# Examples:
# "synthetic_pretraining"
# "f3_finetuning"
# "ablation_study"
# "uncertainty_analysis"