from utils.config import Config

print("=" * 50)
print("Project Configuration Test")
print("=" * 50)

print("Device:", Config.DEVICE)
print("Batch Size:", Config.BATCH_SIZE)
print("Epochs:", Config.NUM_EPOCHS)
print("Learning Rate:", Config.LEARNING_RATE)
print("Dropout:", Config.DROPOUT)
print("Checkpoint Directory:", Config.CHECKPOINT_DIR)
print("Results Directory:", Config.RESULTS_DIR)
print("=" * 50)