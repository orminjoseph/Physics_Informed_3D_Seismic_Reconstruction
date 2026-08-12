from dataset.build_dataset import build_dataset

dataset = build_dataset()

input_cube, target_cube, mask, velocity_cube = dataset[0]

print()
print("Input Shape:", input_cube.shape)
print("Target Shape:", target_cube.shape)
print("Mask Shape:", mask.shape)
print("Velocity Shape:", velocity_cube.shape)
print()

print("Input Min:", input_cube.min().item())
print("Input Max:", input_cube.max().item())

print("Target Min:", target_cube.min().item())
print("Target Max:", target_cube.max().item())

print("Velocity Min:", velocity_cube.min().item())
print("Velocity Max:", velocity_cube.max().item())