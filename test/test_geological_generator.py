from dataset.geological_generator import GeologicalGenerator

generator = GeologicalGenerator()

cube = generator.generate()

print("=" * 60)
print("Generated Cube Shape:", cube.shape)
print("Minimum Amplitude   :", cube.min().item())
print("Maximum Amplitude   :", cube.max().item())
print("=" * 60)