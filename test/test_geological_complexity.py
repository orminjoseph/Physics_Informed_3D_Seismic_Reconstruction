from dataset.geological_generator import GeologicalGenerator
import matplotlib.pyplot as plt

generator = GeologicalGenerator()

complexities = {
    "horizontal":
        generator.generate_horizontal_layers(),

    "dipping":
        generator.generate_dipping_layers(),

    "faulted":
        generator.generate_faulted_layers(),

    "folded":
        generator.generate_folded_layers(),

    "complex":
        generator.generate_complex_structure(),

    "highly_complex":
        generator.generate_highly_complex_structure()
}

for name, cube in complexities.items():

    print(f"{name}: {cube.shape}")

    plt.figure(figsize=(6,4))

    plt.imshow(
        cube[0, :, 64, :],
        cmap="seismic",
        aspect="auto"
    )

    plt.title(name)

    plt.show()