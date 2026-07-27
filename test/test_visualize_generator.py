import matplotlib.pyplot as plt

from dataset.geological_generator import GeologicalGenerator
from utils.plotting import save_figure

generator = GeologicalGenerator()

cube = generator.generate(dipping=True)

# Middle inline slice
slice_image = cube[0, :, 64, :]

plt.figure(figsize=(8, 6))
plt.imshow(slice_image,
           cmap="gray",
           aspect="auto")

plt.title("Synthetic Geological Model")
plt.xlabel("Crossline")
plt.ylabel("Depth")

plt.colorbar()

save_figure(
    "dipping_layers.png",
    category="geology"
)

plt.show()