import matplotlib.pyplot as plt

from dataset.geological_generator import GeologicalGenerator

generator = GeologicalGenerator()

modes = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8)
)

for ax, mode in zip(
        axes.flatten(),
        modes
):

    cube = generator.generate(
        mode=mode
    )

    slice_ = cube[0, :, 64, :]

    im = ax.imshow(
        slice_,
        cmap="seismic",
        aspect="auto"
    )

    ax.set_title(mode)

plt.tight_layout()

plt.savefig(
    "outputs/reports/geological_complexity_gallery.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()