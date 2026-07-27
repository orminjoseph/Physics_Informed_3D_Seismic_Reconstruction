import segyio
import numpy as np


class SegyLoader:
    """
    Loads a 3D SEG-Y seismic volume into a NumPy array.

    Output shape:
        (inline, crossline, samples)
    """

    def __init__(self, filename):

        self.filename = filename

    def load(self):
        print()

        print("=" * 60)
        print("Loading SEG-Y Volume")
        print("=" * 60)

        print(f"File: {self.filename}")

        with segyio.open(
                self.filename,
                "r",
                ignore_geometry=False
        ) as segy:
            segy.mmap()

            cube = segyio.tools.cube(segy)

        cube = cube.astype(np.float32)

        print()

        print("SEG-Y Volume Loaded Successfully.")

        print(f"Cube Shape : {cube.shape}")

        print(f"Minimum    : {cube.min():.6f}")

        print(f"Maximum    : {cube.max():.6f}")

        return cube
