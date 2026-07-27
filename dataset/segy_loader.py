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

        self.metadata = {}

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

            inline_numbers = segy.ilines

            crossline_numbers = segy.xlines

            sample_axis = segy.samples

            trace_count = segy.tracecount

            cube = segyio.tools.cube(segy)

        cube = cube.astype(np.float32)

        self.metadata = {

            "inline_numbers": inline_numbers,

            "crossline_numbers": crossline_numbers,

            "sample_axis": sample_axis,

            "trace_count": trace_count,

            "cube_shape": cube.shape

        }

        print()

        print("SEG-Y Volume Loaded Successfully.")

        print(f"Cube Shape : {cube.shape}")

        print(f"Trace Count : {trace_count}")

        print(f"Number of Inlines : {len(inline_numbers)}")

        print(f"Number of Crosslines : {len(crossline_numbers)}")

        print(f"Samples per Trace : {len(sample_axis)}")

        print(f"Minimum    : {cube.min():.6f}")

        print(f"Maximum    : {cube.max():.6f}")

        return cube

    def get_metadata(self):
        if not self.metadata:
            raise RuntimeError(
                "No metadata available. "
                "Call load() first."
            )

        return self.metadata


