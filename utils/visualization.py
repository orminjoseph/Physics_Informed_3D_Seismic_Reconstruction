import os

import numpy as np

import matplotlib.pyplot as plt

class Visualizer:
    """
    Visualization utilities for seismic reconstruction.

    Responsible for

    - saving reconstructed slices,
    - saving uncertainty maps,
    - creating comparison figures,
    - exporting publication-quality images.
    """

    def __init__(

            self,

            output_directory="outputs"

    ):
        """
        Initialise visualisation directories.
        """

        self.output_directory = output_directory

        self.reconstruction_directory = os.path.join(

            self.output_directory,

            "reconstructions"

        )

        self.uncertainty_directory = os.path.join(

            self.output_directory,

            "uncertainty"

        )

        self.comparison_directory = os.path.join(

            self.output_directory,

            "comparisons"

        )

        os.makedirs(

            self.reconstruction_directory,

            exist_ok=True

        )

        os.makedirs(

            self.uncertainty_directory,

            exist_ok=True

        )

        os.makedirs(

            self.comparison_directory,

            exist_ok=True

        )

    def save_slice(

            self,

            volume,

            filename,

            title="Seismic Slice"

    ):
        """
        Save one central seismic slice.

        Parameters
        ----------
        volume : ndarray or torch.Tensor

            3D seismic cube.

        filename : str

            Output filename.

        title : str

            Figure title.
        """

        # ------------------------------------------
        # Convert PyTorch tensor to NumPy
        # ------------------------------------------

        if hasattr(volume, "detach"):
            volume = volume.detach().cpu().numpy()

        # ------------------------------------------
        # Remove channel dimension if present
        # ------------------------------------------

        if volume.ndim == 4:
            volume = volume[0]

        # ------------------------------------------
        # Central inline slice
        # ------------------------------------------

        slice_index = volume.shape[0] // 2

        seismic_slice = volume[slice_index]

        # ------------------------------------------
        # Create figure
        # ------------------------------------------

        plt.figure(figsize=(8, 6))

        plt.imshow(

            seismic_slice,

            cmap="gray",

            aspect="auto"

        )

        plt.title(title)

        plt.colorbar()

        plt.tight_layout()

        # ------------------------------------------
        # Save figure
        # ------------------------------------------

        save_path = os.path.join(

            self.reconstruction_directory,

            filename

        )

        plt.savefig(

            save_path,

            dpi=300

        )

        plt.close()

