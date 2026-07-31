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

    def save_comparison(

            self,

            corrupted,

            target,

            reconstruction,

            uncertainty,

            filename="f3_comparison.png"

    ):
        """
        Save publication-style comparison figure.
        """

        import numpy as np

        if hasattr(corrupted, "detach"):
            corrupted = corrupted.detach().cpu().numpy()

        if hasattr(target, "detach"):
            target = target.detach().cpu().numpy()

        if hasattr(reconstruction, "detach"):
            reconstruction = reconstruction.detach().cpu().numpy()

        if hasattr(uncertainty, "detach"):
            uncertainty = uncertainty.detach().cpu().numpy()

        corrupted = np.squeeze(corrupted)
        target = np.squeeze(target)
        reconstruction = np.squeeze(reconstruction)
        uncertainty = np.squeeze(uncertainty)

        slice_index = corrupted.shape[0] // 2

        corrupted_slice = corrupted[slice_index]
        target_slice = target[slice_index]
        reconstruction_slice = reconstruction[slice_index]
        uncertainty_slice = uncertainty[slice_index]

        difference_slice = np.abs(
            target_slice - reconstruction_slice
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 10)
        )

        axes[0, 0].imshow(
            corrupted_slice,
            cmap="gray",
            aspect="auto"
        )
        axes[0, 0].set_title("Corrupted Input")

        axes[0, 1].imshow(
            target_slice,
            cmap="gray",
            aspect="auto"
        )
        axes[0, 1].set_title("Ground Truth")

        axes[0, 2].imshow(
            reconstruction_slice,
            cmap="gray",
            aspect="auto"
        )
        axes[0, 2].set_title("Reconstruction")

        axes[1, 0].imshow(
            difference_slice,
            cmap="hot",
            aspect="auto"
        )
        axes[1, 0].set_title("Difference")

        axes[1, 1].imshow(
            uncertainty_slice,
            cmap="viridis",
            aspect="auto"
        )
        axes[1, 1].set_title("Uncertainty")

        axes[1, 2].axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            self.comparison_directory,
            filename
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

