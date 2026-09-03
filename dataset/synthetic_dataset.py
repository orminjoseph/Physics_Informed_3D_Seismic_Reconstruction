"""
=========================================================
Synthetic 3D Seismic Dataset
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Generates synthetic seismic volumes together with:

    1. incomplete seismic input
    2. complete seismic target
    3. sampling mask
    4. velocity model
    5. mask type
    6. geological mode

Tensor convention:

    Individual sample:
        [C, D, H, W]

    DataLoader batch:
        [B, C, D, H, W]

The velocity model is supplied to the physics loss and
is NOT predicted by the neural network.

Mask type and geological mode are metadata used for
experiment tracking and analysis.

Author: Ormin Joseph
=========================================================
"""

import numpy as np
import torch

from torch.utils.data import Dataset

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator
from dataset.mask_generator import SeismicMaskGenerator


class SyntheticSeismicDataset(Dataset):
    """
    Synthetic dataset for Physics-Informed 3D
    Seismic Reconstruction.

    Each sample contains:

        input_cube
        target_cube
        mask
        velocity_model
        mask_type
        geological_mode

    Tensor shapes:

        input_cube:
            [C, D, H, W]

        target_cube:
            [C, D, H, W]

        mask:
            [C, D, H, W]

        velocity_model:
            [C, D, H, W]

    Metadata:

        mask_type:
            Type of missing-data pattern.

        geological_mode:
            Geological structure used to generate
            the seismic target.
    """

    # =====================================================
    # CLASS CONSTANTS
    # =====================================================

    VALID_GEOLOGICAL_MODES = [
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    ]

    VALID_MASK_TYPES = [
        "random_voxels",
        "missing_traces",
        "missing_inlines",
        "missing_crosslines",
        "missing_blocks",
    ]

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        num_samples=100,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
    ):
        """
        Parameters
        ----------
        num_samples : int
            Number of synthetic seismic samples.

        cube_size : tuple
            3D seismic volume size:

                (depth, height, width)

        missing_probability : float
            Approximate proportion of seismic data
            to remove.

        geological_mode : str
            Geological structure to generate.

            Options:

                horizontal
                dipping
                faulted
                folded
                complex
                highly_complex
                random

        mask_mode : str
            Missing-data pattern.

            Options:

                random_voxels
                missing_traces
                missing_inlines
                missing_crosslines
                missing_blocks
                random
        """

        super().__init__()

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.num_samples = int(num_samples)

        self.cube_size = tuple(cube_size)

        self.missing_probability = float(
            missing_probability
        )

        self.geological_mode = str(
            geological_mode
        )

        self.mask_mode = str(
            mask_mode
        )

        # =================================================
        # EXPECTED SAMPLE SHAPE
        # =================================================

        self.expected_shape = (
            1,
            *self.cube_size
        )

        # =================================================
        # VALIDATION
        # =================================================

        self._validate_configuration()

        # =================================================
        # GENERATORS
        # =================================================

        self.generator = GeologicalGenerator(
            cube_size=self.cube_size
        )

        self.velocity_generator = VelocityGenerator(
            cube_size=self.cube_size
        )

        self.mask_generator = SeismicMaskGenerator(
            cube_size=self.cube_size,
            missing_probability=self.missing_probability,
        )

        # =================================================
        # DATA STORAGE
        # =================================================

        self.inputs = []
        self.targets = []
        self.masks = []
        self.velocities = []

        # Experiment metadata
        self.mask_types = []
        self.geological_modes = []

        # =================================================
        # DATASET GENERATION
        # =================================================

        self._generate_dataset()

    # =====================================================
    # CONFIGURATION VALIDATION
    # =====================================================

    def _validate_configuration(self):
        """
        Validate dataset configuration.
        """

        if self.num_samples <= 0:

            raise ValueError(
                "num_samples must be greater than zero."
            )

        if len(self.cube_size) != 3:

            raise ValueError(
                "cube_size must contain exactly "
                "(depth, height, width)."
            )

        if any(
            dimension <= 0
            for dimension in self.cube_size
        ):

            raise ValueError(
                "All cube dimensions must be positive."
            )

        if not (
            0.0
            <= self.missing_probability
            < 1.0
        ):

            raise ValueError(
                "missing_probability must be "
                "between 0.0 and 1.0."
            )

        if self.geological_mode not in (
            self.VALID_GEOLOGICAL_MODES
            + ["random"]
        ):

            raise ValueError(
                "Unsupported geological_mode: "
                f"{self.geological_mode}. "
                "Supported modes are: "
                f"{self.VALID_GEOLOGICAL_MODES + ['random']}"
            )

        if self.mask_mode not in (
            self.VALID_MASK_TYPES
            + ["random"]
        ):

            raise ValueError(
                "Unsupported mask_mode: "
                f"{self.mask_mode}. "
                "Supported modes are: "
                f"{self.VALID_MASK_TYPES + ['random']}"
            )

    # =====================================================
    # SELECT GEOLOGICAL MODE
    # =====================================================

    def _select_geological_mode(self):
        """
        Select the geological structure for one sample.
        """

        if self.geological_mode == "random":

            return str(
                np.random.choice(
                    self.VALID_GEOLOGICAL_MODES
                )
            )

        return self.geological_mode

    # =====================================================
    # SELECT MASK TYPE
    # =====================================================

    def _select_mask_type(self):
        """
        Select the missing-data pattern for one sample.
        """

        if self.mask_mode == "random":

            return str(
                np.random.choice(
                    self.VALID_MASK_TYPES
                )
            )

        return self.mask_mode

    # =====================================================
    # TENSOR CONVERSION
    # =====================================================

    @staticmethod
    def _to_float_tensor(data):
        """
        Convert data to a float32 PyTorch tensor.
        """

        if isinstance(data, torch.Tensor):

            return data.float()

        return torch.tensor(
            data,
            dtype=torch.float32
        )

    # =====================================================
    # VALIDATE TENSOR
    # =====================================================

    def _validate_tensor(
        self,
        tensor,
        name,
    ):
        """
        Validate tensor shape and finite values.
        """

        if tuple(tensor.shape) != self.expected_shape:

            raise RuntimeError(
                f"{name} has an unexpected shape.\n"
                f"Expected: {self.expected_shape}\n"
                f"Received: {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # VALIDATE VELOCITY
    # =====================================================

    def _validate_velocity(
        self,
        velocity,
    ):
        """
        Validate the velocity model.
        """

        self._validate_tensor(
            velocity,
            "Generated velocity model"
        )

        if torch.any(velocity <= 0):

            raise RuntimeError(
                "Velocity model must contain "
                "strictly positive velocities."
            )

    # =====================================================
    # VALIDATE MASK
    # =====================================================

    def _validate_mask(
        self,
        mask,
    ):
        """
        Validate the sampling mask.
        """

        self._validate_tensor(
            mask,
            "Generated sampling mask"
        )

        unique_values = torch.unique(mask)

        if not torch.all(
            (unique_values == 0.0)
            |
            (unique_values == 1.0)
        ):

            raise RuntimeError(
                "Sampling mask must contain "
                "only 0.0 and 1.0 values."
            )

    # =====================================================
    # GENERATE DATASET
    # =====================================================

    def _generate_dataset(self):
        """
        Generate all synthetic dataset samples.
        """

        print()
        print("=" * 60)
        print("GENERATING SYNTHETIC DATASET")
        print("=" * 60)

        print(
            f"Number of Samples    : "
            f"{self.num_samples}"
        )

        print(
            f"Cube Size            : "
            f"{self.cube_size}"
        )

        print(
            f"Missing Probability  : "
            f"{self.missing_probability}"
        )

        print(
            f"Geological Mode      : "
            f"{self.geological_mode}"
        )

        print(
            f"Mask Mode            : "
            f"{self.mask_mode}"
        )

        # =================================================
        # GENERATE EACH SAMPLE
        # =================================================

        for sample_index in range(
            self.num_samples
        ):

            # ---------------------------------------------
            # Geological structure
            # ---------------------------------------------

            geological_mode = (
                self._select_geological_mode()
            )

            # ---------------------------------------------
            # Complete seismic target
            # ---------------------------------------------

            target = self.generator.generate(
                mode=geological_mode
            )

            target = self._to_float_tensor(
                target
            )

            self._validate_tensor(
                target,
                "Generated seismic target"
            )

            # ---------------------------------------------
            # Velocity model
            # ---------------------------------------------

            velocity = (
                self.velocity_generator.generate()
            )

            velocity = self._to_float_tensor(
                velocity
            )

            self._validate_velocity(
                velocity
            )

            # ---------------------------------------------
            # Sampling mask
            # ---------------------------------------------

            mask_type = (
                self._select_mask_type()
            )

            mask = (
                self.mask_generator.generate(
                    mask_type=mask_type
                )
            )

            mask = self._to_float_tensor(
                mask
            )

            self._validate_mask(
                mask
            )

            # ---------------------------------------------
            # Simulate incomplete acquisition
            # ---------------------------------------------

            input_cube = target * mask

            self._validate_tensor(
                input_cube,
                "Generated input seismic volume"
            )

            # ---------------------------------------------
            # Store sample
            # ---------------------------------------------

            self.inputs.append(
                input_cube
            )

            self.targets.append(
                target
            )

            self.masks.append(
                mask
            )

            self.velocities.append(
                velocity
            )

            # Store metadata
            self.mask_types.append(
                mask_type
            )

            self.geological_modes.append(
                geological_mode
            )

            # ---------------------------------------------
            # Progress information
            # ---------------------------------------------

            print(
                f"Generated sample "
                f"{sample_index + 1}/"
                f"{self.num_samples}"
                f" | Geology: {geological_mode}"
                f" | Mask: {mask_type}"
            )

        # =================================================
        # COMPLETION MESSAGE
        # =================================================

        print("=" * 60)

        print(
            "Synthetic dataset generation completed."
        )

        print("=" * 60)

    # =====================================================
    # DATASET LENGTH
    # =====================================================

    def __len__(self):
        """
        Return the number of samples.
        """

        return self.num_samples

    # =====================================================
    # GET SAMPLE
    # =====================================================

    def __getitem__(
        self,
        idx
    ):
        """
        Return one synthetic seismic sample.

        Returns
        -------

        input_cube
            Incomplete seismic volume.

        target_cube
            Complete seismic volume.

        mask
            Sampling mask.

        velocity_model
            Corresponding velocity model.

        mask_type
            Missing-data pattern.

        geological_mode
            Geological structure.

        Tensor shapes:

            input_cube:
                [C, D, H, W]

            target_cube:
                [C, D, H, W]

            mask:
                [C, D, H, W]

            velocity_model:
                [C, D, H, W]
        """

        return (
            self.inputs[idx],
            self.targets[idx],
            self.masks[idx],
            self.velocities[idx],
            self.mask_types[idx],
            self.geological_modes[idx],
        )