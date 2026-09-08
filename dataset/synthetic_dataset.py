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

IMPORTANT:
    Samples are generated lazily inside __getitem__.
    The complete dataset is NOT stored in RAM.

Reproducibility:
    Every sample is generated deterministically from:

        base dataset seed
        +
        sample index

    Separate deterministic seeds are assigned to the
    geological, velocity, and mask generation processes.

Author: Ormin Joseph
=========================================================
"""

import random

import numpy as np
import torch

from torch.utils.data import Dataset

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator
from dataset.mask_generator import SeismicMaskGenerator


class SyntheticSeismicDataset(Dataset):
    """
    Lazy synthetic dataset for Physics-Informed 3D
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

    Samples are generated on demand. Therefore, the
    dataset does not retain all seismic volumes in RAM.
    """

    # =====================================================
    # CLASS CONSTANTS
    # =====================================================

    VALID_GEOLOGICAL_MODES = (
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    )

    VALID_MASK_TYPES = (
        "random_voxels",
        "missing_traces",
        "missing_inlines",
        "missing_crosslines",
        "missing_blocks",
    )

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
        seed=42,
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

        seed : int
            Base random seed used to make sample generation
            deterministic with respect to the sample index.
        """

        super().__init__()

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.num_samples = int(num_samples)

        self.cube_size = tuple(
            int(dimension)
            for dimension in cube_size
        )

        self.missing_probability = float(
            missing_probability
        )

        self.geological_mode = str(
            geological_mode
        )

        self.mask_mode = str(
            mask_mode
        )

        self.seed = int(seed)

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
        # GEOLOGICAL GENERATOR
        # =================================================
        #
        # GeologicalGenerator does not maintain an
        # independent random generator. Its stochastic
        # operations therefore use the deterministic
        # PyTorch/NumPy seeds established for each sample.
        # =================================================

        self.generator = GeologicalGenerator(
            cube_size=self.cube_size
        )

        # =================================================
        # METADATA CACHE
        # =================================================
        #
        # IMPORTANT:
        # We do NOT store seismic tensors here.
        #
        # Only the generated metadata are retained.
        #
        # This requires very little RAM compared with storing
        # complete seismic volumes.
        # =================================================

        self.mask_types = [None] * self.num_samples

        self.geological_modes = [None] * self.num_samples

        # =================================================
        # DATASET INFORMATION
        # =================================================

        print()
        print("=" * 60)
        print("LAZY SYNTHETIC DATASET INITIALIZED")
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

        print(
            f"Random Seed          : "
            f"{self.seed}"
        )

        print(
            "Generation Strategy  : "
            "On-demand / lazy"
        )

        print("=" * 60)
        print()

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
            + ("random",)
        ):

            raise ValueError(
                "Unsupported geological_mode: "
                f"{self.geological_mode}. "
                "Supported modes are: "
                f"{self.VALID_GEOLOGICAL_MODES + ('random',)}"
            )

        if self.mask_mode not in (
            self.VALID_MASK_TYPES
            + ("random",)
        ):

            raise ValueError(
                "Unsupported mask_mode: "
                f"{self.mask_mode}. "
                "Supported modes are: "
                f"{self.VALID_MASK_TYPES + ('random',)}"
            )

    # =====================================================
    # SAMPLE SEED
    # =====================================================

    def _get_sample_seed(self, idx):
        """
        Generate a deterministic base seed for one sample.

        The seed depends only on:

            dataset seed
            sample index

        Therefore, the same dataset seed and sample index
        always produce the same sample.
        """

        return (
            self.seed
            + int(idx) * 100003
        )

    # =====================================================
    # SET GLOBAL SAMPLE SEEDS
    # =====================================================

    def _set_sample_seed(self, idx):
        """
        Set deterministic global random seeds for one sample.

        These seeds control randomness used by components
        that rely on the global random number generators.

        Independent generators such as VelocityGenerator
        and SeismicMaskGenerator receive their own explicit
        sample-specific seeds elsewhere.
        """

        sample_seed = self._get_sample_seed(idx)

        # -------------------------------------------------
        # Python random
        # -------------------------------------------------

        random.seed(sample_seed)

        # -------------------------------------------------
        # NumPy random
        # -------------------------------------------------

        np.random.seed(
            sample_seed % (2**32 - 1)
        )

        # -------------------------------------------------
        # PyTorch random
        # -------------------------------------------------

        torch.manual_seed(sample_seed)

        return sample_seed

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

        Existing tensors are converted without moving
        them to GPU.
        """

        if isinstance(data, torch.Tensor):

            return data.to(
                dtype=torch.float32,
                device="cpu",
            )

        return torch.as_tensor(
            data,
            dtype=torch.float32,
            device="cpu",
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

        if not isinstance(
            tensor,
            torch.Tensor
        ):

            raise RuntimeError(
                f"{name} must be a PyTorch tensor."
            )

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
            "Generated velocity model",
        )

        if torch.any(
            velocity <= 0
        ):

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
            "Generated sampling mask",
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
    # GENERATE ONE SAMPLE
    # =====================================================

    def _generate_sample(self, idx):
        """
        Generate one synthetic seismic sample.

        Nothing is permanently stored in the dataset.

        Every random component is deterministically seeded
        from the sample index.

        Returns
        -------
        tuple
            input_cube,
            target_cube,
            mask,
            velocity_model,
            mask_type,
            geological_mode
        """

        # -------------------------------------------------
        # Establish deterministic sample seed
        # -------------------------------------------------

        sample_seed = self._set_sample_seed(idx)

        # -------------------------------------------------
        # Use separate deterministic seeds for components
        # -------------------------------------------------
        #
        # Keeping separate seed streams prevents one generator
        # from changing the random sequence of another.
        # -------------------------------------------------

        velocity_seed = (
            sample_seed + 1
        )

        mask_seed = (
            sample_seed + 2
        )

        # -------------------------------------------------
        # Geological structure
        # -------------------------------------------------

        geological_mode = (
            self._select_geological_mode()
        )

        # -------------------------------------------------
        # Complete seismic target
        # -------------------------------------------------

        target = self.generator.generate(
            mode=geological_mode
        )

        target = self._to_float_tensor(
            target
        )

        self._validate_tensor(
            target,
            "Generated seismic target",
        )

        # -------------------------------------------------
        # Velocity model
        # -------------------------------------------------
        #
        # IMPORTANT:
        # Create a sample-specific generator with an explicit
        # seed. This avoids dependence on the generator's
        # previous internal random state.
        # -------------------------------------------------

        velocity_generator = VelocityGenerator(
            cube_size=self.cube_size,
            seed=velocity_seed,
        )

        velocity = (
            velocity_generator.generate()
        )

        velocity = self._to_float_tensor(
            velocity
        )

        self._validate_velocity(
            velocity
        )

        # -------------------------------------------------
        # Sampling mask type
        # -------------------------------------------------

        mask_type = (
            self._select_mask_type()
        )

        # -------------------------------------------------
        # Sampling mask
        # -------------------------------------------------
        #
        # Create a sample-specific mask generator with an
        # explicit seed so repeated calls to dataset[idx]
        # produce the same mask.
        # -------------------------------------------------

        mask_generator = SeismicMaskGenerator(
            cube_size=self.cube_size,
            missing_probability=self.missing_probability,
            seed=mask_seed,
        )

        mask = (
            mask_generator.generate(
                mask_type=mask_type
            )
        )

        mask = self._to_float_tensor(
            mask
        )

        self._validate_mask(
            mask
        )

        # -------------------------------------------------
        # Simulate incomplete acquisition
        # -------------------------------------------------

        input_cube = (
            target * mask
        )

        self._validate_tensor(
            input_cube,
            "Generated input seismic volume",
        )

        # -------------------------------------------------
        # Store only metadata
        # -------------------------------------------------

        self.mask_types[idx] = mask_type

        self.geological_modes[idx] = (
            geological_mode
        )

        # -------------------------------------------------
        # Return current sample
        # -------------------------------------------------

        return (
            input_cube,
            target,
            mask,
            velocity,
            mask_type,
            geological_mode,
        )

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
        idx,
    ):
        """
        Generate and return one synthetic seismic sample.

        Samples are generated on demand.

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

        # -------------------------------------------------
        # Handle tensor indices
        # -------------------------------------------------

        if isinstance(
            idx,
            torch.Tensor
        ):

            idx = idx.item()

        # -------------------------------------------------
        # Convert index to integer
        # -------------------------------------------------

        idx = int(idx)

        # -------------------------------------------------
        # Validate index
        # -------------------------------------------------

        if idx < 0:

            idx += self.num_samples

        if idx < 0 or idx >= self.num_samples:

            raise IndexError(
                f"Dataset index {idx} is out of range "
                f"for dataset of size {self.num_samples}."
            )

        # -------------------------------------------------
        # Generate sample on demand
        # -------------------------------------------------

        return self._generate_sample(idx)