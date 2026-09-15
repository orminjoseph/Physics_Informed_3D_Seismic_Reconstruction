"""
=========================================================
Synthetic 3D Seismic Dataset
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction
in Complex Geological Settings

Purpose
-------
Generates scientifically controlled synthetic 3D seismic
reconstruction samples containing:

    1. incomplete seismic input
    2. complete seismic target
    3. sampling mask
    4. geologically conditioned velocity model
    5. mask type
    6. geological mode

The dataset is designed for:

    - supervised seismic reconstruction
    - physics-informed learning
    - Eikonal-based physical constraints
    - uncertainty-aware reconstruction
    - geological-complexity experiments
    - missing-data robustness experiments
    - reproducible PhD-level numerical experiments

Tensor convention
-----------------
Individual sample:

    [C, D, H, W]

DataLoader batch:

    [B, C, D, H, W]

where:

    C = seismic channel
    D = depth/time-sample dimension
    H = crossline/spatial dimension
    W = inline/spatial dimension

Important
---------
The velocity model is supplied to the physics-informed loss.
It is NOT predicted by the neural network.

The velocity model is conditioned on the same geological
scenario used to generate the seismic target.

Samples are generated lazily inside __getitem__.
The complete dataset is therefore NOT stored in RAM.

Reproducibility
---------------
Every sample receives a deterministic seed derived from:

    dataset_seed + sample_index

Independent deterministic seed streams are used for:

    - geological generation
    - velocity generation
    - mask generation

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# STANDARD LIBRARY
# =========================================================

import random


# =========================================================
# NUMERICAL / DEEP LEARNING LIBRARIES
# =========================================================

import numpy as np
import torch

from torch.utils.data import Dataset


# =========================================================
# PROJECT MODULES
# =========================================================

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator
from dataset.mask_generator import SeismicMaskGenerator


# =========================================================
# DATASET CLASS
# =========================================================

class SyntheticSeismicDataset(Dataset):
    """
    Lazy synthetic 3D seismic dataset.

    Each generated sample contains:

        input_cube
        target_cube
        mask
        velocity_model
        mask_type
        geological_mode

    Tensor shape:

        [C, D, H, W]

    where C is normally 1 for the present framework.

    The dataset does not retain complete seismic samples in
    memory. Samples are generated only when requested.
    """

    # =====================================================
    # VALID GEOLOGICAL MODES
    # =====================================================

    VALID_GEOLOGICAL_MODES = (
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    )

    # =====================================================
    # VALID MISSING-DATA MODES
    # =====================================================

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
            3D seismic volume dimensions:

                (depth, height, width)

        missing_probability : float
            Fraction of seismic observations removed
            by the sampling mask.

        geological_mode : str
            Geological scenario.

            Options:

                horizontal
                dipping
                faulted
                folded
                complex
                highly_complex
                random

        mask_mode : str
            Missing-data mechanism.

            Options:

                random_voxels
                missing_traces
                missing_inlines
                missing_crosslines
                missing_blocks
                random

        seed : int
            Base seed controlling reproducibility.
        """

        # -------------------------------------------------
        # Initialize Dataset parent class
        # -------------------------------------------------

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
        # EXPECTED TENSOR SHAPE
        # =================================================

        self.expected_shape = (
            1,
            *self.cube_size,
        )

        # =================================================
        # VALIDATE CONFIGURATION
        # =================================================

        self._validate_configuration()

        # =================================================
        # GEOLOGICAL GENERATOR
        # =================================================
        #
        # This generator creates the complete synthetic
        # seismic target.
        #
        # It is reused because its generation is controlled
        # by the deterministic seeds established for each
        # sample.
        # =================================================

        self.generator = GeologicalGenerator(
            cube_size=self.cube_size
        )

        # =================================================
        # METADATA CACHE
        # =================================================
        #
        # Only lightweight metadata are retained.
        #
        # Complete seismic cubes are NOT stored.
        # =================================================

        self.mask_types = [
            None
            for _ in range(self.num_samples)
        ]

        self.geological_modes = [
            None
            for _ in range(self.num_samples)
        ]

        # =================================================
        # DATASET INFORMATION
        # =================================================

        print()
        print("=" * 65)
        print("PHYSICS-INFORMED SYNTHETIC 3D SEISMIC DATASET")
        print("=" * 65)

        print(
            f"Number of Samples    : {self.num_samples}"
        )

        print(
            f"Cube Size            : {self.cube_size}"
        )

        print(
            f"Missing Probability  : "
            f"{self.missing_probability:.2f}"
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

        print(
            "Velocity Coupling    : "
            "Geological-mode conditioned"
        )

        print("=" * 65)
        print()

    # =====================================================
    # CONFIGURATION VALIDATION
    # =====================================================

    def _validate_configuration(self):
        """
        Validate all dataset configuration parameters.
        """

        # -------------------------------------------------
        # Number of samples
        # -------------------------------------------------

        if self.num_samples <= 0:

            raise ValueError(
                "num_samples must be greater than zero."
            )

        # -------------------------------------------------
        # Cube dimensions
        # -------------------------------------------------

        if len(self.cube_size) != 3:

            raise ValueError(
                "cube_size must contain exactly "
                "(depth, height, width)."
            )

        # -------------------------------------------------
        # Positive dimensions
        # -------------------------------------------------

        if any(
            dimension <= 0
            for dimension in self.cube_size
        ):

            raise ValueError(
                "All cube dimensions must be positive."
            )

        # -------------------------------------------------
        # Missing-data probability
        # -------------------------------------------------

        if not (
            0.0
            <= self.missing_probability
            < 1.0
        ):

            raise ValueError(
                "missing_probability must be between "
                "0.0 and 1.0."
            )

        # -------------------------------------------------
        # Geological mode
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Mask mode
        # -------------------------------------------------

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
        Generate a deterministic seed for a sample.

        The seed depends only on:

            dataset seed
            sample index

        Therefore:

            dataset(seed=42)[10]

        will always generate the same sample as long as
        the underlying generators remain unchanged.
        """

        return (
            self.seed
            + int(idx) * 100003
        )

    # =====================================================
    # SET SAMPLE RANDOM SEEDS
    # =====================================================

    def _set_sample_seed(self, idx):
        """
        Establish deterministic global random states.

        These seeds control components that rely on:

            Python random
            NumPy random
            PyTorch random
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

        torch.manual_seed(
            sample_seed
        )

        return sample_seed

    # =====================================================
    # SELECT GEOLOGICAL MODE
    # =====================================================

    def _select_geological_mode(self):
        """
        Select the geological scenario for one sample.

        If geological_mode == "random", the mode is selected
        deterministically from the sample seed.
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
        Select the missing-data mechanism.

        If mask_mode == "random", the mechanism is selected
        deterministically from the sample seed.
        """

        if self.mask_mode == "random":

            return str(
                np.random.choice(
                    self.VALID_MASK_TYPES
                )
            )

        return self.mask_mode

    # =====================================================
    # CONVERT DATA TO FLOAT32 TENSOR
    # =====================================================

    @staticmethod
    def _to_float_tensor(data):
        """
        Convert input data to a CPU float32 tensor.

        GPU transfer is intentionally NOT performed here.

        The DataLoader/training pipeline is responsible for
        transferring batches to the selected device.
        """

        if isinstance(
            data,
            torch.Tensor,
        ):

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
    # GENERAL TENSOR VALIDATION
    # =====================================================

    def _validate_tensor(
        self,
        tensor,
        name,
    ):
        """
        Validate tensor type, shape, and numerical validity.
        """

        # -------------------------------------------------
        # Tensor type
        # -------------------------------------------------

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise RuntimeError(
                f"{name} must be a PyTorch tensor."
            )

        # -------------------------------------------------
        # Tensor shape
        # -------------------------------------------------

        if tuple(tensor.shape) != self.expected_shape:

            raise RuntimeError(
                f"{name} has an unexpected shape.\n"
                f"Expected: {self.expected_shape}\n"
                f"Received: {tuple(tensor.shape)}"
            )

        # -------------------------------------------------
        # Numerical validity
        # -------------------------------------------------

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    def _validate_velocity(
        self,
        velocity,
    ):
        """
        Validate the generated velocity model.

        Requirements:

            correct tensor shape
            finite values
            strictly positive velocity
        """

        self._validate_tensor(
            velocity,
            "Generated velocity model",
        )

        # -------------------------------------------------
        # Physical requirement
        # -------------------------------------------------
        #
        # P-wave velocity must be strictly positive.
        # -------------------------------------------------

        if torch.any(
            velocity <= 0
        ):

            raise RuntimeError(
                "Velocity model must contain "
                "strictly positive velocities."
            )

    # =====================================================
    # MASK VALIDATION
    # =====================================================

    def _validate_mask(
        self,
        mask,
    ):
        """
        Validate the seismic sampling mask.
        """

        self._validate_tensor(
            mask,
            "Generated sampling mask",
        )

        # -------------------------------------------------
        # Extract unique values
        # -------------------------------------------------

        unique_values = torch.unique(
            mask
        )

        # -------------------------------------------------
        # Binary mask requirement
        # -------------------------------------------------

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

    def _generate_sample(
        self,
        idx,
    ):
        """
        Generate one complete synthetic training sample.

        Generation sequence
        -------------------
        1. establish deterministic sample seed
        2. create independent component seeds
        3. select geological scenario
        4. generate complete seismic target
        5. generate corresponding velocity model
        6. select missing-data mechanism
        7. generate sampling mask
        8. simulate incomplete acquisition
        9. validate all outputs
        10. return sample and metadata

        Returns
        -------
        tuple

            input_cube
            target_cube
            mask
            velocity_model
            mask_type
            geological_mode
        """

        # =================================================
        # STEP 1: SAMPLE SEED
        # =================================================

        sample_seed = self._set_sample_seed(
            idx
        )

        # =================================================
        # STEP 2: INDEPENDENT SEED STREAMS
        # =================================================
        #
        # Different components receive different seeds.
        #
        # This prevents changes in one generator from
        # unintentionally changing the random sequence of
        # another generator.
        # =================================================

        geological_seed = (
            sample_seed + 1
        )

        velocity_seed = (
            sample_seed + 2
        )

        mask_seed = (
            sample_seed + 3
        )

        # =================================================
        # STEP 3: GEOLOGICAL SCENARIO
        # =================================================

        geological_mode = (
            self._select_geological_mode()
        )

        # =================================================
        # STEP 4: COMPLETE SEISMIC TARGET
        # =================================================

        # -------------------------------------------------
        # Re-establish geological-specific deterministic
        # random state.
        # -------------------------------------------------

        random.seed(
            geological_seed
        )

        np.random.seed(
            geological_seed % (2**32 - 1)
        )

        torch.manual_seed(
            geological_seed
        )

        # -------------------------------------------------
        # Generate complete geological seismic cube.
        # -------------------------------------------------

        target = self.generator.generate(
            mode=geological_mode
        )

        # -------------------------------------------------
        # Convert to float32 tensor.
        # -------------------------------------------------

        target = self._to_float_tensor(
            target
        )

        # -------------------------------------------------
        # Validate target.
        # -------------------------------------------------

        self._validate_tensor(
            target,
            "Generated seismic target",
        )

        # =================================================
        # STEP 5: GEOLOGICALLY CONDITIONED VELOCITY MODEL
        # =================================================
        #
        # IMPORTANT:
        #
        # The velocity model is now generated using the SAME
        # resolved geological mode as the seismic target.
        #
        # Example:
        #
        #     target      -> faulted
        #     velocity    -> faulted
        #
        # rather than:
        #
        #     target      -> faulted
        #     velocity    -> random unrelated model
        #
        # This creates a controlled geological relationship
        # between the seismic target and the physical model
        # used by the Eikonal constraint.
        # =================================================

        # -------------------------------------------------
        # Create sample-specific velocity generator.
        # -------------------------------------------------

        velocity_generator = VelocityGenerator(
            cube_size=self.cube_size,
            seed=velocity_seed,
        )

        # -------------------------------------------------
        # Generate velocity using the SAME geological mode.
        # -------------------------------------------------

        velocity = (
            velocity_generator.generate(
                mode=geological_mode
            )
        )

        # -------------------------------------------------
        # Convert to float32 tensor.
        # -------------------------------------------------

        velocity = self._to_float_tensor(
            velocity
        )

        # -------------------------------------------------
        # Validate velocity.
        # -------------------------------------------------

        self._validate_velocity(
            velocity
        )

        # =================================================
        # STEP 6: SELECT MISSING-DATA MECHANISM
        # =================================================

        # -------------------------------------------------
        # Restore sample-level random state before mask
        # selection so the selected mask type is reproducible.
        # -------------------------------------------------

        random.seed(
            mask_seed
        )

        np.random.seed(
            mask_seed % (2**32 - 1)
        )

        torch.manual_seed(
            mask_seed
        )

        mask_type = (
            self._select_mask_type()
        )

        # =================================================
        # STEP 7: GENERATE SAMPLING MASK
        # =================================================

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

        # -------------------------------------------------
        # Convert mask to float32 tensor.
        # -------------------------------------------------

        mask = self._to_float_tensor(
            mask
        )

        # -------------------------------------------------
        # Validate mask.
        # -------------------------------------------------

        self._validate_mask(
            mask
        )

        # =================================================
        # STEP 8: SIMULATE INCOMPLETE ACQUISITION
        # =================================================
        #
        # Observed samples remain.
        #
        # Missing samples are replaced by zero.
        #
        # input = target × mask
        # =================================================

        input_cube = (
            target * mask
        )

        # -------------------------------------------------
        # Validate incomplete seismic input.
        # -------------------------------------------------

        self._validate_tensor(
            input_cube,
            "Generated input seismic volume",
        )

        # =================================================
        # STEP 9: STORE LIGHTWEIGHT METADATA
        # =================================================

        self.mask_types[idx] = (
            mask_type
        )

        self.geological_modes[idx] = (
            geological_mode
        )

        # =================================================
        # STEP 10: RETURN SAMPLE
        # =================================================

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
        Return number of samples in the dataset.
        """

        return self.num_samples

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx,
    ):
        """
        Generate one sample on demand.

        Returns
        -------

        input_cube
            Incomplete seismic volume.

        target_cube
            Complete seismic volume.

        mask
            Binary sampling mask.

        velocity_model
            Geologically conditioned velocity model.

        mask_type
            Missing-data mechanism.

        geological_mode
            Geological scenario.

        Tensor shape for all four tensors:

            [C, D, H, W]
        """

        # =================================================
        # HANDLE TENSOR INDEX
        # =================================================

        if isinstance(
            idx,
            torch.Tensor,
        ):

            idx = idx.item()

        # =================================================
        # CONVERT INDEX TO INTEGER
        # =================================================

        idx = int(idx)

        # =================================================
        # SUPPORT NEGATIVE INDEXING
        # =================================================

        if idx < 0:

            idx += self.num_samples

        # =================================================
        # VALIDATE INDEX
        # =================================================

        if idx < 0 or idx >= self.num_samples:

            raise IndexError(
                f"Dataset index {idx} is out of range "
                f"for dataset of size {self.num_samples}."
            )

        # =================================================
        # GENERATE SAMPLE LAZILY
        # =================================================

        return self._generate_sample(
            idx
        )


# =========================================================
# STANDALONE DATASET TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("SYNTHETIC DATASET STANDARD VALIDATION")
    print("=" * 65)

    # -----------------------------------------------------
    # Create a small test dataset.
    # -----------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=5,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    # -----------------------------------------------------
    # Generate first sample.
    # -----------------------------------------------------

    sample_a = dataset[0]

    input_a = sample_a[0]
    target_a = sample_a[1]
    mask_a = sample_a[2]
    velocity_a = sample_a[3]
    mask_type_a = sample_a[4]
    geological_mode_a = sample_a[5]

    # -----------------------------------------------------
    # Print sample information.
    # -----------------------------------------------------

    print()
    print("Sample 0")
    print("-" * 65)

    print(
        "Input shape       :",
        tuple(input_a.shape)
    )

    print(
        "Target shape      :",
        tuple(target_a.shape)
    )

    print(
        "Mask shape        :",
        tuple(mask_a.shape)
    )

    print(
        "Velocity shape    :",
        tuple(velocity_a.shape)
    )

    print(
        "Mask type         :",
        mask_type_a
    )

    print(
        "Geological mode   :",
        geological_mode_a
    )

    print(
        "Velocity minimum  :",
        float(velocity_a.min())
    )

    print(
        "Velocity maximum  :",
        float(velocity_a.max())
    )

    # =====================================================
    # REPRODUCIBILITY TEST
    # =====================================================

    print()
    print("=" * 65)
    print("REPRODUCIBILITY TEST")
    print("=" * 65)

    sample_b = dataset[0]

    print(
        "Input identical    :",
        torch.equal(sample_a[0], sample_b[0])
    )

    print(
        "Target identical   :",
        torch.equal(sample_a[1], sample_b[1])
    )

    print(
        "Mask identical     :",
        torch.equal(sample_a[2], sample_b[2])
    )

    print(
        "Velocity identical :",
        torch.equal(sample_a[3], sample_b[3])
    )

    print(
        "Mask type identical:",
        sample_a[4] == sample_b[4]
    )

    print(
        "Geology identical  :",
        sample_a[5] == sample_b[5]
    )

    # =====================================================
    # VALIDATION RESULT
    # =====================================================

    assert torch.equal(
        sample_a[0],
        sample_b[0],
    )

    assert torch.equal(
        sample_a[1],
        sample_b[1],
    )

    assert torch.equal(
        sample_a[2],
        sample_b[2],
    )

    assert torch.equal(
        sample_a[3],
        sample_b[3],
    )

    assert sample_a[4] == sample_b[4]

    assert sample_a[5] == sample_b[5]

    print()
    print("STATUS: DATASET VALIDATION PASSED")
    print("=" * 65)