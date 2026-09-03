"""
=========================================================
Unified Seismic Dataset
=========================================================

Handles seismic cubes from multiple sources:

• Synthetic geological generator
• SEG-Y seismic volumes
• Future F3 Netherlands
• Future Marmousi

Pipeline

Cube
 ↓
Cube Shape Normalization
 ↓
Patch Extraction
 ↓
Mask Generation
 ↓
PyTorch Dataset

Each sample contains:

• Input seismic patch
• Target seismic patch
• Sampling mask

Author: Ormin Joseph
=========================================================
"""

import numpy as np
import torch

from torch.utils.data import Dataset

from dataset.segy_loader import SegyLoader
from dataset.patch_extractor import PatchExtractor
from dataset.mask_generator import MaskGenerator
from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator

from utils.config import (
    DATASET_MODE,
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY,
)


class SeismicDataset(Dataset):

    """
    Unified PyTorch Dataset for 3D seismic reconstruction.

    Supported data sources:

    1. Synthetic geological data
    2. SEG-Y data
    3. Directly supplied NumPy seismic cube

    Expected internal cube shape:

        (depth, inline, crossline)

    Each returned sample has shape:

        (1, depth, inline, crossline)

    where the leading 1 is the PyTorch channel dimension.
    """

    def __init__(
        self,
        cube=None,
        segy_file=None,
        synthetic=False,
        dataset_mode=None,
    ):

        """
        Parameters
        ----------
        cube : numpy.ndarray, optional
            A seismic cube supplied directly.

        segy_file : str, optional
            Path to a SEG-Y seismic file.

        synthetic : bool
            If True, generate a synthetic geological cube.

        dataset_mode : str, optional
            Dataset mode. If omitted, DATASET_MODE
            from utils.config is used.
        """

        # =================================================
        # DATASET MODE
        # =================================================

        if dataset_mode is None:
            dataset_mode = DATASET_MODE

        dataset_mode = dataset_mode.lower()

        self.dataset_mode = dataset_mode

        # =================================================
        # METADATA
        # =================================================

        self.metadata = {}

        # =================================================
        # SELECT DATA SOURCE
        # =================================================

        if synthetic or dataset_mode == "synthetic":

            # ---------------------------------------------
            # Generate synthetic seismic cube
            # ---------------------------------------------

            generator = GeologicalGenerator()

            cube = generator.generate()

            patch_size = SYNTHETIC_PATCH_SIZE

            stride = SYNTHETIC_PATCH_SIZE

            missing_probability = (
                SYNTHETIC_MISSING_PROBABILITY
            )

        elif segy_file is not None:

            # ---------------------------------------------
            # Load SEG-Y seismic volume
            # ---------------------------------------------

            loader = SegyLoader(
                segy_file
            )

            cube = loader.load()

            self.metadata = loader.get_metadata()

            patch_size = F3_PATCH_SIZE

            stride = F3_STRIDE

            missing_probability = (
                F3_MISSING_PROBABILITY
            )

        elif dataset_mode == "f3":

            raise ValueError(
                "F3 dataset mode requires a SEG-Y file. "
                "Provide segy_file='path/to/file.sgy'."
            )

        elif cube is not None:

            # ---------------------------------------------
            # Directly supplied seismic cube
            # ---------------------------------------------

            patch_size = SYNTHETIC_PATCH_SIZE

            stride = SYNTHETIC_PATCH_SIZE

            missing_probability = (
                SYNTHETIC_MISSING_PROBABILITY
            )

        else:

            raise ValueError(
                "No seismic data source was provided. "
                "Use synthetic=True, provide cube=..., "
                "or provide segy_file=...."
            )

        # =================================================
        # VALIDATE DATA
        # =================================================

        if cube is None:

            raise ValueError(
                "Seismic cube could not be created or loaded."
            )

        # Convert to NumPy float32
        cube = np.asarray(
            cube,
            dtype=np.float32
        )

        # =================================================
        # NORMALIZE CUBE DIMENSIONS
        # =================================================

        # GeologicalGenerator may return:
        #
        # (1, D, H, W)
        #
        # where 1 is a channel dimension.
        #
        # PatchExtractor requires:
        #
        # (D, H, W)

        if cube.ndim == 4:

            if cube.shape[0] == 1:

                cube = cube[0]

            else:

                raise ValueError(
                    "4D seismic cube must have a singleton "
                    f"leading channel dimension. "
                    f"Received shape: {cube.shape}"
                )

        # After normalization, cube MUST be 3D.

        if cube.ndim != 3:

            raise ValueError(
                "Seismic cube must be 3-dimensional after "
                "normalization. "
                f"Received shape: {cube.shape}"
            )

        # =================================================
        # STORE FULL SEISMIC CUBE
        # =================================================

        self.cube = cube

        # =================================================
        # VELOCITY MODEL
        # =================================================

        # Synthetic training uses a synthetic geological
        # velocity model with the same dimensions as the
        # seismic cube.

        if self.dataset_mode == "synthetic":

            velocity_generator = VelocityGenerator(
                cube_size=cube.shape
            )

            self.velocity_cube = (
                velocity_generator.generate()
            )

            # Remove the channel dimension so that the
            # velocity cube has shape:
            #
            # (D, H, W)
            #
            # This matches the seismic cube and allows
            # corresponding patches to be extracted using
            # identical spatial coordinates.

            if self.velocity_cube.ndim == 4:

                if self.velocity_cube.shape[0] == 1:

                    self.velocity_cube = (
                        self.velocity_cube[0]
                    )

                else:

                    raise ValueError(
                        "Synthetic velocity cube must have "
                        "a singleton channel dimension."
                    )

            # Convert to NumPy for consistency with the
            # seismic patch extraction workflow.

            self.velocity_cube = (
                self.velocity_cube
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # Shape validation
            # -------------------------------------------------

            if self.velocity_cube.shape != cube.shape:
                raise RuntimeError(
                    "Synthetic seismic cube and velocity cube "
                    "must have identical shapes.\n"
                    f"Seismic cube: {cube.shape}\n"
                    f"Velocity cube: {self.velocity_cube.shape}"
                )

        else:

            # F3 velocity integration will be handled
            # separately after the authentic F3 velocity
            # model is validated and loaded.

            self.velocity_cube = None

        # =================================================
        # PATCH EXTRACTION
        # =================================================

        extractor = PatchExtractor(

            patch_size=patch_size,

            stride=stride

        )

        self.patches = extractor.extract(
            cube
        )

        # =================================================
        # CHECK PATCH EXTRACTION
        # =================================================

        if len(self.patches) == 0:

            raise ValueError(
                "No seismic patches were extracted. "
                f"Cube shape: {cube.shape}, "
                f"Patch size: {patch_size}, "
                f"Stride: {stride}"
            )

        # =================================================
        # MASK GENERATOR
        # =================================================

        self.mask_generator = MaskGenerator(

            missing_probability=missing_probability,

            mask_type="random_trace"

        )

    # =====================================================
    # DATASET LENGTH
    # =====================================================

    def __len__(self):

        """
        Return the number of seismic patches.
        """

        return len(
            self.patches
        )

    # =====================================================
    # GET ONE SAMPLE
    # =====================================================

    def __getitem__(
            self,
            index
    ):

        """
        Load one seismic patch.

        Returns
        -------
        dict

            {
                "input": input seismic patch,
                "target": ground-truth patch,
                "mask": sampling mask,
                "position": (z, y, x)
            }

        Tensor shape:

            (1, D, H, W)
        """

        # --------------------------------------------------
        # Extract patch and its spatial position
        # --------------------------------------------------

        target_patch, z, y, x = self.patches[index]

        # --------------------------------------------------
        # Extract corresponding velocity patch
        # --------------------------------------------------

        if self.velocity_cube is not None:

            depth = target_patch.shape[0]

            height = target_patch.shape[1]

            width = target_patch.shape[2]

            velocity_patch = self.velocity_cube[

                z:z + depth,

                y:y + height,

                x:x + width

            ]

        else:

            velocity_patch = None

        # --------------------------------------------------
        # Generate sampling mask
        # --------------------------------------------------

        mask = self.mask_generator.generate_mask(
            target_patch.shape
        )

        # --------------------------------------------------
        # Create corrupted seismic input
        # --------------------------------------------------

        input_patch = target_patch * mask

        # --------------------------------------------------
        # Convert NumPy arrays to PyTorch tensors
        # --------------------------------------------------

        input_patch = torch.from_numpy(
            input_patch
        ).float()

        target_patch = torch.from_numpy(
            target_patch
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        # --------------------------------------------------
        # Convert velocity patch to PyTorch tensor
        # --------------------------------------------------

        if velocity_patch is not None:
            velocity_patch = torch.from_numpy(
                velocity_patch
            ).float()

        # --------------------------------------------------
        # Add channel dimension
        # --------------------------------------------------

        input_patch = input_patch.unsqueeze(0)

        target_patch = target_patch.unsqueeze(0)

        mask = mask.unsqueeze(0)

        if velocity_patch is not None:
            velocity_patch = (
                velocity_patch.unsqueeze(0)
            )

        # --------------------------------------------------
        # Return sample
        # --------------------------------------------------

        return {

            "input":
                input_patch,

            "target":
                target_patch,

            "mask":
                mask,

            "velocity_model":
                velocity_patch,

            "position": (
                z,
                y,
                x
            ),
        }

    # =====================================================
    # METADATA
    # =====================================================

    def get_metadata(self):

        """
        Return seismic dataset metadata.
        """

        return self.metadata