"""
=========================================================
3D Seismic Sampling Mask Generator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generates binary sampling masks for simulating incomplete
3D seismic acquisition.

Mask convention
---------------

    1.0 = observed seismic sample

    0.0 = missing seismic sample

Tensor shape
------------

    [C, D, H, W]

Supported mask types
--------------------

    random_voxels
        Randomly removes individual seismic voxels.

    missing_traces
        Removes complete seismic traces.

    missing_inlines
        Removes complete inline sections.

    missing_crosslines
        Removes complete crossline sections.

    missing_blocks
        Removes contiguous 3D regions.

    random
        Randomly selects one of the above patterns.

Author:
Ormin Joseph
=========================================================
"""

import random

import torch


class SeismicMaskGenerator:
    """
    Generate binary masks for incomplete 3D seismic data.

    The generated mask has the same spatial dimensions as
    the seismic cube.

    Shape:

        [C, D, H, W]

    where:

        C = channel
        D = depth
        H = inline
        W = crossline
    """

    # =====================================================
    # SUPPORTED MASK TYPES
    # =====================================================

    SUPPORTED_MASK_TYPES = (
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
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        seed=None,
    ):
        """
        Parameters
        ----------
        cube_size : tuple
            Spatial dimensions:

                (depth, height, width)

        missing_probability : float
            Approximate proportion of data to remove.

        seed : int or None
            Optional seed for reproducible mask generation.
        """

        # =================================================
        # VALIDATE CUBE SIZE
        # =================================================

        if (
            not isinstance(cube_size, tuple)
            or len(cube_size) != 3
        ):
            raise ValueError(
                "cube_size must be a tuple "
                "(depth, height, width)."
            )

        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in cube_size
        ):
            raise ValueError(
                "All cube dimensions must be "
                "positive integers."
            )

        # =================================================
        # VALIDATE MISSING PROBABILITY
        # =================================================

        if not isinstance(
            missing_probability,
            (int, float),
        ) or isinstance(
            missing_probability,
            bool,
        ):
            raise TypeError(
                "missing_probability must be a real number."
            )

        if not (
            0.0
            <= float(missing_probability)
            < 1.0
        ):
            raise ValueError(
                "missing_probability must be "
                "between 0.0 and 1.0."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.cube_size = cube_size

        self.depth = cube_size[0]
        self.height = cube_size[1]
        self.width = cube_size[2]

        self.missing_probability = float(
            missing_probability
        )

        self.seed = seed

        # =================================================
        # LOCAL RANDOM-NUMBER GENERATOR
        # =================================================

        # A local RNG prevents this generator from changing
        # Python's global random-number state.
        #
        # This is particularly useful for the lazy dataset,
        # where each sample can be generated independently.
        self.rng = random.Random(seed)

        # -------------------------------------------------
        # PyTorch generator for tensor-based random
        # operations.
        # -------------------------------------------------

        self.torch_generator = torch.Generator()

        if seed is not None:
            self.torch_generator.manual_seed(seed)

    # =====================================================
    # CREATE COMPLETE MASK
    # =====================================================

    def _ones_mask(self):
        """
        Create a mask representing completely observed data.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        return torch.ones(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
        )

    # =====================================================
    # RANDOM VOXEL MASK
    # =====================================================

    def random_voxels(self):
        """
        Randomly remove individual seismic voxels.

        This produces an independent Bernoulli sampling mask.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        random_values = torch.rand(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            generator=self.torch_generator,
            dtype=torch.float32,
        )

        mask = (
            random_values
            >= self.missing_probability
        )

        return mask.to(dtype=torch.float32)

    # =====================================================
    # MISSING SEISMIC TRACES
    # =====================================================

    def missing_traces(self):
        """
        Remove complete seismic traces.

        A seismic trace extends along the depth dimension.

        Therefore, for a selected trace location:

            mask[:, :, h, w] = 0

        The approximate number of removed traces is
        determined by missing_probability.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        total_traces = (
            self.height
            *
            self.width
        )

        number_missing = int(
            round(
                self.missing_probability
                *
                total_traces
            )
        )

        number_missing = min(
            number_missing,
            total_traces,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            total_traces,
            generator=self.torch_generator,
        )[:number_missing]

        inline_indices = (
            selected
            //
            self.width
        )

        crossline_indices = (
            selected
            %
            self.width
        )

        mask[
            :,
            :,
            inline_indices,
            crossline_indices,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING INLINE SECTIONS
    # =====================================================

    def missing_inlines(self):
        """
        Remove complete inline sections.

        For a selected inline index h:

            mask[:, :, h, :] = 0

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        number_missing = int(
            round(
                self.missing_probability
                *
                self.height
            )
        )

        number_missing = min(
            number_missing,
            self.height,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            self.height,
            generator=self.torch_generator,
        )[:number_missing]

        mask[
            :,
            :,
            selected,
            :,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING CROSSLINE SECTIONS
    # =====================================================

    def missing_crosslines(self):
        """
        Remove complete crossline sections.

        For a selected crossline index w:

            mask[:, :, :, w] = 0

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        number_missing = int(
            round(
                self.missing_probability
                *
                self.width
            )
        )

        number_missing = min(
            number_missing,
            self.width,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            self.width,
            generator=self.torch_generator,
        )[:number_missing]

        mask[
            :,
            :,
            :,
            selected,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING CONTIGUOUS BLOCK
    # =====================================================

    def missing_blocks(self):
        """
        Remove a contiguous approximately cubic 3D region.

        The block dimensions are chosen to approximately
        correspond to the requested missing probability.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        total_voxels = (
            self.depth
            *
            self.height
            *
            self.width
        )

        target_missing = int(
            round(
                self.missing_probability
                *
                total_voxels
            )
        )

        if target_missing <= 0:
            return mask

        # =================================================
        # DETERMINE APPROXIMATELY CUBIC BLOCK DIMENSIONS
        # =================================================

        scale = (
            self.missing_probability
            ** (1.0 / 3.0)
        )

        block_depth = max(
            1,
            min(
                self.depth,
                int(
                    round(
                        self.depth
                        *
                        scale
                    )
                ),
            ),
        )

        block_height = max(
            1,
            min(
                self.height,
                int(
                    round(
                        self.height
                        *
                        scale
                    )
                ),
            ),
        )

        block_width = max(
            1,
            min(
                self.width,
                int(
                    round(
                        self.width
                        *
                        scale
                    )
                ),
            ),
        )

        # =================================================
        # RANDOM STARTING POSITION
        # =================================================

        if self.depth == block_depth:
            start_depth = 0
        else:
            start_depth = self.rng.randint(
                0,
                self.depth - block_depth,
            )

        if self.height == block_height:
            start_height = 0
        else:
            start_height = self.rng.randint(
                0,
                self.height - block_height,
            )

        if self.width == block_width:
            start_width = 0
        else:
            start_width = self.rng.randint(
                0,
                self.width - block_width,
            )

        # =================================================
        # REMOVE CONTIGUOUS BLOCK
        # =================================================

        mask[
            :,
            start_depth:
            start_depth + block_depth,
            start_height:
            start_height + block_height,
            start_width:
            start_width + block_width,
        ] = 0.0

        return mask

    # =====================================================
    # RANDOM MASK TYPE
    # =====================================================

    def generate(
        self,
        mask_type="random",
    ):
        """
        Generate a binary seismic sampling mask.

        Parameters
        ----------
        mask_type : str

            random_voxels
            missing_traces
            missing_inlines
            missing_crosslines
            missing_blocks
            random

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        # =================================================
        # VALIDATE MASK TYPE
        # =================================================

        if not isinstance(
            mask_type,
            str,
        ):
            raise TypeError(
                "mask_type must be a string."
            )

        # =================================================
        # RANDOMLY SELECT MASK TYPE
        # =================================================

        if mask_type == "random":

            mask_type = self.rng.choice(
                self.SUPPORTED_MASK_TYPES
            )

        # =================================================
        # GENERATE SELECTED MASK
        # =================================================

        if mask_type == "random_voxels":

            return self.random_voxels()

        if mask_type == "missing_traces":

            return self.missing_traces()

        if mask_type == "missing_inlines":

            return self.missing_inlines()

        if mask_type == "missing_crosslines":

            return self.missing_crosslines()

        if mask_type == "missing_blocks":

            return self.missing_blocks()

        raise ValueError(
            "Unsupported mask_type: "
            f"{mask_type}. "
            "Supported types are: "
            f"{list(self.SUPPORTED_MASK_TYPES) + ['random']}"
        )

    # =====================================================
    # MASK VALIDATION
    # =====================================================

    @staticmethod
    def validate_mask(
        mask,
    ):
        """
        Validate a generated seismic sampling mask.

        Parameters
        ----------
        mask : torch.Tensor
            Expected shape:

                [1, D, H, W]

        Returns
        -------
        bool
            True when the mask is valid.
        """

        # -------------------------------------------------
        # Type validation
        # -------------------------------------------------

        if not isinstance(
            mask,
            torch.Tensor,
        ):
            raise TypeError(
                "mask must be a torch.Tensor."
            )

        # -------------------------------------------------
        # Dimension validation
        # -------------------------------------------------

        if mask.ndim != 4:
            raise ValueError(
                "mask must have shape "
                "[C,D,H,W]. "
                f"Received: {tuple(mask.shape)}."
            )

        # -------------------------------------------------
        # Single-channel validation
        # -------------------------------------------------

        if mask.shape[0] != 1:
            raise ValueError(
                "mask must contain exactly one "
                f"channel. Received {mask.shape[0]}."
            )

        # -------------------------------------------------
        # Numerical validation
        # -------------------------------------------------

        if not torch.isfinite(mask).all():
            raise ValueError(
                "Mask contains NaN or Inf."
            )

        # -------------------------------------------------
        # Binary-value validation
        #
        # Valid values:
        #
        #       0.0 = missing
        #       1.0 = observed
        # -------------------------------------------------

        if not torch.all(
            (mask == 0.0)
            |
            (mask == 1.0)
        ):
            raise ValueError(
                "Mask must contain only binary values "
                "0.0 and 1.0."
            )

        return True