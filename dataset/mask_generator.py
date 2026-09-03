"""
=========================================================
3D Seismic Sampling Mask Generator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

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

Author: Ormin Joseph
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
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        cube_size=(64, 128, 128),
        missing_probability=0.30
    ):
        """
        Parameters
        ----------
        cube_size : tuple
            Spatial dimensions:

                (depth, height, width)

        missing_probability : float
            Approximate proportion of data to remove.
        """

        self.cube_size = tuple(
            cube_size
        )

        self.depth = (
            self.cube_size[0]
        )

        self.height = (
            self.cube_size[1]
        )

        self.width = (
            self.cube_size[2]
        )

        self.missing_probability = float(
            missing_probability
        )

        # =================================================
        # VALIDATION
        # =================================================

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

    # =====================================================
    # CREATE COMPLETE MASK
    # =====================================================

    def _ones_mask(self):
        """
        Create a mask representing completely observed data.
        """

        return torch.ones(
            1,
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

    # =====================================================
    # RANDOM VOXEL MASK
    # =====================================================

    def random_voxels(self):
        """
        Randomly remove individual seismic voxels.

        This produces an independent Bernoulli sampling mask.
        """

        mask = (
            torch.rand(
                1,
                self.depth,
                self.height,
                self.width
            )
            >
            self.missing_probability
        )

        return mask.float()

    # =====================================================
    # MISSING SEISMIC TRACES
    # =====================================================

    def missing_traces(self):
        """
        Remove complete seismic traces.

        A seismic trace extends along the depth dimension.
        Therefore, for a selected trace location:

            mask[:, :, h, w] = 0

        The approximate number of removed traces is determined
        by missing_probability.
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
            total_traces
        )

        if number_missing == 0:

            return mask

        selected = torch.randperm(
            total_traces
        )[
            :number_missing
        ]

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
            crossline_indices
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
            self.height
        )

        if number_missing == 0:

            return mask

        selected = torch.randperm(
            self.height
        )[
            :number_missing
        ]

        mask[
            :,
            :,
            selected,
            :
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
            self.width
        )

        if number_missing == 0:

            return mask

        selected = torch.randperm(
            self.width
        )[
            :number_missing
        ]

        mask[
            :,
            :,
            :,
            selected
        ] = 0.0

        return mask

    # =====================================================
    # MISSING CONTIGUOUS BLOCK
    # =====================================================

    def missing_blocks(self):
        """
        Remove a contiguous approximately cubic 3D region.

        The block volume is chosen to approximately match the
        requested missing probability.
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

        # -------------------------------------------------
        # Determine approximately cubic block dimensions.
        # -------------------------------------------------

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
                )
            )
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
                )
            )
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
                )
            )
        )

        # -------------------------------------------------
        # Random starting position.
        # -------------------------------------------------

        start_depth = random.randint(
            0,
            self.depth - block_depth
        )

        start_height = random.randint(
            0,
            self.height - block_height
        )

        start_width = random.randint(
            0,
            self.width - block_width
        )

        # -------------------------------------------------
        # Remove contiguous block.
        # -------------------------------------------------

        mask[
            :,
            start_depth:
            start_depth + block_depth,
            start_height:
            start_height + block_height,
            start_width:
            start_width + block_width
        ] = 0.0

        return mask

    # =====================================================
    # RANDOM MASK TYPE
    # =====================================================

    def generate(
        self,
        mask_type="random"
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

        mask : torch.Tensor

            Shape:

                [1, D, H, W]
        """

        available_types = [
            "random_voxels",
            "missing_traces",
            "missing_inlines",
            "missing_crosslines",
            "missing_blocks"
        ]

        # -------------------------------------------------
        # RANDOMLY SELECT MASK TYPE
        # -------------------------------------------------

        if mask_type == "random":

            mask_type = random.choice(
                available_types
            )

        # -------------------------------------------------
        # GENERATE SELECTED MASK
        # -------------------------------------------------

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
            f"{available_types + ['random']}"
        )
