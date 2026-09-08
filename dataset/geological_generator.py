"""
=========================================================
Geological Generator
=========================================================

Synthetic 3D geological model generator for seismic
reconstruction experiments.

Supported geological structures:
    - Horizontal layers
    - Dipping layers
    - Faulted layers
    - Folded layers
    - Complex faulted + folded structures
    - Highly complex structures with faults and salt body

Output shape:
    [1, depth, height, width]

Author: Ormin Joseph
=========================================================
"""

import math

import numpy as np
import torch


class GeologicalGenerator:
    """
    Generates synthetic 3D geological models.

    Parameters
    ----------
    cube_size : tuple
        Cube dimensions as:
        (depth, height, width)

    num_layers : int
        Number of geological layers.
    """

    def __init__(
        self,
        cube_size=(64, 128, 128),
        num_layers=8
    ):

        self.depth, self.height, self.width = cube_size
        self.num_layers = num_layers

        if self.depth <= 0:
            raise ValueError("depth must be positive.")

        if self.height <= 0:
            raise ValueError("height must be positive.")

        if self.width <= 0:
            raise ValueError("width must be positive.")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

    # -----------------------------------------------------
    # Horizontal layers
    # -----------------------------------------------------

    def generate_horizontal_layers(self):

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        amplitude = 0.2

        for layer in range(self.num_layers):

            start = layer * layer_thickness

            end = min(
                (layer + 1) * layer_thickness,
                self.depth
            )

            if start >= self.depth:
                break

            cube[start:end] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Dipping layers
    # -----------------------------------------------------

    def generate_dipping_layers(
        self,
        dip=0.20
    ):
        """
        Generate dipping geological layers.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                shift = int(dip * x)

                start = (
                    layer * layer_thickness
                    + shift
                )

                end = (
                    start
                    + layer_thickness
                )

                if start >= self.depth:
                    continue

                start = max(start, 0)
                end = min(end, self.depth)

                if start < end:

                    cube[
                        start:end,
                        :,
                        x
                    ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Faulted layers
    # -----------------------------------------------------

    def generate_faulted_layers(
        self,
        dip=0.20,
        fault_x=None,
        throw=8
    ):
        """
        Generate dipping layers with a single fault.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        if fault_x is None:
            fault_x = self.width // 2

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                shift = int(dip * x)

                if x > fault_x:
                    shift += throw

                start = (
                    layer * layer_thickness
                    + shift
                )

                end = (
                    start
                    + layer_thickness
                )

                if start >= self.depth:
                    continue

                start = max(start, 0)
                end = min(end, self.depth)

                if start < end:

                    cube[
                        start:end,
                        :,
                        x
                    ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Folded layers
    # -----------------------------------------------------

    def generate_folded_layers(
        self,
        amplitude_fold=8,
        frequency=0.05
    ):
        """
        Generate folded geological layers.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                fold_shift = int(
                    amplitude_fold
                    * math.sin(frequency * x)
                )

                start = (
                    layer * layer_thickness
                    + fold_shift
                )

                end = (
                    start
                    + layer_thickness
                )

                start = max(start, 0)

                if start >= self.depth:
                    continue

                end = min(end, self.depth)

                if start < end:

                    cube[
                        start:end,
                        :,
                        x
                    ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Complex structure
    # -----------------------------------------------------

    def generate_complex_structure(
        self,
        amplitude_fold=8,
        frequency=0.05,
        throw=8
    ):
        """
        Generate faulted + folded geological structure.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        fault_x = self.width // 2

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                fold_shift = int(
                    amplitude_fold
                    * math.sin(frequency * x)
                )

                shift = fold_shift

                if x > fault_x:
                    shift += throw

                start = (
                    layer * layer_thickness
                    + shift
                )

                end = (
                    start
                    + layer_thickness
                )

                start = max(start, 0)

                if start >= self.depth:
                    continue

                end = min(end, self.depth)

                if start < end:

                    cube[
                        start:end,
                        :,
                        x
                    ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Highly complex structure
    # -----------------------------------------------------

    def generate_highly_complex_structure(self):
        """
        Generate a highly complex geological structure:

        - Dipping layers
        - Strong folding
        - Two faults
        - Salt-dome deformation
        - Salt body
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

        layer_thickness = max(
            1,
            self.depth // self.num_layers
        )

        # -------------------------------------------------
        # Dipping + folded layers
        # -------------------------------------------------

        for layer in range(self.num_layers):

            amplitude = (
                0.2
                if layer % 2 == 0
                else -0.2
            )

            for x in range(self.width):

                dip_shift = int(
                    0.25 * x
                )

                fold_shift = int(
                    10
                    * np.sin(
                        4
                        * np.pi
                        * x
                        / self.width
                    )
                )

                total_shift = (
                    dip_shift
                    + fold_shift
                )

                start = (
                    layer * layer_thickness
                    + total_shift
                )

                end = (
                    start
                    + layer_thickness
                )

                start = max(start, 0)

                if start >= self.depth:
                    continue

                end = min(
                    end,
                    self.depth
                )

                if start < end:

                    cube[
                        start:end,
                        :,
                        x
                    ] = amplitude

        # -------------------------------------------------
        # Fault 1
        # -------------------------------------------------

        fault1 = int(
            self.width * 0.30
        )

        throw1 = 10

        if throw1 < self.depth:

            cube[
                throw1:,
                :,
                fault1:
            ] = cube[
                :-throw1,
                :,
                fault1:
            ]

        # -------------------------------------------------
        # Salt dome deformation
        # -------------------------------------------------

        center_x = int(
            self.width * 0.50
        )

        radius = 15

        for x in range(self.width):

            distance = abs(
                x - center_x
            )

            if distance < radius:

                uplift = int(
                    12
                    * (
                        1
                        - distance / radius
                    )
                )

                if uplift > 0:

                    column = cube[
                        :,
                        :,
                        x
                    ].clone()

                    cube[
                        :,
                        :,
                        x
                    ] = 0

                    cube[
                        uplift:,
                        :,
                        x
                    ] = column[
                        :-uplift,
                        :
                    ]

        # -------------------------------------------------
        # Salt body
        # -------------------------------------------------

        center_z = int(
            self.depth * 0.50
        )

        for z in range(self.depth):

            for x in range(self.width):

                distance_squared = (
                    (x - center_x) ** 2
                    +
                    (z - center_z) ** 2
                )

                if distance_squared < radius ** 2:

                    cube[
                        z,
                        :,
                        x
                    ] = 0.35

        # -------------------------------------------------
        # Fault 2
        # -------------------------------------------------

        fault2 = int(
            self.width * 0.65
        )

        throw2 = 15

        if throw2 < self.depth:

            cube[
                throw2:,
                :,
                fault2:
            ] = cube[
                :-throw2,
                :,
                fault2:
            ]

        return cube.unsqueeze(0)

    # -----------------------------------------------------
    # Main interface
    # -----------------------------------------------------

    def generate(
        self,
        mode="horizontal"
    ):

        if mode == "horizontal":
            return self.generate_horizontal_layers()

        elif mode == "dipping":
            return self.generate_dipping_layers()

        elif mode == "faulted":
            return self.generate_faulted_layers()

        elif mode == "folded":
            return self.generate_folded_layers()

        elif mode == "complex":
            return self.generate_complex_structure()

        elif mode == "highly_complex":
            return self.generate_highly_complex_structure()

        else:

            raise ValueError(
                f"Unknown geological mode: {mode}"
            )