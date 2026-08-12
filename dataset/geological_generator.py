import torch
import math
import numpy as np
class GeologicalGenerator:
    """
    Generates simple layered geological models.
    """

    def __init__(self,
                 cube_size=(64, 128, 128),
                 num_layers=8):

        self.depth, self.height, self.width = cube_size
        self.num_layers = num_layers

    def generate_horizontal_layers(self):

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = self.depth // self.num_layers

        amplitude = 0.2

        for layer in range(self.num_layers):

            start = layer * layer_thickness

            end = min(
                (layer + 1) * layer_thickness,
                self.depth
            )

            cube[start:end] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate_dipping_layers(self, dip=0.20):
        """
        Generate dipping geological layers.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = self.depth // self.num_layers

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                shift = int(dip * x)

                start = layer * layer_thickness + shift
                end = start + layer_thickness

                if start >= self.depth:
                    continue

                end = min(end, self.depth)

                cube[start:end, :, x] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate_faulted_layers(
            self,
            dip=0.20,
            fault_x=None,
            throw=8
    ):
        """
        Dipping layers with a fault.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        if fault_x is None:
            fault_x = self.width // 2

        layer_thickness = (
                self.depth //
                self.num_layers
        )

        amplitude = 0.2

        for layer in range(
                self.num_layers
        ):

            for x in range(
                    self.width
            ):

                shift = int(
                    dip * x
                )

                if x > fault_x:
                    shift += throw

                start = (
                        layer *
                        layer_thickness +
                        shift
                )

                end = (
                        start +
                        layer_thickness
                )

                if start >= self.depth:
                    continue

                end = min(
                    end,
                    self.depth
                )

                cube[
                    start:end,
                    :,
                    x
                ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate_folded_layers(
            self,
            amplitude_fold=8,
            frequency=0.05
    ):
        """
        Folded geological layers.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = (
                self.depth //
                self.num_layers
        )

        amplitude = 0.2

        for layer in range(
                self.num_layers
        ):

            for x in range(
                    self.width
            ):

                fold_shift = int(
                    amplitude_fold *
                    math.sin(
                        frequency * x
                    )
                )

                start = (
                        layer *
                        layer_thickness +
                        fold_shift
                )

                end = (
                        start +
                        layer_thickness
                )

                if start < 0:
                    start = 0

                if start >= self.depth:
                    continue

                end = min(
                    end,
                    self.depth
                )

                cube[
                    start:end,
                    :,
                    x
                ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate_complex_structure(
            self,
            amplitude_fold=8,
            frequency=0.05,
            throw=8
    ):
        """
        Faulted + folded geology.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        fault_x = (
                self.width // 2
        )

        layer_thickness = (
                self.depth //
                self.num_layers
        )

        amplitude = 0.2

        for layer in range(
                self.num_layers
        ):

            for x in range(
                    self.width
            ):

                fold_shift = int(
                    amplitude_fold *
                    math.sin(
                        frequency * x
                    )
                )

                shift = fold_shift

                if x > fault_x:
                    shift += throw

                start = (
                        layer *
                        layer_thickness +
                        shift
                )

                end = (
                        start +
                        layer_thickness
                )

                if start < 0:
                    start = 0

                if start >= self.depth:
                    continue

                end = min(
                    end,
                    self.depth
                )

                cube[
                    start:end,
                    :,
                    x
                ] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate(
            self,
            mode="horizontal"
    ):

        if mode == "horizontal":
            return self.generate_horizontal_layers()

        if mode == "dipping":
            return self.generate_dipping_layers()

        if mode == "faulted":
            return self.generate_faulted_layers()

        if mode == "folded":
            return self.generate_folded_layers()

        if mode == "complex":
            return self.generate_complex_structure()

        if mode == "highly_complex":
            return self.generate_highly_complex_structure()

        raise ValueError(
            f"Unknown mode: {mode}"
        )
    def generate_highly_complex_structure(self):
        """
        Highly complex geological setting:
        - dipping layers
        - strong folding
        - multiple faults
        - salt dome
        - constant amplitudes
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = (
                self.depth // self.num_layers
        )

        for layer in range(self.num_layers):

            amplitude = (
                0.2 if layer % 2 == 0
                else -0.2
            )

            for x in range(self.width):

                dip_shift = int(
                    0.25 * x
                )

                fold_shift = int(
                    10 * np.sin(
                        4 * np.pi * x / self.width
                    )
                )

                total_shift = (
                        dip_shift +
                        fold_shift
                )

                start = (
                        layer * layer_thickness
                        + total_shift
                )

                end = (
                        start + layer_thickness
                )

                if start >= self.depth:
                    continue

                start = max(
                    start,
                    0
                )

                end = min(
                    end,
                    self.depth
                )

                cube[
                    start:end,
                    :,
                    x
                ] = amplitude

        # Fault 1
        fault1 = int(
            self.width * 0.30
        )

        throw1 = 10

        cube[
            throw1:,
            :,
            fault1:
        ] = cube[
            :-throw1,
            :,
            fault1:
        ]

        # Salt dome deformation

        center_x = int(self.width * 0.50)
        center_z = int(self.depth * 0.50)

        radius = 15

        for x in range(self.width):

            distance = abs(x - center_x)

            if distance < radius:

                uplift = int(
                    12 * (
                            1 - distance / radius
                    )
                )

                if uplift > 0:
                    column = cube[:, :, x].clone()

                    cube[:, :, x] = 0

                    cube[
                        uplift:,
                        :,
                        x
                    ] = column[
                        :-uplift,
                        :
                    ]
        # Salt body

        for z in range(self.depth):

            for x in range(self.width):

                if (
                        (x - center_x) ** 2
                        +
                        (z - center_z) ** 2
                ) < radius ** 2:
                    cube[
                        z,
                        :,
                        x
                    ] = 0.35
        # Fault 2

        fault2 = int(
            self.width * 0.65
        )

        throw2 = 15

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
