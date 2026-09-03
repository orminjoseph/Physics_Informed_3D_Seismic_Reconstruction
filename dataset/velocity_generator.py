"""
=========================================================
Velocity Model Generator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generates physically valid synthetic 3D seismic velocity
models for development, testing, and controlled validation
of the physics-informed reconstruction framework.

Velocity models currently supported
-----------------------------------

1. Horizontally layered velocity model
2. Linear velocity-gradient model

The velocity model is used by the Eikonal physics loss:

        V^2 |∇T|^2 - 1 = 0

where

        V = seismic velocity [m/s]
        T = seismic travel time [s]

Tensor convention
-----------------

Output:

        [C, D, H, W]

where

        C = velocity channel
        D = depth
        H = crossline
        W = inline

The training pipeline adds the batch dimension:

        [B, C, D, H, W]

Author:
Ormin Joseph
=========================================================
"""

import random

import torch


class VelocityGenerator:
    """
    Generate synthetic 3D seismic velocity models.

    Parameters
    ----------
    cube_size : tuple
        Spatial dimensions of the velocity cube:

            (D, H, W)

    min_velocity : float
        Minimum seismic velocity in m/s.

    max_velocity : float
        Maximum seismic velocity in m/s.

    seed : int or None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        cube_size=(64, 128, 128),
        min_velocity=1800.0,
        max_velocity=3500.0,
        seed=None
    ):

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
            not isinstance(value, int)
            or value <= 0
            for value in cube_size
        ):
            raise ValueError(
                "All cube dimensions must be "
                "positive integers."
            )

        # =================================================
        # VALIDATE VELOCITY RANGE
        # =================================================

        if min_velocity <= 0:
            raise ValueError(
                "min_velocity must be greater than zero."
            )

        if max_velocity <= min_velocity:
            raise ValueError(
                "max_velocity must be greater than "
                "min_velocity."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.cube_size = cube_size

        self.depth = cube_size[0]
        self.height = cube_size[1]
        self.width = cube_size[2]

        self.min_velocity = float(
            min_velocity
        )

        self.max_velocity = float(
            max_velocity
        )

        # =================================================
        # REPRODUCIBILITY
        # =================================================

        self.rng = random.Random(seed)

    # =====================================================
    # VALIDATE NUMBER OF LAYERS
    # =====================================================

    def _validate_number_of_layers(
        self,
        number_of_layers
    ):

        if not isinstance(
            number_of_layers,
            int
        ):
            raise TypeError(
                "number_of_layers must be an integer."
            )

        if number_of_layers < 1:
            raise ValueError(
                "number_of_layers must be at least 1."
            )

        if number_of_layers > self.depth:
            raise ValueError(
                "number_of_layers cannot exceed "
                "the depth of the velocity cube."
            )

    # =====================================================
    # LAYER BOUNDARIES
    # =====================================================

    def _generate_layer_boundaries(
        self,
        number_of_layers
    ):
        """
        Generate unique layer boundaries.

        Returns
        -------
        list

            Sorted boundaries containing:

                0
                internal boundaries
                depth
        """

        self._validate_number_of_layers(
            number_of_layers
        )

        # -------------------------------------------------
        # One layer requires no internal boundary.
        # -------------------------------------------------

        if number_of_layers == 1:

            return [
                0,
                self.depth
            ]

        # -------------------------------------------------
        # Choose unique internal boundaries.
        #
        # We use sampling without replacement so that
        # zero-thickness layers cannot occur.
        # -------------------------------------------------

        internal_boundaries = self.rng.sample(
            range(
                1,
                self.depth
            ),
            number_of_layers - 1
        )

        internal_boundaries.sort()

        return (
            [0]
            +
            internal_boundaries
            +
            [self.depth]
        )

    # =====================================================
    # CREATE EMPTY VELOCITY CUBE
    # =====================================================

    def _empty_velocity_cube(self):
        """
        Create an empty single-channel velocity cube.
        """

        return torch.empty(
            1,
            self.depth,
            self.height,
            self.width,
            dtype=torch.float32
        )

    # =====================================================
    # HORIZONTALLY LAYERED MODEL
    # =====================================================

    def generate_layered_model(
        self,
        number_of_layers=4
    ):
        """
        Generate a horizontally layered velocity model.

        Velocity increases with depth.

        Parameters
        ----------
        number_of_layers : int
            Number of geological velocity layers.

        Returns
        -------
        torch.Tensor

            Shape:

                [1, D, H, W]

            Units:

                m/s
        """

        boundaries = (
            self._generate_layer_boundaries(
                number_of_layers
            )
        )

        velocity = self._empty_velocity_cube()

        # -------------------------------------------------
        # Velocity increment between layers.
        # -------------------------------------------------

        if number_of_layers == 1:

            velocity_increment = 0.0

        else:

            velocity_increment = (
                self.max_velocity
                -
                self.min_velocity
            ) / (
                number_of_layers - 1
            )

        # -------------------------------------------------
        # Assign velocity to each layer.
        # -------------------------------------------------

        for layer_index in range(
            number_of_layers
        ):

            top = boundaries[layer_index]

            bottom = boundaries[
                layer_index + 1
            ]

            layer_velocity = (
                self.min_velocity
                +
                layer_index
                *
                velocity_increment
            )

            velocity[
                :,
                top:bottom,
                :,
                :
            ] = layer_velocity

        return velocity

    # =====================================================
    # LINEAR VELOCITY-GRADIENT MODEL
    # =====================================================

    def generate_gradient_model(self):
        """
        Generate a velocity model with a linear
        increase in velocity with depth.

        Returns
        -------
        torch.Tensor

            Shape:

                [1, D, H, W]

            Units:

                m/s
        """

        velocity = self._empty_velocity_cube()

        # -------------------------------------------------
        # Avoid division by zero for a one-sample depth.
        # -------------------------------------------------

        if self.depth == 1:

            velocity[
                :,
                0,
                :,
                :
            ] = self.min_velocity

            return velocity

        # -------------------------------------------------
        # Create normalized depth coordinate.
        # -------------------------------------------------

        depth_coordinate = torch.linspace(
            0.0,
            1.0,
            self.depth,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # Linear velocity variation.
        # -------------------------------------------------

        velocity_profile = (
            self.min_velocity
            +
            (
                self.max_velocity
                -
                self.min_velocity
            )
            *
            depth_coordinate
        )

        # -------------------------------------------------
        # Expand the 1D velocity profile into a 3D cube.
        # -------------------------------------------------

        velocity[:] = (
            velocity_profile
            .view(
                1,
                self.depth,
                1,
                1
            )
            .expand(
                1,
                self.depth,
                self.height,
                self.width
            )
        )

        return velocity

    # =====================================================
    # RANDOM MODEL SELECTION
    # =====================================================

    def generate(
        self,
        number_of_layers=4
    ):
        """
        Randomly generate either a layered or gradient
        velocity model.

        Returns
        -------
        torch.Tensor

            Shape:

                [1, D, H, W]

            Units:

                m/s
        """

        if self.rng.random() < 0.5:

            return self.generate_layered_model(
                number_of_layers=number_of_layers
            )

        return self.generate_gradient_model()

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    @staticmethod
    def validate_velocity(
        velocity
    ):
        """
        Validate a generated velocity model.

        Parameters
        ----------
        velocity : torch.Tensor

            Expected shape:

                [C,D,H,W]

        Returns
        -------
        bool
        """

        if not isinstance(
            velocity,
            torch.Tensor
        ):
            raise TypeError(
                "velocity must be a torch.Tensor."
            )

        if velocity.ndim != 4:
            raise ValueError(
                "velocity must have shape "
                "[C,D,H,W]. "
                f"Received: {tuple(velocity.shape)}."
            )

        if not torch.isfinite(
            velocity
        ).all():
            raise ValueError(
                "Velocity model contains NaN or Inf."
            )

        if torch.any(
            velocity <= 0
        ):
            raise ValueError(
                "Velocity values must be strictly "
                "greater than zero."
            )

        return True