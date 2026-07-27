"""
=========================================================
Velocity Model Generator
=========================================================

Generates synthetic velocity models for Physics-Informed
3D Seismic Reconstruction.

The velocity model represents subsurface geological layers
with different seismic velocities.

Author: Ormin Joseph
=========================================================
"""

import random
import torch


class VelocityGenerator:
    """
    Generate synthetic 3D velocity cubes.
    """

    def __init__(
            self,
            cube_size=(64, 128, 128),
            min_velocity=1800.0,
            max_velocity=3500.0
    ):

        self.cube_size = cube_size

        self.depth = cube_size[0]
        self.height = cube_size[1]
        self.width = cube_size[2]

        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    # --------------------------------------------------
    # Layered velocity model
    # --------------------------------------------------

    def generate_layered_model(
            self,
            number_of_layers=4
    ):
        """
        Generate horizontally layered velocity model.

        Returns
        -------
        velocity_cube
            Shape:
            (1, depth, height, width)
        """

        velocity = torch.zeros(
            1,
            self.depth,
            self.height,
            self.width
        )

        # ------------------------------------------
        # Layer boundaries
        # ------------------------------------------

        boundaries = [0]

        for _ in range(number_of_layers - 1):

            boundary = random.randint(
                5,
                self.depth - 5
            )

            boundaries.append(boundary)

        boundaries.append(self.depth)

        boundaries = sorted(boundaries)

        # ------------------------------------------
        # Assign increasing velocity with depth
        # ------------------------------------------

        current_velocity = self.min_velocity

        velocity_increment = (

            self.max_velocity
            -
            self.min_velocity

        ) / max(1, number_of_layers - 1)

        for i in range(number_of_layers):

            top = boundaries[i]

            bottom = boundaries[i + 1]

            velocity[
                :,
                top:bottom,
                :,
                :
            ] = current_velocity

            current_velocity += velocity_increment

        return velocity

    # --------------------------------------------------
    # Linear velocity gradient
    # --------------------------------------------------

    def generate_gradient_model(self):
        """
        Velocity increases gradually with depth.
        """

        velocity = torch.zeros(
            1,
            self.depth,
            self.height,
            self.width
        )

        for z in range(self.depth):

            value = (

                self.min_velocity

                +

                (
                    self.max_velocity
                    -
                    self.min_velocity
                )

                *

                z

                /

                (self.depth - 1)

            )

            velocity[
                :,
                z,
                :,
                :
            ] = value

        return velocity

    # --------------------------------------------------
    # Random selection
    # --------------------------------------------------

    def generate(self):
        """
        Randomly choose a velocity model.
        """

        if random.random() < 0.5:

            return self.generate_layered_model()

        return self.generate_gradient_model()