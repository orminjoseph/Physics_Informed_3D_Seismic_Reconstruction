"""
=========================================================
Synthetic 3D Seismic Dataset
=========================================================

Generates synthetic seismic cubes together with
velocity models for physics-informed learning.

Author: Ormin Joseph
=========================================================
"""

import torch
from torch.utils.data import Dataset
import numpy as np

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator


class SyntheticSeismicDataset(Dataset):

    def __init__(
            self,
            num_samples=100,
            cube_size=(64, 128, 128),
            missing_probability=0.30,
            geological_mode="random"
    ):
        super().__init__()

        self.num_samples = num_samples
        self.cube_size = cube_size
        self.missing_probability = missing_probability
        self.geological_mode = geological_mode

        self.generator = GeologicalGenerator(
            cube_size=cube_size
        )
        self.num_samples = num_samples

        self.cube_size = cube_size

        self.missing_probability = missing_probability

        self.geological_mode = geological_mode

        # ------------------------------------------
        # Geological model generator
        # ------------------------------------------

        self.generator = GeologicalGenerator(
            cube_size=cube_size
        )

        # ------------------------------------------
        # Velocity model generator
        # ------------------------------------------

        self.velocity_generator = VelocityGenerator(
            cube_size=cube_size
        )

        # ------------------------------------------
        # Store generated samples
        # ------------------------------------------

        self.inputs = []

        self.targets = []

        self.masks = []

        self.velocities = []

        # ------------------------------------------
        # Generate dataset once
        # ------------------------------------------

        for _ in range(num_samples):

            if self.geological_mode == "random":

                mode = np.random.choice([

                    "horizontal",

                    "dipping",

                    "faulted",

                    "folded",

                    "complex",

                    "highly_complex"
                ])

            else:

                mode = self.geological_mode

            target = self.generator.generate(
                mode=mode
            )

            # ------------------------------
            # Velocity cube
            # ------------------------------

            velocity = self.velocity_generator.generate()

            # ------------------------------
            # Missing trace mask
            # ------------------------------

            mask = (
                torch.rand_like(target)
                >
                self.missing_probability
            ).float()

            # ------------------------------
            # Simulated acquisition
            # ------------------------------

            input_cube = target * mask

            self.inputs.append(input_cube)

            self.targets.append(target)

            self.masks.append(mask)

            self.velocities.append(velocity)

    # --------------------------------------------------

    def __len__(self):

        return self.num_samples

    # --------------------------------------------------

    def __getitem__(self, idx):
        """
        Return one sample.

        Returns
        -------
        input_cube
        target_cube
        mask
        velocity_cube
        """

        return (

            self.inputs[idx],

            self.targets[idx],

            self.masks[idx],

            self.velocities[idx]

        )