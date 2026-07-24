"""
=========================================================
Dataset Generator
=========================================================

Generates complete synthetic seismic datasets for training,
validation, and testing of the Physics-Informed 3D
Encoder–Decoder Framework with Predictive Uncertainty.

Each generated sample contains

• Ground Truth Seismic Cube
• Corrupted Seismic Cube
• Binary Sampling Mask

The generated dataset is stored on disk as NumPy arrays.

Author: Ormin Joseph
=========================================================
"""
import os

import numpy as np

from data.synthetic_generator import SyntheticGenerator

class DatasetGenerator:
    """
    DatasetGenerator

    Automatically generates complete synthetic seismic
    datasets for deep learning.

    Each sample consists of

    • Ground Truth Cube
    • Corrupted Cube
    • Binary Mask

    The generated files are saved to disk for later
    loading by the PyTorch Dataset class.
    """

    def __init__(
            self,
            output_directory="datasets",
            number_of_samples=100,
            cube_size=(64, 64, 64),
            random_seed=42
    ):
        # ------------------------------------------
        # Dataset parameters
        # ------------------------------------------

        self.output_directory = output_directory

        self.number_of_samples = number_of_samples

        self.cube_size = cube_size

        self.random_seed = random_seed
        # ------------------------------------------
        # Synthetic seismic generator
        # ------------------------------------------

        self.generator = SyntheticGenerator(

            cube_size=self.cube_size,

            random_seed=self.random_seed

        )
        # ------------------------------------------
        # Create dataset directory
        # ------------------------------------------

        os.makedirs(

            self.output_directory,

            exist_ok=True

        )

    # --------------------------------------------------
    # Generate Single Sample
    # --------------------------------------------------

    def generate_single_sample(
            self
    ):
        """
        Generate one complete seismic training sample.

        Returns
        -------
        tuple

            (
                ground_truth,
                corrupted,
                mask
            )
        """
        # ------------------------------------------
        # Generate one sample
        # ------------------------------------------

        ground_truth, corrupted, mask = (

            self.generator.generate_training_sample()

        )
        return (

            ground_truth,

            corrupted,

            mask

        )

    # --------------------------------------------------
    # Generate Complete Dataset
    # --------------------------------------------------

    def generate_dataset(
            self
    ):
        """
        Generate an entire synthetic seismic dataset.

        Each generated sample is saved as

        • Ground Truth
        • Corrupted Cube
        • Binary Mask
        """

        # ------------------------------------------
        # Generate all samples
        # ------------------------------------------

        for sample in range(
                self.number_of_samples
        ):
            # ------------------------------------------
            # Generate one sample
            # ------------------------------------------

            ground_truth, corrupted, mask = (

                self.generate_single_sample()

            )

            # ------------------------------------------
            # Sample filename
            # ------------------------------------------

            sample_name = (

                f"sample_{sample:05d}"

            )

            # ------------------------------------------
            # Save ground truth
            # ------------------------------------------

            np.save(

                os.path.join(

                    self.output_directory,

                    sample_name + "_gt.npy"

                ),

                ground_truth

            )

            # ------------------------------------------
            # Save corrupted cube
            # ------------------------------------------

            np.save(

                os.path.join(

                    self.output_directory,

                    sample_name + "_corrupted.npy"

                ),

                corrupted

            )

            # ------------------------------------------
            # Save binary mask
            # ------------------------------------------

            np.save(

                os.path.join(

                    self.output_directory,

                    sample_name + "_mask.npy"

                ),

                mask

            )

            # ------------------------------------------
            # Progress
            # ------------------------------------------

            print(

                f"Generated {sample + 1}/{self.number_of_samples}"

            )

        print()

        print("Dataset generation completed.")

        print(

            "Saved to:",

            self.output_directory

        )