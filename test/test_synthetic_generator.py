"""
=========================================================
Test Synthetic Seismic Generator
=========================================================

Tests the complete synthetic seismic generation pipeline.

Workflow

1. Create generator
2. Generate complete synthetic seismic cube
3. Display basic statistics

Author: Ormin Joseph
=========================================================
"""

import numpy as np

from data.synthetic_generator import SyntheticGenerator


def main():

    # ------------------------------------------
    # Create generator
    # ------------------------------------------

    generator = SyntheticGenerator(

        cube_size=(64, 64, 64),

        mask_type="shot_lines",

        shot_line_interval=4,


        random_seed=42

    )



    # ------------------------------------------
    # Generate complete seismic cube
    # ------------------------------------------

    corrupted_cube, ground_truth_cube, mask = (
        generator.generate_seismic_cube()
    )

    # ------------------------------------------
    # Display statistics
    # ------------------------------------------

    print("Ground Truth Shape :", ground_truth_cube.shape)

    print("Corrupted Shape :", corrupted_cube.shape)

    print("Mask Shape :", mask.shape)

    print("Ground Truth Max :", ground_truth_cube.max())

    print("Corrupted Max :", corrupted_cube.max())

    print("Mask Unique Values :", np.unique(mask))

    print("Missing Trace Percentage :",
          np.mean(mask == 0.0) * 100)

if __name__ == "__main__":

    main()