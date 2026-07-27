import numpy as np

from dataset.mask_generator import MaskGenerator
from utils.config import (
    MASK_TYPE,
    MISSING_PROBABILITY
)


def main():

    cube = np.ones(

        (64, 64, 64),

        dtype=np.float32

    )

    generator = MaskGenerator(

        missing_probability=MISSING_PROBABILITY,

        mask_type=MASK_TYPE

    )


    mask = generator.generate(cube)

    print()

    print(f"Mask Type: {MASK_TYPE}")

    print("=" * 60)

    print("Mask Generator Test")

    print("=" * 60)

    print()

    print("Cube Shape:")

    print(cube.shape)

    print()

    print("Mask Shape:")

    print(mask.shape)

    print()

    print("Available Voxels:")

    print(np.sum(mask))

    print()

    print("Missing Voxels:")

    print(mask.size - np.sum(mask))

    missing_percentage = 100.0 * (

            mask.size - np.sum(mask)

    ) / mask.size

    print()

    print(f"Missing Percentage: {missing_percentage:.2f}%")


if __name__ == "__main__":

    main()