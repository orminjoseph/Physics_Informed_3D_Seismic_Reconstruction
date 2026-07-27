import numpy as np

from dataset.patch_extractor import PatchExtractor


def main():

    cube = np.random.randn(

        128,

        128,

        128

    ).astype(np.float32)

    extractor = PatchExtractor(

        patch_size=(64, 64, 64),

        stride=(64, 64, 64)

    )

    patches = extractor.extract(cube)

    print()

    print("=" * 60)

    print("Patch Extractor Test")

    print("=" * 60)

    print()

    print("Input Cube Shape:")

    print(cube.shape)

    print()

    print("Number of Patches:")

    print(len(patches))

    print()

    print("Patch Shape:")

    print(patches[0].shape)


if __name__ == "__main__":

    main()