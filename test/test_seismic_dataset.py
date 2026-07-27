import numpy as np

from dataset.seismic_dataset import SeismicDataset


def main():

    cube = np.random.randn(

        128,

        128,

        128

    ).astype(np.float32)

    dataset = SeismicDataset(

        cube=cube

    )

    sample = dataset[0]

    print()

    print("=" * 60)

    print("Unified Dataset Test")

    print("=" * 60)

    print()

    print("Input Shape")

    print(sample["input"].shape)

    print()

    print("Target Shape")

    print(sample["target"].shape)

    print()

    print("Mask Shape")

    print(sample["mask"].shape)

    print()

    print("Dataset Size")

    print(len(dataset))


if __name__ == "__main__":

    main()