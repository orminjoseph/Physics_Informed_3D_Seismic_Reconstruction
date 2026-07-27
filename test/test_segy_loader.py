from dataset.segy_loader import SegyLoader


def main():

    filename = "datasets/sample.sgy"

    loader = SegyLoader(filename)

    cube = loader.load()

    print()

    print("=" * 60)

    print("SEG-Y Loader Test")

    print("=" * 60)

    print()

    print("Cube Shape:")

    print(cube.shape)

    print()

    print("Minimum Amplitude:")

    print(cube.min())

    print()

    print("Maximum Amplitude:")

    print(cube.max())


if __name__ == "__main__":
    main()