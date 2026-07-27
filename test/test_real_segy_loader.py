import os

from dataset.segy_loader import SegyLoader


def main():

    filename = "data/F3/F3.sgy"

    if not os.path.exists(filename):

        print()

        print("SEG-Y file not found.")

        print("Skipping test.")

        return

    loader = SegyLoader(filename)

    cube = loader.load()

    metadata = loader.get_metadata()

    print()

    print("=" * 60)

    print("SEG-Y Metadata")

    print("=" * 60)

    for key, value in metadata.items():

        print(f"{key}: {value}")


if __name__ == "__main__":

    main()