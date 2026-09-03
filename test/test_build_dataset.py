# test/test_build_dataset.py

from dataset.build_dataset import build_dataset
from dataset.synthetic_dataset import SyntheticSeismicDataset


def test_build_dataset():

    dataset = build_dataset()

    print()
    print("=" * 60)
    print("DATASET FACTORY TEST")
    print("=" * 60)

    print()
    print("Dataset Type:")
    print(type(dataset))

    print()
    print("Dataset Size:")
    print(len(dataset))

    # -------------------------------------------------
    # Verify that the dataset factory returns the
    # expected Synthetic dataset when DATASET_MODE
    # is set to "synthetic".
    # -------------------------------------------------

    assert isinstance(
        dataset,
        SyntheticSeismicDataset
    )

    assert len(dataset) > 0

    # -------------------------------------------------
    # Get one sample.
    #
    # Current dataset convention:
    #
    # input, target, mask, velocity
    # -------------------------------------------------

    input_cube, target, mask, velocity = dataset[0]

    print()
    print("Input Shape:")
    print(input_cube.shape)

    print()
    print("Target Shape:")
    print(target.shape)

    print()
    print("Mask Shape:")
    print(mask.shape)

    print()
    print("Velocity Shape:")
    print(velocity.shape)

    # -------------------------------------------------
    # Shape consistency checks
    # -------------------------------------------------

    assert input_cube.shape == target.shape
    assert input_cube.shape == mask.shape
    assert input_cube.shape == velocity.shape

    # -------------------------------------------------
    # Tensor dimensionality
    #
    # Individual dataset sample should be:
    #
    # [C, D, H, W]
    # -------------------------------------------------

    assert input_cube.ndim == 4
    assert target.ndim == 4
    assert mask.ndim == 4
    assert velocity.ndim == 4