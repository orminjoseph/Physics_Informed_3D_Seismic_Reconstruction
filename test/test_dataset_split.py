from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset


def test_dataset_split():

    dataset = build_dataset()

    train_dataset, validation_dataset = (
        split_dataset(dataset)
    )

    print()
    print("=" * 60)
    print("DATASET SPLIT TEST")
    print("=" * 60)

    print()
    print("Total:", len(dataset))
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print()

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    assert len(dataset) > 0

    assert len(train_dataset) > 0

    assert len(validation_dataset) > 0

    # -----------------------------------------------------
    # Ensure all samples are accounted for
    # -----------------------------------------------------

    assert (
        len(train_dataset) + len(validation_dataset)
        == len(dataset)
    )

    # -----------------------------------------------------
    # Ensure both subsets are Dataset objects
    # -----------------------------------------------------

    assert hasattr(train_dataset, "__len__")
    assert hasattr(train_dataset, "__getitem__")

    assert hasattr(validation_dataset, "__len__")
    assert hasattr(validation_dataset, "__getitem__")

    print("DATASET SPLIT TEST PASSED")