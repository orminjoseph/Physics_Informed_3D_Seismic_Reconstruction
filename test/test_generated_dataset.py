import os

from dataset.dataset_generator import DatasetGenerator
from dataset.generated_dataset import SeismicDataset


def test_generated_dataset(tmp_path):

    # ------------------------------------------
    # Generate a small synthetic dataset
    # ------------------------------------------

    dataset_directory = tmp_path / "datasets"

    generator = DatasetGenerator(
        output_directory=str(dataset_directory),
        number_of_samples=5,
        cube_size=(64, 64, 64),
        random_seed=42
    )

    generator.generate_dataset()

    # ------------------------------------------
    # Load generated dataset
    # ------------------------------------------

    dataset = SeismicDataset(
        dataset_directory=str(dataset_directory)
    )

    # ------------------------------------------
    # Basic dataset validation
    # ------------------------------------------

    assert len(dataset) == 5

    # ------------------------------------------
    # Load first sample
    # ------------------------------------------

    sample = dataset[0]

    # ------------------------------------------
    # Validate dictionary structure
    # ------------------------------------------

    assert "ground_truth" in sample
    assert "corrupted" in sample
    assert "mask" in sample

    # ------------------------------------------
    # Validate tensor types
    # ------------------------------------------

    assert sample["ground_truth"].dtype.is_floating_point
    assert sample["corrupted"].dtype.is_floating_point
    assert sample["mask"].dtype.is_floating_point

    # ------------------------------------------
    # Validate shapes
    # ------------------------------------------

    expected_shape = (1, 64, 64, 64)

    assert sample["ground_truth"].shape == expected_shape
    assert sample["corrupted"].shape == expected_shape
    assert sample["mask"].shape == expected_shape

    # ------------------------------------------
    # Validate finite values
    # ------------------------------------------

    assert sample["ground_truth"].isfinite().all()
    assert sample["corrupted"].isfinite().all()
    assert sample["mask"].isfinite().all()

    # ------------------------------------------
    # Validate mask
    # ------------------------------------------

    unique_mask_values = sample["mask"].unique()

    assert all(
        value.item() in (0.0, 1.0)
        for value in unique_mask_values
    )