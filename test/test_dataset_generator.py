from pathlib import Path

from dataset.dataset_generator import DatasetGenerator


def test_dataset_generator(tmp_path):
    """
    Test that DatasetGenerator can generate a synthetic dataset.
    """

    output_directory = tmp_path / "datasets"

    dataset = DatasetGenerator(
        output_directory=str(output_directory),
        number_of_samples=5,
        cube_size=(64, 64, 64),
        random_seed=42
    )

    dataset.generate_dataset()

    # Confirm that the output directory was created
    assert output_directory.exists()

    # Confirm that files were generated
    generated_files = list(output_directory.iterdir())

    assert len(generated_files) > 0