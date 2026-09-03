from dataset.build_dataset import build_dataset


def test_dataset_sample():

    dataset = build_dataset()

    input_cube, target_cube, mask, velocity_cube = dataset[0]

    print()
    print("=" * 60)
    print("DATASET SAMPLE TEST")
    print("=" * 60)

    print()
    print("Input Shape:", input_cube.shape)
    print("Target Shape:", target_cube.shape)
    print("Mask Shape:", mask.shape)
    print("Velocity Shape:", velocity_cube.shape)
    print()

    print("Input Min:", input_cube.min().item())
    print("Input Max:", input_cube.max().item())

    print("Target Min:", target_cube.min().item())
    print("Target Max:", target_cube.max().item())

    print("Velocity Min:", velocity_cube.min().item())
    print("Velocity Max:", velocity_cube.max().item())

    # -----------------------------------------------------
    # Basic shape checks
    # -----------------------------------------------------

    assert input_cube.shape == target_cube.shape

    assert input_cube.shape == mask.shape

    assert input_cube.shape == velocity_cube.shape

    # -----------------------------------------------------
    # Channel dimension
    # -----------------------------------------------------

    assert input_cube.shape[0] == 1

    # -----------------------------------------------------
    # Expected 3D seismic patch dimensions
    # -----------------------------------------------------

    assert input_cube.shape[1:] == (64, 128, 128)

    # -----------------------------------------------------
    # Data type checks
    # -----------------------------------------------------

    assert input_cube.dtype.is_floating_point

    assert target_cube.dtype.is_floating_point

    assert mask.dtype.is_floating_point

    assert velocity_cube.dtype.is_floating_point

    # -----------------------------------------------------
    # Finite-value checks
    # -----------------------------------------------------

    assert input_cube.isfinite().all()

    assert target_cube.isfinite().all()

    assert mask.isfinite().all()

    assert velocity_cube.isfinite().all()

    # -----------------------------------------------------
    # Mask validation
    #
    # 0 = missing
    # 1 = observed
    # -----------------------------------------------------

    assert mask.min().item() >= 0.0

    assert mask.max().item() <= 1.0

    # -----------------------------------------------------
    # Velocity must be physically positive
    # -----------------------------------------------------

    assert velocity_cube.min().item() > 0.0

    # -----------------------------------------------------
    # Input must contain the target only at observed voxels
    # -----------------------------------------------------

    assert (
        (input_cube[mask == 1] - target_cube[mask == 1])
        .abs()
        .max()
        .item()
        < 1e-6
    )

    # -----------------------------------------------------
    # Missing voxels should be zero in the input
    # -----------------------------------------------------

    assert (
        input_cube[mask == 0]
        .abs()
        .max()
        .item()
        < 1e-6
    )

    print()
    print("DATASET SAMPLE TEST PASSED")