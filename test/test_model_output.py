"""
=========================================================
Test 3D Model Output
=========================================================

Verifies that the current F3 dataset can be passed through
the current 3D Encoder-Decoder network.

This test does NOT require a trained checkpoint.

It verifies:

    F3 Dataset
        ↓
    Input seismic patch
        ↓
    Network3D
        ↓
    Reconstruction output
    Uncertainty output

Author: Ormin Joseph
=========================================================
"""

import torch

from dataset.f3_dataset import F3Dataset
from models.network import Network3D


# =========================================================
# F3 DATASET PATH
# =========================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)


# =========================================================
# TEST
# =========================================================

def test_model_output():

    # -----------------------------------------------------
    # Create F3 dataset
    # -----------------------------------------------------

    dataset = F3Dataset(

        segy_path=F3_PATH,

        patch_size=(
            64,
            64,
            64
        ),

        stride=(
            64,
            64,
            64
        ),

        missing_probability=0.30
    )

    # -----------------------------------------------------
    # Verify dataset contains samples
    # -----------------------------------------------------

    assert len(dataset) > 0

    # -----------------------------------------------------
    # Load first sample
    # -----------------------------------------------------

    sample = dataset[0]

    # -----------------------------------------------------
    # Support the current F3Dataset output format
    # -----------------------------------------------------

    if isinstance(sample, dict):

        input_cube = sample["input"]

    else:

        input_cube = sample[0]

    # -----------------------------------------------------
    # Verify input is a PyTorch tensor
    # -----------------------------------------------------

    assert isinstance(
        input_cube,
        torch.Tensor
    )

    # -----------------------------------------------------
    # Verify expected patch shape
    # -----------------------------------------------------

    assert input_cube.shape == (
        1,
        64,
        64,
        64
    )

    # -----------------------------------------------------
    # Create a fresh model
    # -----------------------------------------------------

    model = Network3D()

    # -----------------------------------------------------
    # Evaluation mode
    # -----------------------------------------------------

    model.eval()

    # -----------------------------------------------------
    # Add batch dimension
    #
    # Current shape:
    #
    # (1, 64, 64, 64)
    #
    # becomes:
    #
    # (1, 1, 64, 64, 64)
    # -----------------------------------------------------

    input_batch = input_cube.unsqueeze(0)

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            input_batch
        )

    # -----------------------------------------------------
    # Display output information
    # -----------------------------------------------------

    print()

    print("=" * 60)

    print("MODEL OUTPUT TEST")

    print("=" * 60)

    print()

    print(
        "Input Shape :",
        input_batch.shape
    )

    print()

    print(
        "Output Type :",
        type(output)
    )

    print()

    # -----------------------------------------------------
    # Handle multiple model outputs
    # -----------------------------------------------------

    if isinstance(
        output,
        (tuple, list)
    ):

        print(
            "Number of Outputs :",
            len(output)
        )

        print()

        for i, item in enumerate(output):

            print(
                f"Output {i} Shape :",
                item.shape
            )

            # ---------------------------------------------
            # Every output should retain the batch,
            # channel and spatial dimensions.
            # ---------------------------------------------

            assert item.shape[0] == 1

            assert item.shape[2:] == (
                64,
                64,
                64
            )

    # -----------------------------------------------------
    # Handle dictionary model outputs
    # -----------------------------------------------------

    elif isinstance(
        output,
        dict
    ):

        print(
            "Output Keys :",
            list(output.keys())
        )

        print()

        for key, item in output.items():

            print(
                f"{key} Shape :",
                item.shape
            )

            assert item.shape[0] == 1

            assert item.shape[2:] == (
                64,
                64,
                64
            )

    # -----------------------------------------------------
    # Handle single tensor output
    # -----------------------------------------------------

    elif isinstance(
        output,
        torch.Tensor
    ):

        print(
            "Output Shape :",
            output.shape
        )

        assert output.shape[0] == 1

        assert output.shape[2:] == (
            64,
            64,
            64
        )

    else:

        raise TypeError(
            "Unsupported model output type: "
            f"{type(output)}"
        )

    print()

    print(
        "MODEL FORWARD PASS: PASSED"
    )

    print("=" * 60)

    print()