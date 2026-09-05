"""
============================================================
F3 DATASET TEST
============================================================
"""

from dataset.f3_dataset import F3Dataset


def test_f3_dataset():

    dataset = F3Dataset(

        segy_path=(
            r"C:\Users\ormin\Desktop\SEG_FILES"
            r"\F3_Demo_2023 (1)"
            r"\F3_Demo_2023"
            r"\Rawdata"
            r"\Seismic_data.sgy"
        ),

        patch_size=(64, 64, 64),

        stride=(64, 64, 64),

        missing_probability=0.30
    )

    print()
    print("=" * 60)
    print("F3 DATASET TEST")
    print("=" * 60)

    print(
        "Dataset Size :",
        len(dataset)
    )

    corrupted, target, mask, velocity = dataset[0][:4]

    print()
    print(
        "Corrupted Shape :",
        corrupted.shape
    )

    print(
        "Target Shape    :",
        target.shape
    )

    print(
        "Mask Shape      :",
        mask.shape
    )

    print("Velocity Shape :", velocity.shape)

    print()
    print(
        "Missing Voxels :",
        (mask == 0).sum().item()
    )

    print()
    print(
        "F3 Dataset Test: PASSED"
    )


if __name__ == "__main__":

    test_f3_dataset()