"""
=========================================================
Test PyTorch DataLoader
=========================================================
"""

from dataset.dataloader import create_dataloader


def main():

    # ------------------------------------------
    # Create DataLoader
    # ------------------------------------------

    dataloader = create_dataloader(

        dataset_directory="datasets",

        batch_size=2,

        shuffle=True

    )

    # ------------------------------------------
    # Load one mini-batch
    # ------------------------------------------

    ground_truth, corrupted, mask = next(iter(dataloader))

    # ------------------------------------------
    # Display information
    # ------------------------------------------

    print()

    print("Ground Truth Batch Shape :", ground_truth.shape)

    print("Corrupted Batch Shape    :", corrupted.shape)

    print("Mask Batch Shape         :", mask.shape)

    print()

    print("Ground Truth Type :", ground_truth.dtype)

    print("Corrupted Type    :", corrupted.dtype)

    print("Mask Type         :", mask.dtype)


if __name__ == "__main__":

    main()