"""
=========================================================
Test Seismic Dataset
=========================================================

Verifies that the PyTorch SeismicDataset correctly

• Loads the generated dataset
• Returns the correct number of samples
• Loads one sample
• Converts arrays to tensors

=========================================================
"""

from dataset.generated_dataset import SeismicDataset


def main():

    dataset = SeismicDataset()

    print()

    print("Number of Samples :", len(dataset))

    print()

    ground_truth, corrupted, mask = dataset[0]

    print("Ground Truth Shape :", ground_truth.shape)

    print("Corrupted Shape    :", corrupted.shape)

    print("Mask Shape         :", mask.shape)

    print()

    print("Ground Truth Type :", ground_truth.dtype)

    print("Corrupted Type    :", corrupted.dtype)

    print("Mask Type         :", mask.dtype)


if __name__ == "__main__":

    main()