import numpy as np
import segyio
import torch

from torch.utils.data import Dataset

from dataset.mask_generator import MaskGenerator


class Marmousi2PatchDataset(Dataset):
    """
    Marmousi2 P-wave velocity model dataset.

    Converts the original 2D Marmousi2 velocity model
    into a pseudo-3D volume by repeating it along
    a depth dimension.

    Output patch shape:
        (32, 256, 256)
    """

    def __init__(
            self,
            segy_path,
            patch_size=(256, 256),
            pseudo_depth=32,
            missing_rate=0.30,
            mask_type="random_trace"
    ):

        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

        self.mask_generator = MaskGenerator(
            missing_probability=missing_rate,
            mask_type=mask_type
        )

        # ----------------------------------
        # Load SEG-Y velocity model
        # ----------------------------------

        with segyio.open(
                segy_path,
                "r",
                ignore_geometry=True
        ) as f:

            velocity_model = np.asarray(
                [trace for trace in f.trace]
            ).astype(np.float32)

        # ----------------------------------
        # Convert 2D Marmousi2 into pseudo-3D
        # ----------------------------------

        self.velocity_model = np.repeat(
            velocity_model[np.newaxis, :, :],
            pseudo_depth,
            axis=0
        )

        # ----------------------------------
        # Volume dimensions
        # ----------------------------------

        self.depth = self.velocity_model.shape[0]
        self.height = self.velocity_model.shape[1]
        self.width = self.velocity_model.shape[2]

        # ----------------------------------
        # Patch locations
        # ----------------------------------

        self.patch_locations = []

        for i in range(
                0,
                self.height - self.patch_h + 1,
                self.patch_h
        ):

            for j in range(
                    0,
                    self.width - self.patch_w + 1,
                    self.patch_w
            ):

                self.patch_locations.append((i, j))


    def __len__(self):

        return len(self.patch_locations)

    def __getitem__(self, idx):

        i, j = self.patch_locations[idx]

        # ----------------------------------
        # Extract 3D patch
        # Shape:
        # (32, 256, 256)
        # ----------------------------------

        target = self.velocity_model[
            :,
            i:i + self.patch_h,
            j:j + self.patch_w
        ]

        target = (
                         target - 1500.0
                 ) / (
                         4500.0 - 1500.0
                 )
        # ----------------------------------
        # Generate mask
        # ----------------------------------

        mask = self.mask_generator.generate_mask(
            target.shape
        )

        # ----------------------------------
        # Apply mask
        # ----------------------------------

        input_data = target * mask

        # ----------------------------------
        # Return same format as synthetic data
        # ----------------------------------

        return (

            torch.tensor(
                input_data,
                dtype=torch.float32
            ).unsqueeze(0),

            torch.tensor(
                target,
                dtype=torch.float32
            ).unsqueeze(0),

            torch.tensor(
                mask,
                dtype=torch.float32
            ).unsqueeze(0),

            torch.tensor(
                target,
                dtype=torch.float32
            ).unsqueeze(0)

        )