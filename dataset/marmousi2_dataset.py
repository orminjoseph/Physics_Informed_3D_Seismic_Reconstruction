import numpy as np
import segyio
import torch
from torch.utils.data import Dataset

from dataset.mask_generator import MaskGenerator


class Marmousi2Dataset(Dataset):
    """
    Marmousi2 velocity model dataset.

    Loads Marmousi2 P-wave velocity model from SEG-Y format
    and converts it into a pseudo-3D seismic volume.

    Returns:
        {
            "input": Tensor,
            "target": Tensor,
            "mask": Tensor
        }
    """

    def __init__(
            self,
            segy_path,
            mask_type="random_trace",
            missing_rate=0.30,
            pseudo_depth=32
    ):

        self.segy_path = segy_path

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
            )

        # ----------------------------------
        # Convert 2D Marmousi2 into pseudo-3D
        # ----------------------------------

        self.volume = np.repeat(
            velocity_model[np.newaxis, :, :],
            pseudo_depth,
            axis=0
        ).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        target = self.volume

        mask = self.mask_generator.generate_mask(
            target.shape
        )

        input_data = target * mask

        return {
            "input": torch.tensor(
                input_data,
                dtype=torch.float32
            ),

            "target": torch.tensor(
                target,
                dtype=torch.float32
            ),

            "mask": torch.tensor(
                mask,
                dtype=torch.float32
            )
        }