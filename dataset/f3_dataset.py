"""
=========================================================
F3 Netherlands Dataset
=========================================================

Loads the F3 seismic volume and prepares
training patches for reconstruction.

Author: Ormin Joseph
=========================================================
"""

import torch
import numpy as np
import segyio

from torch.utils.data import Dataset

from dataset.patch_extractor import PatchExtractor
from dataset.mask_generator import MaskGenerator


class F3Dataset(Dataset):

    def __init__(
        self,
        segy_path,
        patch_size=(64, 64, 64),
        stride=(64, 64, 64),
        missing_probability=0.30
    ):

        self.segy_path = segy_path

        self.patch_size = patch_size

        self.stride = stride

        self.missing_probability = (
            missing_probability
        )

        self.volume = self._load_volume()

        self.extractor = PatchExtractor(
            patch_size=patch_size,
            stride=stride
        )

        self.mask_generator = (
            MaskGenerator(
                missing_probability
            )
        )

        self.patches = (
            self.extractor.extract(
                self.volume
            )
        )

    def _load_volume(self):

        with segyio.open(
                self.segy_path,
                ignore_geometry=True
        ) as segy:

            print()
            print("=" * 60)
            print("LOADING F3 VOLUME")
            print("=" * 60)

            trace_count = segy.tracecount

            samples = len(segy.samples)

            print(
                "Trace Count:",
                trace_count
            )

            print(
                "Samples:",
                samples
            )

            inline_values = []

            crossline_values = []

            # ----------------------------------
            # Read survey geometry
            # ----------------------------------

            for i in range(trace_count):
                header = segy.header[i]

                inline_values.append(
                    header[
                        segyio.TraceField.INLINE_3D
                    ]
                )

                crossline_values.append(
                    header[
                        segyio.TraceField.CROSSLINE_3D
                    ]
                )

            inline_values = np.array(
                inline_values
            )

            crossline_values = np.array(
                crossline_values
            )

            unique_inlines = np.unique(
                inline_values
            )

            unique_crosslines = np.unique(
                crossline_values
            )

            n_inline = len(
                unique_inlines
            )

            n_crossline = len(
                unique_crosslines
            )

            print(
                "Unique Inlines:",
                n_inline
            )

            print(
                "Unique Crosslines:",
                n_crossline
            )

            # ----------------------------------
            # Mapping tables
            # ----------------------------------

            inline_map = {

                il: idx

                for idx, il in enumerate(
                    unique_inlines
                )
            }

            crossline_map = {

                xl: idx

                for idx, xl in enumerate(
                    unique_crosslines
                )
            }

            # ----------------------------------
            # Allocate volume
            # ----------------------------------

            volume = np.zeros(

                (
                    n_inline,
                    n_crossline,
                    samples
                ),

                dtype=np.float32

            )

            # ----------------------------------
            # Insert traces
            # ----------------------------------

            for i in range(trace_count):
                iline = inline_values[i]

                xline = crossline_values[i]

                il_idx = inline_map[iline]

                xl_idx = crossline_map[xline]

                volume[
                    il_idx,
                    xl_idx,
                    :
                ] = segy.trace[i]

        # ----------------------------------
        # Normalize
        # ----------------------------------

        volume = (

                         volume - volume.mean()

                 ) / (

                         volume.std() + 1e-8

                 )

        print()

        print(
            "Volume Shape:",
            volume.shape
        )

        return volume


    def __len__(self):

        return len(
            self.patches
        )

    def __getitem__(self, idx):

        patch, z, y, x = self.patches[idx]

        patch = torch.tensor(
            patch,
            dtype=torch.float32
        )

        mask = torch.tensor(
            self.mask_generator.generate(
                patch.shape
            ),
            dtype=torch.float32
        )

        corrupted = patch * mask

        normalized_patch = (
                torch.abs(patch)
                /
                (torch.max(torch.abs(patch)) + 1e-8)
        )

        velocity_model = (
                1800.0
                +
                2200.0 * normalized_patch
        )

        return (
            corrupted.unsqueeze(0),
            patch.unsqueeze(0),
            mask.unsqueeze(0),
            velocity_model.unsqueeze(0)
        )
