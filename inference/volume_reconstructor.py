"""
=========================================================
Volume Reconstructor
=========================================================

Reconstructs a full seismic volume by
processing overlapping patches.

Author: Ormin Joseph
=========================================================
"""

import torch
import numpy as np

from dataset.patch_extractor import PatchExtractor


class VolumeReconstructor:

    def __init__(
        self,
        predictor,
        patch_size,
        stride
    ):

        self.predictor = predictor

        self.patch_size = patch_size

        self.stride = stride

    def reconstruct(
        self,
        volume
    ):

        extractor = PatchExtractor(
            patch_size=self.patch_size,
            stride=self.stride
        )

        patches = extractor.extract(volume)

        reconstructed_patches = []

        for patch in patches:

            patch_tensor = (
                torch.from_numpy(patch)
                .float()
                .unsqueeze(0)
            )

            reconstruction, uncertainty = (
                self.predictor.predict(
                    patch_tensor
                )
            )

            reconstructed_patches.append(
                reconstruction.squeeze().numpy()
            )

        return reconstructed_patches