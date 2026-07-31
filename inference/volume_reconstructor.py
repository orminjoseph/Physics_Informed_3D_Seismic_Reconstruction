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

        depth, height, width = volume.shape

        reconstructed_volume = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        counter_volume = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        uncertainty_volume = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        uncertainty_counter = np.zeros(
            (depth, height, width),
            dtype=np.float32
        )

        patches = extractor.extract(volume)

        for patch, z, y, x in patches:

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

            reconstructed_patch = (
                reconstruction.squeeze()
                .cpu()
                .numpy()
            )

            uncertainty_patch = (
                uncertainty.squeeze()
                .cpu()
                .numpy()
            )

            pd, ph, pw = reconstructed_patch.shape

            reconstructed_volume[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += reconstructed_patch

            counter_volume[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += 1

            uncertainty_volume[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += uncertainty_patch

            uncertainty_counter[
                z:z + pd,
                y:y + ph,
                x:x + pw
            ] += 1

        counter_volume[
            counter_volume == 0
            ] = 1

        uncertainty_counter[
            uncertainty_counter == 0
            ] = 1

        reconstructed_volume /= counter_volume

        uncertainty_volume /= uncertainty_counter

        return (
            reconstructed_volume,
            uncertainty_volume
        )
 