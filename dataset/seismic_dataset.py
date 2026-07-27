"""
=========================================================
Unified Seismic Dataset
=========================================================

Handles seismic cubes from multiple sources:

• Synthetic geological generator
• SEG-Y seismic volumes
• Future F3 Netherlands
• Future Marmousi

Pipeline

Cube
 ↓
Patch Extraction
 ↓
Mask Generation
 ↓
PyTorch Dataset

Author: Ormin Joseph
=========================================================
"""

import numpy as np
import torch

from dataset.segy_loader import SegyLoader

from torch.utils.data import Dataset

from dataset.patch_extractor import PatchExtractor
from dataset.mask_generator import MaskGenerator
from dataset.geological_generator import GeologicalGenerator

from utils.config import (
    PATCH_SIZE,
    PATCH_STRIDE,
    MASK_TYPE,
    MISSING_PROBABILITY
)
class SeismicDataset(Dataset):
    def __init__(

            self,

            cube=None,

            segy_file=None,

            synthetic=False

    ):


        if synthetic:
            generator = GeologicalGenerator()

            cube = generator.generate()
        if segy_file is not None:
            loader = SegyLoader(

                segy_file

            )

            cube = loader.load()

            self.metadata = loader.get_metadata()
        if segy_file is None:
            self.metadata = {}

        self.cube = cube

        extractor = PatchExtractor(

            patch_size=PATCH_SIZE,

            stride=PATCH_STRIDE

        )

        self.patches = extractor.extract(

            cube

        )

        self.mask_generator = MaskGenerator(

            missing_probability=MISSING_PROBABILITY,

            mask_type=MASK_TYPE

        )


    def __len__(self):
        return len(self.patches)

    def __getitem__(

            self,

            index

    ):
        target_patch = self.patches[index]

        mask = self.mask_generator.generate(

            target_patch

        )

        input_patch = target_patch * mask

        input_patch = torch.from_numpy(

            input_patch

        ).float()

        target_patch = torch.from_numpy(

            target_patch

        ).float()

        mask = torch.from_numpy(

            mask

        ).float()

        return {

            "input": input_patch,

            "target": target_patch,

            "mask": mask

        }

    def get_metadata(self):

        return self.metadata

