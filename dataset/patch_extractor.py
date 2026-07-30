import numpy as np


class PatchExtractor:
    """
    Extracts overlapping or non-overlapping
    3D patches from a seismic cube.
    """

    def __init__(

            self,

            patch_size=(64, 64, 64),

            stride=(64, 64, 64)

    ):

        self.patch_size = patch_size

        self.stride = stride

    def extract(

            self,

            cube

    ):

        patches = []

        d, h, w = cube.shape

        pd, ph, pw = self.patch_size

        sd, sh, sw = self.stride

        for z in range(0, d - pd + 1, sd):

            for y in range(0, h - ph + 1, sh):

                for x in range(0, w - pw + 1, sw):

                    patch = cube[

                        z:z + pd,

                        y:y + ph,

                        x:x + pw

                    ]

                    patches.append(
                        (
                            patch,
                            z,
                            y,
                            x
                        )
                    )


        return patches