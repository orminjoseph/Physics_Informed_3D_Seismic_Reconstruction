import numpy as np


class MaskGenerator:

    def __init__(

            self,

            missing_probability=0.30,

            mask_type="random_trace"

    ):
        self.missing_probability = missing_probability

        self.mask_type = mask_type

    def generate(

            self,

            cube

    ):

        inline, crossline, samples = cube.shape

        # -------------------------------------
        # Random Trace Mask
        # -------------------------------------

        if self.mask_type == "random_trace":

            trace_mask = (

                    np.random.rand(

                        inline,

                        crossline

                    ) > self.missing_probability

            )

        # -------------------------------------
        # Regular Trace Mask
        # -------------------------------------

        elif self.mask_type == "regular_trace":

            trace_mask = np.ones(

                (inline, crossline),

                dtype=np.float32

            )

            spacing = int(

                1 / self.missing_probability

            )

            trace_mask[:, ::spacing] = 0

        # -------------------------------------
        # Missing Inline Strip
        # -------------------------------------

        elif self.mask_type == "inline_strip":

            trace_mask = np.ones(

                (inline, crossline),

                dtype=np.float32

            )

            width = max(

                1,

                int(inline * self.missing_probability)

            )

            start = np.random.randint(

                0,

                inline - width

            )

            trace_mask[start:start + width, :] = 0

        # -------------------------------------
        # Missing Crossline Strip
        # -------------------------------------

        elif self.mask_type == "crossline_strip":

            trace_mask = np.ones(

                (inline, crossline),

                dtype=np.float32

            )

            width = max(

                1,

                int(crossline * self.missing_probability)

            )

            start = np.random.randint(

                0,

                crossline - width

            )

            trace_mask[:, start:start + width] = 0

        # -------------------------------------
        # Checkerboard
        # -------------------------------------

        elif self.mask_type == "checkerboard":

            trace_mask = np.indices(

                (inline, crossline)

            ).sum(axis=0) % 2

        else:

            raise ValueError(

                f"Unknown mask type: {self.mask_type}"

            )

        mask = np.repeat(

            trace_mask[:, :, np.newaxis],

            samples,

            axis=2

        )

        return mask.astype(np.float32)
