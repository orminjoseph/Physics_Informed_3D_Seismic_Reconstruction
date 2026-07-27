import torch


class GeologicalGenerator:
    """
    Generates simple layered geological models.
    """

    def __init__(self,
                 cube_size=(64, 128, 128),
                 num_layers=8):

        self.depth, self.height, self.width = cube_size
        self.num_layers = num_layers

    def generate_horizontal_layers(self):

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = self.depth // self.num_layers

        amplitude = 0.2

        for layer in range(self.num_layers):

            start = layer * layer_thickness

            end = min(
                (layer + 1) * layer_thickness,
                self.depth
            )

            cube[start:end] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate_dipping_layers(self, dip=0.20):
        """
        Generate dipping geological layers.
        """

        cube = torch.zeros(
            self.depth,
            self.height,
            self.width
        )

        layer_thickness = self.depth // self.num_layers

        amplitude = 0.2

        for layer in range(self.num_layers):

            for x in range(self.width):

                shift = int(dip * x)

                start = layer * layer_thickness + shift
                end = start + layer_thickness

                if start >= self.depth:
                    continue

                end = min(end, self.depth)

                cube[start:end, :, x] = amplitude

            amplitude *= -1

        return cube.unsqueeze(0)

    def generate(self, dipping=False):

        if dipping:
            return self.generate_dipping_layers()

        return self.generate_horizontal_layers()
    