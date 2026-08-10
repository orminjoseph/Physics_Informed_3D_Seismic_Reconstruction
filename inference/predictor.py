"""
Inference module

Loads a trained network and reconstructs
missing seismic data.
"""

import torch


class Predictor:

    def __init__(

            self,

            model,

            checkpoint,

            device

    ):

        self.device = device

        self.model = model.to(device)

        checkpoint_data = torch.load(

            checkpoint,

            map_location=device

        )

        self.model.load_state_dict(

            checkpoint_data["model_state_dict"]

        )

        self.model.eval()

    def predict(self, corrupted_cube):
        """
        Perform inference on one corrupted seismic cube.

        Parameters
        ----------
        corrupted_cube : torch.Tensor
            Shape:
            (1, D, H, W)
            or
            (B, 1, D, H, W)

        Returns
        -------
        reconstruction
        uncertainty
        """

        self.model.eval()

        with torch.no_grad():
            # If channel dimension is missing
            if corrupted_cube.dim() == 4:
                corrupted_cube = corrupted_cube.unsqueeze(0)

            corrupted_cube = corrupted_cube.to(self.device)

            reconstruction, log_variance = self.model(corrupted_cube)

            print(
                "LogVar Min:",
                log_variance.min().item()
            )

            print(
                "LogVar Max:",
                log_variance.max().item()
            )

            log_variance = torch.clamp(
                log_variance,
                min=-10.0,
                max=10.0
            )

            uncertainty = torch.exp(
                0.5 * log_variance
            )

        return reconstruction.cpu(), uncertainty.cpu()

