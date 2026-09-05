import torch
import matplotlib.pyplot as plt

from dataset.f3_dataset import F3Dataset
from models.network import Network3D

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

DEVICE = torch.device("cpu")

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64,64,64),
    stride=(64,64,64),
    missing_probability=0.30
)

input_cube, target_cube, mask, velocity_model = dataset[0][:4]

model = Network3D()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.no_grad():

    reconstruction, log_variance = model(
        input_cube.unsqueeze(0)
    )

reconstruction = reconstruction.squeeze()

uncertainty = torch.exp(
    0.5 * log_variance.squeeze()
)

slice_index = 32

original_slice = target_cube[0, slice_index]

corrupted_slice = input_cube[0, slice_index]

reconstructed_slice = reconstruction[slice_index]

error_slice = torch.abs(
    original_slice -
    reconstructed_slice
)

uncertainty_slice = uncertainty[slice_index]

velocity_slice = velocity_model[0, slice_index]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,10)
)

axes[0,0].imshow(
    original_slice,
    cmap="seismic"
)
axes[0,0].set_title("Original")

axes[0,1].imshow(
    corrupted_slice,
    cmap="seismic"
)
axes[0,1].set_title("Corrupted")

axes[0,2].imshow(
    reconstructed_slice,
    cmap="seismic"
)
axes[0,2].set_title("Reconstructed")

axes[1,0].imshow(
    error_slice,
    cmap="hot"
)
axes[1,0].set_title("Error Map")

axes[1,1].imshow(
    uncertainty_slice,
    cmap="viridis"
)
axes[1,1].set_title("Uncertainty")

axes[1,2].imshow(
    velocity_slice,
    cmap="jet"
)
axes[1,2].set_title("Velocity Model")

plt.tight_layout()

plt.savefig(
    "outputs/reconstruction_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()