import torch

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

dataset = F3Dataset(
    segy_path=F3_PATH,
    patch_size=(64,64,64),
    stride=(64,64,64),
    missing_probability=0.30
)

input_cube, target_cube, mask, velocity_model = dataset[0]

model = Network3D()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.no_grad():

    output = model(
        input_cube.unsqueeze(0)
    )

print(type(output))

if isinstance(output, (tuple, list)):
    print("Number of outputs:", len(output))

    for i, item in enumerate(output):
        print(
            f"Output {i}:",
            item.shape
        )
else:
    print(output.shape)