from dataset.f3_dataset import F3Dataset

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

sample = dataset[0]

print("Type:", type(sample))

if isinstance(sample, (tuple, list)):
    print("Number of items:", len(sample))

    for i, item in enumerate(sample):
        try:
            print(f"Item {i}: shape = {item.shape}")
        except AttributeError:
            print(f"Item {i}: type = {type(item)}")
