from utils.visualization import Visualizer

import numpy as np

visualizer = Visualizer()

cube = np.random.randn(64, 64, 64)

visualizer.save_slice(

    cube,

    filename="test_slice.png",

    title="Synthetic Seismic"

)

print("Slice saved successfully.")