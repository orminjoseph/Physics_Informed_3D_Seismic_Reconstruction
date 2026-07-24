
# ==========================================
# Test Python Environment for Deep Learning
# ==========================================

import torch
import numpy as np
import pandas as pd
import scipy
import sklearn
import cv2
import h5py
import segyio
import matplotlib

print("=" * 50)
print("Deep Learning Environment Test")
print("=" * 50)

print(f"Python Version   : OK")
print(f"PyTorch Version  : {torch.__version__}")
print(f"NumPy Version    : {np.__version__}")
print(f"Pandas Version   : {pd.__version__}")
print(f"SciPy Version    : {scipy.__version__}")
print(f"Scikit-learn     : {sklearn.__version__}")
print(f"OpenCV Version   : {cv2.__version__}")
print(f"h5py Version     : {h5py.__version__}")
print(f"Matplotlib       : {matplotlib.__version__}")

print("\nChecking PyTorch...")

# Create a tensor
x = torch.tensor([[1, 2],
                  [3, 4]])

print("Tensor:")
print(x)

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU.")

print("\nSEG-Y Library Loaded Successfully.")

print("\nEverything is working correctly!")
print("=" * 50)

