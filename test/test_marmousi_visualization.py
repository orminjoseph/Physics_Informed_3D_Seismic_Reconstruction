import segyio
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\ormin\Desktop\SEG_FILES\elastic-marmousi-model\elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy\MODEL_P-WAVE_VELOCITY_1.25m.segy"

with segyio.open(path, "r", ignore_geometry=True) as f:
    data = np.asarray([tr for tr in f.trace])

print("Shape:", data.shape)

plt.figure(figsize=(10,6))

plt.imshow(
    data.T,
    aspect="auto",
    cmap="jet",
    origin="upper"
)

plt.colorbar(label="Velocity (m/s)")
plt.title("Marmousi2 P-Wave Velocity Model")

plt.xlabel("Trace Number")
plt.ylabel("Depth Sample")

plt.show()