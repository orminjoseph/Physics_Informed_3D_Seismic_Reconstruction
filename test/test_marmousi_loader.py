import segyio
import numpy as np

path = r"C:\Users\ormin\Desktop\SEG_FILES\elastic-marmousi-model\elastic-marmousi-model\model\MODEL_P-WAVE_VELOCITY_1.25m.segy\MODEL_P-WAVE_VELOCITY_1.25m.segy"

with segyio.open(path, "r", ignore_geometry=True) as f:

    print("Trace Count:", f.tracecount)

    trace = f.trace[0]

    print("Samples Per Trace:", len(trace))

    print("Min Velocity:", trace.min())
    print("Max Velocity:", trace.max())

    data = np.asarray([tr for tr in f.trace])

    print("Raw Array Shape:", data.shape)