import segyio
import numpy as np

FILE_PATH = (
    r"C:\Users\ormin\Desktop\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

with segyio.open(
        FILE_PATH,
        ignore_geometry=True
) as segy:

    print()
    print("=" * 60)
    print("F3 VOLUME SHAPE TEST")
    print("=" * 60)

    trace_count = segy.tracecount
    samples = len(segy.samples)

    print("Trace Count :", trace_count)
    print("Samples     :", samples)

    # Read first 1000 traces only for verification
    traces = []

    limit = min(1000, trace_count)

    for i in range(limit):
        traces.append(segy.trace[i])

    traces = np.asarray(traces)

    print()
    print("Loaded Trace Block Shape :", traces.shape)

    print()
    print("Amplitude Statistics")
    print("--------------------")
    print("Min  :", traces.min())
    print("Max  :", traces.max())
    print("Mean :", traces.mean())
    print("Std  :", traces.std())