import segyio

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
    print("F3 DATASET TEST")
    print("=" * 60)

    print("Trace Count :", segy.tracecount)

    first_trace = segy.trace[0]

    print("Samples per Trace :", len(first_trace))

    print("Min Amplitude :", first_trace.min())
    print("Max Amplitude :", first_trace.max())