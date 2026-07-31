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

    inlines = []
    crosslines = []

    for i in range(segy.tracecount):

        header = segy.header[i]

        inlines.append(
            header[segyio.TraceField.INLINE_3D]
        )

        crosslines.append(
            header[segyio.TraceField.CROSSLINE_3D]
        )

    inlines = np.unique(inlines)
    crosslines = np.unique(crosslines)

    print()
    print("=" * 60)
    print("F3 GEOMETRY V2")
    print("=" * 60)

    print("Unique Inlines    :", len(inlines))
    print("Unique Crosslines :", len(crosslines))
    print("Samples           :", len(segy.samples))

    print()

    print("Inline Range:")
    print(inlines[0], "->", inlines[-1])

    print()

    print("Crossline Range:")
    print(crosslines[0], "->", crosslines[-1])