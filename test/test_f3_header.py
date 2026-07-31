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
    print("F3 HEADER TEST")
    print("=" * 60)

    print(
        "Trace Count:",
        segy.tracecount
    )

    print(
        "Samples:",
        len(segy.samples)
    )

    header = segy.header[0]

    print()
    print("Header Keys:")

    for key in header.keys():
        print(key, header[key])
