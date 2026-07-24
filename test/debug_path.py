import os
import sys

print("Current working directory:")
print(os.getcwd())

print("\nCurrent script:")
print(__file__)

print("\nPython search path:")
for p in sys.path:
    print(p)