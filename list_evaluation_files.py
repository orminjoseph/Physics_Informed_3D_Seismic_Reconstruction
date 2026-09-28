"""
=========================================================
LIST ALL FILES UNDER evaluation/
=========================================================

This script recursively lists every file and folder
inside the project's evaluation/ directory.

It does NOT modify any project files.

Author: Ormin Joseph
=========================================================
"""

from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

# This file should be placed in the PROJECT ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# EVALUATION DIRECTORY
# =========================================================

EVALUATION_DIR = PROJECT_ROOT / "evaluation"


# =========================================================
# CHECK THAT evaluation/ EXISTS
# =========================================================

if not EVALUATION_DIR.exists():

    print()
    print("[ERROR] evaluation/ directory was not found.")
    print()
    print("Expected location:")
    print(EVALUATION_DIR)
    print()

    raise SystemExit(1)


# =========================================================
# DISPLAY LOCATION
# =========================================================

print()
print("=" * 80)
print("EVALUATION DIRECTORY CONTENTS")
print("=" * 80)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Evaluation directory:")
print(EVALUATION_DIR)

print()


# =========================================================
# LIST DIRECTORIES
# =========================================================

directories = sorted(
    [
        path
        for path in EVALUATION_DIR.rglob("*")
        if path.is_dir()
    ]
)


print("=" * 80)
print("DIRECTORIES")
print("=" * 80)

print()

if not directories:

    print("No subdirectories found.")

else:

    print("evaluation/")

    for directory in directories:

        relative_path = directory.relative_to(
            PROJECT_ROOT
        )

        depth = len(relative_path.parts) - 1

        indentation = "    " * depth

        print(
            f"{indentation}├── {directory.name}/"
        )


# =========================================================
# LIST FILES
# =========================================================

files = sorted(
    [
        path
        for path in EVALUATION_DIR.rglob("*")
        if path.is_file()
    ]
)


print()
print("=" * 80)
print("FILES")
print("=" * 80)

print()

if not files:

    print("No files found.")

else:

    print("evaluation/")

    for file_path in files:

        relative_path = file_path.relative_to(
            PROJECT_ROOT
        )

        depth = len(relative_path.parts) - 1

        indentation = "    " * depth

        print(
            f"{indentation}├── {file_path.name}"
        )


# =========================================================
# PYTHON FILES ONLY
# =========================================================

python_files = sorted(
    [
        path
        for path in EVALUATION_DIR.rglob("*.py")
        if path.is_file()
    ]
)


print()
print("=" * 80)
print("PYTHON FILES ONLY")
print("=" * 80)

print()

if not python_files:

    print("No Python files found.")

else:

    for number, file_path in enumerate(
        python_files,
        start=1
    ):

        relative_path = file_path.relative_to(
            PROJECT_ROOT
        )

        print(
            f"{number:03d}. {relative_path}"
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print(
    f"Total directories : {len(directories)}"
)

print(
    f"Total files       : {len(files)}"
)

print(
    f"Python files      : {len(python_files)}"
)

print()
print("=" * 80)
print("SCAN COMPLETE")
print("=" * 80)