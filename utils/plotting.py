"""
=========================================================
Plotting Utilities
=========================================================

Helper functions for saving and displaying figures.

Author: Ormin Joseph
=========================================================
"""

import os
import matplotlib.pyplot as plt


def create_output_directories():
    """
    Create all output folders.
    """

    folders = [

        "outputs/checkpoints",

        "outputs/figures/geology",

        "outputs/figures/predictions",

        "outputs/figures/training",

        "outputs/figures/uncertainty",

        "outputs/figures/comparison",

        "outputs/logs",

        "outputs/metrics",

        "outputs/predictions"

    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_figure(filename,
                category="geology",
                dpi=300):
    """
    Save the current matplotlib figure.
    """

    create_output_directories()

    filepath = os.path.join(
        "outputs",
        "figures",
        category,
        filename
    )

    plt.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(f"Figure saved to:\n{filepath}")