"""
=========================================================
Helper Functions
=========================================================

Project:
Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for
Seismic Data Reconstruction

Author:
Ormin Joseph
=========================================================
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Set Random Seed
# =========================================================

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# Select Device
# =========================================================

def get_device():
    """
    Returns CPU or GPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# Count Trainable Parameters
# =========================================================

def count_parameters(model):
    """
    Counts trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# Create Folder
# =========================================================

def create_directory(path):
    """
    Creates a directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# =========================================================
# Weight Initialization
# =========================================================

def initialize_weights(model):
    """
    Initializes neural network weights.
    """

    for layer in model.modules():

        if isinstance(layer, nn.Conv3d):

            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.BatchNorm3d):

            nn.init.constant_(layer.weight, 1)

            nn.init.constant_(layer.bias, 0)