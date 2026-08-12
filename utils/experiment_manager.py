"""
Experiment Manager

Dataset-specific experiment workspace

Author: Ormin Joseph
"""

import os

from utils.config import DATASET_MODE


class ExperimentManager:

    def __init__(self):

        # ------------------------------------------
        # Dataset-specific root folder
        # ------------------------------------------

        self.root = os.path.join(
            "outputs",
            DATASET_MODE.lower()
        )

        self.checkpoints = os.path.join(
            self.root,
            "checkpoints"
        )

        self.logs = os.path.join(
            self.root,
            "logs"
        )

        self.reports = os.path.join(
            self.root,
            "reports"
        )

        self.plots = os.path.join(
            self.root,
            "plots"
        )

        self.reconstructions = os.path.join(
            self.root,
            "reconstructions"
        )

        self.tensorboard = os.path.join(
            self.root,
            "tensorboard"
        )

        self.training_progress = os.path.join(
            self.root,
            "training_progress"
        )

        self.global_checkpoints = self.checkpoints

        self.create()

    def create(self):

        folders = [

            self.root,

            self.checkpoints,

            self.logs,

            self.reports,

            self.plots,

            self.reconstructions,

            self.tensorboard,

            self.training_progress

        ]

        for folder in folders:

            os.makedirs(
                folder,
                exist_ok=True
            )