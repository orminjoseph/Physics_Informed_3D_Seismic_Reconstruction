"""
=========================================================
Experiment Manager
=========================================================

Persistent experiment workspace

Author: Ormin Joseph
=========================================================
"""

import os


class ExperimentManager:

    def __init__(self):

        self.root = os.path.join(
            "outputs",
            "current_experiment"
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

            self.tensorboard

        ]

        for folder in folders:

            os.makedirs(
                folder,
                exist_ok=True
            )