"""
=========================================================
Experiment Manager
=========================================================

Creates unique folders for every experiment.

Author: Ormin Joseph
=========================================================
"""

import os

from datetime import datetime


class ExperimentManager:

    def __init__(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.root = os.path.join(

            "outputs",

            f"experiment_{timestamp}"

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

        self.global_checkpoints = "checkpoints"

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

            self.global_checkpoints

        ]

        for folder in folders:
            os.makedirs(

                folder,

                exist_ok=True

            )