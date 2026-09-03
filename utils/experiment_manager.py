"""
=========================================================
Experiment Manager
=========================================================

Manages directories and output locations for experiments.

Default behavior:
    outputs/<EXPERIMENT_NAME>/

Optional behavior:
    A custom root directory can be supplied for isolated
    experiments such as ablation studies.

=========================================================
"""

import os

from utils.config import (
    OUTPUT_ROOT,
    EXPERIMENT_NAME
)


class ExperimentManager:
    """
    Manage directories and output locations for one experiment.
    """

    def __init__(
        self,
        root=None
    ):
        """
        Parameters
        ----------
        root : str or None
            Optional custom experiment root.

            If None:
                outputs/<EXPERIMENT_NAME>/

            If supplied:
                the supplied directory is used directly.
        """

        # =================================================
        # EXPERIMENT ROOT
        # =================================================

        if root is None:

            self.root = os.path.join(
                OUTPUT_ROOT,
                EXPERIMENT_NAME
            )

        else:

            self.root = root

        # =================================================
        # CHECKPOINTS
        # =================================================

        self.checkpoints = os.path.join(
            self.root,
            "checkpoints"
        )

        # =================================================
        # LOGS
        # =================================================

        self.logs = os.path.join(
            self.root,
            "logs"
        )

        # =================================================
        # REPORTS
        # =================================================

        self.reports = os.path.join(
            self.root,
            "reports"
        )

        # =================================================
        # PLOTS
        # =================================================

        self.plots = os.path.join(
            self.root,
            "plots"
        )

        # =================================================
        # RECONSTRUCTIONS
        # =================================================

        self.reconstructions = os.path.join(
            self.root,
            "reconstructions"
        )

        # =================================================
        # TENSORBOARD
        # =================================================

        self.tensorboard = os.path.join(
            self.root,
            "tensorboard"
        )

        # =================================================
        # TRAINING PROGRESS
        # =================================================

        self.training_progress = os.path.join(
            self.root,
            "training_progress"
        )

        # =================================================
        # GLOBAL CHECKPOINT REFERENCE
        # =================================================

        self.global_checkpoints = (
            self.checkpoints
        )

        # =================================================
        # CREATE DIRECTORIES
        # =================================================

        self.create()

    # =====================================================
    # CREATE DIRECTORIES
    # =====================================================

    def create(self):

        directories = [

            self.root,

            self.checkpoints,

            self.logs,

            self.reports,

            self.plots,

            self.reconstructions,

            self.tensorboard,

            self.training_progress

        ]

        for directory in directories:

            os.makedirs(
                directory,
                exist_ok=True
            )