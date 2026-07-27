from utils.experiment_manager import ExperimentManager


def main():

    manager = ExperimentManager()

    print()

    print("=" * 60)

    print("Experiment Manager Test")

    print("=" * 60)

    print()

    print("Experiment Folder")

    print(manager.root)

    print()

    print("Checkpoint Folder")

    print(manager.checkpoints)

    print()

    print("TensorBoard Folder")

    print(manager.tensorboard)

    print()

    print("Reports Folder")

    print(manager.reports)


if __name__ == "__main__":

    main()