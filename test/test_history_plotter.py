from utils.history_plotter import HistoryPlotter


def main():

    plotter = HistoryPlotter(

        "outputs/logs/training_history.csv"

    )

    plotter.plot_loss()

    plotter.plot_psnr()

    print()

    print("Plots created successfully.")

    print()

    print("Saved to:")

    print("outputs/plots")


if __name__ == "__main__":

    main()