import os
import pandas as pd
import matplotlib.pyplot as plt


class HistoryPlotter:

    def __init__(

            self,

            csv_file

    ):

        self.csv_file = csv_file

        os.makedirs(

            "outputs/plots",

            exist_ok=True

        )

    def plot_loss(self):

        history = pd.read_csv(self.csv_file)

        plt.figure(figsize=(8, 5))

        plt.plot(

            history["Epoch"],

            history["Train_Total"],

            label="Training"

        )

        plt.plot(

            history["Epoch"],

            history["Validation_Total"],

            label="Validation"

        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training History")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            "outputs/plots/loss_curve.png"

        )

        plt.close()

    def plot_psnr(self):

        history = pd.read_csv(self.csv_file)

        plt.figure(figsize=(8, 5))

        plt.plot(

            history["Epoch"],

            history["PSNR"]

        )

        plt.xlabel("Epoch")

        plt.ylabel("PSNR (dB)")

        plt.title("PSNR")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            "outputs/plots/psnr_curve.png"

        )

        plt.close()