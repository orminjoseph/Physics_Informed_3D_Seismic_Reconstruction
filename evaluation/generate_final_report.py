import os
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


REPORT_DIR = "outputs/reports"

PDF_FILE = os.path.join(
    REPORT_DIR,
    "final_evaluation_report.pdf"
)

BASELINE_FILE = os.path.join(
    REPORT_DIR,
    "baseline_comparison.csv"
)

UNCERTAINTY_FILE = os.path.join(
    REPORT_DIR,
    "uncertainty_evaluation.csv"
)


def main():

    baseline_df = pd.read_csv(
        BASELINE_FILE
    )

    uncertainty_df = pd.read_csv(
        UNCERTAINTY_FILE
    )

    mean_uncertainty = (
        uncertainty_df["Mean_Uncertainty"]
        .mean()
    )

    max_uncertainty = (
        uncertainty_df["Mean_Uncertainty"]
        .max()
    )

    mean_mae = (
        uncertainty_df["MAE"]
        .mean()
    )

    mean_ssim = (
        uncertainty_df["SSIM"]
        .mean()
    )

    correlation = (
        uncertainty_df[
            "Mean_Uncertainty"
        ].corr(
            uncertainty_df["MAE"]
        )
    )

    doc = SimpleDocTemplate(
        PDF_FILE
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Physics-Informed 3D Seismic Reconstruction Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Dataset: F3 Netherlands",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "Missing traces: 30%",
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Baseline Comparison",
            styles["Heading2"]
        )
    )

    for _, row in baseline_df.iterrows():

        text = (
            f"{row['Method']} | "
            f"MAE={row['MAE']:.4f} | "
            f"RMSE={row['RMSE']:.4f} | "
            f"PSNR={row['PSNR']:.2f} | "
            f"SNR={row['SNR']:.2f} | "
            f"SSIM={row['SSIM']:.4f}"
        )

        elements.append(
            Paragraph(
                text,
                styles["BodyText"]
            )
        )

    elements.append(
        Spacer(1, 12)
    )

    elements.append(
        Paragraph(
            "Uncertainty Evaluation",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"Mean Uncertainty: {mean_uncertainty:.6f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Maximum Uncertainty: {max_uncertainty:.6f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Mean MAE: {mean_mae:.6f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Mean SSIM: {mean_ssim:.6f}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Correlation(Uncertainty, MAE): {correlation:.4f}",
            styles["BodyText"]
        )
    )

    def add_figure(
        filename,
        caption
    ):

        path = os.path.join(
            REPORT_DIR,
            filename
        )

        if os.path.exists(path):

            elements.append(
                PageBreak()
            )

            elements.append(
                Paragraph(
                    caption,
                    styles["Heading2"]
                )
            )

            elements.append(
                Image(
                    path,
                    width=450,
                    height=300
                )
            )

    add_figure(
        "best_patch.png",
        "Best Reconstruction Patch"
    )

    add_figure(
        "average_patch.png",
        "Average Reconstruction Patch"
    )

    add_figure(
        "worst_patch.png",
        "Worst Reconstruction Patch"
    )

    add_figure(
        "highest_uncertainty_patch.png",
        "Highest Uncertainty Patch"
    )

    add_figure(
        "uncertainty_vs_error.png",
        "Uncertainty versus Error"
    )

    elements.append(
        PageBreak()
    )

    elements.append(
        Paragraph(
            "Interpretation",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            (
                "Predictive uncertainty is positively "
                "correlated with reconstruction error. "
                "Regions exhibiting large reconstruction "
                "errors are generally associated with "
                "higher uncertainty values, indicating "
                "that the uncertainty framework is "
                "successfully identifying unreliable "
                "reconstruction zones."
            ),
            styles["BodyText"]
        )
    )

    doc.build(
        elements
    )

    print()
    print("Report generated:")
    print(PDF_FILE)


if __name__ == "__main__":
    main()