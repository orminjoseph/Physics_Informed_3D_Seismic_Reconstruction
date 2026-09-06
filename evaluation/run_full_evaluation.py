"""
FULL EVALUATION PIPELINE

Runs all evaluation modules automatically.

Outputs:
1. Quantitative Metrics
2. Reconstruction Gallery
3. Uncertainty Analysis
4. Baseline Comparison
5. Statistical Significance
6. Ablation Study
7. Thesis Tables
8. Final PDF Report

Author: Ormin Joseph
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from evaluation.evaluate_model import evaluate

from evaluation.reconstruction_gallery import (
    generate_gallery
)

from evaluation.uncertainty_analysis import (
    analyze_uncertainty
)

from evaluation.compare_with_baselines import (
    main as baseline_comparison
)

from evaluation.statistical_significance import (
    run_significance_test
)

from evaluation.final_thesis_tables import (
    generate_thesis_tables
)

from evaluation.generate_final_report import (
    generate_final_report
)

from evaluation.ablation_study import (
    run_ablation
)


def run_step(step_name, function, *args, **kwargs):
    """
    Runs a single evaluation step safely.
    """

    print()
    print("-" * 80)
    print(step_name)
    print("-" * 80)

    try:

        function(*args, **kwargs)

        print()
        print(f"[SUCCESS] {step_name}")

        return True

    except Exception as error:

        print()
        print(f"[FAILED] {step_name}")
        print(error)

        return False


def run_full_evaluation():

    print()
    print("=" * 80)
    print("FULL EVALUATION PIPELINE")
    print("=" * 80)

    completed_steps = []

    # --------------------------------------------------
    # STEP 1
    # --------------------------------------------------

    if run_step(
        "STEP 1 : MODEL EVALUATION",
        evaluate
    ):
        completed_steps.append(
            "Model Evaluation"
        )

    # --------------------------------------------------
    # STEP 2
    # --------------------------------------------------

    if run_step(
        "STEP 2 : RECONSTRUCTION GALLERY",
        generate_gallery,
        number_of_samples=5
    ):
        completed_steps.append(
            "Reconstruction Gallery"
        )

    # --------------------------------------------------
    # STEP 3
    # --------------------------------------------------

    if run_step(
        "STEP 3 : UNCERTAINTY ANALYSIS",
        analyze_uncertainty
    ):
        completed_steps.append(
            "Uncertainty Analysis"
        )

    # --------------------------------------------------
    # STEP 4
    # --------------------------------------------------

    if run_step(
        "STEP 4 : BASELINE COMPARISON",
        baseline_comparison
    ):
        completed_steps.append(
            "Baseline Comparison"
        )

    # --------------------------------------------------
    # STEP 5
    # --------------------------------------------------

    if run_step(
        "STEP 5 : STATISTICAL SIGNIFICANCE",
        run_significance_test
    ):
        completed_steps.append(
            "Statistical Significance"
        )

    # --------------------------------------------------
    # STEP 6
    # --------------------------------------------------

    if run_step(
        "STEP 6 : ABLATION STUDY",
        run_ablation
    ):
        completed_steps.append(
            "Ablation Study"
        )

    # --------------------------------------------------
    # STEP 7
    # --------------------------------------------------

    if run_step(
        "STEP 7 : THESIS TABLES",
        generate_thesis_tables
    ):
        completed_steps.append(
            "Thesis Tables"
        )

    # --------------------------------------------------
    # STEP 8
    # --------------------------------------------------

    if run_step(
        "STEP 8 : FINAL REPORT",
        generate_final_report
    ):
        completed_steps.append(
            "Final Report"
        )

    print()
    print("=" * 80)
    print("FULL EVALUATION COMPLETE")
    print("=" * 80)

    print()
    print("Completed Steps:")

    for item in completed_steps:
        print(f"  - {item}")

    print()
    print(
        f"Total Successful Steps: "
        f"{len(completed_steps)}/8"
    )


if __name__ == "__main__":

    run_full_evaluation()