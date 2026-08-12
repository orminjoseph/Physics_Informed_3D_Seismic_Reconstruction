"""
FINAL REPORT GENERATOR
"""

from evaluation.evaluate_model import evaluate

from evaluation.reconstruction_gallery import (
    generate_gallery
)

from evaluation.uncertainty_analysis import (
    analyze_uncertainty
)

from evaluation.compare_with_baselines import (
    main as compare_baselines
)

from evaluation.statistical_significance import (
    run_significance_test
)

from evaluation.final_thesis_tables import (
    generate_thesis_tables
)

def generate_final_report():

    print()
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Evaluation Metrics
    # --------------------------------------------------

    print()
    print("STEP 1: EVALUATION METRICS")

    evaluate()

    # --------------------------------------------------
    # 2. Reconstruction Gallery
    # --------------------------------------------------

    print()
    print("STEP 2: RECONSTRUCTION GALLERY")

    generate_gallery(
        number_of_samples=5
    )

    # --------------------------------------------------
    # 3. Uncertainty Analysis
    # --------------------------------------------------

    print()
    print("STEP 3: UNCERTAINTY ANALYSIS")

    analyze_uncertainty()

    # --------------------------------------------------
    # 4. Baseline Comparison
    # --------------------------------------------------

    print()
    print("STEP 4: BASELINE COMPARISON")

    compare_baselines()

    # --------------------------------------------------
    # 5. Statistical Significance
    # --------------------------------------------------

    print()
    print("STEP 5: STATISTICAL SIGNIFICANCE")

    run_significance_test()

    # --------------------------------------------------
    # 6. Thesis Tables
    # --------------------------------------------------

    print()
    print("STEP 6: THESIS TABLES")

    generate_thesis_tables()

    print()
    print("=" * 70)
    print("FINAL REPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    generate_final_report()