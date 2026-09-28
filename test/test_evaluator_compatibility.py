"""
======================================================================
Evaluator Compatibility and Smoke Test
======================================================================

Purpose
-------
This test verifies compatibility between:

    evaluation/metrics.py
    evaluation/evaluator.py
    models/network.py
    dataset/build_dataset.py

The test is deliberately performed before the full evaluation pipeline
is executed.

Tests performed
---------------
1. EvaluationMetrics API compatibility.
2. Synthetic dataset loading.
3. Network3D forward compatibility.
4. Deterministic Evaluator execution.
5. Reconstruction shape validation.
6. Missing-region metric calculation.
7. Observed-region metric calculation.
8. Observed-data preservation.
9. Numerical finite-value validation.

MC-Dropout testing is intentionally handled separately after the
deterministic evaluator has passed.

Author: Ormin Joseph
======================================================================
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from evaluation.metrics import EvaluationMetrics

from evaluation.evaluator import Evaluator

from models.network import Network3D

from dataset.build_dataset import build_dataset


# =====================================================================
# TEST CONFIGURATION
# =====================================================================

DEVICE = torch.device(
    "cpu"
)

TEST_MC_SAMPLES = 1


# =====================================================================
# DATASET ADAPTER
# =====================================================================

class EvaluationDatasetAdapter(Dataset):
    """
    Convert the project's dataset tuple convention into the dictionary
    convention expected by Evaluator.

    Original dataset output:

        input_cube
        target_cube
        mask
        velocity_model
        mask_type
        geological_mode

    Evaluator convention:

        {
            "input": ...,
            "target": ...,
            "mask": ...
        }
    """

    def __init__(
        self,
        base_dataset
    ):
        self.base_dataset = (
            base_dataset
        )

    def __len__(self):
        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index
    ):
        (
            input_cube,
            target_cube,
            mask,
            velocity_model,
            mask_type,
            geological_mode
        ) = self.base_dataset[
            index
        ]

        return {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
            "velocity": velocity_model,
            "mask_type": mask_type,
            "geological_mode": geological_mode
        }


# =====================================================================
# PASS / FAIL HELPER
# =====================================================================

def report(
    test_name,
    passed
):
    """
    Print a standardized test result.
    """

    if passed:
        print(
            f"[PASS] {test_name}"
        )
    else:
        print(
            f"[FAIL] {test_name}"
        )


# =====================================================================
# TEST 1 — METRICS API
# =====================================================================

def test_metrics_api():
    """
    Verify that every metric required by Evaluator exists.
    """

    required_methods = [
        "mae",
        "mse",
        "rmse",
        "relative_error",
        "snr",
        "psnr",
        "ssim",
        "missing_mae",
        "missing_rmse",
        "observed_mae",
        "observed_rmse",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
    ]

    print()
    print("=" * 70)
    print("TEST 1 — METRICS API COMPATIBILITY")
    print("=" * 70)

    missing_methods = []

    for method_name in required_methods:

        if not hasattr(
            EvaluationMetrics,
            method_name
        ):
            missing_methods.append(
                method_name
            )

    if missing_methods:

        print(
            "Missing metric methods:"
        )

        for method_name in missing_methods:
            print(
                f"  - {method_name}"
            )

        report(
            "Metrics API compatibility",
            False
        )

        return False

    print(
        f"Required methods found: "
        f"{len(required_methods)}"
    )

    print(
        f"DATA_RANGE = "
        f"{EvaluationMetrics.DATA_RANGE}"
    )

    report(
        "Metrics API compatibility",
        True
    )

    return True


# =====================================================================
# TEST 2 — BASIC METRIC CALCULATION
# =====================================================================

def test_basic_metrics():
    """
    Verify that the core metrics return finite values.
    """

    print()
    print("=" * 70)
    print("TEST 2 — BASIC METRIC CALCULATION")
    print("=" * 70)

    target = torch.tensor(
        [[[[[1.0, 0.5],
           [0.0, -0.5]]]]]
    )

    prediction = torch.tensor(
        [[[[[0.8, 0.4],
           [0.1, -0.4]]]]]
    )

    metrics = {
        "MAE":
            EvaluationMetrics.mae(
                prediction,
                target
            ),

        "MSE":
            EvaluationMetrics.mse(
                prediction,
                target
            ),

        "RMSE":
            EvaluationMetrics.rmse(
                prediction,
                target
            ),

        "Relative Error":
            EvaluationMetrics.relative_error(
                prediction,
                target
            ),

        "SNR":
            EvaluationMetrics.snr(
                prediction,
                target
            ),

        "PSNR":
            EvaluationMetrics.psnr(
                prediction,
                target
            ),

        "SSIM":
            EvaluationMetrics.ssim(
                prediction,
                target
            ),
    }

    for name, value in metrics.items():

        print(
            f"{name:18s}: "
            f"{value.item():.6f}"
        )

        if not torch.isfinite(
            value
        ):
            report(
                f"{name} finite-value test",
                False
            )

            return False

    report(
        "Basic metric calculation",
        True
    )

    return True


# =====================================================================
# TEST 3 — REGIONAL METRICS
# =====================================================================

def test_regional_metrics():
    """
    Verify missing-region and observed-region metrics.
    """

    print()
    print("=" * 70)
    print("TEST 3 — REGIONAL METRICS")
    print("=" * 70)

    target = torch.tensor(
        [[[[[1.0, 1.0],
           [1.0, 1.0]]]]]
    )

    prediction = torch.tensor(
        [[[[[1.0, 0.0],
           [0.5, 0.0]]]]]
    )

    # 1 = observed
    # 0 = missing
    mask = torch.tensor(
        [[[[[1.0, 0.0],
           [1.0, 0.0]]]]]
    )

    missing_mae = (
        EvaluationMetrics.missing_mae(
            prediction,
            target,
            mask
        )
    )

    missing_rmse = (
        EvaluationMetrics.missing_rmse(
            prediction,
            target,
            mask
        )
    )

    observed_mae = (
        EvaluationMetrics.observed_mae(
            prediction,
            target,
            mask
        )
    )

    observed_rmse = (
        EvaluationMetrics.observed_rmse(
            prediction,
            target,
            mask
        )
    )

    print(
        f"Missing MAE      : "
        f"{missing_mae.item():.6f}"
    )

    print(
        f"Missing RMSE     : "
        f"{missing_rmse.item():.6f}"
    )

    print(
        f"Observed MAE     : "
        f"{observed_mae.item():.6f}"
    )

    print(
        f"Observed RMSE    : "
        f"{observed_rmse.item():.6f}"
    )

    values = [
        missing_mae,
        missing_rmse,
        observed_mae,
        observed_rmse
    ]

    for value in values:

        if not torch.isfinite(
            value
        ):
            report(
                "Regional metric finite-value test",
                False
            )

            return False

    report(
        "Regional metrics",
        True
    )

    return True


# =====================================================================
# TEST 4 — UNCERTAINTY METRICS
# =====================================================================

def test_uncertainty_metrics():
    """
    Verify aleatoric, epistemic, and predictive uncertainty
    calculations independently of the neural network.
    """

    print()
    print("=" * 70)
    print("TEST 4 — UNCERTAINTY METRICS")
    print("=" * 70)

    log_variance = torch.zeros(
        1,
        1,
        2,
        2,
        2
    )

    aleatoric_variance = (
        EvaluationMetrics.aleatoric_variance(
            log_variance
        )
    )

    reconstruction_samples = torch.stack(
        [
            torch.zeros(
                1,
                1,
                2,
                2,
                2
            ),

            torch.ones(
                1,
                1,
                2,
                2,
                2
            )
        ],
        dim=0
    )

    epistemic_variance = (
        EvaluationMetrics.epistemic_variance(
            reconstruction_samples
        )
    )

    predictive_variance = (
        EvaluationMetrics.predictive_variance(
            aleatoric_variance,
            epistemic_variance
        )
    )

    predictive_std = (
        EvaluationMetrics.predictive_std(
            predictive_variance
        )
    )

    print(
        "Mean aleatoric variance : "
        f"{aleatoric_variance.mean().item():.6f}"
    )

    print(
        "Mean epistemic variance : "
        f"{epistemic_variance.mean().item():.6f}"
    )

    print(
        "Mean predictive variance: "
        f"{predictive_variance.mean().item():.6f}"
    )

    print(
        "Mean predictive std     : "
        f"{predictive_std.mean().item():.6f}"
    )

    tensors = [
        aleatoric_variance,
        epistemic_variance,
        predictive_variance,
        predictive_std
    ]

    for tensor in tensors:

        if not torch.isfinite(
            tensor
        ).all():

            report(
                "Uncertainty metric finite-value test",
                False
            )

            return False

        if torch.any(
            tensor < 0
        ):

            report(
                "Uncertainty non-negativity test",
                False
            )

            return False

    report(
        "Uncertainty metrics",
        True
    )

    return True


# =====================================================================
# TEST 5 — SYNTHETIC DATASET
# =====================================================================

def test_dataset():
    """
    Verify that the configured synthetic dataset can be built.
    """

    print()
    print("=" * 70)
    print("TEST 5 — DATASET COMPATIBILITY")
    print("=" * 70)

    dataset = build_dataset()

    if len(dataset) == 0:

        report(
            "Synthetic dataset loading",
            False
        )

        return None

    print(
        f"Dataset samples: "
        f"{len(dataset)}"
    )

    sample = dataset[0]

    if not isinstance(
        sample,
        (tuple, list)
    ):

        print(
            "Unexpected dataset output type:"
        )

        print(
            type(sample)
        )

        report(
            "Synthetic dataset loading",
            False
        )

        return None

    if len(sample) < 3:

        print(
            "Dataset sample does not contain "
            "input, target, and mask."
        )

        report(
            "Synthetic dataset loading",
            False
        )

        return None

    input_cube = sample[0]
    target_cube = sample[1]
    mask = sample[2]

    print(
        f"Input shape : "
        f"{tuple(input_cube.shape)}"
    )

    print(
        f"Target shape: "
        f"{tuple(target_cube.shape)}"
    )

    print(
        f"Mask shape  : "
        f"{tuple(mask.shape)}"
    )

    if input_cube.shape != target_cube.shape:

        report(
            "Dataset input-target shape compatibility",
            False
        )

        return None

    if input_cube.shape != mask.shape:

        report(
            "Dataset input-mask shape compatibility",
            False
        )

        return None

    if not torch.isfinite(
        input_cube
    ).all():

        report(
            "Dataset finite-value validation",
            False
        )

        return None

    if not torch.isfinite(
        target_cube
    ).all():

        report(
            "Dataset target finite-value validation",
            False
        )

        return None

    if not torch.isfinite(
        mask
    ).all():

        report(
            "Dataset mask finite-value validation",
            False
        )

        return None

    if not torch.all(
        (mask == 0) | (mask == 1)
    ):

        report(
            "Dataset binary-mask validation",
            False
        )

        return None

    report(
        "Synthetic dataset loading",
        True
    )

    return dataset


# =====================================================================
# TEST 6 — DATASET ADAPTER
# =====================================================================

def test_dataset_adapter(
    dataset
):
    """
    Verify conversion from the project's six-element tuple to the
    dictionary expected by Evaluator.
    """

    print()
    print("=" * 70)
    print("TEST 6 — DATASET ADAPTER")
    print("=" * 70)

    adapter = (
        EvaluationDatasetAdapter(
            dataset
        )
    )

    sample = adapter[0]

    required_keys = {
        "input",
        "target",
        "mask"
    }

    if not required_keys.issubset(
        sample.keys()
    ):

        report(
            "Dataset adapter",
            False
        )

        return None

    print(
        "Adapter keys:"
    )

    for key in sample.keys():
        print(
            f"  - {key}"
        )

    report(
        "Dataset adapter",
        True
    )

    return adapter


# =====================================================================
# TEST 7 — NETWORK FORWARD PASS
# =====================================================================

def test_network(
    adapter
):
    """
    Verify Network3D forward compatibility using one real synthetic
    sample.
    """

    print()
    print("=" * 70)
    print("TEST 7 — NETWORK FORWARD COMPATIBILITY")
    print("=" * 70)

    sample = adapter[0]

    input_cube = sample[
        "input"
    ].unsqueeze(
        0
    )

    input_cube = input_cube.to(
        DEVICE
    )

    print(
        f"Network input shape: "
        f"{tuple(input_cube.shape)}"
    )

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        output = model(
            input_cube
        )

    print(
        f"Output type: "
        f"{type(output)}"
    )

    if not isinstance(
        output,
        (tuple, list)
    ):

        print(
            "Network output is not a tuple/list."
        )

        report(
            "Network forward compatibility",
            False
        )

        return None

    print(
        f"Number of outputs: "
        f"{len(output)}"
    )

    for index, item in enumerate(
        output
    ):

        if isinstance(
            item,
            torch.Tensor
        ):

            print(
                f"Output[{index}] shape: "
                f"{tuple(item.shape)}"
            )

            if not torch.isfinite(
                item
            ).all():

                print(
                    f"Output[{index}] contains "
                    "NaN or Inf."
                )

                report(
                    "Network finite-output validation",
                    False
                )

                return None

        else:

            print(
                f"Output[{index}] type: "
                f"{type(item)}"
            )

    if len(output) < 3:

        print(
            "Expected at least three outputs:"
        )

        print(
            "  reconstruction"
        )

        print(
            "  travel_time"
        )

        print(
            "  log_variance"
        )

        report(
            "Network output convention",
            False
        )

        return None

    reconstruction = output[0]
    log_variance = output[2]

    if reconstruction.shape != input_cube.shape:

        print(
            "Reconstruction shape mismatch."
        )

        report(
            "Network reconstruction shape",
            False
        )

        return None

    if log_variance.shape != input_cube.shape:

        print(
            "Log-variance shape mismatch."
        )

        report(
            "Network log-variance shape",
            False
        )

        return None

    report(
        "Network forward compatibility",
        True
    )

    return model


# =====================================================================
# TEST 8 — DETERMINISTIC EVALUATOR
# =====================================================================

def test_deterministic_evaluator(
    model,
    adapter
):
    """
    Run the corrected Evaluator in deterministic mode.
    """

    print()
    print("=" * 70)
    print("TEST 8 — DETERMINISTIC EVALUATOR")
    print("=" * 70)

    dataloader = DataLoader(
        adapter,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    evaluator = Evaluator(
        model=model,
        device=DEVICE,
        mc_samples=TEST_MC_SAMPLES
    )

    results = evaluator.evaluate(
        dataloader
    )

    print()
    print("Evaluation results:")
    print("-" * 70)

    display_keys = [
        "mae",
        "mse",
        "rmse",
        "relative_error",
        "snr",
        "psnr",
        "ssim",
        "missing_mae",
        "missing_rmse",
        "observed_mae",
        "observed_rmse",
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
        "observed_preservation_error",
        "measured_missing_rate",
        "num_samples",
    ]

    for key in display_keys:

        if key in results:

            print(
                f"{key:32s}: "
                f"{results[key]}"
            )

    required_results = [
        "mae",
        "mse",
        "rmse",
        "relative_error",
        "snr",
        "psnr",
        "ssim",
        "missing_mae",
        "missing_rmse",
        "observed_mae",
        "observed_rmse",
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
        "observed_preservation_error",
        "measured_missing_rate",
    ]

    for key in required_results:

        if key not in results:

            print(
                f"Missing evaluator result: "
                f"{key}"
            )

            report(
                "Evaluator result completeness",
                False
            )

            return False

        value = results[key]

        if not isinstance(
            value,
            (int, float)
        ):

            print(
                f"Result '{key}' has unexpected "
                f"type: {type(value)}"
            )

            report(
                "Evaluator result type validation",
                False
            )

            return False

        if isinstance(
            value,
            float
        ):

            if not torch.isfinite(
                torch.tensor(value)
            ):

                print(
                    f"Result '{key}' is not finite."
                )

                report(
                    "Evaluator finite-result validation",
                    False
                )

                return False

    report(
        "Deterministic evaluator",
        True
    )

    return True


# =====================================================================
# MAIN TEST RUNNER
# =====================================================================

def main():
    """
    Run all compatibility tests in sequence.
    """

    print()
    print("=" * 70)
    print(
        "EVALUATOR COMPATIBILITY AND SMOKE TEST"
    )
    print("=" * 70)

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"MC samples: "
        f"{TEST_MC_SAMPLES}"
    )

    print()

    # ---------------------------------------------------------------
    # Test 1
    # ---------------------------------------------------------------

    if not test_metrics_api():
        print()
        print(
            "[STOP] Metrics API is not compatible "
            "with Evaluator."
        )
        return False

    # ---------------------------------------------------------------
    # Test 2
    # ---------------------------------------------------------------

    if not test_basic_metrics():
        print()
        print(
            "[STOP] Basic metrics failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 3
    # ---------------------------------------------------------------

    if not test_regional_metrics():
        print()
        print(
            "[STOP] Regional metrics failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 4
    # ---------------------------------------------------------------

    if not test_uncertainty_metrics():
        print()
        print(
            "[STOP] Uncertainty metrics failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 5
    # ---------------------------------------------------------------

    dataset = test_dataset()

    if dataset is None:
        print()
        print(
            "[STOP] Dataset compatibility failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 6
    # ---------------------------------------------------------------

    adapter = test_dataset_adapter(
        dataset
    )

    if adapter is None:
        print()
        print(
            "[STOP] Dataset adapter failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 7
    # ---------------------------------------------------------------

    model = test_network(
        adapter
    )

    if model is None:
        print()
        print(
            "[STOP] Network compatibility failed."
        )
        return False

    # ---------------------------------------------------------------
    # Test 8
    # ---------------------------------------------------------------

    if not test_deterministic_evaluator(
        model,
        adapter
    ):

        print()
        print(
            "[STOP] Deterministic evaluator failed."
        )
        return False

    # ---------------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ALL DETERMINISTIC EVALUATOR TESTS PASSED"
    )
    print("=" * 70)

    print()
    print(
        "The following interfaces are compatible:"
    )

    print(
        "  Metrics"
    )

    print(
        "      ↓"
    )

    print(
        "  Evaluator"
    )

    print(
        "      ↓"
    )

    print(
        "  Network3D"
    )

    print(
        "      ↓"
    )

    print(
        "  Synthetic Dataset"
    )

    print()
    print(
        "Next step: controlled MC-Dropout "
        "uncertainty test."
    )

    return True


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    success = main()

    if not success:
        raise SystemExit(1)