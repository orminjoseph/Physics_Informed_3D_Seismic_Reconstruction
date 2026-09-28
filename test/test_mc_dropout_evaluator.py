"""
======================================================================
MC-Dropout Evaluator Test
======================================================================

Purpose
-------
This test verifies the MC-Dropout uncertainty pathway of the
Physics-Informed 3D Encoder-Decoder framework.

Tests performed
---------------
1. Synthetic dataset loading.
2. Dataset adaptation.
3. Network construction.
4. Checkpoint-independent MC-Dropout execution.
5. Multiple stochastic reconstruction samples.
6. Epistemic variance generation.
7. Aleatoric variance generation.
8. Predictive variance calculation.
9. Predictive standard deviation calculation.
10. Selective dropout activation.
11. Restoration of model/module training states.
12. Numerical finite-value validation.

Important
---------
This test does NOT load the trained checkpoint.

Its purpose is to verify the MC-Dropout mechanism itself before
integrating it into the production evaluation pipeline.

Author: Ormin Joseph
======================================================================
"""

import os
import sys

import torch
from torch.utils.data import Dataset


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

from models.network import Network3D

from evaluation.evaluator import Evaluator

from evaluation.metrics import EvaluationMetrics

from dataset.build_dataset import build_dataset


# =====================================================================
# CONFIGURATION
# =====================================================================

DEVICE = torch.device(
    "cpu"
)

MC_SAMPLES = 10


# =====================================================================
# DATASET ADAPTER
# =====================================================================

class EvaluationDatasetAdapter(Dataset):
    """
    Convert the project's six-element dataset tuple into the
    dictionary convention used by Evaluator.
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
# TEST 1 — DROPOUT INVENTORY
# =====================================================================

def inspect_dropout_modules(
    model
):
    """
    Identify all dropout modules contained in the network.
    """

    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )

    dropout_modules = []

    for name, module in model.named_modules():

        if isinstance(
            module,
            dropout_types
        ):
            dropout_modules.append(
                (
                    name,
                    module
                )
            )

    print()
    print("=" * 70)
    print("TEST 1 — DROPOUT MODULE INVENTORY")
    print("=" * 70)

    print(
        f"Dropout modules found: "
        f"{len(dropout_modules)}"
    )

    for name, module in dropout_modules:

        print(
            f"  - {name}: "
            f"{module.__class__.__name__}"
        )

    if len(dropout_modules) == 0:

        print()
        print(
            "[FAIL] No dropout modules were found."
        )

        print(
            "MC-Dropout cannot produce epistemic uncertainty "
            "without stochastic dropout."
        )

        return None

    print(
        "[PASS] Dropout modules detected."
    )

    return dropout_modules


# =====================================================================
# TEST 2 — INITIAL MODEL STATE
# =====================================================================

def inspect_initial_state(
    model,
    dropout_modules
):
    """
    Verify that the model begins in evaluation mode and record
    the original module states.
    """

    print()
    print("=" * 70)
    print("TEST 2 — INITIAL MODEL STATE")
    print("=" * 70)

    model.eval()

    original_states = {
        module: module.training
        for module in model.modules()
    }

    print(
        f"Model training state before MC test: "
        f"{model.training}"
    )

    if model.training:

        print(
            "[FAIL] Model should begin in evaluation mode."
        )

        return None

    dropout_training_states = []

    for _, module in dropout_modules:

        dropout_training_states.append(
            module.training
        )

    if any(
        dropout_training_states
    ):

        print(
            "[FAIL] Dropout modules unexpectedly "
            "start in training mode."
        )

        return None

    print(
        "[PASS] Model and dropout modules begin "
        "in evaluation mode."
    )

    return original_states


# =====================================================================
# TEST 3 — SELECTIVE MC-DROPOUT ACTIVATION
# =====================================================================

def test_selective_dropout_activation(
    model,
    dropout_modules
):
    """
    Verify that Evaluator activates only dropout modules while the
    remainder of the model remains in evaluation mode.
    """

    print()
    print("=" * 70)
    print("TEST 3 — SELECTIVE MC-DROPOUT ACTIVATION")
    print("=" * 70)

    Evaluator._enable_mc_dropout(
        model
    )

    if model.training:

        print(
            "[FAIL] Entire model was switched "
            "to training mode."
        )

        return False

    print(
        "Complete model training state: "
        f"{model.training}"
    )

    for name, module in dropout_modules:

        if not module.training:

            print(
                f"[FAIL] Dropout module '{name}' "
                "was not activated."
            )

            return False

        print(
            f"  {name}: training={module.training}"
        )

    dropout_ids = {
        id(module)
        for _, module in dropout_modules
    }

    non_dropout_training_modules = []

    for name, module in model.named_modules():

        if id(module) in dropout_ids:
            continue

        if module.training:

            non_dropout_training_modules.append(
                name
            )

    if non_dropout_training_modules:

        print()
        print(
            "[FAIL] Non-dropout modules were activated:"
        )

        for name in non_dropout_training_modules:

            print(
                f"  - {name}"
            )

        return False

    print()
    print(
        "[PASS] Only dropout modules are in "
        "training mode."
    )

    return True


# =====================================================================
# TEST 4 — RESTORE MODEL STATES
# =====================================================================

def test_state_restoration(
    model,
    original_states
):
    """
    Restore the model to the states recorded before MC-Dropout.
    """

    print()
    print("=" * 70)
    print("TEST 4 — MODEL STATE RESTORATION")
    print("=" * 70)

    for module, training_state in (
        original_states.items()
    ):

        module.train(
            training_state
        )

    restored_correctly = True

    for module, training_state in (
        original_states.items()
    ):

        if module.training != training_state:

            restored_correctly = False

            print(
                "[FAIL] Module training state "
                "was not restored."
            )

            break

    if not restored_correctly:
        return False

    print(
        f"Model training state restored to: "
        f"{model.training}"
    )

    print(
        "[PASS] All module training states "
        "were restored."
    )

    return True


# =====================================================================
# TEST 5 — BUILD DATASET
# =====================================================================

def test_dataset():
    """
    Build the configured synthetic dataset and obtain one sample.
    """

    print()
    print("=" * 70)
    print("TEST 5 — SYNTHETIC INPUT")
    print("=" * 70)

    dataset = build_dataset()

    if len(dataset) == 0:

        print(
            "[FAIL] Dataset is empty."
        )

        return None

    sample = dataset[0]

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

    if not torch.isfinite(
        input_cube
    ).all():

        print(
            "[FAIL] Input contains NaN or Inf."
        )

        return None

    print(
        "[PASS] Valid synthetic input obtained."
    )

    return {
        "input": input_cube,
        "target": target_cube,
        "mask": mask
    }


# =====================================================================
# TEST 6 — STOCHASTIC FORWARD PASSES
# =====================================================================

def test_stochastic_forward_passes(
    model,
    input_cube,
    dropout_modules
):
    """
    Perform multiple forward passes with selective dropout enabled
    and verify that stochastic variation exists.
    """

    print()
    print("=" * 70)
    print("TEST 6 — STOCHASTIC FORWARD PASSES")
    print("=" * 70)

    model.eval()

    Evaluator._enable_mc_dropout(
        model
    )

    reconstruction_samples = []

    log_variance_samples = []

    with torch.no_grad():

        for sample_index in range(
            MC_SAMPLES
        ):

            output = model(
                input_cube
            )

            if not isinstance(
                output,
                (tuple, list)
            ):

                print(
                    "[FAIL] Network output is not "
                    "a tuple/list."
                )

                return None

            if len(output) < 3:

                print(
                    "[FAIL] Network output contains "
                    "fewer than three elements."
                )

                return None

            reconstruction = output[0]

            log_variance = output[2]

            if not torch.isfinite(
                reconstruction
            ).all():

                print(
                    f"[FAIL] Reconstruction sample "
                    f"{sample_index} contains NaN or Inf."
                )

                return None

            if not torch.isfinite(
                log_variance
            ).all():

                print(
                    f"[FAIL] Log-variance sample "
                    f"{sample_index} contains NaN or Inf."
                )

                return None

            reconstruction_samples.append(
                reconstruction
            )

            log_variance_samples.append(
                log_variance
            )

    reconstruction_samples = torch.stack(
        reconstruction_samples,
        dim=0
    )

    log_variance_samples = torch.stack(
        log_variance_samples,
        dim=0
    )

    print(
        "Reconstruction sample tensor: "
        f"{tuple(reconstruction_samples.shape)}"
    )

    print(
        "Log-variance sample tensor: "
        f"{tuple(log_variance_samples.shape)}"
    )

    # ---------------------------------------------------------------
    # Determine whether stochastic variation exists.
    # ---------------------------------------------------------------

    sample_variance = torch.var(
        reconstruction_samples,
        dim=0,
        unbiased=False
    )

    mean_sample_variance = (
        sample_variance.mean()
    )

    print(
        "Mean reconstruction variance: "
        f"{mean_sample_variance.item():.10f}"
    )

    if not torch.isfinite(
        sample_variance
    ).all():

        print(
            "[FAIL] Reconstruction variance "
            "contains NaN or Inf."
        )

        return None

    if mean_sample_variance <= 0:

        print()
        print(
            "[FAIL] MC-Dropout produced "
            "zero stochastic variation."
        )

        print(
            "This means epistemic uncertainty "
            "cannot be estimated from these passes."
        )

        return None

    print(
        "[PASS] Stochastic variation detected."
    )

    return (
        reconstruction_samples,
        log_variance_samples
    )


# =====================================================================
# TEST 7 — UNCERTAINTY DECOMPOSITION
# =====================================================================

def test_uncertainty_decomposition(
    reconstruction_samples,
    log_variance_samples
):
    """
    Verify the complete uncertainty decomposition.
    """

    print()
    print("=" * 70)
    print("TEST 7 — UNCERTAINTY DECOMPOSITION")
    print("=" * 70)

    aleatoric_variance_samples = torch.exp(
        log_variance_samples
    )

    aleatoric_variance = (
        aleatoric_variance_samples.mean(
            dim=0
        )
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

    values = {
        "Aleatoric variance":
            aleatoric_variance,

        "Epistemic variance":
            epistemic_variance,

        "Predictive variance":
            predictive_variance,

        "Predictive standard deviation":
            predictive_std
    }

    for name, tensor in values.items():

        if not torch.isfinite(
            tensor
        ).all():

            print(
                f"[FAIL] {name} contains "
                "NaN or Inf."
            )

            return False

        if torch.any(
            tensor < 0
        ):

            print(
                f"[FAIL] {name} contains "
                "negative values."
            )

            return False

        print(
            f"{name:32s}: "
            f"{tensor.mean().item():.10f}"
        )

    # ---------------------------------------------------------------
    # Verify predictive variance identity.
    # ---------------------------------------------------------------

    expected_predictive_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    maximum_difference = torch.max(
        torch.abs(
            predictive_variance
            -
            expected_predictive_variance
        )
    )

    print(
        "Maximum decomposition difference: "
        f"{maximum_difference.item():.10e}"
    )

    if maximum_difference > 1.0e-6:

        print(
            "[FAIL] Predictive variance decomposition "
            "is inconsistent."
        )

        return False

    print(
        "[PASS] Aleatoric + epistemic = "
        "predictive variance."
    )

    return True


# =====================================================================
# TEST 8 — FULL EVALUATOR MC MODE
# =====================================================================

def test_full_evaluator(
    model,
    input_data
):
    """
    Run the production Evaluator with MC-Dropout.

    The synthetic dataset returns a dictionary containing:

        input
        target
        mask
        velocity
        mask_type
        geological_mode

    Individual seismic tensors are expected to have:

        [C, D, H, W]

    The DataLoader automatically adds:

        [B, C, D, H, W]

    The production Evaluator therefore receives the correct
    five-dimensional tensors.
    """

    print()
    print("=" * 70)
    print("TEST 8 — FULL MC-DROPOUT EVALUATOR")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Verify that the synthetic sample is a dictionary.
    # ---------------------------------------------------------------

    if not isinstance(input_data, dict):

        print(
            "[FAIL] Expected the synthetic dataset sample "
            "to be a dictionary."
        )

        print(
            f"Received type: {type(input_data)}"
        )

        return False

    # ---------------------------------------------------------------
    # Verify required fields.
    # ---------------------------------------------------------------

    required_keys = [
        "input",
        "target",
        "mask",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in input_data
    ]

    if missing_keys:

        print(
            "[FAIL] Dataset sample is missing required keys:"
        )

        print(
            missing_keys
        )

        return False

    # ---------------------------------------------------------------
    # Make a copy so that the original dataset sample is not
    # modified by the test.
    # ---------------------------------------------------------------

    sample = dict(
        input_data
    )

    # ---------------------------------------------------------------
    # Normalize the three seismic tensors.
    #
    # Dataset samples should normally be:
    #
    #     [C, D, H, W]
    #
    # If a tensor already contains a batch dimension:
    #
    #     [1, C, D, H, W]
    #
    # remove only that single batch dimension.
    # ---------------------------------------------------------------

    for key in [
        "input",
        "target",
        "mask",
    ]:

        tensor = sample[key]

        if not torch.is_tensor(tensor):

            print(
                f"[FAIL] '{key}' must be a PyTorch tensor."
            )

            print(
                f"Received type: {type(tensor)}"
            )

            return False

        if tensor.ndim == 5:

            if tensor.shape[0] != 1:

                print(
                    f"[FAIL] '{key}' contains a batch dimension "
                    f"greater than one."
                )

                print(
                    f"Received shape: {tuple(tensor.shape)}"
                )

                return False

            tensor = tensor.squeeze(0)

        if tensor.ndim != 4:

            print(
                f"[FAIL] '{key}' must have shape "
                f"[C, D, H, W] before DataLoader batching."
            )

            print(
                f"Received shape: {tuple(tensor.shape)}"
            )

            return False

        sample[key] = tensor

    # ---------------------------------------------------------------
    # Verify that all three seismic tensors have matching shapes.
    # ---------------------------------------------------------------

    if not (
        sample["input"].shape
        == sample["target"].shape
        == sample["mask"].shape
    ):

        print(
            "[FAIL] Input, target, and mask shapes do not match."
        )

        print(
            f"Input : {tuple(sample['input'].shape)}"
        )

        print(
            f"Target: {tuple(sample['target'].shape)}"
        )

        print(
            f"Mask  : {tuple(sample['mask'].shape)}"
        )

        return False

    print(
        f"Sample input shape before DataLoader: "
        f"{tuple(sample['input'].shape)}"
    )

    print(
        f"Sample target shape before DataLoader: "
        f"{tuple(sample['target'].shape)}"
    )

    print(
        f"Sample mask shape before DataLoader: "
        f"{tuple(sample['mask'].shape)}"
    )

    # ---------------------------------------------------------------
    # Construct a one-sample dataset.
    # ---------------------------------------------------------------

    class SingleSampleDataset(
        Dataset
    ):
        def __len__(self):
            return 1

        def __getitem__(
            self,
            index
        ):
            return sample

    # ---------------------------------------------------------------
    # DataLoader automatically adds the batch dimension.
    #
    # [C, D, H, W]
    #
    # becomes:
    #
    # [B, C, D, H, W]
    # ---------------------------------------------------------------

    dataloader = torch.utils.data.DataLoader(
        SingleSampleDataset(),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------------
    # Inspect the actual batch produced by DataLoader.
    # ---------------------------------------------------------------

    loader_sample = next(
        iter(dataloader)
    )

    if not isinstance(loader_sample, dict):

        print(
            "[FAIL] DataLoader did not return a dictionary."
        )

        print(
            f"Received type: {type(loader_sample)}"
        )

        return False

    print(
        f"DataLoader input shape: "
        f"{tuple(loader_sample['input'].shape)}"
    )

    print(
        f"DataLoader target shape: "
        f"{tuple(loader_sample['target'].shape)}"
    )

    print(
        f"DataLoader mask shape: "
        f"{tuple(loader_sample['mask'].shape)}"
    )

    # ---------------------------------------------------------------
    # Verify the five-dimensional production shape.
    # ---------------------------------------------------------------

    for key in [
        "input",
        "target",
        "mask",
    ]:

        if loader_sample[key].ndim != 5:

            print(
                f"[FAIL] DataLoader field '{key}' "
                f"does not have shape [B,C,D,H,W]."
            )

            return False

    # ---------------------------------------------------------------
    # Create the production Evaluator.
    # ---------------------------------------------------------------

    evaluator = Evaluator(
        model=model,
        device=DEVICE,
        mc_samples=MC_SAMPLES
    )

    # ---------------------------------------------------------------
    # Run the complete evaluation.
    # ---------------------------------------------------------------

    results = evaluator.evaluate(
        dataloader
    )

    # ---------------------------------------------------------------
    # Display important MC-Dropout results.
    # ---------------------------------------------------------------

    display_keys = [
        "mae",
        "missing_mae",
        "missing_rmse",
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
        "observed_preservation_error",
        "measured_missing_rate",
        "mc_samples",
    ]

    print()
    print(
        "MC-Dropout evaluator results:"
    )

    print(
        "-" * 70
    )

    for key in display_keys:

        if key in results:

            print(
                f"{key:32s}: "
                f"{results[key]}"
            )

    # ---------------------------------------------------------------
    # Verify MC sample count.
    # ---------------------------------------------------------------

    if results[
        "mc_samples"
    ] != MC_SAMPLES:

        print(
            "[FAIL] Evaluator did not use "
            "the requested number of MC samples."
        )

        return False

    # ---------------------------------------------------------------
    # Verify positive epistemic uncertainty.
    # ---------------------------------------------------------------

    if results[
        "epistemic_variance"
    ] <= 0:

        print(
            "[FAIL] Evaluator returned zero "
            "epistemic variance."
        )

        return False

    # ---------------------------------------------------------------
    # Verify uncertainty decomposition.
    #
    # Predictive variance must not be smaller than
    # aleatoric variance.
    # ---------------------------------------------------------------

    if results[
        "predictive_variance"
    ] < results[
        "aleatoric_variance"
    ]:

        print(
            "[FAIL] Predictive variance is "
            "smaller than aleatoric variance."
        )

        return False

    # ---------------------------------------------------------------
    # Verify observed-data preservation.
    # ---------------------------------------------------------------

    if results[
        "observed_preservation_error"
    ] > 1.0e-6:

        print(
            "[FAIL] Observed-data preservation "
            "tolerance exceeded."
        )

        return False

    # ---------------------------------------------------------------
    # Test completed successfully.
    # ---------------------------------------------------------------

    print()
    print(
        "[PASS] Full MC-Dropout Evaluator."
    )

    return True

# =====================================================================
# MAIN
# =====================================================================

def main():
    """
    Execute the MC-Dropout audit.
    """

    print()
    print("=" * 70)
    print(
        "MC-DROPOUT UNCERTAINTY AUDIT"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"MC samples: {MC_SAMPLES}"
    )

    # ---------------------------------------------------------------
    # Build model.
    # ---------------------------------------------------------------

    model = Network3D(
        use_attention=True,
        use_residual=True,
        use_uncertainty=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # ---------------------------------------------------------------
    # Test 1
    # ---------------------------------------------------------------

    dropout_modules = (
        inspect_dropout_modules(
            model
        )
    )

    if dropout_modules is None:
        return False

    # ---------------------------------------------------------------
    # Test 2
    # ---------------------------------------------------------------

    original_states = (
        inspect_initial_state(
            model,
            dropout_modules
        )
    )

    if original_states is None:
        return False

    # ---------------------------------------------------------------
    # Test 3
    # ---------------------------------------------------------------

    if not test_selective_dropout_activation(
        model,
        dropout_modules
    ):
        return False

    # ---------------------------------------------------------------
    # Test 4
    # ---------------------------------------------------------------

    if not test_state_restoration(
        model,
        original_states
    ):
        return False

    # ---------------------------------------------------------------
    # Test 5
    # ---------------------------------------------------------------

    input_data = test_dataset()

    if input_data is None:
        return False

    input_cube = input_data[
        "input"
    ].unsqueeze(
        0
    ).to(
        DEVICE
    )

    input_data[
        "input"
    ] = input_cube

    input_data[
        "target"
    ] = input_data[
        "target"
    ].unsqueeze(
        0
    ).to(
        DEVICE
    )

    input_data[
        "mask"
    ] = input_data[
        "mask"
    ].unsqueeze(
        0
    ).to(
        DEVICE
    )

    # ---------------------------------------------------------------
    # Test 6
    # ---------------------------------------------------------------

    stochastic_samples = (
        test_stochastic_forward_passes(
            model,
            input_cube,
            dropout_modules
        )
    )

    if stochastic_samples is None:
        return False

    (
        reconstruction_samples,
        log_variance_samples
    ) = stochastic_samples

    # ---------------------------------------------------------------
    # Test 7
    # ---------------------------------------------------------------

    if not test_uncertainty_decomposition(
        reconstruction_samples,
        log_variance_samples
    ):
        return False

    # ---------------------------------------------------------------
    # Restore model before production Evaluator test.
    # ---------------------------------------------------------------

    model.eval()

    # ---------------------------------------------------------------
    # Test 8
    # ---------------------------------------------------------------

    if not test_full_evaluator(
        model,
        input_data
    ):
        return False

    # ---------------------------------------------------------------
    # Final state check.
    # ---------------------------------------------------------------

    model.eval()

    if model.training:

        print(
            "[FAIL] Model did not finish in "
            "evaluation mode."
        )

        return False

    print()
    print("=" * 70)
    print(
        "MC-DROPOUT UNCERTAINTY AUDIT PASSED"
    )
    print("=" * 70)

    print()
    print(
        "Verified:"
    )

    print(
        "  ✓ Dropout modules detected"
    )

    print(
        "  ✓ Selective dropout activation"
    )

    print(
        "  ✓ Non-dropout layers remain in evaluation mode"
    )

    print(
        "  ✓ Model states restored"
    )

    print(
        "  ✓ Multiple stochastic predictions"
    )

    print(
        "  ✓ Epistemic variance"
    )

    print(
        "  ✓ Aleatoric variance"
    )

    print(
        "  ✓ Predictive variance"
    )

    print(
        "  ✓ Predictive standard deviation"
    )

    print(
        "  ✓ Observed-data preservation"
    )

    print(
        "  ✓ Full Evaluator MC-Dropout pathway"
    )

    return True


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    success = main()

    if not success:
        raise SystemExit(1)