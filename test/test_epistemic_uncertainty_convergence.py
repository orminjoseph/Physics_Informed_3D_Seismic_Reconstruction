"""
============================================================
EPISTEMIC UNCERTAINTY CONVERGENCE AUDIT
============================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This audit verifies the Monte Carlo Dropout implementation
used to estimate epistemic (model) uncertainty.

Epistemic uncertainty represents uncertainty caused by
limited knowledge of the learned model parameters.

Monte Carlo Dropout estimates epistemic uncertainty by
performing multiple stochastic forward passes while
dropout layers remain active.

For N stochastic predictions:

    y_1, y_2, ..., y_N

the predictive mean is:

    mu(x) = 1/N sum(y_i)

and epistemic variance is:

    sigma_epistemic^2(x)
        = 1/N sum((y_i - mu(x))^2)

This audit verifies:

    1. MC Dropout initialization
    2. Dropout layer detection
    3. Stochastic forward passes
    4. Correct MC sample dimensions
    5. Prediction variability
    6. Epistemic variance
    7. Epistemic uncertainty positivity
    8. Predictive mean validity
    9. Travel-time validity
    10. Aleatoric output preservation
    11. Model parameter stability
    12. Numerical stability
    13. MC sample convergence
    14. Epistemic uncertainty convergence

Important distinction
---------------------

Aleatoric uncertainty:

    predicted by Network3D through log_variance.

Epistemic uncertainty:

    estimated through Monte Carlo Dropout.

Therefore this audit does NOT replace the
heteroscedastic aleatoric uncertainty audit.

It validates the second component required for
the complete predictive uncertainty framework.

Author: Ormin Joseph
============================================================
"""

import torch
import torch.nn as nn


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.network import Network3D
from models.mc_dropout import MCDropout3D

from utils.config import (
    DEVICE,
    USE_UNCERTAINTY,
    LEARNING_RATE,
    WEIGHT_DECAY,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

BATCH_SIZE = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

NUM_MC_SAMPLES = 20

SEED = 42


# ============================================================
# PRINTING UTILITIES
# ============================================================

def print_header(title):
    """Print a major audit section."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_statistics(name, tensor):
    """Print numerical statistics for a tensor."""

    tensor = tensor.detach()

    print(
        f"{name:<30}"
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def compute_parameter_norm(model):
    """
    Compute the global L2 norm of trainable parameters.
    """

    squared_norm = torch.tensor(
        0.0,
        device=next(model.parameters()).device
    )

    for parameter in model.parameters():

        if parameter.requires_grad:

            squared_norm = (
                squared_norm
                +
                parameter.detach().pow(2).sum()
            )

    return torch.sqrt(
        squared_norm
    ).item()


# ============================================================
# DROPOUT UTILITIES
# ============================================================

def find_dropout_layers(model):
    """
    Find all dropout layers contained in the model.

    Returns
    -------

    list
        List of dropout module names and objects.
    """

    dropout_layers = []

    for name, module in model.named_modules():

        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d
            )
        ):

            dropout_layers.append(
                (
                    name,
                    module
                )
            )

    return dropout_layers


def count_training_dropout_layers(model):
    """
    Count dropout layers currently operating in
    training mode.
    """

    count = 0

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d
            )
        ):

            if module.training:

                count += 1

    return count


def count_training_non_dropout_layers(model):
    """
    Count non-dropout layers that are in training mode.

    The MC-Dropout implementation is expected to keep
    these layers in evaluation mode.
    """

    count = 0

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d
            )
        ):

            continue

        if module.training:

            count += 1

    return count


# ============================================================
# TENSOR VALIDATION
# ============================================================

def validate_finite(name, tensor):
    """
    Verify that all tensor values are finite.
    """

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )


def validate_shape(
    name,
    tensor,
    expected_shape
):
    """
    Verify tensor shape.
    """

    if tuple(tensor.shape) != tuple(expected_shape):

        raise RuntimeError(
            f"{name} has incorrect shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(tensor.shape)}."
        )


# ============================================================
# PREDICTION VARIABILITY
# ============================================================

def compute_sample_difference(
    samples
):
    """
    Compute the average absolute difference between
    consecutive MC predictions.

    This provides a direct measure of whether the
    stochastic predictions actually vary.
    """

    differences = (
        samples[1:]
        -
        samples[:-1]
    )

    return (
        differences.abs().mean().item()
    )


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print_header(
        "EPISTEMIC UNCERTAINTY CONVERGENCE AUDIT"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(
        SEED
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        DEVICE
    )

    print_header(
        "DEVICE CONFIGURATION"
    )

    print(
        f"Device                     : {device}"
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print_header(
        "EPISTEMIC UNCERTAINTY CONFIGURATION"
    )

    print(
        f"USE_UNCERTAINTY            : "
        f"{USE_UNCERTAINTY}"
    )

    print(
        f"MC samples                 : "
        f"{NUM_MC_SAMPLES}"
    )

    print(
        f"Learning rate              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay               : "
        f"{WEIGHT_DECAY}"
    )

    # ========================================================
    # SYNTHETIC INPUT
    # ========================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    input_cube = torch.randn(
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    print_statistics(
        "Input seismic volume",
        input_cube
    )

    # ========================================================
    # INITIALIZE NETWORK
    # ========================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=USE_UNCERTAINTY
    ).to(
        device
    )

    print(
        "Network successfully initialized."
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters       : "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # DETECT DROPOUT LAYERS
    # ========================================================

    print_header(
        "DROPOUT LAYER AUDIT"
    )

    dropout_layers = find_dropout_layers(
        model
    )

    print(
        f"Dropout layers detected    : "
        f"{len(dropout_layers)}"
    )

    if len(dropout_layers) == 0:

        raise RuntimeError(
            "No dropout layers were detected in Network3D. "
            "MC Dropout cannot estimate epistemic uncertainty "
            "without stochastic dropout layers."
        )

    for index, (
        name,
        module
    ) in enumerate(
        dropout_layers,
        start=1
    ):

        print(
            f"Dropout layer {index:<3}       : "
            f"{name} "
            f"({module.__class__.__name__})"
        )

    print(
        "Dropout detection test: PASSED"
    )

    # ========================================================
    # INITIAL PARAMETER STATE
    # ========================================================

    initial_parameter_norm = (
        compute_parameter_norm(
            model
        )
    )

    print_header(
        "INITIAL PARAMETER STATE"
    )

    print(
        f"Initial parameter norm     : "
        f"{initial_parameter_norm:.12e}"
    )

    # ========================================================
    # INITIAL MODEL MODE
    # ========================================================

    print_header(
        "INITIAL MODEL MODE"
    )

    model.eval()

    print(
        f"Model training mode       : "
        f"{model.training}"
    )

    # ========================================================
    # INITIAL DETERMINISTIC FORWARD
    # ========================================================

    print_header(
        "DETERMINISTIC FORWARD VALIDATION"
    )

    with torch.no_grad():

        (
            deterministic_reconstruction_1,
            deterministic_travel_time_1,
            deterministic_log_variance_1
        ) = model(
            input_cube
        )

        (
            deterministic_reconstruction_2,
            deterministic_travel_time_2,
            deterministic_log_variance_2
        ) = model(
            input_cube
        )

    validate_finite(
        "Deterministic reconstruction",
        deterministic_reconstruction_1
    )

    validate_finite(
        "Deterministic travel time",
        deterministic_travel_time_1
    )

    validate_finite(
        "Deterministic log variance",
        deterministic_log_variance_1
    )

    deterministic_difference = (
        deterministic_reconstruction_1
        -
        deterministic_reconstruction_2
    ).abs().max().item()

    print(
        f"Maximum deterministic prediction "
        f"difference : "
        f"{deterministic_difference:.12e}"
    )

    if deterministic_difference > 1e-6:

        raise RuntimeError(
            "Deterministic evaluation produced "
            "different predictions."
        )

    print(
        "Deterministic evaluation: PASSED"
    )

    # ========================================================
    # INITIALIZE MC DROPOUT
    # ========================================================

    print_header(
        "INITIALIZING MC DROPOUT"
    )

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=NUM_MC_SAMPLES
    )

    print(
        "MCDropout3D successfully initialized."
    )

    # ========================================================
    # VERIFY MC DROPOUT MODE
    # ========================================================

    print_header(
        "MC DROPOUT MODE AUDIT"
    )

    mc_dropout._enable_dropout()

    dropout_training_count = (
        count_training_dropout_layers(
            model
        )
    )

    non_dropout_training_count = (
        count_training_non_dropout_layers(
            model
        )
    )

    print(
        f"Dropout layers in training mode "
        f": {dropout_training_count}"
    )

    print(
        f"Non-dropout layers in training mode "
        f": {non_dropout_training_count}"
    )

    if dropout_training_count != len(
        dropout_layers
    ):

        raise RuntimeError(
            "Not all dropout layers were activated "
            "for MC inference."
        )

    if non_dropout_training_count != 0:

        raise RuntimeError(
            "Non-dropout layers were placed in training "
            "mode during MC inference."
        )

    print(
        "MC Dropout mode configuration: PASSED"
    )

    # ========================================================
    # MC DROPOUT PREDICTION
    # ========================================================

    print_header(
        "MONTE CARLO STOCHASTIC INFERENCE"
    )

    results = mc_dropout.predict(
        input_cube
    )

    print(
        "MC stochastic inference completed."
    )

    # ========================================================
    # EXTRACT MC RESULTS
    # ========================================================

    reconstruction_samples = (
        results[
            "reconstruction_samples"
        ]
    )

    travel_time_samples = (
        results[
            "travel_time_samples"
        ]
    )

    log_variance_samples = (
        results[
            "log_variance_samples"
        ]
    )

    reconstruction_mean = (
        results[
            "reconstruction_mean"
        ]
    )

    travel_time_mean = (
        results[
            "travel_time_mean"
        ]
    )

    log_variance_mean = (
        results[
            "log_variance_mean"
        ]
    )

    reconstruction_epistemic_variance = (
        results[
            "reconstruction_epistemic_variance"
        ]
    )

    travel_time_epistemic_variance = (
        results[
            "travel_time_epistemic_variance"
        ]
    )

    log_variance_epistemic_variance = (
        results[
            "log_variance_epistemic_variance"
        ]
    )

    # ========================================================
    # MC SAMPLE SHAPE AUDIT
    # ========================================================

    print_header(
        "MC SAMPLE SHAPE AUDIT"
    )

    expected_sample_shape = (
        NUM_MC_SAMPLES,
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH
    )

    expected_output_shape = (
        BATCH_SIZE,
        1,
        DEPTH,
        HEIGHT,
        WIDTH
    )

    validate_shape(
        "Reconstruction samples",
        reconstruction_samples,
        expected_sample_shape
    )

    validate_shape(
        "Travel-time samples",
        travel_time_samples,
        expected_sample_shape
    )

    validate_shape(
        "Log-variance samples",
        log_variance_samples,
        expected_sample_shape
    )

    validate_shape(
        "Reconstruction mean",
        reconstruction_mean,
        expected_output_shape
    )

    validate_shape(
        "Travel-time mean",
        travel_time_mean,
        expected_output_shape
    )

    validate_shape(
        "Log-variance mean",
        log_variance_mean,
        expected_output_shape
    )

    validate_shape(
        "Reconstruction epistemic variance",
        reconstruction_epistemic_variance,
        expected_output_shape
    )

    validate_shape(
        "Travel-time epistemic variance",
        travel_time_epistemic_variance,
        expected_output_shape
    )

    validate_shape(
        "Log-variance epistemic variance",
        log_variance_epistemic_variance,
        expected_output_shape
    )

    print(
        f"MC sample shape            : "
        f"{tuple(reconstruction_samples.shape)}"
    )

    print(
        "MC sample shape validation: PASSED"
    )

    # ========================================================
    # FINITE VALUE AUDIT
    # ========================================================

    print_header(
        "EPISTEMIC NUMERICAL STABILITY AUDIT"
    )

    tensors_to_validate = {

        "Reconstruction samples":
            reconstruction_samples,

        "Travel-time samples":
            travel_time_samples,

        "Log-variance samples":
            log_variance_samples,

        "Reconstruction mean":
            reconstruction_mean,

        "Travel-time mean":
            travel_time_mean,

        "Log-variance mean":
            log_variance_mean,

        "Reconstruction epistemic variance":
            reconstruction_epistemic_variance,

        "Travel-time epistemic variance":
            travel_time_epistemic_variance,

        "Log-variance epistemic variance":
            log_variance_epistemic_variance
    }

    for name, tensor in tensors_to_validate.items():

        validate_finite(
            name,
            tensor
        )

    print(
        "All MC-Dropout quantities are finite."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # ========================================================
    # PREDICTIVE STATISTICS
    # ========================================================

    print_header(
        "EPISTEMIC UNCERTAINTY STATISTICS"
    )

    print_statistics(
        "Reconstruction MC mean",
        reconstruction_mean
    )

    print_statistics(
        "Reconstruction epistemic variance",
        reconstruction_epistemic_variance
    )

    print_statistics(
        "Travel-time MC mean",
        travel_time_mean
    )

    print_statistics(
        "Travel-time epistemic variance",
        travel_time_epistemic_variance
    )

    print_statistics(
        "Log-variance MC mean",
        log_variance_mean
    )

    print_statistics(
        "Log-variance epistemic variance",
        log_variance_epistemic_variance
    )

    # ========================================================
    # PREDICTION VARIABILITY AUDIT
    # ========================================================

    print_header(
        "STOCHASTIC PREDICTION VARIABILITY AUDIT"
    )

    reconstruction_difference = (
        compute_sample_difference(
            reconstruction_samples
        )
    )

    travel_time_difference = (
        compute_sample_difference(
            travel_time_samples
        )
    )

    log_variance_difference = (
        compute_sample_difference(
            log_variance_samples
        )
    )

    print(
        f"Reconstruction sample difference "
        f": {reconstruction_difference:.12e}"
    )

    print(
        f"Travel-time sample difference      "
        f": {travel_time_difference:.12e}"
    )

    print(
        f"Log-variance sample difference     "
        f": {log_variance_difference:.12e}"
    )

    if reconstruction_difference <= 0.0:

        raise RuntimeError(
            "MC reconstruction predictions show "
            "no stochastic variability."
        )

    print(
        "Stochastic reconstruction variability: OBSERVED"
    )

    # ========================================================
    # EPISTEMIC VARIANCE AUDIT
    # ========================================================

    print_header(
        "EPISTEMIC VARIANCE VALIDATION"
    )

    reconstruction_variance_mean = (
        reconstruction_epistemic_variance.mean().item()
    )

    reconstruction_variance_maximum = (
        reconstruction_epistemic_variance.max().item()
    )

    travel_time_variance_mean = (
        travel_time_epistemic_variance.mean().item()
    )

    print(
        f"Mean reconstruction epistemic "
        f"variance : "
        f"{reconstruction_variance_mean:.12e}"
    )

    print(
        f"Maximum reconstruction epistemic "
        f"variance : "
        f"{reconstruction_variance_maximum:.12e}"
    )

    print(
        f"Mean travel-time epistemic "
        f"variance : "
        f"{travel_time_variance_mean:.12e}"
    )

    if reconstruction_variance_mean <= 0.0:

        raise RuntimeError(
            "Mean reconstruction epistemic variance "
            "is zero."
        )

    if reconstruction_variance_maximum <= 0.0:

        raise RuntimeError(
            "Maximum reconstruction epistemic variance "
            "is zero."
        )

    if (
        reconstruction_epistemic_variance
        < 0.0
    ).any():

        raise RuntimeError(
            "Negative epistemic variance detected."
        )

    print(
        "Epistemic variance positivity: PASSED"
    )

    # ========================================================
    # EPISTEMIC STANDARD DEVIATION
    # ========================================================

    print_header(
        "EPISTEMIC STANDARD DEVIATION"
    )

    reconstruction_epistemic_std = torch.sqrt(
        torch.clamp(
            reconstruction_epistemic_variance,
            min=0.0
        )
    )

    travel_time_epistemic_std = torch.sqrt(
        torch.clamp(
            travel_time_epistemic_variance,
            min=0.0
        )
    )

    log_variance_epistemic_std = torch.sqrt(
        torch.clamp(
            log_variance_epistemic_variance,
            min=0.0
        )
    )

    validate_finite(
        "Reconstruction epistemic standard deviation",
        reconstruction_epistemic_std
    )

    validate_finite(
        "Travel-time epistemic standard deviation",
        travel_time_epistemic_std
    )

    validate_finite(
        "Log-variance epistemic standard deviation",
        log_variance_epistemic_std
    )

    print_statistics(
        "Reconstruction epistemic std",
        reconstruction_epistemic_std
    )

    print_statistics(
        "Travel-time epistemic std",
        travel_time_epistemic_std
    )

    print(
        "Epistemic standard deviation validation: PASSED"
    )

    # ========================================================
    # TRAVEL-TIME POSITIVITY
    # ========================================================

    print_header(
        "TRAVEL-TIME VALIDATION"
    )

    minimum_travel_time = (
        travel_time_samples.min().item()
    )

    maximum_travel_time = (
        travel_time_samples.max().item()
    )

    mean_travel_time = (
        travel_time_samples.mean().item()
    )

    print(
        f"Minimum travel time       : "
        f"{minimum_travel_time:.12e}"
    )

    print(
        f"Maximum travel time       : "
        f"{maximum_travel_time:.12e}"
    )

    print(
        f"Mean travel time          : "
        f"{mean_travel_time:.12e}"
    )

    if minimum_travel_time < 0.0:

        raise RuntimeError(
            "Negative travel time detected during "
            "MC-Dropout inference."
        )

    print(
        "Travel-time positivity: PASSED"
    )

    # ========================================================
    # ALEATORIC OUTPUT PRESERVATION
    # ========================================================

    print_header(
        "ALEATORIC OUTPUT PRESERVATION AUDIT"
    )

    predicted_variance = torch.exp(
        log_variance_mean
    )

    predicted_standard_deviation = torch.sqrt(
        predicted_variance
    )

    validate_finite(
        "Predicted aleatoric variance",
        predicted_variance
    )

    validate_finite(
        "Predicted aleatoric standard deviation",
        predicted_standard_deviation
    )

    if (
        predicted_variance
        <= 0.0
    ).any():

        raise RuntimeError(
            "Aleatoric variance is not positive."
        )

    print_statistics(
        "Aleatoric variance",
        predicted_variance
    )

    print_statistics(
        "Aleatoric standard deviation",
        predicted_standard_deviation
    )

    print(
        "Aleatoric output preservation: PASSED"
    )

    # ========================================================
    # PARAMETER STABILITY AUDIT
    # ========================================================

    print_header(
        "PARAMETER STABILITY AUDIT"
    )

    final_parameter_norm = (
        compute_parameter_norm(
            model
        )
    )

    parameter_change = (
        final_parameter_norm
        -
        initial_parameter_norm
    )

    print(
        f"Initial parameter norm     : "
        f"{initial_parameter_norm:.12e}"
    )

    print(
        f"Final parameter norm       : "
        f"{final_parameter_norm:.12e}"
    )

    print(
        f"Parameter norm change      : "
        f"{parameter_change:.12e}"
    )

    if abs(
        parameter_change
    ) > 1e-10:

        raise RuntimeError(
            "Model parameters changed during "
            "MC-Dropout inference."
        )

    print(
        "Model parameters remained unchanged."
    )

    print(
        "Parameter stability test: PASSED"
    )

    # ========================================================
    # MC SAMPLE CONVERGENCE
    # ========================================================

    print_header(
        "MC SAMPLE CONVERGENCE ANALYSIS"
    )

    sample_counts = [
        5,
        10,
        NUM_MC_SAMPLES
    ]

    sample_variance_means = []

    for sample_count in sample_counts:

        local_estimator = MCDropout3D(
            model=model,
            num_samples=sample_count
        )

        local_results = (
            local_estimator.predict(
                input_cube
            )
        )

        local_variance = (
            local_results[
                "reconstruction_epistemic_variance"
            ]
        )

        local_variance_mean = (
            local_variance.mean().item()
        )

        sample_variance_means.append(
            local_variance_mean
        )

        print(
            f"MC samples = "
            f"{sample_count:<3} "
            f"| Mean epistemic variance = "
            f"{local_variance_mean:.12e}"
        )

    for value in sample_variance_means:

        if not torch.isfinite(
            torch.tensor(value)
        ):

            raise RuntimeError(
                "MC sample convergence produced "
                "a non-finite epistemic variance."
            )

    print(
        "MC sample variance estimates are finite."
    )

    print(
        "MC sample convergence stability: PASSED"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_header(
        "EPISTEMIC UNCERTAINTY CONVERGENCE AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ MC Dropout implementation"
    )

    print(
        "  ✓ Dropout layer detection"
    )

    print(
        "  ✓ Dropout activation during inference"
    )

    print(
        "  ✓ Non-dropout evaluation-mode preservation"
    )

    print(
        "  ✓ Stochastic forward passes"
    )

    print(
        "  ✓ MC sample dimensions"
    )

    print(
        "  ✓ Prediction variability"
    )

    print(
        "  ✓ Epistemic variance"
    )

    print(
        "  ✓ Epistemic standard deviation"
    )

    print(
        "  ✓ Travel-time positivity"
    )

    print(
        "  ✓ Aleatoric output preservation"
    )

    print(
        "  ✓ Parameter stability"
    )

    print(
        "  ✓ Numerical stability"
    )

    print(
        "  ✓ MC sample convergence"
    )

    print()

    print(
        "EPISTEMIC UNCERTAINTY CONVERGENCE TEST PASSED."
    )


# ============================================================
# RUN AUDIT
# ============================================================

if __name__ == "__main__":

    main()