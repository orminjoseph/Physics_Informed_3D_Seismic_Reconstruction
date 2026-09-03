"""
============================================================
FULL PREDICTIVE UNCERTAINTY AUDIT
============================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

This audit evaluates the complete predictive uncertainty
pipeline and its relationship with reconstruction error.

The audit combines:

    1. Network-predicted aleatoric uncertainty

    2. Monte Carlo Dropout epistemic uncertainty

    3. Predictive uncertainty decomposition

and compares uncertainty quantities with reconstruction error.

Scientific question
-------------------

Do regions with larger reconstruction errors generally
correspond to regions with larger predicted uncertainty?

Uncertainty decomposition
-------------------------

    predictive_variance
        =
    aleatoric_variance
        +
    epistemic_variance

where:

    aleatoric_variance
        =
    exp(log_variance)

and:

    epistemic_variance
        =
    variance of MC Dropout reconstruction samples.

The audit verifies:

    ✓ Network output validity
    ✓ Reconstruction error computation
    ✓ Aleatoric uncertainty
    ✓ MC Dropout epistemic uncertainty
    ✓ Predictive uncertainty decomposition
    ✓ Uncertainty numerical stability
    ✓ Uncertainty positivity
    ✓ Error-uncertainty correlation
    ✓ High-error versus low-error uncertainty behavior
    ✓ Predictive uncertainty dominance
    ✓ Model parameter stability during inference

Author:
Ormin Joseph
============================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)

from utils.config import (
    USE_UNCERTAINTY,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MC_DROPOUT_SAMPLES,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(42)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    """
    Print a formatted section heading.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_statistics(name, tensor):
    """
    Print numerical statistics for a tensor.
    """

    tensor = tensor.detach()

    print(
        f"{name:<32} "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


def parameter_norm(model):
    """
    Calculate the global L2 norm of model parameters.
    """

    total_norm = 0.0

    for parameter in model.parameters():

        total_norm += (
            parameter.detach()
            .pow(2)
            .sum()
            .item()
        )

    return total_norm ** 0.5


def pearson_correlation(
    x,
    y,
    eps=1.0e-12,
):
    """
    Calculate Pearson correlation between two tensors.

    The tensors are flattened into one-dimensional vectors.
    """

    x = x.detach().reshape(-1)
    y = y.detach().reshape(-1)

    x = x - x.mean()
    y = y - y.mean()

    numerator = torch.sum(
        x * y
    )

    denominator = torch.sqrt(
        torch.sum(x ** 2)
        *
        torch.sum(y ** 2)
    )

    correlation = (
        numerator
        /
        torch.clamp(
            denominator,
            min=eps,
        )
    )

    return correlation.item()


# ============================================================
# AUDIT START
# ============================================================

print_section(
    "FULL PREDICTIVE UNCERTAINTY AUDIT"
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

print_section(
    "DEVICE CONFIGURATION"
)

print(
    f"Device                     : "
    f"{DEVICE}"
)


# ============================================================
# CONFIGURATION
# ============================================================

print_section(
    "FULL UNCERTAINTY CONFIGURATION"
)

print(
    f"USE_UNCERTAINTY            : "
    f"{USE_UNCERTAINTY}"
)

print(
    f"MC samples                 : "
    f"{MC_DROPOUT_SAMPLES}"
)

print(
    f"Learning rate              : "
    f"{LEARNING_RATE}"
)

print(
    f"Weight decay               : "
    f"{WEIGHT_DECAY}"
)


# ============================================================
# CREATE SYNTHETIC SEISMIC DATA
# ============================================================

print_section(
    "CREATING SYNTHETIC SEISMIC DATA"
)

input_cube = torch.randn(
    1,
    1,
    64,
    128,
    128,
    device=DEVICE,
)

target_cube = (
    input_cube
    +
    0.10
    *
    torch.randn_like(
        input_cube
    )
)

print_statistics(
    "Input seismic volume",
    input_cube,
)

print_statistics(
    "Target seismic volume",
    target_cube,
)


# ============================================================
# INITIALIZE NETWORK
# ============================================================

print_section(
    "INITIALIZING 3D NETWORK"
)

model = Network3D(
    in_channels=1,
    out_channels=1,
    use_uncertainty=USE_UNCERTAINTY,
    use_residual=True,
    use_attention=True,
).to(
    DEVICE
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


# ============================================================
# INITIAL PARAMETER STATE
# ============================================================

print_section(
    "INITIAL PARAMETER STATE"
)

initial_parameter_norm = (
    parameter_norm(model)
)

print(
    f"Initial parameter norm     : "
    f"{initial_parameter_norm:.12e}"
)


# ============================================================
# DETERMINISTIC NETWORK PREDICTION
# ============================================================

print_section(
    "DETERMINISTIC NETWORK OUTPUT"
)

model.eval()

with torch.no_grad():

    (
        reconstruction,
        travel_time,
        log_variance,
    ) = model(
        input_cube
    )

print(
    "Network returned three outputs."
)


# ============================================================
# NETWORK OUTPUT VALIDATION
# ============================================================

print_section(
    "NETWORK OUTPUT VALIDATION"
)

expected_shape = input_cube.shape

print(
    f"Expected output shape      : "
    f"{tuple(expected_shape)}"
)

print(
    f"Reconstruction shape       : "
    f"{tuple(reconstruction.shape)}"
)

print(
    f"Travel-time shape          : "
    f"{tuple(travel_time.shape)}"
)

print(
    f"Log-variance shape         : "
    f"{tuple(log_variance.shape)}"
)

assert reconstruction.shape == expected_shape
assert travel_time.shape == expected_shape
assert log_variance.shape == expected_shape

print(
    "Network output shape test: PASSED"
)


# ============================================================
# NETWORK NUMERICAL STABILITY
# ============================================================

print_section(
    "NETWORK NUMERICAL AUDIT"
)

assert torch.isfinite(
    reconstruction
).all()

assert torch.isfinite(
    travel_time
).all()

assert torch.isfinite(
    log_variance
).all()

print(
    "All network outputs are finite."
)

print(
    "Network numerical stability test: PASSED"
)


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

print_section(
    "RECONSTRUCTION ERROR"
)

absolute_error = torch.abs(
    reconstruction
    -
    target_cube
)

squared_error = (
    reconstruction
    -
    target_cube
) ** 2

print_statistics(
    "Absolute reconstruction error",
    absolute_error,
)

print_statistics(
    "Squared reconstruction error",
    squared_error,
)

assert torch.isfinite(
    absolute_error
).all()

assert torch.isfinite(
    squared_error
).all()

print(
    "Reconstruction error computation: PASSED"
)


# ============================================================
# INITIALIZE PREDICTIVE UNCERTAINTY ESTIMATOR
# ============================================================

print_section(
    "INITIALIZING PREDICTIVE UNCERTAINTY ESTIMATOR"
)

uncertainty_estimator = (
    PredictiveUncertaintyEstimator()
    .to(DEVICE)
)

print(
    "PredictiveUncertaintyEstimator "
    "successfully initialized."
)


# ============================================================
# INITIALIZE MC DROPOUT
# ============================================================

print_section(
    "INITIALIZING MC DROPOUT"
)

mc_dropout = MCDropout3D(
    model=model,
    num_samples=MC_DROPOUT_SAMPLES,
)

print(
    "MCDropout3D successfully initialized."
)


# ============================================================
# MONTE CARLO STOCHASTIC INFERENCE
# ============================================================

print_section(
    "MONTE CARLO STOCHASTIC INFERENCE"
)

mc_results = mc_dropout.predict(
    input_cube
)

print(
    "MC stochastic inference completed."
)


# ============================================================
# EXTRACT MC RECONSTRUCTION SAMPLES
# ============================================================

reconstruction_samples = (
    mc_results[
        "reconstruction_samples"
    ]
)

expected_mc_shape = (
    MC_DROPOUT_SAMPLES,
    *input_cube.shape,
)

print_section(
    "MC SAMPLE SHAPE AUDIT"
)

print(
    f"Expected MC shape          : "
    f"{expected_mc_shape}"
)

print(
    f"Reconstruction MC shape    : "
    f"{tuple(reconstruction_samples.shape)}"
)

assert (
    reconstruction_samples.shape
    ==
    expected_mc_shape
)

print(
    "MC sample shape validation: PASSED"
)


# ============================================================
# COMPLETE PREDICTIVE UNCERTAINTY ESTIMATION
# ============================================================

print_section(
    "COMPLETE PREDICTIVE UNCERTAINTY ESTIMATION"
)

uncertainty_results = (
    uncertainty_estimator(
        log_variance=log_variance,
        mc_predictions=reconstruction_samples,
    )
)

print(
    "Complete predictive uncertainty "
    "estimation completed."
)


# ============================================================
# EXTRACT UNCERTAINTY COMPONENTS
# ============================================================

aleatoric_variance = (
    uncertainty_results[
        "aleatoric_variance"
    ]
)

epistemic_variance = (
    uncertainty_results[
        "epistemic_variance"
    ]
)

predictive_variance = (
    uncertainty_results[
        "predictive_variance"
    ]
)

aleatoric_std = (
    uncertainty_results[
        "aleatoric_std"
    ]
)

epistemic_std = (
    uncertainty_results[
        "epistemic_std"
    ]
)

predictive_std = (
    uncertainty_results[
        "predictive_std"
    ]
)


# ============================================================
# UNCERTAINTY STATISTICS
# ============================================================

print_section(
    "UNCERTAINTY STATISTICS"
)

print_statistics(
    "Aleatoric variance",
    aleatoric_variance,
)

print_statistics(
    "Epistemic variance",
    epistemic_variance,
)

print_statistics(
    "Predictive variance",
    predictive_variance,
)

print_statistics(
    "Aleatoric standard deviation",
    aleatoric_std,
)

print_statistics(
    "Epistemic standard deviation",
    epistemic_std,
)

print_statistics(
    "Predictive standard deviation",
    predictive_std,
)


# ============================================================
# UNCERTAINTY NUMERICAL VALIDATION
# ============================================================

print_section(
    "UNCERTAINTY NUMERICAL VALIDATION"
)

uncertainty_tensors = {

    "Aleatoric variance":
        aleatoric_variance,

    "Epistemic variance":
        epistemic_variance,

    "Predictive variance":
        predictive_variance,

    "Aleatoric standard deviation":
        aleatoric_std,

    "Epistemic standard deviation":
        epistemic_std,

    "Predictive standard deviation":
        predictive_std,
}

for name, tensor in uncertainty_tensors.items():

    assert torch.isfinite(
        tensor
    ).all()

    assert torch.all(
        tensor >= 0.0
    )

print(
    "All uncertainty quantities are finite."
)

print(
    "All uncertainty quantities are non-negative."
)

print(
    "Uncertainty numerical validation: PASSED"
)


# ============================================================
# UNCERTAINTY DECOMPOSITION CONSISTENCY
# ============================================================

print_section(
    "UNCERTAINTY DECOMPOSITION CONSISTENCY"
)

expected_predictive_variance = (
    aleatoric_variance
    +
    epistemic_variance
)

maximum_decomposition_error = (
    torch.max(
        torch.abs(
            predictive_variance
            -
            expected_predictive_variance
        )
    )
    .item()
)

print(
    f"Maximum decomposition error : "
    f"{maximum_decomposition_error:.12e}"
)

assert torch.allclose(
    predictive_variance,
    expected_predictive_variance,
    rtol=1.0e-6,
    atol=1.0e-8,
)

print(
    "Predictive variance decomposition: PASSED"
)


# ============================================================
# PREDICTIVE VARIANCE DOMINANCE
# ============================================================

print_section(
    "PREDICTIVE VARIANCE DOMINANCE AUDIT"
)

predictive_minus_aleatoric = (
    predictive_variance
    -
    aleatoric_variance
)

minimum_difference = (
    predictive_minus_aleatoric
    .min()
    .item()
)

print(
    "Minimum predictive-minus-aleatoric "
    f"variance : {minimum_difference:.12e}"
)

assert torch.all(
    predictive_variance
    >=
    aleatoric_variance
)

assert torch.all(
    predictive_variance
    >=
    epistemic_variance
)

print(
    "Predictive variance dominance: PASSED"
)


# ============================================================
# ERROR-UNCERTAINTY CORRELATION
# ============================================================

print_section(
    "ERROR-UNCERTAINTY CORRELATION AUDIT"
)

aleatoric_error_correlation = (
    pearson_correlation(
        absolute_error,
        aleatoric_std,
    )
)

epistemic_error_correlation = (
    pearson_correlation(
        absolute_error,
        epistemic_std,
    )
)

predictive_error_correlation = (
    pearson_correlation(
        absolute_error,
        predictive_std,
    )
)

print(
    "Aleatoric uncertainty correlation : "
    f"{aleatoric_error_correlation:.6f}"
)

print(
    "Epistemic uncertainty correlation : "
    f"{epistemic_error_correlation:.6f}"
)

print(
    "Predictive uncertainty correlation: "
    f"{predictive_error_correlation:.6f}"
)

assert (
    -1.0
    <=
    aleatoric_error_correlation
    <=
    1.0
)

assert (
    -1.0
    <=
    epistemic_error_correlation
    <=
    1.0
)

assert (
    -1.0
    <=
    predictive_error_correlation
    <=
    1.0
)

print(
    "Error-uncertainty correlation calculation: PASSED"
)


# ============================================================
# HIGH-ERROR / LOW-ERROR REGION AUDIT
# ============================================================

print_section(
    "HIGH-ERROR / LOW-ERROR UNCERTAINTY AUDIT"
)

error_flat = absolute_error.reshape(
    -1
)

predictive_std_flat = (
    predictive_std.reshape(
        -1
    )
)

low_error_threshold = torch.quantile(
    error_flat,
    0.25,
)

high_error_threshold = torch.quantile(
    error_flat,
    0.75,
)

low_error_mask = (
    error_flat
    <=
    low_error_threshold
)

high_error_mask = (
    error_flat
    >=
    high_error_threshold
)

low_error_uncertainty = (
    predictive_std_flat[
        low_error_mask
    ]
    .mean()
    .item()
)

high_error_uncertainty = (
    predictive_std_flat[
        high_error_mask
    ]
    .mean()
    .item()
)

print(
    f"Low-error threshold       : "
    f"{low_error_threshold.item():.6e}"
)

print(
    f"High-error threshold      : "
    f"{high_error_threshold.item():.6e}"
)

print(
    f"Mean uncertainty in low-error region : "
    f"{low_error_uncertainty:.6e}"
)

print(
    f"Mean uncertainty in high-error region: "
    f"{high_error_uncertainty:.6e}"
)

high_vs_low_difference = (
    high_error_uncertainty
    -
    low_error_uncertainty
)

print(
    f"High-minus-low uncertainty difference: "
    f"{high_vs_low_difference:.6e}"
)

assert torch.isfinite(
    torch.tensor(
        high_vs_low_difference
    )
)

print(
    "High-error / low-error uncertainty "
    "analysis: COMPLETED"
)


# ============================================================
# MEAN UNCERTAINTY CONTRIBUTIONS
# ============================================================

print_section(
    "ALEATORIC / EPISTEMIC CONTRIBUTION ANALYSIS"
)

mean_aleatoric = (
    aleatoric_variance
    .mean()
    .item()
)

mean_epistemic = (
    epistemic_variance
    .mean()
    .item()
)

mean_predictive = (
    predictive_variance
    .mean()
    .item()
)

aleatoric_contribution = (
    mean_aleatoric
    /
    mean_predictive
    *
    100.0
)

epistemic_contribution = (
    mean_epistemic
    /
    mean_predictive
    *
    100.0
)

contribution_sum = (
    aleatoric_contribution
    +
    epistemic_contribution
)

print(
    f"Mean aleatoric variance   : "
    f"{mean_aleatoric:.12e}"
)

print(
    f"Mean epistemic variance   : "
    f"{mean_epistemic:.12e}"
)

print(
    f"Mean predictive variance  : "
    f"{mean_predictive:.12e}"
)

print(
    f"Aleatoric contribution    : "
    f"{aleatoric_contribution:.6f}%"
)

print(
    f"Epistemic contribution    : "
    f"{epistemic_contribution:.6f}%"
)

print(
    f"Contribution sum          : "
    f"{contribution_sum:.6f}%"
)


# ============================================================
# TRAVEL-TIME VALIDATION
# ============================================================

print_section(
    "TRAVEL-TIME VALIDATION"
)

print_statistics(
    "Travel-time field",
    travel_time,
)

minimum_travel_time = (
    travel_time
    .min()
    .item()
)

assert (
    minimum_travel_time
    >=
    0.0
)

assert torch.isfinite(
    travel_time
).all()

print(
    f"Minimum travel time       : "
    f"{minimum_travel_time:.12e}"
)

print(
    "Travel-time positivity: PASSED"
)


# ============================================================
# PARAMETER STABILITY
# ============================================================

print_section(
    "PARAMETER STABILITY AUDIT"
)

final_parameter_norm = (
    parameter_norm(model)
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

assert (
    abs(parameter_change)
    <
    1.0e-10
)

print(
    "Model parameters remained unchanged."
)

print(
    "Parameter stability test: PASSED"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print_section(
    "FULL UNCERTAINTY AUDIT INTERPRETATION"
)

print(
    "Aleatoric uncertainty       : OBSERVED"
)

print(
    "Epistemic uncertainty       : OBSERVED"
)

print(
    "Predictive uncertainty      : OBSERVED"
)

print(
    "Variance decomposition      : CONSISTENT"
)

print(
    "Uncertainty components      : FINITE"
)

print(
    "Error-uncertainty analysis  : COMPLETED"
)

print(
    "High/low error analysis     : COMPLETED"
)

print(
    "Travel-time field           : VALID"
)

print(
    "Model parameters            : STABLE"
)


# ============================================================
# AUDIT COMPLETION
# ============================================================

print_section(
    "FULL PREDICTIVE UNCERTAINTY AUDIT PASSED"
)

print(
    "Verified:"
)

print(
    "  ✓ Network reconstruction output"
)

print(
    "  ✓ Reconstruction error computation"
)

print(
    "  ✓ Aleatoric uncertainty"
)

print(
    "  ✓ MC Dropout epistemic uncertainty"
)

print(
    "  ✓ Predictive uncertainty estimator"
)

print(
    "  ✓ Predictive variance decomposition"
)

print(
    "  ✓ Uncertainty positivity"
)

print(
    "  ✓ Numerical stability"
)

print(
    "  ✓ Error-uncertainty correlation"
)

print(
    "  ✓ High-error / low-error uncertainty analysis"
)

print(
    "  ✓ Predictive variance dominance"
)

print(
    "  ✓ Aleatoric / epistemic contribution analysis"
)

print(
    "  ✓ Travel-time positivity"
)

print(
    "  ✓ Parameter stability"
)

print(
    "\nFULL PREDICTIVE UNCERTAINTY "
    "AUDIT PASSED."
)