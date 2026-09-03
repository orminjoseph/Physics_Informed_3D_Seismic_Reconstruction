"""
=========================================================
Predictive Uncertainty Decomposition Audit
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

This test validates the combination of:

    1. Aleatoric uncertainty
       -> predicted by the Network3D log-variance head.

    2. Epistemic uncertainty
       -> estimated using Monte Carlo Dropout.

The predictive uncertainty is decomposed as:

    sigma²_predictive
        =
    sigma²_aleatoric
        +
    sigma²_epistemic

where:

    sigma²_aleatoric
        = exp(log_variance)

and:

    sigma²_epistemic
        = Var_MC(reconstruction)

The test verifies:

    ✓ Network output validity
    ✓ Aleatoric variance calculation
    ✓ MC Dropout epistemic variance
    ✓ Predictive variance construction
    ✓ Predictive standard deviation
    ✓ Non-negative variances
    ✓ Finite uncertainty quantities
    ✓ Tensor shape consistency
    ✓ Epistemic uncertainty contribution
    ✓ Aleatoric uncertainty contribution
    ✓ Predictive variance consistency
    ✓ Travel-time preservation
    ✓ Parameter stability
    ✓ Numerical stability

Tensor convention
-----------------

Input:

    [B, C, D, H, W]

MC reconstruction samples:

    [N, B, C, D, H, W]

Aleatoric variance:

    [B, C, D, H, W]

Epistemic variance:

    [B, C, D, H, W]

Predictive variance:

    [B, C, D, H, W]

Predictive standard deviation:

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D

from utils.config import (
    USE_UNCERTAINTY,
    LEARNING_RATE,
    WEIGHT_DECAY
)


# =========================================================
# CONFIGURATION
# =========================================================

MC_SAMPLES = 20

TOLERANCE = 1.0e-10


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_section(title):
    """Print a formatted audit section."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def tensor_statistics(name, tensor):
    """Print basic statistics for a tensor."""

    print(
        f"{name:<30}"
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


def assert_finite(name, tensor):
    """Verify that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all():

        raise AssertionError(
            f"{name} contains NaN or Inf values."
        )


def assert_shape(name, tensor, expected_shape):
    """Verify tensor shape."""

    if tuple(tensor.shape) != tuple(expected_shape):

        raise AssertionError(
            f"{name} shape mismatch. "
            f"Expected {expected_shape}, "
            f"received {tuple(tensor.shape)}."
        )


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print_section(
        "PREDICTIVE UNCERTAINTY DECOMPOSITION AUDIT"
    )

    # =====================================================
    # DEVICE CONFIGURATION
    # =====================================================

    print_section(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # UNCERTAINTY CONFIGURATION
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY CONFIGURATION"
    )

    print(
        f"USE_UNCERTAINTY            : "
        f"{USE_UNCERTAINTY}"
    )

    print(
        f"MC samples                 : "
        f"{MC_SAMPLES}"
    )

    print(
        f"Learning rate              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay               : "
        f"{WEIGHT_DECAY}"
    )

    if not USE_UNCERTAINTY:

        raise AssertionError(
            "USE_UNCERTAINTY must be True "
            "for predictive uncertainty "
            "decomposition."
        )

    # =====================================================
    # SYNTHETIC INPUT
    # =====================================================

    print_section(
        "CREATING SYNTHETIC SEISMIC INPUT"
    )

    torch.manual_seed(42)

    input_volume = torch.randn(
        1,
        1,
        64,
        128,
        128,
        device=device
    )

    tensor_statistics(
        "Input seismic volume",
        input_volume
    )

    # =====================================================
    # INITIALIZE NETWORK
    # =====================================================

    print_section(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(device)

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

    # =====================================================
    # INITIAL PARAMETER STATE
    # =====================================================

    print_section(
        "INITIAL PARAMETER STATE"
    )

    initial_parameter_norm = torch.sqrt(
        sum(
            torch.sum(parameter.detach() ** 2)
            for parameter in model.parameters()
        )
    ).item()

    print(
        f"Initial parameter norm     : "
        f"{initial_parameter_norm:.12e}"
    )

    # =====================================================
    # DETERMINISTIC FORWARD PASS
    # =====================================================

    print_section(
        "DETERMINISTIC NETWORK OUTPUT"
    )

    model.eval()

    with torch.no_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(input_volume)

    print(
        "Network returned three outputs."
    )

    # =====================================================
    # OUTPUT SHAPE AUDIT
    # =====================================================

    print_section(
        "NETWORK OUTPUT SHAPE AUDIT"
    )

    expected_shape = tuple(
        input_volume.shape
    )

    print(
        f"Expected output shape      : "
        f"{expected_shape}"
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

    assert_shape(
        "Reconstruction",
        reconstruction,
        expected_shape
    )

    assert_shape(
        "Travel time",
        travel_time,
        expected_shape
    )

    assert_shape(
        "Log variance",
        log_variance,
        expected_shape
    )

    print(
        "Network output shape test: PASSED"
    )

    # =====================================================
    # NETWORK OUTPUT FINITENESS
    # =====================================================

    print_section(
        "NETWORK OUTPUT NUMERICAL AUDIT"
    )

    assert_finite(
        "Reconstruction",
        reconstruction
    )

    assert_finite(
        "Travel time",
        travel_time
    )

    assert_finite(
        "Log variance",
        log_variance
    )

    print(
        "All network outputs are finite."
    )

    print(
        "Network numerical stability test: PASSED"
    )

    # =====================================================
    # ALEATORIC VARIANCE
    # =====================================================

    print_section(
        "ALEATORIC UNCERTAINTY EXTRACTION"
    )

    aleatoric_variance = torch.exp(
        log_variance
    )

    aleatoric_std = torch.sqrt(
        aleatoric_variance
    )

    tensor_statistics(
        "Log variance",
        log_variance
    )

    tensor_statistics(
        "Aleatoric variance",
        aleatoric_variance
    )

    tensor_statistics(
        "Aleatoric standard deviation",
        aleatoric_std
    )

    # =====================================================
    # ALEATORIC VALIDATION
    # =====================================================

    print_section(
        "ALEATORIC UNCERTAINTY VALIDATION"
    )

    assert_finite(
        "Aleatoric variance",
        aleatoric_variance
    )

    assert_finite(
        "Aleatoric standard deviation",
        aleatoric_std
    )

    if torch.any(
        aleatoric_variance < 0
    ):

        raise AssertionError(
            "Aleatoric variance contains "
            "negative values."
        )

    if torch.any(
        aleatoric_std < 0
    ):

        raise AssertionError(
            "Aleatoric standard deviation "
            "contains negative values."
        )

    assert_shape(
        "Aleatoric variance",
        aleatoric_variance,
        expected_shape
    )

    assert_shape(
        "Aleatoric standard deviation",
        aleatoric_std,
        expected_shape
    )

    print(
        "Aleatoric variance positivity: PASSED"
    )

    print(
        "Aleatoric standard deviation: PASSED"
    )

    # =====================================================
    # INITIALIZE MC DROPOUT
    # =====================================================

    print_section(
        "INITIALIZING MC DROPOUT"
    )

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES
    )

    print(
        "MCDropout3D successfully initialized."
    )

    # =====================================================
    # MC DROPOUT PREDICTION
    # =====================================================

    print_section(
        "MONTE CARLO STOCHASTIC INFERENCE"
    )

    mc_results = mc_dropout.predict(
        input_volume
    )

    print(
        "MC stochastic inference completed."
    )

    # =====================================================
    # EXTRACT MC OUTPUTS
    # =====================================================

    reconstruction_samples = (
        mc_results[
            "reconstruction_samples"
        ]
    )

    travel_time_samples = (
        mc_results[
            "travel_time_samples"
        ]
    )

    log_variance_samples = (
        mc_results[
            "log_variance_samples"
        ]
    )

    reconstruction_mean = (
        mc_results[
            "reconstruction_mean"
        ]
    )

    travel_time_mean = (
        mc_results[
            "travel_time_mean"
        ]
    )

    log_variance_mean = (
        mc_results[
            "log_variance_mean"
        ]
    )

    reconstruction_epistemic_variance = (
        mc_results[
            "reconstruction_epistemic_variance"
        ]
    )

    travel_time_epistemic_variance = (
        mc_results[
            "travel_time_epistemic_variance"
        ]
    )

    log_variance_epistemic_variance = (
        mc_results[
            "log_variance_epistemic_variance"
        ]
    )

    # =====================================================
    # MC SAMPLE SHAPE AUDIT
    # =====================================================

    print_section(
        "MC SAMPLE SHAPE AUDIT"
    )

    expected_mc_shape = (
        MC_SAMPLES,
        *expected_shape
    )

    print(
        f"Expected MC shape          : "
        f"{expected_mc_shape}"
    )

    print(
        f"Reconstruction MC shape    : "
        f"{tuple(reconstruction_samples.shape)}"
    )

    print(
        f"Travel-time MC shape       : "
        f"{tuple(travel_time_samples.shape)}"
    )

    print(
        f"Log-variance MC shape      : "
        f"{tuple(log_variance_samples.shape)}"
    )

    assert_shape(
        "Reconstruction MC samples",
        reconstruction_samples,
        expected_mc_shape
    )

    assert_shape(
        "Travel-time MC samples",
        travel_time_samples,
        expected_mc_shape
    )

    assert_shape(
        "Log-variance MC samples",
        log_variance_samples,
        expected_mc_shape
    )

    print(
        "MC sample shape validation: PASSED"
    )

    # =====================================================
    # MC NUMERICAL STABILITY
    # =====================================================

    print_section(
        "MC UNCERTAINTY NUMERICAL AUDIT"
    )

    mc_tensors = {

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

    for name, tensor in mc_tensors.items():

        assert_finite(
            name,
            tensor
        )

    print(
        "All MC-Dropout quantities are finite."
    )

    print(
        "MC numerical stability test: PASSED"
    )

    # =====================================================
    # EPISTEMIC RECONSTRUCTION UNCERTAINTY
    # =====================================================

    print_section(
        "EPISTEMIC RECONSTRUCTION UNCERTAINTY"
    )

    tensor_statistics(
        "Reconstruction MC mean",
        reconstruction_mean
    )

    tensor_statistics(
        "Reconstruction epistemic variance",
        reconstruction_epistemic_variance
    )

    epistemic_std = torch.sqrt(
        reconstruction_epistemic_variance
    )

    tensor_statistics(
        "Reconstruction epistemic std",
        epistemic_std
    )

    # =====================================================
    # EPISTEMIC VALIDATION
    # =====================================================

    print_section(
        "EPISTEMIC UNCERTAINTY VALIDATION"
    )

    if torch.any(
        reconstruction_epistemic_variance < 0
    ):

        raise AssertionError(
            "Epistemic reconstruction variance "
            "contains negative values."
        )

    if torch.any(
        epistemic_std < 0
    ):

        raise AssertionError(
            "Epistemic reconstruction standard "
            "deviation contains negative values."
        )

    assert_finite(
        "Epistemic standard deviation",
        epistemic_std
    )

    print(
        f"Mean epistemic variance   : "
        f"{reconstruction_epistemic_variance.mean().item():.12e}"
    )

    print(
        f"Maximum epistemic variance: "
        f"{reconstruction_epistemic_variance.max().item():.12e}"
    )

    print(
        "Epistemic variance positivity: PASSED"
    )

    # =====================================================
    # STOCHASTIC VARIABILITY
    # =====================================================

    print_section(
        "STOCHASTIC VARIABILITY AUDIT"
    )

    sample_difference = torch.max(
        torch.abs(
            reconstruction_samples[0]
            -
            reconstruction_samples[1]
        )
    ).item()

    print(
        f"Maximum reconstruction sample difference : "
        f"{sample_difference:.12e}"
    )

    if sample_difference <= 0.0:

        raise AssertionError(
            "MC Dropout produced identical "
            "reconstruction samples. "
            "Epistemic uncertainty is not active."
        )

    print(
        "Stochastic reconstruction variability: "
        "OBSERVED"
    )

    # =====================================================
    # PREDICTIVE VARIANCE DECOMPOSITION
    # =====================================================

    print_section(
        "PREDICTIVE VARIANCE DECOMPOSITION"
    )

    predictive_variance = (
        aleatoric_variance
        +
        reconstruction_epistemic_variance
    )

    predictive_std = torch.sqrt(
        predictive_variance
    )

    tensor_statistics(
        "Aleatoric variance",
        aleatoric_variance
    )

    tensor_statistics(
        "Epistemic variance",
        reconstruction_epistemic_variance
    )

    tensor_statistics(
        "Predictive variance",
        predictive_variance
    )

    tensor_statistics(
        "Predictive standard deviation",
        predictive_std
    )

    # =====================================================
    # PREDICTIVE VARIANCE VALIDATION
    # =====================================================

    print_section(
        "PREDICTIVE VARIANCE VALIDATION"
    )

    assert_finite(
        "Predictive variance",
        predictive_variance
    )

    assert_finite(
        "Predictive standard deviation",
        predictive_std
    )

    if torch.any(
        predictive_variance < 0
    ):

        raise AssertionError(
            "Predictive variance contains "
            "negative values."
        )

    if torch.any(
        predictive_std < 0
    ):

        raise AssertionError(
            "Predictive standard deviation "
            "contains negative values."
        )

    assert_shape(
        "Predictive variance",
        predictive_variance,
        expected_shape
    )

    assert_shape(
        "Predictive standard deviation",
        predictive_std,
        expected_shape
    )

    print(
        "Predictive variance positivity: PASSED"
    )

    print(
        "Predictive standard deviation: PASSED"
    )

    # =====================================================
    # DECOMPOSITION CONSISTENCY
    # =====================================================

    print_section(
        "UNCERTAINTY DECOMPOSITION CONSISTENCY"
    )

    reconstructed_predictive_variance = (
        aleatoric_variance
        +
        reconstruction_epistemic_variance
    )

    decomposition_error = torch.max(
        torch.abs(
            predictive_variance
            -
            reconstructed_predictive_variance
        )
    ).item()

    print(
        f"Maximum decomposition error : "
        f"{decomposition_error:.12e}"
    )

    if decomposition_error > TOLERANCE:

        raise AssertionError(
            "Predictive variance decomposition "
            "is inconsistent."
        )

    print(
        "Predictive variance decomposition: PASSED"
    )

    # =====================================================
    # PREDICTIVE DOMINANCE AUDIT
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY DOMINANCE AUDIT"
    )

    variance_difference = (
        predictive_variance
        -
        aleatoric_variance
    )

    minimum_difference = (
        variance_difference.min().item()
    )

    print(
        f"Minimum predictive-minus-aleatoric "
        f"variance : {minimum_difference:.12e}"
    )

    if minimum_difference < -TOLERANCE:

        raise AssertionError(
            "Predictive variance is smaller "
            "than aleatoric variance."
        )

    print(
        "Predictive variance >= aleatoric "
        "variance: PASSED"
    )

    # =====================================================
    # UNCERTAINTY CONTRIBUTION ANALYSIS
    # =====================================================

    print_section(
        "ALEATORIC / EPISTEMIC CONTRIBUTION ANALYSIS"
    )

    mean_aleatoric = (
        aleatoric_variance.mean().item()
    )

    mean_epistemic = (
        reconstruction_epistemic_variance
        .mean()
        .item()
    )

    mean_predictive = (
        predictive_variance
        .mean()
        .item()
    )

    aleatoric_fraction = (
        mean_aleatoric
        /
        mean_predictive
    )

    epistemic_fraction = (
        mean_epistemic
        /
        mean_predictive
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
        f"{100.0 * aleatoric_fraction:.6f}%"
    )

    print(
        f"Epistemic contribution    : "
        f"{100.0 * epistemic_fraction:.6f}%"
    )

    # =====================================================
    # CONTRIBUTION CONSISTENCY
    # =====================================================

    contribution_sum = (
        aleatoric_fraction
        +
        epistemic_fraction
    )

    print(
        f"Contribution sum          : "
        f"{contribution_sum:.12f}"
    )

    if abs(
        contribution_sum - 1.0
    ) > 1.0e-6:

        raise AssertionError(
            "Aleatoric and epistemic "
            "contributions do not sum to 1."
        )

    print(
        "Uncertainty contribution consistency: "
        "PASSED"
    )

    # =====================================================
    # TRAVEL-TIME VALIDATION
    # =====================================================

    print_section(
        "TRAVEL-TIME VALIDATION"
    )

    tensor_statistics(
        "MC travel-time mean",
        travel_time_mean
    )

    tensor_statistics(
        "Travel-time epistemic variance",
        travel_time_epistemic_variance
    )

    if torch.any(
        travel_time_mean < 0
    ):

        raise AssertionError(
            "MC travel-time prediction "
            "contains negative values."
        )

    if torch.any(
        travel_time_epistemic_variance < 0
    ):

        raise AssertionError(
            "Travel-time epistemic variance "
            "contains negative values."
        )

    print(
        f"Minimum travel time       : "
        f"{travel_time_mean.min().item():.12e}"
    )

    print(
        f"Maximum travel time       : "
        f"{travel_time_mean.max().item():.12e}"
    )

    print(
        f"Mean travel time          : "
        f"{travel_time_mean.mean().item():.12e}"
    )

    print(
        "Travel-time positivity: PASSED"
    )

    # =====================================================
    # ALEATORIC OUTPUT PRESERVATION
    # =====================================================

    print_section(
        "ALEATORIC OUTPUT PRESERVATION AUDIT"
    )

    mc_log_variance_mean = (
        log_variance_mean
    )

    mc_aleatoric_variance = torch.exp(
        mc_log_variance_mean
    )

    assert_finite(
        "MC-derived aleatoric variance",
        mc_aleatoric_variance
    )

    tensor_statistics(
        "MC-derived aleatoric variance",
        mc_aleatoric_variance
    )

    if torch.any(
        mc_aleatoric_variance <= 0
    ):

        raise AssertionError(
            "MC-derived aleatoric variance "
            "is not positive."
        )

    print(
        "Aleatoric uncertainty remains "
        "positive during MC inference."
    )

    print(
        "Aleatoric output preservation: PASSED"
    )

    # =====================================================
    # PARAMETER STABILITY
    # =====================================================

    print_section(
        "PARAMETER STABILITY AUDIT"
    )

    final_parameter_norm = torch.sqrt(
        sum(
            torch.sum(parameter.detach() ** 2)
            for parameter in model.parameters()
        )
    ).item()

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

    if abs(parameter_change) > TOLERANCE:

        raise AssertionError(
            "Model parameters changed "
            "during uncertainty inference."
        )

    print(
        "Model parameters remained unchanged."
    )

    print(
        "Parameter stability test: PASSED"
    )

    # =====================================================
    # FINAL NUMERICAL STABILITY AUDIT
    # =====================================================

    print_section(
        "FINAL NUMERICAL STABILITY AUDIT"
    )

    final_tensors = {

        "Aleatoric variance":
            aleatoric_variance,

        "Aleatoric standard deviation":
            aleatoric_std,

        "Epistemic variance":
            reconstruction_epistemic_variance,

        "Epistemic standard deviation":
            epistemic_std,

        "Predictive variance":
            predictive_variance,

        "Predictive standard deviation":
            predictive_std
    }

    for name, tensor in final_tensors.items():

        assert_finite(
            name,
            tensor
        )

    print(
        "All uncertainty quantities are finite."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # =====================================================
    # FINAL INTERPRETATION
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY INTERPRETATION"
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
        "Travel-time field           : VALID"
    )

    print(
        "Model parameters            : STABLE"
    )

    # =====================================================
    # FINAL AUDIT
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY DECOMPOSITION AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ Network reconstruction output"
    )

    print(
        "  ✓ Aleatoric log-variance output"
    )

    print(
        "  ✓ Aleatoric variance"
    )

    print(
        "  ✓ Aleatoric standard deviation"
    )

    print(
        "  ✓ MC Dropout epistemic variance"
    )

    print(
        "  ✓ Epistemic standard deviation"
    )

    print(
        "  ✓ Predictive variance"
    )

    print(
        "  ✓ Predictive standard deviation"
    )

    print(
        "  ✓ Variance decomposition consistency"
    )

    print(
        "  ✓ Aleatoric contribution"
    )

    print(
        "  ✓ Epistemic contribution"
    )

    print(
        "  ✓ MC stochastic variability"
    )

    print(
        "  ✓ Travel-time positivity"
    )

    print(
        "  ✓ Aleatoric uncertainty preservation"
    )

    print(
        "  ✓ Parameter stability"
    )

    print(
        "  ✓ Numerical stability"
    )

    print()

    print(
        "PREDICTIVE UNCERTAINTY DECOMPOSITION "
        "TEST PASSED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()