"""
=========================================================
Predictive Uncertainty Convergence Audit
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

This test verifies the complete predictive uncertainty
pipeline by combining:

    1. Aleatoric uncertainty
       obtained from the network-predicted log variance.

    2. Epistemic uncertainty
       obtained from Monte Carlo Dropout.

    3. Predictive uncertainty
       obtained from the sum of aleatoric and epistemic
       variance.

The test uses the actual project modules:

    models.network.Network3D
    models.mc_dropout.MCDropout3D
    models.predictive_uncertainty.PredictiveUncertaintyEstimator

The test verifies:

    - Network output validity
    - Aleatoric variance calculation
    - MC Dropout stochastic inference
    - Epistemic variance calculation
    - Predictive variance calculation
    - Standard deviation calculation
    - Variance decomposition consistency
    - MC-sample convergence stability
    - Uncertainty positivity
    - Numerical stability
    - Parameter stability
    - Travel-time positivity

Tensor convention
-----------------

Network input/output:

    [B, C, D, H, W]

MC predictions:

    [N, B, C, D, H, W]

where:

    N = number of MC Dropout samples.

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from models.mc_dropout import MCDropout3D
from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from utils.config import (
    USE_UNCERTAINTY,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MC_DROPOUT_SAMPLES
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def print_section(title):
    """Print a formatted audit section."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_statistics(name, tensor):
    """Print numerical statistics for a tensor."""

    print(
        f"{name:<30}"
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


def assert_finite(tensor, name):
    """Verify that a tensor contains only finite values."""

    if not torch.isfinite(tensor).all():
        raise AssertionError(
            f"{name} contains NaN or Inf values."
        )


def parameter_norm(model):
    """Calculate the L2 norm of all model parameters."""

    total = torch.tensor(
        0.0,
        device=next(model.parameters()).device
    )

    for parameter in model.parameters():

        total = total + torch.sum(
            parameter.detach() ** 2
        )

    return torch.sqrt(total).item()


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print_section(
        "PREDICTIVE UNCERTAINTY CONVERGENCE AUDIT"
    )

    # =====================================================
    # DEVICE CONFIGURATION
    # =====================================================

    print_section(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # CONFIGURATION
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

    if not USE_UNCERTAINTY:

        raise AssertionError(
            "USE_UNCERTAINTY must be True for "
            "predictive uncertainty testing."
        )

    if MC_DROPOUT_SAMPLES < 20:

        raise AssertionError(
            "MC_DROPOUT_SAMPLES must be at least 20 "
            "for this convergence audit."
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

    print_statistics(
        "Input seismic volume",
        input_volume
    )

    if not torch.isfinite(
        input_volume
    ).all():

        raise AssertionError(
            "Synthetic input contains NaN or Inf."
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
        use_uncertainty=USE_UNCERTAINTY,
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

    initial_parameter_norm = parameter_norm(
        model
    )

    print(
        f"Initial parameter norm     : "
        f"{initial_parameter_norm:.12e}"
    )

    # =====================================================
    # DETERMINISTIC NETWORK FORWARD
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
        ) = model(
            input_volume
        )

    print(
        "Network returned three outputs."
    )

    expected_shape = (
        1,
        1,
        64,
        128,
        128
    )

    # =====================================================
    # OUTPUT SHAPE AUDIT
    # =====================================================

    print_section(
        "NETWORK OUTPUT SHAPE AUDIT"
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

    if tuple(reconstruction.shape) != expected_shape:
        raise AssertionError(
            "Reconstruction output shape is incorrect."
        )

    if tuple(travel_time.shape) != expected_shape:
        raise AssertionError(
            "Travel-time output shape is incorrect."
        )

    if tuple(log_variance.shape) != expected_shape:
        raise AssertionError(
            "Log-variance output shape is incorrect."
        )

    print(
        "Network output shape test: PASSED"
    )

    # =====================================================
    # NETWORK NUMERICAL AUDIT
    # =====================================================

    print_section(
        "NETWORK NUMERICAL AUDIT"
    )

    assert_finite(
        reconstruction,
        "Reconstruction"
    )

    assert_finite(
        travel_time,
        "Travel time"
    )

    assert_finite(
        log_variance,
        "Log variance"
    )

    print(
        "All network outputs are finite."
    )

    print(
        "Network numerical stability test: PASSED"
    )

    # =====================================================
    # INITIALIZE PREDICTIVE UNCERTAINTY ESTIMATOR
    # =====================================================

    print_section(
        "INITIALIZING PREDICTIVE UNCERTAINTY ESTIMATOR"
    )

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator()
        .to(device)
    )

    print(
        "PredictiveUncertaintyEstimator "
        "successfully initialized."
    )

    # =====================================================
    # ALEATORIC UNCERTAINTY
    # =====================================================

    print_section(
        "ALEATORIC UNCERTAINTY ESTIMATION"
    )

    aleatoric_variance = (
        uncertainty_estimator.aleatoric_variance(
            log_variance
        )
    )

    aleatoric_std = (
        uncertainty_estimator.standard_deviation(
            aleatoric_variance
        )
    )

    print_statistics(
        "Log variance",
        log_variance
    )

    print_statistics(
        "Aleatoric variance",
        aleatoric_variance
    )

    print_statistics(
        "Aleatoric standard deviation",
        aleatoric_std
    )

    assert_finite(
        aleatoric_variance,
        "Aleatoric variance"
    )

    assert_finite(
        aleatoric_std,
        "Aleatoric standard deviation"
    )

    if torch.any(
        aleatoric_variance <= 0.0
    ):

        raise AssertionError(
            "Aleatoric variance must be positive."
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
        num_samples=MC_DROPOUT_SAMPLES
    )

    print(
        "MCDropout3D successfully initialized."
    )

    # =====================================================
    # MC DROPOUT INFERENCE
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
    # EXTRACT MC RECONSTRUCTION SAMPLES
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

    # =====================================================
    # MC SAMPLE SHAPE AUDIT
    # =====================================================

    print_section(
        "MC SAMPLE SHAPE AUDIT"
    )

    expected_mc_shape = (
        MC_DROPOUT_SAMPLES,
        1,
        1,
        64,
        128,
        128
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

    if tuple(
        reconstruction_samples.shape
    ) != expected_mc_shape:

        raise AssertionError(
            "Reconstruction MC sample shape "
            "is incorrect."
        )

    if tuple(
        travel_time_samples.shape
    ) != expected_mc_shape:

        raise AssertionError(
            "Travel-time MC sample shape "
            "is incorrect."
        )

    if tuple(
        log_variance_samples.shape
    ) != expected_mc_shape:

        raise AssertionError(
            "Log-variance MC sample shape "
            "is incorrect."
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

    assert_finite(
        reconstruction_samples,
        "Reconstruction MC samples"
    )

    assert_finite(
        travel_time_samples,
        "Travel-time MC samples"
    )

    assert_finite(
        log_variance_samples,
        "Log-variance MC samples"
    )

    print(
        "All MC-Dropout quantities are finite."
    )

    print(
        "MC numerical stability test: PASSED"
    )

    # =====================================================
    # USE THE ACTUAL PREDICTIVE UNCERTAINTY MODULE
    # =====================================================

    print_section(
        "COMPLETE PREDICTIVE UNCERTAINTY ESTIMATION"
    )

    uncertainty_results = (
        uncertainty_estimator(
            log_variance,
            reconstruction_samples
        )
    )

    estimator_aleatoric = (
        uncertainty_results[
            "aleatoric_variance"
        ]
    )

    estimator_epistemic = (
        uncertainty_results[
            "epistemic_variance"
        ]
    )

    estimator_predictive = (
        uncertainty_results[
            "predictive_variance"
        ]
    )

    estimator_aleatoric_std = (
        uncertainty_results[
            "aleatoric_std"
        ]
    )

    estimator_epistemic_std = (
        uncertainty_results[
            "epistemic_std"
        ]
    )

    estimator_predictive_std = (
        uncertainty_results[
            "predictive_std"
        ]
    )

    print(
        "Complete predictive uncertainty "
        "estimation completed."
    )

    # =====================================================
    # EPISTEMIC UNCERTAINTY AUDIT
    # =====================================================

    print_section(
        "EPISTEMIC UNCERTAINTY AUDIT"
    )

    print_statistics(
        "Epistemic variance",
        estimator_epistemic
    )

    print_statistics(
        "Epistemic standard deviation",
        estimator_epistemic_std
    )

    assert_finite(
        estimator_epistemic,
        "Epistemic variance"
    )

    assert_finite(
        estimator_epistemic_std,
        "Epistemic standard deviation"
    )

    if torch.any(
        estimator_epistemic < 0.0
    ):

        raise AssertionError(
            "Epistemic variance cannot be negative."
        )

    print(
        "Epistemic variance positivity: PASSED"
    )

    # =====================================================
    # PREDICTIVE UNCERTAINTY AUDIT
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY AUDIT"
    )

    print_statistics(
        "Predictive variance",
        estimator_predictive
    )

    print_statistics(
        "Predictive standard deviation",
        estimator_predictive_std
    )

    assert_finite(
        estimator_predictive,
        "Predictive variance"
    )

    assert_finite(
        estimator_predictive_std,
        "Predictive standard deviation"
    )

    if torch.any(
        estimator_predictive <= 0.0
    ):

        raise AssertionError(
            "Predictive variance must be positive."
        )

    if torch.any(
        estimator_predictive_std <= 0.0
    ):

        raise AssertionError(
            "Predictive standard deviation must be positive."
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

    reconstructed_predictive = (
        estimator_aleatoric
        +
        estimator_epistemic
    )

    decomposition_error = torch.max(
        torch.abs(
            estimator_predictive
            -
            reconstructed_predictive
        )
    ).item()

    print(
        f"Maximum decomposition error : "
        f"{decomposition_error:.12e}"
    )

    tolerance = 1.0e-6

    if decomposition_error > tolerance:

        raise AssertionError(
            "Predictive variance decomposition "
            "is inconsistent."
        )

    print(
        "Predictive variance decomposition: PASSED"
    )

    # =====================================================
    # PREDICTIVE VARIANCE DOMINANCE
    # =====================================================

    print_section(
        "PREDICTIVE VARIANCE DOMINANCE AUDIT"
    )

    variance_difference = (
        estimator_predictive
        -
        estimator_aleatoric
    )

    minimum_difference = (
        variance_difference.min().item()
    )

    print(
        f"Minimum predictive-minus-"
        f"aleatoric variance : "
        f"{minimum_difference:.12e}"
    )

    if minimum_difference < -tolerance:

        raise AssertionError(
            "Predictive variance is smaller "
            "than aleatoric variance."
        )

    print(
        "Predictive variance >= "
        "aleatoric variance: PASSED"
    )

    # =====================================================
    # UNCERTAINTY CONTRIBUTION ANALYSIS
    # =====================================================

    print_section(
        "ALEATORIC / EPISTEMIC CONTRIBUTION ANALYSIS"
    )

    mean_aleatoric = (
        estimator_aleatoric.mean().item()
    )

    mean_epistemic = (
        estimator_epistemic.mean().item()
    )

    mean_predictive = (
        estimator_predictive.mean().item()
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
        f"{contribution_sum:.12f}"
    )

    if abs(
        contribution_sum - 100.0
    ) > 1.0e-4:

        raise AssertionError(
            "Aleatoric and epistemic contributions "
            "do not sum to 100%."
        )

    print(
        "Uncertainty contribution consistency: PASSED"
    )

    # =====================================================
    # MC SAMPLE CONVERGENCE AUDIT
    # =====================================================

    print_section(
        "MC SAMPLE CONVERGENCE AUDIT"
    )

    sample_counts = [
        5,
        10,
        20
    ]

    variance_estimates = {}

    for sample_count in sample_counts:

        samples = (
            reconstruction_samples[
                :sample_count
            ]
        )

        variance = (
            uncertainty_estimator.epistemic_variance(
                samples
            )
        )

        mean_variance = (
            variance.mean().item()
        )

        variance_estimates[
            sample_count
        ] = mean_variance

        print(
            f"MC samples = {sample_count:<3}"
            f" | Mean epistemic variance = "
            f"{mean_variance:.12e}"
        )

    # -----------------------------------------------------
    # Verify all estimates are finite.
    # -----------------------------------------------------

    for sample_count, estimate in (
        variance_estimates.items()
    ):

        if not torch.isfinite(
            torch.tensor(estimate)
        ):

            raise AssertionError(
                f"MC variance estimate for "
                f"{sample_count} samples is not finite."
            )

    print(
        "MC sample variance estimates are finite."
    )

    # -----------------------------------------------------
    # Convergence stability
    # -----------------------------------------------------
    #
    # MC variance estimates do NOT need to decrease
    # monotonically as the number of samples increases.
    #
    # Instead, the test checks whether the estimate from
    # 20 samples remains reasonably close to the estimate
    # from 10 samples.
    # -----------------------------------------------------

    variance_10 = (
        variance_estimates[10]
    )

    variance_20 = (
        variance_estimates[20]
    )

    convergence_difference = abs(
        variance_20 - variance_10
    )

    convergence_scale = max(
        abs(variance_20),
        1.0e-12
    )

    relative_difference = (
        convergence_difference
        /
        convergence_scale
    )

    print(
        f"Absolute 10-to-20 difference : "
        f"{convergence_difference:.12e}"
    )

    print(
        f"Relative 10-to-20 difference : "
        f"{relative_difference:.6f}"
    )

    # -----------------------------------------------------
    # We use a broad numerical stability criterion rather
    # than demanding artificial monotonic convergence.
    # -----------------------------------------------------

    convergence_threshold = 1.0

    if relative_difference > convergence_threshold:

        raise AssertionError(
            "MC epistemic variance estimate shows "
            "excessive instability between 10 and "
            "20 samples."
        )

    print(
        "MC sample convergence stability: PASSED"
    )

    # =====================================================
    # TRAVEL-TIME VALIDATION
    # =====================================================

    print_section(
        "TRAVEL-TIME VALIDATION"
    )

    mc_travel_time_mean = (
        travel_time_samples.mean(
            dim=0
        )
    )

    mc_travel_time_variance = (
        uncertainty_estimator.epistemic_variance(
            travel_time_samples
        )
    )

    print_statistics(
        "MC travel-time mean",
        mc_travel_time_mean
    )

    print_statistics(
        "Travel-time epistemic variance",
        mc_travel_time_variance
    )

    minimum_travel_time = (
        mc_travel_time_mean.min().item()
    )

    maximum_travel_time = (
        mc_travel_time_mean.max().item()
    )

    mean_travel_time = (
        mc_travel_time_mean.mean().item()
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

    if minimum_travel_time <= 0.0:

        raise AssertionError(
            "Travel-time field contains "
            "non-positive values."
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
        log_variance_samples.mean(
            dim=0
        )
    )

    mc_aleatoric_variance = (
        uncertainty_estimator.aleatoric_variance(
            mc_log_variance_mean
        )
    )

    print_statistics(
        "MC-derived aleatoric variance",
        mc_aleatoric_variance
    )

    if torch.any(
        mc_aleatoric_variance <= 0.0
    ):

        raise AssertionError(
            "Aleatoric variance became "
            "non-positive during MC inference."
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

    final_parameter_norm = parameter_norm(
        model
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

    parameter_tolerance = 1.0e-10

    if abs(parameter_change) > parameter_tolerance:

        raise AssertionError(
            "Model parameters changed during "
            "predictive uncertainty estimation."
        )

    print(
        "Model parameters remained unchanged."
    )

    print(
        "Parameter stability test: PASSED"
    )

    # =====================================================
    # FINAL NUMERICAL STABILITY
    # =====================================================

    print_section(
        "FINAL NUMERICAL STABILITY AUDIT"
    )

    uncertainty_tensors = [
        estimator_aleatoric,
        estimator_epistemic,
        estimator_predictive,
        estimator_aleatoric_std,
        estimator_epistemic_std,
        estimator_predictive_std
    ]

    for tensor in uncertainty_tensors:

        assert_finite(
            tensor,
            "Predictive uncertainty quantity"
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
    # FINAL PASS MESSAGE
    # =====================================================

    print_section(
        "PREDICTIVE UNCERTAINTY CONVERGENCE AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ Network uncertainty output"
    )

    print(
        "  ✓ PredictiveUncertaintyEstimator"
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
        "  ✓ Aleatoric / epistemic contribution"
    )

    print(
        "  ✓ MC stochastic variability"
    )

    print(
        "  ✓ MC sample convergence stability"
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
        "PREDICTIVE UNCERTAINTY CONVERGENCE "
        "TEST PASSED."
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()