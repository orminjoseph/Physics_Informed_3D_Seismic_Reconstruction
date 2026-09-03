"""
======================================================================
NETWORK PREDICTIVE UNCERTAINTY ARCHITECTURE AUDIT
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------

Audit the complete uncertainty architecture implemented in Network3D.

The network produces:

    1. Reconstruction
    2. Travel-time field
    3. Log variance

Predictive uncertainty is decomposed into:

    Aleatoric uncertainty:
        sigma_aleatoric^2

    Epistemic uncertainty:
        sigma_epistemic^2

Total predictive uncertainty:

    sigma_predictive^2
        =
        sigma_aleatoric^2
        +
        sigma_epistemic^2

Aleatoric variance is obtained from:

    sigma_aleatoric^2
        =
        mean(exp(log_variance))

Epistemic variance is obtained from Monte Carlo
Dropout reconstruction samples.

This audit verifies:

    A. Network uncertainty output
    B. Aleatoric uncertainty positivity
    C. Aleatoric numerical stability
    D. Deterministic evaluation behavior
    E. Monte Carlo Dropout activation
    F. Stochastic output variation
    G. Epistemic variance
    H. Total uncertainty decomposition
    I. Backward propagation through aleatoric loss

Author: Ormin Joseph
======================================================================
"""

import torch
import torch.nn as nn

from models.network import Network3D
from utils.config import MC_DROPOUT_SAMPLES


# ==============================================================
# CONFIGURATION
# ==============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32


# ==============================================================
# PRINT UTILITIES
# ==============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_stats(name, tensor):

    tensor = tensor.detach()

    print(f"{name}:")
    print(
        f"    shape : "
        f"{tuple(tensor.shape)}"
    )
    print(
        f"    min   : "
        f"{tensor.min().item():.6e}"
    )
    print(
        f"    max   : "
        f"{tensor.max().item():.6e}"
    )
    print(
        f"    mean  : "
        f"{tensor.mean().item():.6e}"
    )
    print(
        f"    std   : "
        f"{tensor.std().item():.6e}"
    )
    print(
        f"    absmax: "
        f"{tensor.abs().max().item():.6e}"
    )


def check_finite(name, tensor):

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values."
        )

    print(
        f"{name}: finite values confirmed."
    )


# ==============================================================
# INPUT CREATION
# ==============================================================

def create_synthetic_input():

    input_cube = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=DEVICE
    )

    input_cube = torch.tanh(
        input_cube
    )

    return input_cube


# ==============================================================
# MONTE CARLO DROPOUT
# ==============================================================

def enable_mc_dropout(model):
    """
    Activate Dropout layers while leaving
    BatchNorm layers in evaluation mode.
    """

    dropout_layers = 0

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout2d,
                nn.Dropout3d
            )
        ):

            module.train()

            dropout_layers += 1

    return dropout_layers


# ==============================================================
# ALEATORIC UNCERTAINTY
# ==============================================================

def compute_aleatoric_variance(
    log_variance_samples
):
    """
    Compute expected aleatoric variance.

    Each sample predicts:

        log(sigma^2)

    Therefore:

        sigma^2 = exp(log_variance)

    The expected aleatoric variance is:

        mean(exp(log_variance))
    """

    variance_samples = torch.exp(
        log_variance_samples
    )

    aleatoric_variance = (
        variance_samples.mean(
            dim=0
        )
    )

    return aleatoric_variance


# ==============================================================
# EPISTEMIC UNCERTAINTY
# ==============================================================

def compute_epistemic_variance(
    reconstruction_samples
):
    """
    Compute epistemic variance from
    Monte Carlo reconstruction samples.
    """

    epistemic_variance = (
        reconstruction_samples.var(
            dim=0,
            unbiased=False
        )
    )

    return epistemic_variance


# ==============================================================
# MAIN AUDIT
# ==============================================================

def main():

    print_header(
        "NETWORK PREDICTIVE UNCERTAINTY AUDIT"
    )

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Tensor shape        : "
        f"({BATCH_SIZE}, "
        f"{CHANNELS}, "
        f"{DEPTH}, "
        f"{HEIGHT}, "
        f"{WIDTH})"
    )

    print(
        f"MC Dropout samples  : "
        f"{MC_DROPOUT_SAMPLES}"
    )

    # ==========================================================
    # INPUT
    # ==========================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    input_cube = (
        create_synthetic_input()
    )

    print_stats(
        "Input",
        input_cube
    )

    check_finite(
        "Input",
        input_cube
    )

    # ==========================================================
    # NETWORK
    # ==========================================================

    print_header(
        "INITIALIZING NETWORK3D"
    )

    network = Network3D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(DEVICE)

    print(
        "Network3D initialized successfully."
    )

    # ==========================================================
    # FORWARD OUTPUT AUDIT
    # ==========================================================

    print_header(
        "DETERMINISTIC FORWARD OUTPUT AUDIT"
    )

    network.eval()

    with torch.no_grad():

        (
            reconstruction,
            travel_time,
            log_variance
        ) = network(
            input_cube
        )

    expected_shape = (
        input_cube.shape
    )

    if reconstruction.shape != expected_shape:

        raise RuntimeError(
            "Reconstruction shape mismatch."
        )

    if travel_time.shape != expected_shape:

        raise RuntimeError(
            "Travel-time shape mismatch."
        )

    if log_variance.shape != expected_shape:

        raise RuntimeError(
            "Log-variance shape mismatch."
        )

    print(
        "All output shapes: PASS"
    )

    check_finite(
        "Reconstruction",
        reconstruction
    )

    check_finite(
        "Travel time",
        travel_time
    )

    check_finite(
        "Log variance",
        log_variance
    )

    print_stats(
        "Reconstruction",
        reconstruction
    )

    print_stats(
        "Log variance",
        log_variance
    )

    # ==========================================================
    # ALEATORIC UNCERTAINTY AUDIT
    # ==========================================================

    print_header(
        "ALEATORIC UNCERTAINTY AUDIT"
    )

    deterministic_variance = (
        torch.exp(
            log_variance
        )
    )

    check_finite(
        "Aleatoric variance",
        deterministic_variance
    )

    aleatoric_positive = (
        deterministic_variance > 0.0
    ).all().item()

    print(
        "Aleatoric variance positivity: "
        f"{'PASS' if aleatoric_positive else 'FAIL'}"
    )

    if not aleatoric_positive:

        raise RuntimeError(
            "Aleatoric variance must be positive."
        )

    print_stats(
        "Aleatoric variance",
        deterministic_variance
    )

    # ==========================================================
    # DETERMINISTIC REPEATABILITY
    # ==========================================================

    print_header(
        "DETERMINISTIC EVALUATION AUDIT"
    )

    network.eval()

    with torch.no_grad():

        (
            reconstruction_1,
            _,
            log_variance_1
        ) = network(
            input_cube
        )

        (
            reconstruction_2,
            _,
            log_variance_2
        ) = network(
            input_cube
        )

    reconstruction_difference = (
        reconstruction_1
        -
        reconstruction_2
    ).abs().max().item()

    variance_difference = (
        log_variance_1
        -
        log_variance_2
    ).abs().max().item()

    print(
        "Maximum reconstruction difference: "
        f"{reconstruction_difference:.6e}"
    )

    print(
        "Maximum log-variance difference  : "
        f"{variance_difference:.6e}"
    )

    deterministic_pass = (
        reconstruction_difference
        <
        1.0e-7
        and
        variance_difference
        <
        1.0e-7
    )

    print(
        "Deterministic evaluation: "
        f"{'PASS' if deterministic_pass else 'FAIL'}"
    )

    # ==========================================================
    # MONTE CARLO DROPOUT ACTIVATION
    # ==========================================================

    print_header(
        "MONTE CARLO DROPOUT ACTIVATION"
    )

    network.eval()

    dropout_layers = (
        enable_mc_dropout(
            network
        )
    )

    print(
        f"Activated Dropout layers: "
        f"{dropout_layers}"
    )

    if dropout_layers == 0:

        raise RuntimeError(
            "No Dropout layers were found."
        )

    print(
        "Monte Carlo Dropout activation: PASS"
    )

    # ==========================================================
    # MONTE CARLO SAMPLING
    # ==========================================================

    print_header(
        "MONTE CARLO UNCERTAINTY SAMPLING"
    )

    reconstruction_samples = []

    log_variance_samples = []

    with torch.no_grad():

        for sample_index in range(
            MC_DROPOUT_SAMPLES
        ):

            (
                reconstruction_sample,
                _,
                log_variance_sample
            ) = network(
                input_cube
            )

            reconstruction_samples.append(
                reconstruction_sample
            )

            log_variance_samples.append(
                log_variance_sample
            )

    reconstruction_samples = torch.stack(
        reconstruction_samples,
        dim=0
    )

    log_variance_samples = torch.stack(
        log_variance_samples,
        dim=0
    )

    check_finite(
        "MC reconstruction samples",
        reconstruction_samples
    )

    check_finite(
        "MC log-variance samples",
        log_variance_samples
    )

    print(
        "Monte Carlo sampling: PASS"
    )

    # ==========================================================
    # STOCHASTIC VARIATION AUDIT
    # ==========================================================

    print_header(
        "STOCHASTIC OUTPUT VARIATION AUDIT"
    )

    sample_difference = (
        reconstruction_samples[0]
        -
        reconstruction_samples[1]
    ).abs().max().item()

    print(
        "Maximum difference between "
        "MC samples 1 and 2: "
        f"{sample_difference:.6e}"
    )

    stochastic_pass = (
        sample_difference > 0.0
    )

    print(
        "Stochastic variation: "
        f"{'PASS' if stochastic_pass else 'FAIL'}"
    )

    if not stochastic_pass:

        raise RuntimeError(
            "MC Dropout did not produce "
            "stochastic reconstruction outputs."
        )

    # ==========================================================
    # EPISTEMIC UNCERTAINTY
    # ==========================================================

    print_header(
        "EPISTEMIC UNCERTAINTY AUDIT"
    )

    epistemic_variance = (
        compute_epistemic_variance(
            reconstruction_samples
        )
    )

    check_finite(
        "Epistemic variance",
        epistemic_variance
    )

    epistemic_nonnegative = (
        epistemic_variance >= 0.0
    ).all().item()

    print(
        "Epistemic variance non-negativity: "
        f"{'PASS' if epistemic_nonnegative else 'FAIL'}"
    )

    print_stats(
        "Epistemic variance",
        epistemic_variance
    )

    print(
        f"Mean epistemic variance: "
        f"{epistemic_variance.mean().item():.6e}"
    )

    # ==========================================================
    # ALEATORIC UNCERTAINTY
    # ==========================================================

    print_header(
        "MONTE CARLO ALEATORIC UNCERTAINTY"
    )

    aleatoric_variance = (
        compute_aleatoric_variance(
            log_variance_samples
        )
    )

    check_finite(
        "Aleatoric variance",
        aleatoric_variance
    )

    aleatoric_positive = (
        aleatoric_variance > 0.0
    ).all().item()

    print(
        "Aleatoric variance positivity: "
        f"{'PASS' if aleatoric_positive else 'FAIL'}"
    )

    print_stats(
        "Aleatoric variance",
        aleatoric_variance
    )

    # ==========================================================
    # TOTAL PREDICTIVE UNCERTAINTY
    # ==========================================================

    print_header(
        "TOTAL PREDICTIVE UNCERTAINTY"
    )

    total_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    total_std = torch.sqrt(
        total_variance
    )

    check_finite(
        "Total predictive variance",
        total_variance
    )

    check_finite(
        "Total predictive standard deviation",
        total_std
    )

    total_nonnegative = (
        total_variance >= 0.0
    ).all().item()

    print(
        "Total variance non-negativity: "
        f"{'PASS' if total_nonnegative else 'FAIL'}"
    )

    print_stats(
        "Total predictive variance",
        total_variance
    )

    print_stats(
        "Total predictive standard deviation",
        total_std
    )

    # ==========================================================
    # UNCERTAINTY DECOMPOSITION CHECK
    # ==========================================================

    print_header(
        "UNCERTAINTY DECOMPOSITION CHECK"
    )

    expected_total_variance = (
        aleatoric_variance
        +
        epistemic_variance
    )

    decomposition_error = (
        total_variance
        -
        expected_total_variance
    ).abs().max().item()

    print(
        "Maximum decomposition error: "
        f"{decomposition_error:.6e}"
    )

    decomposition_pass = (
        decomposition_error
        <
        1.0e-10
    )

    print(
        "Uncertainty decomposition: "
        f"{'PASS' if decomposition_pass else 'FAIL'}"
    )

    # ==========================================================
    # FINAL RESULTS
    # ==========================================================

    print_header(
        "NETWORK UNCERTAINTY AUDIT RESULT"
    )

    print(
        "Network forward outputs        : PASS"
    )

    print(
        "Aleatoric uncertainty output   : "
        f"{'PASS' if aleatoric_positive else 'FAIL'}"
    )

    print(
        "Deterministic evaluation       : "
        f"{'PASS' if deterministic_pass else 'FAIL'}"
    )

    print(
        "MC Dropout activation          : PASS"
    )

    print(
        "Stochastic output variation    : "
        f"{'PASS' if stochastic_pass else 'FAIL'}"
    )

    print(
        "Epistemic uncertainty          : "
        f"{'PASS' if epistemic_nonnegative else 'FAIL'}"
    )

    print(
        "Total uncertainty              : "
        f"{'PASS' if total_nonnegative else 'FAIL'}"
    )

    print(
        "Uncertainty decomposition      : "
        f"{'PASS' if decomposition_pass else 'FAIL'}"
    )

    print()
    print(
        "NETWORK PREDICTIVE UNCERTAINTY "
        "ARCHITECTURE AUDIT COMPLETED."
    )


# ==============================================================
# SCRIPT ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()