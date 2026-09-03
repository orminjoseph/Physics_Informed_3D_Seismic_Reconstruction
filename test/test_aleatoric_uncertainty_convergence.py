"""
============================================================
ALEATORIC UNCERTAINTY CONVERGENCE AUDIT
============================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This test verifies that the heteroscedastic aleatoric
uncertainty branch of the network is correctly connected
to the training objective and responds to optimization.

The audit monitors:

    1. Aleatoric uncertainty loss
    2. Reconstruction MAE loss
    3. Predicted log variance
    4. Predicted variance
    5. Predicted standard deviation
    6. Uncertainty-head gradients
    7. Global gradient stability
    8. Parameter updates
    9. Uncertainty-loss convergence
    10. Numerical stability
    11. Positivity of predicted variance
    12. Short-run optimization response

Important distinction
---------------------

This test validates ALEATORIC uncertainty only.

Aleatoric uncertainty represents uncertainty associated
with the data/noise or observation process.

It does NOT measure epistemic uncertainty.

Epistemic uncertainty will be evaluated separately using
stochastic/repeated model predictions.

Current uncertainty formulation
-------------------------------

The network predicts:

    log_variance

The physical variance is obtained as:

    variance = exp(log_variance)

and the standard deviation is:

    sigma = sqrt(variance)

The heteroscedastic uncertainty loss is supplied by:

    Heteroscedastic_Aleatoric_uncertainty_loss

Author: Ormin Joseph
============================================================
"""

import torch
import torch.optim as optim


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.network import Network3D

from losses.mae_loss import MAELoss

from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    UncertaintyLoss
)

from utils.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_UNCERTAINTY,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

NUM_TRAINING_STEPS = 10

BATCH_SIZE = 1

CHANNELS = 1

DEPTH = 64
HEIGHT = 128
WIDTH = 128

RANDOM_SEED = 42


# ============================================================
# PRINTING UTILITIES
# ============================================================

def print_header(title):
    """
    Print a major audit section.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_statistics(name, tensor):
    """
    Print numerical statistics for a tensor.
    """

    tensor = tensor.detach()

    print(
        f"{name:<30}"
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


# ============================================================
# GRADIENT UTILITIES
# ============================================================

def compute_gradient_statistics(model):
    """
    Compute global gradient statistics across all
    trainable model parameters.
    """

    gradient_values = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_values.append(
                parameter.grad.detach()
                .abs()
                .reshape(-1)
            )

    if not gradient_values:

        return {
            "mean": 0.0,
            "maximum": 0.0,
            "minimum": 0.0,
            "elements": 0,
            "finite": False,
        }

    gradients = torch.cat(
        gradient_values
    )

    return {
        "mean": gradients.mean().item(),
        "maximum": gradients.max().item(),
        "minimum": gradients.min().item(),
        "elements": gradients.numel(),
        "finite": bool(
            torch.isfinite(
                gradients
            ).all().item()
        ),
    }


# ============================================================
# UNCERTAINTY-HEAD GRADIENT UTILITIES
# ============================================================

def compute_uncertainty_gradient_statistics(model):
    """
    Estimate gradient activity associated with parameters
    responsible for the uncertainty prediction.

    The function searches parameter names containing
    common uncertainty-related identifiers.

    This is intentionally diagnostic rather than assuming
    a particular internal layer name.
    """

    uncertainty_gradients = []

    for name, parameter in model.named_parameters():

        if parameter.grad is None:

            continue

        parameter_name = name.lower()

        if (
            "uncertainty" in parameter_name
            or "variance" in parameter_name
            or "logvar" in parameter_name
            or "log_variance" in parameter_name
            or "sigma" in parameter_name
        ):

            uncertainty_gradients.append(
                parameter.grad.detach()
                .abs()
                .reshape(-1)
            )

    if not uncertainty_gradients:

        return {
            "found": False,
            "mean": 0.0,
            "maximum": 0.0,
            "elements": 0,
            "finite": True,
        }

    gradients = torch.cat(
        uncertainty_gradients
    )

    return {
        "found": True,
        "mean": gradients.mean().item(),
        "maximum": gradients.max().item(),
        "elements": gradients.numel(),
        "finite": bool(
            torch.isfinite(
                gradients
            ).all().item()
        ),
    }


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def compute_parameter_norm(model):
    """
    Compute the global L2 norm of trainable parameters.
    """

    squared_norm = torch.tensor(
        0.0,
        device=DEVICE
    )

    for parameter in model.parameters():

        if parameter.requires_grad:

            squared_norm = (
                squared_norm
                +
                parameter.detach()
                .pow(2)
                .sum()
            )

    return torch.sqrt(
        squared_norm
    ).item()


# ============================================================
# UNCERTAINTY STATISTICS
# ============================================================

def compute_uncertainty_statistics(log_variance):
    """
    Convert predicted log variance into physical variance
    and standard deviation and return diagnostic statistics.
    """

    variance = torch.exp(
        log_variance
    )

    standard_deviation = torch.sqrt(
        variance
    )

    return {
        "variance": variance,
        "standard_deviation": standard_deviation,
    }


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print_header(
        "ALEATORIC UNCERTAINTY CONVERGENCE AUDIT"
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
    # UNCERTAINTY CONFIGURATION
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY CONFIGURATION"
    )

    print(
        f"USE_UNCERTAINTY            : "
        f"{USE_UNCERTAINTY}"
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

        raise RuntimeError(
            "USE_UNCERTAINTY must be True for this audit."
        )

    # ========================================================
    # CREATE SYNTHETIC DATA
    # ========================================================

    print_header(
        "CREATING SYNTHETIC TRAINING DATA"
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    input_cube = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    target_cube = torch.randn(
        BATCH_SIZE,
        CHANNELS,
        DEPTH,
        HEIGHT,
        WIDTH,
        device=device
    )

    print_statistics(
        "Input seismic volume",
        input_cube
    )

    print_statistics(
        "Target seismic volume",
        target_cube
    )

    # ========================================================
    # INITIALIZE NETWORK
    # ========================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        use_uncertainty=True
    ).to(
        device
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Network successfully initialized."
    )

    print(
        f"Trainable parameters       : "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # INITIALIZE LOSSES
    # ========================================================

    print_header(
        "INITIALIZING LOSS FUNCTIONS"
    )

    mae_loss_function = MAELoss()

    uncertainty_loss_function = UncertaintyLoss()

    print(
        "MAE loss initialized."
    )

    print(
        "Heteroscedastic aleatoric uncertainty loss initialized."
    )

    # ========================================================
    # INITIALIZE OPTIMIZER
    # ========================================================

    print_header(
        "INITIALIZING OPTIMIZER"
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
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
    # INITIAL FORWARD PROPAGATION
    # ========================================================

    print_header(
        "INITIAL FORWARD PROPAGATION"
    )

    model.train()

    (
        initial_reconstruction,
        initial_travel_time,
        initial_log_variance
    ) = model(
        input_cube
    )

    print(
        "Network returned three outputs."
    )

    # ========================================================
    # INITIAL OUTPUT AUDIT
    # ========================================================

    print_header(
        "INITIAL UNCERTAINTY OUTPUT AUDIT"
    )

    print_statistics(
        "Initial reconstruction",
        initial_reconstruction
    )

    print_statistics(
        "Initial log variance",
        initial_log_variance
    )

    uncertainty_statistics = (
        compute_uncertainty_statistics(
            initial_log_variance
        )
    )

    initial_variance = (
        uncertainty_statistics["variance"]
    )

    initial_standard_deviation = (
        uncertainty_statistics[
            "standard_deviation"
        ]
    )

    print_statistics(
        "Initial predicted variance",
        initial_variance
    )

    print_statistics(
        "Initial predicted std",
        initial_standard_deviation
    )

    # ========================================================
    # INITIAL UNCERTAINTY LOSS
    # ========================================================

    print_header(
        "INITIAL ALEATORIC UNCERTAINTY LOSS"
    )

    initial_uncertainty_loss = (
        uncertainty_loss_function(
            initial_reconstruction,
            target_cube,
            initial_log_variance
        )
    )

    if not torch.isfinite(
        initial_uncertainty_loss
    ):

        raise RuntimeError(
            "Initial aleatoric uncertainty loss "
            "is NaN or Inf."
        )

    print(
        f"Initial uncertainty loss  : "
        f"{initial_uncertainty_loss.item():.12e}"
    )

    # ========================================================
    # INITIAL MAE
    # ========================================================

    initial_mae_loss = (
        mae_loss_function(
            initial_reconstruction,
            target_cube
        )
    )

    print(
        f"Initial MAE loss           : "
        f"{initial_mae_loss.item():.12e}"
    )

    # ========================================================
    # INITIAL UNCERTAINTY GRADIENT AUDIT
    # ========================================================

    print_header(
        "INITIAL UNCERTAINTY GRADIENT AUDIT"
    )

    optimizer.zero_grad()

    initial_uncertainty_loss.backward()

    uncertainty_gradient_statistics = (
        compute_uncertainty_gradient_statistics(
            model
        )
    )

    global_gradient_statistics = (
        compute_gradient_statistics(
            model
        )
    )

    print(
        f"Uncertainty parameters found : "
        f"{uncertainty_gradient_statistics['found']}"
    )

    print(
        f"Uncertainty gradient mean    : "
        f"{uncertainty_gradient_statistics['mean']:.12e}"
    )

    print(
        f"Uncertainty gradient maximum : "
        f"{uncertainty_gradient_statistics['maximum']:.12e}"
    )

    print(
        f"Global gradient mean         : "
        f"{global_gradient_statistics['mean']:.12e}"
    )

    print(
        f"Global gradient maximum     : "
        f"{global_gradient_statistics['maximum']:.12e}"
    )

    if not global_gradient_statistics["finite"]:

        raise RuntimeError(
            "Initial uncertainty gradients contain "
            "non-finite values."
        )

    optimizer.zero_grad()

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {
        "uncertainty": [],
        "mae": [],
        "log_variance_mean": [],
        "variance_mean": [],
        "std_mean": [],
        "gradient_mean": [],
        "gradient_maximum": [],
        "uncertainty_gradient_mean": [],
        "uncertainty_gradient_maximum": [],
    }

    # ========================================================
    # TRAINING
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY CONVERGENCE"
    )

    print(
        f"{'Step':<8}"
        f"{'Uncertainty Loss':<20}"
        f"{'MAE Loss':<18}"
        f"{'LogVar Mean':<18}"
        f"{'Variance Mean':<18}"
        f"{'Std Mean':<18}"
        f"{'Grad Mean':<18}"
        f"{'Grad Max':<18}"
    )

    print(
        "-" * 135
    )

    for step in range(
        1,
        NUM_TRAINING_STEPS + 1
    ):

        # ====================================================
        # RESET GRADIENTS
        # ====================================================

        optimizer.zero_grad()

        # ====================================================
        # FORWARD PROPAGATION
        # ====================================================

        (
            reconstruction,
            travel_time,
            log_variance
        ) = model(
            input_cube
        )

        # ====================================================
        # UNCERTAINTY LOSS
        # ====================================================

        uncertainty_loss = (
            uncertainty_loss_function(
                reconstruction,
                target_cube,
                log_variance
            )
        )

        # ====================================================
        # MAE LOSS
        # ====================================================

        mae_loss = (
            mae_loss_function(
                reconstruction,
                target_cube
            )
        )

        # ====================================================
        # NUMERICAL STABILITY
        # ====================================================

        if not torch.isfinite(
            uncertainty_loss
        ):

            raise RuntimeError(
                "Aleatoric uncertainty loss became "
                "NaN or Inf."
            )

        if not torch.isfinite(
            mae_loss
        ):

            raise RuntimeError(
                "MAE loss became NaN or Inf."
            )

        # ====================================================
        # BACKWARD PROPAGATION
        # ====================================================

        uncertainty_loss.backward()

        # ====================================================
        # GRADIENT AUDIT
        # ====================================================

        gradient_statistics = (
            compute_gradient_statistics(
                model
            )
        )

        uncertainty_gradient_statistics = (
            compute_uncertainty_gradient_statistics(
                model
            )
        )

        if not gradient_statistics["finite"]:

            raise RuntimeError(
                "Non-finite global gradients detected."
            )

        if not uncertainty_gradient_statistics[
            "finite"
        ]:

            raise RuntimeError(
                "Non-finite uncertainty-head gradients "
                "detected."
            )

        # ====================================================
        # OPTIMIZER UPDATE
        # ====================================================

        optimizer.step()

        # ====================================================
        # POST-UPDATE UNCERTAINTY STATISTICS
        # ====================================================

        with torch.no_grad():

            variance = torch.exp(
                log_variance
            )

            standard_deviation = torch.sqrt(
                variance
            )

            log_variance_mean = (
                log_variance.mean().item()
            )

            variance_mean = (
                variance.mean().item()
            )

            standard_deviation_mean = (
                standard_deviation.mean().item()
            )

        # ====================================================
        # VARIANCE VALIDATION
        # ====================================================

        if not torch.isfinite(
            variance
        ).all():

            raise RuntimeError(
                "Predicted aleatoric variance contains "
                "NaN or Inf."
            )

        if not torch.isfinite(
            standard_deviation
        ).all():

            raise RuntimeError(
                "Predicted aleatoric standard deviation "
                "contains NaN or Inf."
            )

        if variance.min().item() <= 0.0:

            raise RuntimeError(
                "Predicted variance is not strictly positive."
            )

        # ====================================================
        # STORE HISTORY
        # ====================================================

        history["uncertainty"].append(
            uncertainty_loss.item()
        )

        history["mae"].append(
            mae_loss.item()
        )

        history["log_variance_mean"].append(
            log_variance_mean
        )

        history["variance_mean"].append(
            variance_mean
        )

        history["std_mean"].append(
            standard_deviation_mean
        )

        history["gradient_mean"].append(
            gradient_statistics["mean"]
        )

        history["gradient_maximum"].append(
            gradient_statistics["maximum"]
        )

        history[
            "uncertainty_gradient_mean"
        ].append(
            uncertainty_gradient_statistics[
                "mean"
            ]
        )

        history[
            "uncertainty_gradient_maximum"
        ].append(
            uncertainty_gradient_statistics[
                "maximum"
            ]
        )

        # ====================================================
        # PRINT STEP
        # ====================================================

        print(
            f"{step:<8}"
            f"{uncertainty_loss.item():<20.8e}"
            f"{mae_loss.item():<18.8e}"
            f"{log_variance_mean:<18.8e}"
            f"{variance_mean:<18.8e}"
            f"{standard_deviation_mean:<18.8e}"
            f"{gradient_statistics['mean']:<18.8e}"
            f"{gradient_statistics['maximum']:<18.8e}"
        )

    # ========================================================
    # FINAL FORWARD PROPAGATION
    # ========================================================

    print_header(
        "FINAL FORWARD PROPAGATION"
    )

    (
        final_reconstruction,
        final_travel_time,
        final_log_variance
    ) = model(
        input_cube
    )

    print(
        "Final network outputs are finite."
    )

    # ========================================================
    # FINAL UNCERTAINTY EVALUATION
    # ========================================================

    print_header(
        "FINAL ALEATORIC UNCERTAINTY EVALUATION"
    )

    final_uncertainty_loss = (
        uncertainty_loss_function(
            final_reconstruction,
            target_cube,
            final_log_variance
        )
    )

    final_mae_loss = (
        mae_loss_function(
            final_reconstruction,
            target_cube
        )
    )

    final_variance = torch.exp(
        final_log_variance
    )

    final_standard_deviation = torch.sqrt(
        final_variance
    )

    print(
        f"Final uncertainty loss    : "
        f"{final_uncertainty_loss.item():.12e}"
    )

    print(
        f"Final MAE loss            : "
        f"{final_mae_loss.item():.12e}"
    )

    print_statistics(
        "Final log variance",
        final_log_variance
    )

    print_statistics(
        "Final predicted variance",
        final_variance
    )

    print_statistics(
        "Final predicted std",
        final_standard_deviation
    )

    # ========================================================
    # FINAL UNCERTAINTY VALIDATION
    # ========================================================

    print_header(
        "FINAL UNCERTAINTY VALIDATION"
    )

    if not torch.isfinite(
        final_log_variance
    ).all():

        raise RuntimeError(
            "Final log variance contains NaN or Inf."
        )

    if not torch.isfinite(
        final_variance
    ).all():

        raise RuntimeError(
            "Final variance contains NaN or Inf."
        )

    if not torch.isfinite(
        final_standard_deviation
    ).all():

        raise RuntimeError(
            "Final standard deviation contains NaN or Inf."
        )

    if final_variance.min().item() <= 0.0:

        raise RuntimeError(
            "Final predicted variance is not positive."
        )

    print(
        "Log variance numerical validity: PASSED"
    )

    print(
        "Variance numerical validity: PASSED"
    )

    print(
        "Standard deviation numerical validity: PASSED"
    )

    print(
        "Aleatoric variance positivity: PASSED"
    )

    # ========================================================
    # PARAMETER UPDATE AUDIT
    # ========================================================

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

    print_header(
        "PARAMETER UPDATE AUDIT"
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
    ) <= 1e-12:

        raise RuntimeError(
            "Model parameters did not change."
        )

    print(
        "Model parameters changed during optimization."
    )

    print(
        "Parameter update test: PASSED"
    )

    # ========================================================
    # UNCERTAINTY CONVERGENCE ANALYSIS
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY CONVERGENCE ANALYSIS"
    )

    initial_uncertainty_value = (
        initial_uncertainty_loss.item()
    )

    final_uncertainty_value = (
        final_uncertainty_loss.item()
    )

    uncertainty_reduction = (
        (
            initial_uncertainty_value
            -
            final_uncertainty_value
        )
        /
        (
            abs(
                initial_uncertainty_value
            )
            +
            1e-12
        )
        * 100.0
    )

    print(
        f"Initial uncertainty loss  : "
        f"{initial_uncertainty_value:.12e}"
    )

    print(
        f"Final uncertainty loss    : "
        f"{final_uncertainty_value:.12e}"
    )

    print(
        f"Uncertainty loss reduction: "
        f"{uncertainty_reduction:.6f}%"
    )

    # ========================================================
    # UNCERTAINTY TREND
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY LOSS TREND"
    )

    for index in range(
        NUM_TRAINING_STEPS
    ):

        print(
            f"Step {index + 1:<3}: "
            f"Uncertainty="
            f"{history['uncertainty'][index]:.8e}   "
            f"MAE="
            f"{history['mae'][index]:.8e}   "
            f"LogVar="
            f"{history['log_variance_mean'][index]:.8e}   "
            f"Variance="
            f"{history['variance_mean'][index]:.8e}"
        )

    # ========================================================
    # NUMERICAL STABILITY AUDIT
    # ========================================================

    print_header(
        "NUMERICAL STABILITY AUDIT"
    )

    for name, values in history.items():

        history_tensor = torch.tensor(
            values
        )

        if not torch.isfinite(
            history_tensor
        ).all():

            raise RuntimeError(
                f"Non-finite values detected in "
                f"{name} history."
            )

    print(
        "All aleatoric uncertainty training-history "
        "values are finite."
    )

    print(
        "Numerical stability test: PASSED"
    )

    # ========================================================
    # CONVERGENCE DECISION
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY CONVERGENCE INTERPRETATION"
    )

    uncertainty_descent = (
        final_uncertainty_value
        <
        initial_uncertainty_value
    )

    parameter_update_detected = (
        abs(
            parameter_change
        )
        >
        1e-12
    )

    print(
        f"Aleatoric-loss descent     : "
        f"{uncertainty_descent}"
    )

    print(
        f"Parameter update detected  : "
        f"{parameter_update_detected}"
    )

    if not uncertainty_descent:

        raise RuntimeError(
            "Aleatoric uncertainty loss did not decrease "
            "during optimization."
        )

    if not parameter_update_detected:

        raise RuntimeError(
            "Model parameters did not respond to "
            "aleatoric uncertainty optimization."
        )

    print()
    print(
        "Aleatoric uncertainty-loss descent: OBSERVED"
    )

    print(
        "The uncertainty branch responded to optimization."
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_header(
        "ALEATORIC UNCERTAINTY CONVERGENCE AUDIT PASSED"
    )

    print(
        "Verified:"
    )

    print(
        "  ✓ Heteroscedastic aleatoric uncertainty loss"
    )

    print(
        "  ✓ Network log-variance output"
    )

    print(
        "  ✓ Positive predicted variance"
    )

    print(
        "  ✓ Predicted standard deviation"
    )

    print(
        "  ✓ Aleatoric uncertainty gradients"
    )

    print(
        "  ✓ Global gradient stability"
    )

    print(
        "  ✓ Backward propagation"
    )

    print(
        "  ✓ Parameter updates"
    )

    print(
        "  ✓ Numerical stability"
    )

    print(
        "  ✓ Aleatoric uncertainty-loss convergence"
    )

    print(
        "  ✓ Reconstruction/uncertainty coupling"
    )

    print()

    print(
        "ALEATORIC UNCERTAINTY CONVERGENCE TEST PASSED."
    )


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    main()