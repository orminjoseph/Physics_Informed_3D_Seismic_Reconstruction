"""
=========================================================
PHYSICS LOSS SCALE AUDIT
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data
Reconstruction in Complex Geological Settings.

Purpose
-------

This test investigates the numerical scale of the
Eikonal physics loss relative to the reconstruction
and uncertainty terms.

The audit does NOT modify the model or configuration.

It investigates:

    1. Travel-time magnitude
    2. Travel-time gradient magnitude
    3. Expected physical gradient magnitude
    4. Eikonal residual magnitude
    5. Eikonal loss magnitude
    6. Reconstruction loss magnitude
    7. Uncertainty loss magnitude
    8. Physics contribution to total loss
    9. Relative loss contributions
    10. Physics gradient magnitude
    11. Gradient-to-loss relationship
    12. Potential numerical imbalance

Governing equation
------------------

    |grad T|^2 = 1 / V^2

Equivalent normalized form:

    V^2 |grad T|^2 = 1

Therefore the expected gradient magnitude is
approximately:

    |grad T| = 1 / V

For V = 2000 m/s:

    |grad T| = 5e-4 s/m

Tensor convention
-----------------

    [B, C, D, H, W]

Author: Ormin Joseph
=========================================================
"""

import torch

from models.network import Network3D
from losses.physics_loss import PhysicsLoss

from utils.config import (
    DX,
    DY,
    DZ,
    TRAVEL_TIME_SCALE,
    VELOCITY_MIN,
    VELOCITY_MAX,
    DEVICE,
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    SEISMIC_AMPLITUDE_MIN,
    SEISMIC_AMPLITUDE_MAX
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def print_separator():
    """Print a standard separator."""
    print("=" * 60)


def print_header(title):
    """Print a formatted section header."""
    print()
    print_separator()
    print(title)
    print_separator()


def tensor_statistics(name, tensor):
    """
    Print basic numerical statistics for a tensor.
    """

    print(
        f"{name:<30}: "
        f"min={tensor.min().item():.6e}, "
        f"max={tensor.max().item():.6e}, "
        f"mean={tensor.mean().item():.6e}, "
        f"std={tensor.std().item():.6e}"
    )


def count_parameters(model):
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print_header(
        "PHYSICS LOSS SCALE AUDIT"
    )

    # =====================================================
    # 1. DEVICE CONFIGURATION
    # =====================================================

    print_header(
        "DEVICE CONFIGURATION"
    )

    device = torch.device(
        DEVICE
    )

    print(
        f"Device                     : {device}"
    )

    # =====================================================
    # 2. PHYSICAL CONFIGURATION
    # =====================================================

    print_header(
        "PHYSICAL CONFIGURATION"
    )

    print(
        f"DX                         : {DX}"
    )

    print(
        f"DY                         : {DY}"
    )

    print(
        f"DZ                         : {DZ}"
    )

    print(
        f"Travel-time scale          : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"Velocity minimum           : "
        f"{VELOCITY_MIN} m/s"
    )

    print(
        f"Velocity maximum           : "
        f"{VELOCITY_MAX} m/s"
    )

    print(
        f"Physics loss weight        : "
        f"{LOSS_WEIGHTS['physics']}"
    )

    print(
        f"Eikonal weight             : "
        f"{PHYSICS_LOSS_WEIGHTS['eikonal']}"
    )

    print(
        f"Uncertainty weight         : "
        f"{LOSS_WEIGHTS['uncertainty']}"
    )

    print(
        f"MAE weight                 : "
        f"{LOSS_WEIGHTS['mae']}"
    )

    print(
        f"SSIM weight                : "
        f"{LOSS_WEIGHTS['ssim']}"
    )

    # =====================================================
    # 3. EXPECTED PHYSICAL GRADIENT
    # =====================================================

    print_header(
        "EXPECTED EIKONAL GRADIENT SCALE"
    )

    reference_velocity = (
        0.5
        *
        (
            VELOCITY_MIN
            +
            VELOCITY_MAX
        )
    )

    expected_gradient = (
        1.0
        /
        reference_velocity
    )

    print(
        f"Reference velocity        : "
        f"{reference_velocity:.6e} m/s"
    )

    print(
        f"Expected |grad T|         : "
        f"{expected_gradient:.6e} s/m"
    )

    print(
        "For V = 2000 m/s, the "
        "expected gradient is approximately "
        "5.0e-4 s/m."
    )

    # =====================================================
    # 4. CREATE SYNTHETIC INPUT
    # =====================================================

    print_header(
        "CREATING SYNTHETIC INPUT"
    )

    torch.manual_seed(42)

    shape = (
        1,
        1,
        64,
        128,
        128
    )

    seismic_input = torch.randn(
        shape,
        device=device
    )

    print(
        f"Input shape                : "
        f"{tuple(seismic_input.shape)}"
    )

    tensor_statistics(
        "Input seismic volume",
        seismic_input
    )

    # =====================================================
    # 5. CREATE PHYSICAL VELOCITY
    # =====================================================

    print_header(
        "CREATING PHYSICAL VELOCITY FIELD"
    )

    velocity_value = 2000.0

    velocity = torch.full(
        shape,
        velocity_value,
        device=device
    )

    tensor_statistics(
        "Velocity field",
        velocity
    )

    # =====================================================
    # 6. INITIALIZE NETWORK
    # =====================================================

    print_header(
        "INITIALIZING 3D NETWORK"
    )

    model = Network3D(
        in_channels=1,
        out_channels=1,
        use_uncertainty=True,
        use_residual=True,
        use_attention=True
    ).to(device)

    model.train()

    print(
        "Network successfully initialized."
    )

    print(
        f"Trainable parameters       : "
        f"{count_parameters(model):,}"
    )

    # =====================================================
    # 7. FORWARD PROPAGATION
    # =====================================================

    print_header(
        "FORWARD PROPAGATION"
    )

    (
        reconstruction,
        travel_time,
        log_variance
    ) = model(
        seismic_input
    )

    print(
        "Network returned three outputs."
    )

    # =====================================================
    # 8. OUTPUT STATISTICS
    # =====================================================

    print_header(
        "NETWORK OUTPUT SCALE"
    )

    tensor_statistics(
        "Reconstruction",
        reconstruction
    )

    tensor_statistics(
        "Travel-time",
        travel_time
    )

    tensor_statistics(
        "Log variance",
        log_variance
    )

    # =====================================================
    # 9. TRAVEL-TIME RANGE AUDIT
    # =====================================================

    print_header(
        "TRAVEL-TIME SCALE AUDIT"
    )

    print(
        f"Configured TRAVEL_TIME_SCALE : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print(
        f"Observed travel-time maximum : "
        f"{travel_time.max().item():.6e}"
    )

    print(
        f"Observed travel-time mean    : "
        f"{travel_time.mean().item():.6e}"
    )

    print(
        f"Observed travel-time minimum : "
        f"{travel_time.min().item():.6e}"
    )

    if torch.any(
        travel_time < 0.0
    ):

        raise RuntimeError(
            "Travel-time field contains "
            "negative values."
        )

    print(
        "Travel-time positivity: PASSED"
    )

    # =====================================================
    # 10. PHYSICS LOSS INITIALIZATION
    # =====================================================

    print_header(
        "INITIALIZING PHYSICS LOSS"
    )

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
        eikonal_weight=PHYSICS_LOSS_WEIGHTS[
            "eikonal"
        ],
        source_weight=PHYSICS_LOSS_WEIGHTS[
            "source"
        ],
        travel_time_weight=PHYSICS_LOSS_WEIGHTS[
            "travel_time"
        ]
    )

    print(
        "PhysicsLoss successfully initialized."
    )

    # =====================================================
    # 11. SPATIAL DERIVATIVES
    # =====================================================

    print_header(
        "TRAVEL-TIME GRADIENT SCALE"
    )

    dT_dz = physics_loss._derivative(
        travel_time,
        spacing=DZ,
        dimension=2
    )

    dT_dy = physics_loss._derivative(
        travel_time,
        spacing=DY,
        dimension=3
    )

    dT_dx = physics_loss._derivative(
        travel_time,
        spacing=DX,
        dimension=4
    )

    tensor_statistics(
        "dT/dz",
        dT_dz
    )

    tensor_statistics(
        "dT/dy",
        dT_dy
    )

    tensor_statistics(
        "dT/dx",
        dT_dx
    )

    # =====================================================
    # 12. GRADIENT MAGNITUDE
    # =====================================================

    gradient_magnitude = torch.sqrt(
        dT_dx.pow(2)
        +
        dT_dy.pow(2)
        +
        dT_dz.pow(2)
        +
        1.0e-12
    )

    tensor_statistics(
        "|grad T|",
        gradient_magnitude
    )

    observed_gradient_mean = (
        gradient_magnitude.mean().item()
    )

    print(
        f"Expected gradient scale   : "
        f"{expected_gradient:.6e}"
    )

    print(
        f"Observed gradient mean    : "
        f"{observed_gradient_mean:.6e}"
    )

    gradient_ratio = (
        observed_gradient_mean
        /
        expected_gradient
    )

    print(
        f"Observed / expected ratio : "
        f"{gradient_ratio:.6e}"
    )

    # =====================================================
    # 13. EIKONAL RESIDUAL
    # =====================================================

    print_header(
        "EIKONAL RESIDUAL SCALE"
    )

    residual = physics_loss.eikonal_residual(
        travel_time,
        velocity
    )

    tensor_statistics(
        "Eikonal residual",
        residual
    )

    print(
        f"Maximum absolute residual : "
        f"{residual.abs().max().item():.6e}"
    )

    # =====================================================
    # 14. EIKONAL LOSS
    # =====================================================

    print_header(
        "EIKONAL LOSS SCALE"
    )

    eikonal_loss = physics_loss.eikonal_loss(
        travel_time,
        velocity
    )

    print(
        f"Eikonal loss              : "
        f"{eikonal_loss.item():.12e}"
    )

    # =====================================================
    # 15. RECONSTRUCTION LOSS
    # =====================================================

    print_header(
        "RECONSTRUCTION LOSS SCALE"
    )

    target = torch.randn(
        shape,
        device=device
    )

    reconstruction_mae = (
        reconstruction
        -
        target
    ).abs().mean()

    print(
        f"MAE loss                  : "
        f"{reconstruction_mae.item():.12e}"
    )

    # =====================================================
    # 16. UNCERTAINTY LOSS SCALE
    # =====================================================

    print_header(
        "UNCERTAINTY LOSS SCALE"
    )

    uncertainty_probe = (
        log_variance.abs().mean()
    )

    print(
        f"Mean |log variance|       : "
        f"{uncertainty_probe.item():.12e}"
    )

    # =====================================================
    # 17. STRUCTURAL LOSS SCALE PROXY
    # =====================================================

    print_header(
        "STRUCTURAL-LOSS SCALE PROXY"
    )

    data_range = (
        SEISMIC_AMPLITUDE_MAX
        -
        SEISMIC_AMPLITUDE_MIN
    )

    normalized_difference = (
        (
            reconstruction
            -
            target
        )
        /
        data_range
    )

    structural_proxy = (
        normalized_difference.abs().mean()
    )

    print(
        f"Data range                : "
        f"{data_range:.6e}"
    )

    print(
        f"Structural-loss proxy     : "
        f"{structural_proxy.item():.12e}"
    )

    # =====================================================
    # 18. WEIGHTED LOSS CONTRIBUTIONS
    # =====================================================

    print_header(
        "WEIGHTED LOSS CONTRIBUTIONS"
    )

    mae_weighted = (
        LOSS_WEIGHTS["mae"]
        *
        reconstruction_mae
    )

    physics_weighted = (
        LOSS_WEIGHTS["physics"]
        *
        PHYSICS_LOSS_WEIGHTS["eikonal"]
        *
        eikonal_loss
    )

    uncertainty_weighted = (
        LOSS_WEIGHTS["uncertainty"]
        *
        uncertainty_probe
    )

    ssim_weighted = (
        LOSS_WEIGHTS["ssim"]
        *
        structural_proxy
    )

    total_proxy = (
        mae_weighted
        +
        physics_weighted
        +
        uncertainty_weighted
        +
        ssim_weighted
    )

    print(
        f"MAE contribution         : "
        f"{mae_weighted.item():.12e}"
    )

    print(
        f"Physics contribution     : "
        f"{physics_weighted.item():.12e}"
    )

    print(
        f"Uncertainty contribution : "
        f"{uncertainty_weighted.item():.12e}"
    )

    print(
        f"SSIM proxy contribution  : "
        f"{ssim_weighted.item():.12e}"
    )

    print(
        f"Diagnostic total         : "
        f"{total_proxy.item():.12e}"
    )

    # =====================================================
    # 19. RELATIVE LOSS CONTRIBUTIONS
    # =====================================================

    print_header(
        "RELATIVE LOSS CONTRIBUTIONS"
    )

    total_value = (
        total_proxy.item()
    )

    if total_value <= 0.0:

        raise RuntimeError(
            "Diagnostic total loss must "
            "be positive."
        )

    mae_fraction = (
        mae_weighted.item()
        /
        total_value
    )

    physics_fraction = (
        physics_weighted.item()
        /
        total_value
    )

    uncertainty_fraction = (
        uncertainty_weighted.item()
        /
        total_value
    )

    ssim_fraction = (
        ssim_weighted.item()
        /
        total_value
    )

    print(
        f"MAE fraction             : "
        f"{mae_fraction:.6%}"
    )

    print(
        f"Physics fraction         : "
        f"{physics_fraction:.6%}"
    )

    print(
        f"Uncertainty fraction     : "
        f"{uncertainty_fraction:.6%}"
    )

    print(
        f"SSIM proxy fraction      : "
        f"{ssim_fraction:.6%}"
    )

    # =====================================================
    # 20. PHYSICS / MAE RATIO
    # =====================================================

    print_header(
        "PHYSICS-TO-RECONSTRUCTION SCALE RATIO"
    )

    mae_value = (
        reconstruction_mae.item()
    )

    physics_value = (
        physics_weighted.item()
    )

    if mae_value > 0.0:

        physics_to_mae = (
            physics_value
            /
            mae_value
        )

    else:

        physics_to_mae = float(
            "inf"
        )

    print(
        f"Weighted physics loss    : "
        f"{physics_value:.12e}"
    )

    print(
        f"Weighted MAE loss        : "
        f"{mae_value:.12e}"
    )

    print(
        f"Physics / MAE ratio      : "
        f"{physics_to_mae:.6e}"
    )

    # =====================================================
    # 21. BACKWARD GRADIENT AUDIT
    # =====================================================

    print_header(
        "PHYSICS GRADIENT SCALE AUDIT"
    )

    model.zero_grad(
        set_to_none=True
    )

    physics_value_for_backward = (
        physics_loss.eikonal_loss(
            travel_time,
            velocity
        )
    )

    print(
        f"Physics loss requires grad: "
        f"{physics_value_for_backward.requires_grad}"
    )

    physics_value_for_backward.backward()

    gradient_values = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_values.append(
                parameter.grad.detach()
                .abs()
                .reshape(-1)
            )

    if not gradient_values:

        raise RuntimeError(
            "No network gradients were produced."
        )

    all_gradients = torch.cat(
        gradient_values
    )

    gradient_mean = (
        all_gradients.mean().item()
    )

    gradient_max = (
        all_gradients.max().item()
    )

    gradient_min = (
        all_gradients.min().item()
    )

    print(
        f"Gradient mean            : "
        f"{gradient_mean:.12e}"
    )

    print(
        f"Gradient maximum         : "
        f"{gradient_max:.12e}"
    )

    print(
        f"Gradient minimum         : "
        f"{gradient_min:.12e}"
    )

    print(
        f"Gradient elements        : "
        f"{all_gradients.numel():,}"
    )

    if not torch.isfinite(
        all_gradients
    ).all():

        raise RuntimeError(
            "Physics gradients contain "
            "NaN or Inf."
        )

    print(
        "Physics gradients finite : PASSED"
    )

    # =====================================================
    # 22. DIAGNOSTIC INTERPRETATION
    # =====================================================

    print_header(
        "SCALE AUDIT INTERPRETATION"
    )

    if gradient_ratio > 10.0:

        print(
            "WARNING:"
        )

        print(
            "The observed travel-time gradient is "
            "more than one order of magnitude above "
            "the expected physical gradient scale."
        )

    elif gradient_ratio < 0.1:

        print(
            "WARNING:"
        )

        print(
            "The observed travel-time gradient is "
            "more than one order of magnitude below "
            "the expected physical gradient scale."
        )

    else:

        print(
            "Travel-time gradient is within one "
            "order of magnitude of the expected "
            "physical scale."
        )

    if physics_to_mae > 1000.0:

        print()

        print(
            "WARNING:"
        )

        print(
            "The weighted physics contribution is "
            "more than 1000 times the MAE contribution."
        )

        print(
            "The composite loss is therefore strongly "
            "dominated by the physics term at the "
            "current initialization."
        )

    elif physics_to_mae > 100.0:

        print()

        print(
            "WARNING:"
        )

        print(
            "The weighted physics contribution is "
            "more than 100 times the MAE contribution."
        )

    else:

        print()

        print(
            "Physics-to-MAE scale ratio is not "
            "extremely large."
        )

    # =====================================================
    # 23. FINAL SUMMARY
    # =====================================================

    print_header(
        "PHYSICS LOSS SCALE AUDIT SUMMARY"
    )

    print(
        f"Expected |grad T|         : "
        f"{expected_gradient:.6e}"
    )

    print(
        f"Observed |grad T|         : "
        f"{observed_gradient_mean:.6e}"
    )

    print(
        f"Gradient ratio            : "
        f"{gradient_ratio:.6e}"
    )

    print(
        f"Eikonal loss              : "
        f"{eikonal_loss.item():.6e}"
    )

    print(
        f"Weighted physics loss     : "
        f"{physics_value:.6e}"
    )

    print(
        f"MAE loss                  : "
        f"{mae_value:.6e}"
    )

    print(
        f"Physics / MAE ratio       : "
        f"{physics_to_mae:.6e}"
    )

    print(
        f"Physics gradient mean     : "
        f"{gradient_mean:.6e}"
    )

    print(
        f"Physics gradient maximum  : "
        f"{gradient_max:.6e}"
    )

    print()

    print_separator()

    print(
        "PHYSICS LOSS SCALE AUDIT COMPLETED"
    )

    print_separator()


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()