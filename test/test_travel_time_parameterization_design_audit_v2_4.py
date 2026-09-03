"""
===============================================================================
TRAVEL-TIME PARAMETERIZATION DESIGN AUDIT v2.4
===============================================================================

Purpose
-------
Controlled comparison of alternative travel-time parameterizations.

IMPORTANT:
    This is a DIAGNOSTIC ONLY experiment.

    It does NOT modify:
        - models/network.py
        - losses/physics_loss.py
        - losses/total_loss.py
        - utils/config.py
        - LOSS_WEIGHTS
        - production model parameters

Current production parameterization:

    T = TRAVEL_TIME_SCALE * Softplus(z)

Candidates:

    A: Current Softplus
       T = T_scale * Softplus(z)

    B: Centered Softplus
       T = T_scale * [Softplus(z) - ln(2)]

    C: Exponential
       T = T_scale * exp(z)

    D: Characteristic-time Softplus
       T = T_char * Softplus(z)

    E: Characteristic-time centered Softplus
       T = T_char * [Softplus(z) - ln(2)]
===============================================================================
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from models.network import Network3D
from losses.physics_loss import PhysicsLoss
from utils.config import DX, DY, DZ, TRAVEL_TIME_SCALE


# =============================================================================
# Configuration
# =============================================================================

SEED = 42

DEVICE = torch.device("cpu")

TEST_SHAPE = (1, 1, 16, 32, 32)

DX_VALUE = float(DX)
DY_VALUE = float(DY)
DZ_VALUE = float(DZ)

T_SCALE = float(TRAVEL_TIME_SCALE)

DIAGNOSTIC_VELOCITY = 3250.0

EPS = 1.0e-12


# =============================================================================
# Reproducibility
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# Utility functions
# =============================================================================

def tensor_stats(name, tensor):
    """
    Print descriptive statistics for a tensor.
    """

    tensor = tensor.detach()

    print(
        f"{name:<34}"
        f"mean={tensor.mean().item():.6e} | "
        f"std={tensor.std(unbiased=False).item():.6e} | "
        f"min={tensor.min().item():.6e} | "
        f"max={tensor.max().item():.6e}"
    )


def is_finite(tensor):
    """
    Check whether every element is finite.
    """

    return bool(torch.isfinite(tensor).all().item())

def get_decoder_output(model, x):
    """
    Extract the decoder output using exactly the same
    encoder -> bottleneck -> decoder pathway used by
    Network3D.forward().

    This is a diagnostic helper only. It does not modify
    the production architecture.
    """

    # =====================================================
    # ENCODER
    # =====================================================
    #
    # Encoder3D returns five feature representations.
    #
    # x1, x2, x3, x4:
    #     Skip-connection features used by the decoder.
    #
    # x5:
    #     Deepest encoder representation sent to the
    #     bottleneck.
    # =====================================================

    x1, x2, x3, x4, x5 = model.encoder(x)

    # =====================================================
    # BOTTLENECK
    # =====================================================

    bottleneck_output = model.bottleneck(x5)

    # =====================================================
    # DECODER
    # =====================================================
    #
    # This exactly reproduces Network3D.forward().
    # =====================================================

    decoder_output = model.decoder(
        x1,
        x2,
        x3,
        x4,
        bottleneck_output
    )

    # =====================================================
    # VALIDATE DECODER OUTPUT
    # =====================================================

    if not isinstance(decoder_output, torch.Tensor):

        raise RuntimeError(
            "Decoder did not return a torch.Tensor."
        )

    if decoder_output.ndim != 5:

        raise RuntimeError(
            "Decoder output must have shape "
            "[B, C, D, H, W]. "
            f"Received: {tuple(decoder_output.shape)}"
        )

    return decoder_output

def get_raw_travel_time(model, decoder_output):
    """
    Obtain raw travel-time head output z_T.
    """

    return model.travel_time_head(
        decoder_output
    )


# =============================================================================
# Parameterization definitions
# =============================================================================

def apply_parameterization(
    candidate,
    z,
    scale
):
    """
    Apply one candidate travel-time parameterization.
    """

    if candidate == "A_current_softplus":

        return scale * F.softplus(z)


    elif candidate == "B_centered_softplus":

        return scale * (
            F.softplus(z) - math.log(2.0)
        )


    elif candidate == "C_exponential":

        return scale * torch.exp(z)


    elif candidate == "D_characteristic_softplus":

        return scale * F.softplus(z)


    elif candidate == "E_characteristic_centered_softplus":

        return scale * (
            F.softplus(z) - math.log(2.0)
        )


    else:

        raise ValueError(
            f"Unknown candidate: {candidate}"
        )


# =============================================================================
# Spatial derivatives
# =============================================================================

def compute_eikonal_diagnostics(
    physics_loss,
    travel_time
):
    """
    Compute the stabilized first-order Eikonal residual:

        R = V |grad T| - 1

    using the project's PhysicsLoss derivative implementation.
    """

    dT_dx = physics_loss._derivative(
        travel_time,
        dimension=4,
        spacing=DX_VALUE
    )

    dT_dy = physics_loss._derivative(
        travel_time,
        dimension=3,
        spacing=DY_VALUE
    )

    dT_dz = physics_loss._derivative(
        travel_time,
        dimension=2,
        spacing=DZ_VALUE
    )

    gradient_squared = (
        dT_dx.pow(2)
        + dT_dy.pow(2)
        + dT_dz.pow(2)
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared + EPS
    )

    velocity = torch.as_tensor(
        DIAGNOSTIC_VELOCITY,
        dtype=travel_time.dtype,
        device=travel_time.device
    ).expand_as(travel_time)

    velocity_gradient = (
        velocity * gradient_magnitude
    )

    residual = (
        velocity_gradient - 1.0
    )

    eikonal_loss = torch.mean(
        residual.pow(2)
    )

    return {
        "dT_dx": dT_dx,
        "dT_dy": dT_dy,
        "dT_dz": dT_dz,
        "gradient_magnitude": gradient_magnitude,
        "velocity_gradient": velocity_gradient,
        "residual": residual,
        "loss": eikonal_loss
    }


# =============================================================================
# Gradient audit
# =============================================================================

def gradient_wrt_z(
    candidate,
    z,
    scale,
    physics_loss
):
    """
    Measure d(Eikonal loss)/dz for a candidate.

    This isolates the travel-time parameterization gradient.
    """

    z_work = (
        z.detach()
        .clone()
        .requires_grad_(True)
    )

    travel_time = apply_parameterization(
        candidate,
        z_work,
        scale
    )

    diagnostics = compute_eikonal_diagnostics(
        physics_loss,
        travel_time
    )

    gradient = torch.autograd.grad(
        diagnostics["loss"],
        z_work,
        retain_graph=False,
        allow_unused=False
    )[0]

    return (
        diagnostics,
        gradient
    )


# =============================================================================
# Main audit
# =============================================================================

def main():

    print("=" * 78)
    print(
        "TRAVEL-TIME PARAMETERIZATION DESIGN AUDIT v2.4"
    )
    print("=" * 78)


    # =========================================================================
    # Section 1
    # =========================================================================

    print("\n[1] Configuration")
    print("-" * 78)

    print(f"Device                    : {DEVICE}")
    print(f"Test shape                : {TEST_SHAPE}")
    print(f"Seed                      : {SEED}")

    print(
        f"DX                        : {DX_VALUE}"
    )

    print(
        f"DY                        : {DY_VALUE}"
    )

    print(
        f"DZ                        : {DZ_VALUE}"
    )

    print(
        f"Configured T scale        : "
        f"{T_SCALE:.6e} s"
    )

    print(
        f"Diagnostic velocity       : "
        f"{DIAGNOSTIC_VELOCITY:.6e} m/s"
    )

    print(
        f"Target |grad T|           : "
        f"{1.0 / DIAGNOSTIC_VELOCITY:.6e} s/m"
    )

    print()
    print(
        "IMPORTANT: production configuration "
        "will NOT be modified."
    )


    # =========================================================================
    # Section 2
    # =========================================================================

    print("\n[2] Creating diagnostic input")
    print("-" * 78)

    x = torch.randn(
        TEST_SHAPE,
        device=DEVICE
    )

    print(
        "Input finite              : "
        f"{'PASS' if is_finite(x) else 'FAIL'}"
    )


    # =========================================================================
    # Section 3
    # =========================================================================

    print("\n[3] Creating Network3D")
    print("-" * 78)

    model = Network3D().to(DEVICE)

    model.eval()

    print(
        "Network3D created successfully."
    )


    # =========================================================================
    # Section 4
    # =========================================================================

    print("\n[4] Extracting decoder and travel-time features")
    print("-" * 78)

    with torch.no_grad():

        decoder_output = get_decoder_output(
            model,
            x
        )

        z = get_raw_travel_time(
            model,
            decoder_output
        )

    tensor_stats(
        "Decoder output",
        decoder_output
    )

    tensor_stats(
        "Raw z_T",
        z
    )


    # =========================================================================
    # Section 5
    # =========================================================================

    depth_extent = (
        TEST_SHAPE[2] - 1
    ) * DZ_VALUE

    y_extent = (
        TEST_SHAPE[3] - 1
    ) * DY_VALUE

    x_extent = (
        TEST_SHAPE[4] - 1
    ) * DX_VALUE

    diagonal_extent = math.sqrt(
        depth_extent ** 2
        + y_extent ** 2
        + x_extent ** 2
    )

    characteristic_time = (
        diagonal_extent
        / DIAGNOSTIC_VELOCITY
    )


    print("\n[5] Characteristic physical time")
    print("-" * 78)

    print(
        f"Depth extent              : "
        f"{depth_extent:.6e} m"
    )

    print(
        f"Y extent                  : "
        f"{y_extent:.6e} m"
    )

    print(
        f"X extent                  : "
        f"{x_extent:.6e} m"
    )

    print(
        f"Diagonal extent           : "
        f"{diagonal_extent:.6e} m"
    )

    print(
        f"Characteristic velocity   : "
        f"{DIAGNOSTIC_VELOCITY:.6e} m/s"
    )

    print(
        f"Characteristic T          : "
        f"{characteristic_time:.6e} s"
    )

    print(
        f"T_scale / T_char         : "
        f"{T_SCALE / characteristic_time:.6e}"
    )


    # =========================================================================
    # Candidate definitions
    # =========================================================================

    candidates = [

        (
            "A_current_softplus",
            T_SCALE
        ),

        (
            "B_centered_softplus",
            T_SCALE
        ),

        (
            "C_exponential",
            T_SCALE
        ),

        (
            "D_characteristic_softplus",
            characteristic_time
        ),

        (
            "E_characteristic_centered_softplus",
            characteristic_time
        ),

    ]


    # =========================================================================
    # Section 6
    # =========================================================================

    print("\n[6] Parameterization comparison")
    print("-" * 78)

    print(
        f"{'Candidate':<38}"
        f"{'Mean T':>13}"
        f"{'Std T':>13}"
        f"{'Mean |gradT|':>17}"
        f"{'Mean V|gradT|':>18}"
        f"{'Eikonal':>13}"
        f"{'T/Tchar':>13}"
    )

    print("-" * 125)


    results = {}


    for candidate, scale in candidates:

        z_work = (
            z.detach()
            .clone()
            .requires_grad_(True)
        )

        travel_time = apply_parameterization(
            candidate,
            z_work,
            scale
        )

        diagnostics = (
            compute_eikonal_diagnostics(
                PhysicsLoss(
                    dx=DX_VALUE,
                    dy=DY_VALUE,
                    dz=DZ_VALUE
                ),
                travel_time
            )
        )

        mean_T = (
            travel_time.mean().item()
        )

        std_T = (
            travel_time.std(
                unbiased=False
            ).item()
        )

        mean_gradient = (
            diagnostics[
                "gradient_magnitude"
            ].mean().item()
        )

        mean_velocity_gradient = (
            diagnostics[
                "velocity_gradient"
            ].mean().item()
        )

        loss = (
            diagnostics["loss"].item()
        )

        ratio = (
            abs(mean_T)
            / characteristic_time
        )


        results[candidate] = {
            "scale": scale,
            "travel_time": travel_time.detach(),
            "diagnostics": diagnostics,
        }


        print(
            f"{candidate:<38}"
            f"{mean_T:13.6e}"
            f"{std_T:13.6e}"
            f"{mean_gradient:17.6e}"
            f"{mean_velocity_gradient:18.6e}"
            f"{loss:13.6e}"
            f"{ratio:13.6e}"
        )


    # =========================================================================
    # Section 7
    # =========================================================================

    print("\n[7] Zero-activation behavior")
    print("-" * 78)

    z_zero = torch.zeros(
        1,
        dtype=torch.float32
    )


    for candidate, scale in candidates:

        T_zero = apply_parameterization(
            candidate,
            z_zero,
            scale
        ).item()


        if (
            candidate
            in [
                "A_current_softplus",
                "D_characteristic_softplus"
            ]
        ):

            expected = (
                scale
                * math.log(2.0)
            )


        elif (
            candidate
            in [
                "B_centered_softplus",
                "E_characteristic_centered_softplus"
            ]
        ):

            expected = 0.0


        elif candidate == "C_exponential":

            expected = scale


        else:

            expected = float("nan")


        print(
            f"{candidate:<38}"
            f"T(z=0)={T_zero:.9e} s | "
            f"expected={expected:.9e} s | "
            f"error={abs(T_zero - expected):.3e}"
        )


    # =========================================================================
    # Section 8
    # =========================================================================

    print("\n[8] Physical-gradient proximity")
    print("-" * 78)

    target_gradient = (
        1.0 / DIAGNOSTIC_VELOCITY
    )


    for candidate, item in results.items():

        mean_gradient = (
            item["diagnostics"]
            ["gradient_magnitude"]
            .mean()
            .item()
        )

        ratio = (
            mean_gradient
            / target_gradient
        )

        print(
            f"{candidate:<38}"
            f"ratio = {ratio:.6e} "
            f"({100.0 * ratio:.4f}%)"
        )


    # =========================================================================
    # Section 9
    # =========================================================================

    print("\n[9] Positivity and finite-value audit")
    print("-" * 78)


    for candidate, item in results.items():

        travel_time = (
            item["travel_time"]
        )

        finite = is_finite(
            travel_time
        )

        minimum = (
            travel_time.min().item()
        )

        positive = (
            minimum >= 0.0
        )

        print(
            f"{candidate:<38}"
            f"finite={'PASS' if finite else 'FAIL'} | "
            f"non-negative={'PASS' if positive else 'FAIL'} | "
            f"min={minimum:.6e}"
        )


    # =========================================================================
    # Section 10
    # =========================================================================

    print("\n[10] Travel-time parameterization gradient audit")
    print("-" * 78)

    for candidate, scale in candidates:

        physics_loss = PhysicsLoss(
            dx=DX_VALUE,
            dy=DY_VALUE,
            dz=DZ_VALUE
        )

        diagnostics, gradient = (
            gradient_wrt_z(
                candidate,
                z,
                scale,
                physics_loss
            )
        )

        print(
            f"{candidate:<38}"
            f"Eikonal={diagnostics['loss'].item():.6e} | "
            f"dL/dz norm={gradient.norm().item():.6e} | "
            f"max={gradient.abs().max().item():.6e}"
        )


    # =========================================================================
    # Section 11
    # =========================================================================

    print("\n[11] Relative comparison with current production")
    print("-" * 78)

    baseline_loss = (
        results[
            "A_current_softplus"
        ]["diagnostics"]["loss"].item()
    )


    for candidate, item in results.items():

        loss = (
            item["diagnostics"]["loss"]
            .item()
        )

        ratio = (
            loss
            / baseline_loss
        )

        print(
            f"{candidate:<38}"
            f"Eikonal loss ratio={ratio:.6e}"
        )


    # =========================================================================
    # Section 12
    # =========================================================================

    print("\n[12] Production decision criteria")
    print("-" * 78)

    print(
        """
A candidate should NOT be selected solely because it
has the lowest untrained Eikonal loss.

The preferred candidate should satisfy as many of the
following as possible:

  1. Positive/non-negative travel time.
  2. Physically meaningful characteristic time scale.
  3. Meaningful spatial variation.
  4. Improved |grad T| relative to 1/V.
  5. Stable and finite gradients.
  6. No pathological exponential sensitivity.
  7. A clear physical interpretation.
  8. Compatibility with eventual source/boundary constraints.
"""
    )


    # =========================================================================
    # Section 13
    # =========================================================================

    print("\n[13] AUDIT INTERPRETATION")
    print("-" * 78)

    current = results[
        "A_current_softplus"
    ]

    current_gradient_ratio = (
        current["diagnostics"]
        ["gradient_magnitude"]
        .mean()
        .item()
        / target_gradient
    )

    print(
        f"Current mean |grad T|/(1/V): "
        f"{current_gradient_ratio:.6e}"
    )

    print(
        f"Current mean T/T_char: "
        f"{current['travel_time'].abs().mean().item() / characteristic_time:.6e}"
    )

    print()
    print(
        "The v2.4 experiment is diagnostic only."
    )

    print(
        "No production model or configuration "
        "has been modified."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not modify Network3D until the "
        "candidate results have been reviewed."
    )

    print()
    print(
        "Production files changed: NONE"
    )

    print()
    print("=" * 78)
    print(
        "TRAVEL-TIME PARAMETERIZATION DESIGN AUDIT "
        "v2.4 COMPLETE"
    )
    print("=" * 78)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()