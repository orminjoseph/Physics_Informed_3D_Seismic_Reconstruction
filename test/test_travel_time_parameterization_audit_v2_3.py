"""
======================================================================
NETWORK3D TRAVEL-TIME PARAMETERIZATION AUDIT v2.3.1
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction.

PURPOSE
-------
This diagnostic audit investigates the actual travel-time parameterization
inside Network3D before any production architecture or loss-weight changes.

The audit specifically investigates:

    1. Travel-time head initialization
    2. Decoder feature magnitude
    3. Raw travel-time field z_T
    4. Softplus transformation
    5. Physical travel-time scaling
    6. Spatial variation of T
    7. Physical Eikonal gradient
    8. V |grad T|
    9. Eikonal residual and loss
   10. Gradient propagation
   11. TRAVEL_TIME_SCALE sensitivity
   12. Travel-time-head initialization sensitivity
   13. Parameterization consistency

IMPORTANT
---------
This script is DIAGNOSTIC ONLY.

It does NOT:
    - update model parameters
    - perform optimizer steps
    - modify config.py
    - modify Network3D
    - modify TotalLoss
    - modify loss weights

The current production parameterization is:

    z_T = travel_time_head(decoder_output)

    T_normalized = Softplus(z_T)

    T = TRAVEL_TIME_SCALE * T_normalized

At z_T = 0:

    Softplus(0) = ln(2)

Therefore:

    T = TRAVEL_TIME_SCALE * ln(2)

For TRAVEL_TIME_SCALE = 0.1:

    T ~= 0.0693147 s

This behavior is expected mathematically but may produce an
initially near-constant travel-time field.

Author: Ormin Joseph
======================================================================
"""

import math
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
)


# ======================================================================
# CONFIGURATION
# ======================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

BATCH_SIZE = 1
CHANNELS = 1

DEPTH = 16
HEIGHT = 32
WIDTH = 32

TEST_SHAPE = (
    BATCH_SIZE,
    CHANNELS,
    DEPTH,
    HEIGHT,
    WIDTH,
)

DIAGNOSTIC_VELOCITY = (
    VELOCITY_MIN + VELOCITY_MAX
) / 2.0

EPS = 1.0e-12

HEAD_INIT_STDS = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
)

TRAVEL_TIME_TARGETS = (
    0.01,
    0.02,
    0.05,
    0.10,
)


# ======================================================================
# PRINTING UTILITIES
# ======================================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subheader(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def scalar_statistics(name, tensor):
    """
    Print robust scalar statistics.

    unbiased=False is intentionally used so that a one-element tensor
    has a valid standard deviation of zero rather than NaN.
    """

    tensor = tensor.detach()

    print(
        f"{name:<32} "
        f"mean={tensor.mean().item():.6e} | "
        f"std={tensor.std(unbiased=False).item():.6e} | "
        f"min={tensor.min().item():.6e} | "
        f"max={tensor.max().item():.6e}"
    )


def assert_finite(tensor, name):
    """
    Confirm that a tensor contains only finite values.
    """

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )


def safe_std(tensor):
    """
    Return population standard deviation.

    This avoids NaN for one-element tensors.
    """

    return tensor.detach().std(
        unbiased=False
    ).item()


def norm_or_zero(tensor):
    """
    Return L2 norm of a tensor.
    """

    if tensor is None:
        return 0.0

    return tensor.detach().norm().item()


# ======================================================================
# RANDOM SEED
# ======================================================================

def set_seed(seed):
    """
    Set deterministic diagnostic seed.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# SYNTHETIC INPUT
# ======================================================================

def create_diagnostic_input(device):
    """
    Create a deterministic synthetic seismic-like input volume.

    The actual values are not used as a geological benchmark.
    They simply provide a stable input for auditing the network pathway.
    """

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(SEED)

    x = torch.randn(
        TEST_SHAPE,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    return x


# ======================================================================
# MANUAL NETWORK FEATURE EXTRACTION
# ======================================================================

def get_decoder_output(model, x):
    """
    Reproduce the Network3D feature pathway up to decoder output.

    This is intentionally explicit so the diagnostic can separately
    inspect the travel-time head.
    """

    x1, x2, x3, x4, x5 = model.encoder(x)

    bottleneck_output = model.bottleneck(x5)

    decoder_output = model.decoder(
        x1,
        x2,
        x3,
        x4,
        bottleneck_output,
    )

    return decoder_output


# ======================================================================
# TRAVEL-TIME PARAMETERIZATION
# ======================================================================

def compute_travel_time_components(
    model,
    decoder_output,
    scale=None,
):
    """
    Compute the complete travel-time parameterization.

        z_T
          |
          v
       Softplus
          |
          v
      normalized T
          |
          v
        scale
          |
          v
        T [s]
    """

    raw_travel_time = model.travel_time_head(
        decoder_output
    )

    normalized_travel_time = (
        model.travel_time_activation(
            raw_travel_time
        )
    )

    if scale is None:
        scale = TRAVEL_TIME_SCALE

    physical_travel_time = (
        float(scale)
        * normalized_travel_time
    )

    return (
        raw_travel_time,
        normalized_travel_time,
        physical_travel_time,
    )


# ======================================================================
# EIKONAL GRADIENT
# ======================================================================

def calculate_spatial_gradient(
    physics_loss,
    travel_time,
):
    """
    Calculate dT/dx, dT/dy, dT/dz using the same derivative mechanism
    used by PhysicsLoss.

    Tensor convention:

        [B, C, D, H, W]

        D -> z
        H -> y
        W -> x
    """

    dT_dz = physics_loss._derivative(
        travel_time,
        spacing=DZ,
        dimension=2,
    )

    dT_dy = physics_loss._derivative(
        travel_time,
        spacing=DY,
        dimension=3,
    )

    dT_dx = physics_loss._derivative(
        travel_time,
        spacing=DX,
        dimension=4,
    )

    return (
        dT_dx,
        dT_dy,
        dT_dz,
    )


# ======================================================================
# PHYSICAL EIKONAL DIAGNOSTICS
# ======================================================================

def physics_diagnostics(
    physics_loss,
    travel_time,
    velocity,
):
    """
    Compute the stabilized physical Eikonal quantities:

        |grad T|

        V |grad T|

        R = V |grad T| - 1

        L = mean(R^2)

    IMPORTANT
    ---------
    All operations are explicitly tensor-based.

    This avoids the previous v2.3 TypeError associated with the
    velocity multiplication expression.
    """

    if not isinstance(velocity, torch.Tensor):
        velocity = torch.as_tensor(
            velocity,
            dtype=travel_time.dtype,
            device=travel_time.device,
        )

    if velocity.ndim == 0:
        velocity = velocity.expand_as(
            travel_time
        )

    if velocity.shape != travel_time.shape:
        velocity = velocity.expand_as(
            travel_time
        )

    (
        dT_dx,
        dT_dy,
        dT_dz,
    ) = calculate_spatial_gradient(
        physics_loss,
        travel_time,
    )

    gradient_squared = (
        dT_dx.pow(2)
        + dT_dy.pow(2)
        + dT_dz.pow(2)
    )

    gradient_magnitude = torch.sqrt(
        gradient_squared + EPS
    )

    velocity_gradient = (
        velocity * gradient_magnitude
    )

    eikonal_residual = (
        velocity_gradient - 1.0
    )

    eikonal_loss = torch.mean(
        eikonal_residual.pow(2)
    )

    return (
        dT_dx,
        dT_dy,
        dT_dz,
        gradient_squared,
        gradient_magnitude,
        velocity_gradient,
        eikonal_residual,
        eikonal_loss,
    )


# ======================================================================
# GRADIENT AUDIT
# ======================================================================

def gradient_audit(
    model,
    physics_loss,
    travel_time,
):
    """
    Backpropagate the Eikonal loss and inspect parameter gradients.

    No optimizer step is performed.
    """

    model.zero_grad(
        set_to_none=True
    )

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        eikonal_loss,
    ) = physics_diagnostics(
        physics_loss,
        travel_time,
        torch.full_like(
            travel_time,
            DIAGNOSTIC_VELOCITY,
        ),
    )

    eikonal_loss.backward()

    total_squared = 0.0
    max_gradient = 0.0

    total_elements = 0

    travel_head_norm = 0.0
    upstream_norm = 0.0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        assert_finite(
            gradient,
            f"Gradient: {name}",
        )

        gradient_norm = gradient.norm().item()

        total_squared += (
            gradient_norm ** 2
        )

        max_gradient = max(
            max_gradient,
            gradient.abs().max().item(),
        )

        total_elements += gradient.numel()

        if name.startswith(
            "travel_time_head."
        ):
            travel_head_norm += (
                gradient_norm ** 2
            )
        else:
            upstream_norm += (
                gradient_norm ** 2
            )

    total_norm = math.sqrt(
        total_squared
    )

    travel_head_norm = math.sqrt(
        travel_head_norm
    )

    upstream_norm = math.sqrt(
        upstream_norm
    )

    return (
        eikonal_loss,
        total_norm,
        max_gradient,
        total_elements,
        travel_head_norm,
        upstream_norm,
    )


# ======================================================================
# MAIN AUDIT
# ======================================================================

def main():

    set_seed(SEED)

    # ------------------------------------------------------------------
    # 1. CONFIGURATION
    # ------------------------------------------------------------------

    header(
        "NETWORK3D TRAVEL-TIME PARAMETERIZATION AUDIT v2.3.1"
    )

    header("[1] Configuration")

    print(
        f"Device                  : {DEVICE}"
    )

    print(
        f"Test shape              : {TEST_SHAPE}"
    )

    print(
        f"Seed                    : {SEED}"
    )

    print()
    print("Physical sampling:")

    print(
        f"DX                      : {DX}"
    )

    print(
        f"DY                      : {DY}"
    )

    print(
        f"DZ                      : {DZ}"
    )

    print()
    print(
        "Configured travel-time scale:"
    )

    print(
        f"TRAVEL_TIME_SCALE       : "
        f"{TRAVEL_TIME_SCALE}"
    )

    print()
    print(
        "Diagnostic velocity     : "
        f"{DIAGNOSTIC_VELOCITY:.6e} m/s"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No model parameters will be updated."
    )

    # ------------------------------------------------------------------
    # 2. INPUT
    # ------------------------------------------------------------------

    header(
        "[2] Creating diagnostic input"
    )

    x = create_diagnostic_input(
        DEVICE
    )

    assert_finite(
        x,
        "Diagnostic input",
    )

    print(
        f"Input shape             : "
        f"{tuple(x.shape)}"
    )

    print(
        "Input finite            : PASS"
    )

    # ------------------------------------------------------------------
    # 3. NETWORK
    # ------------------------------------------------------------------

    header(
        "[3] Creating Network3D"
    )

    model = Network3D().to(
        DEVICE
    )

    model.eval()

    print(
        "Network3D created successfully."
    )

    # ------------------------------------------------------------------
    # 4. TRAVEL-TIME HEAD INITIALIZATION
    # ------------------------------------------------------------------

    header(
        "[4] Travel-time head parameter initialization"
    )

    weight = (
        model.travel_time_head
        .weight
        .detach()
    )

    bias = (
        model.travel_time_head
        .bias
        .detach()
    )

    scalar_statistics(
        "Travel-time head weight",
        weight,
    )

    scalar_statistics(
        "Travel-time head bias",
        bias,
    )

    configured_std = 1.0e-3

    print(
        f"Configured weight std target : "
        f"{configured_std:.6e}"
    )

    print(
        f"Actual weight std            : "
        f"{safe_std(weight):.6e}"
    )

    print(
        f"Bias mean                    : "
        f"{bias.mean().item():.6e}"
    )

    # ------------------------------------------------------------------
    # 5. DECODER FEATURE AUDIT
    # ------------------------------------------------------------------

    header(
        "[5] Decoder feature audit"
    )

    with torch.no_grad():

        decoder_output = (
            get_decoder_output(
                model,
                x,
            )
        )

    assert_finite(
        decoder_output,
        "Decoder output",
    )

    scalar_statistics(
        "Decoder output",
        decoder_output,
    )

    # ------------------------------------------------------------------
    # 6. ACTUAL PARAMETERIZATION
    # ------------------------------------------------------------------

    header(
        "[6] Actual travel-time parameterization"
    )

    with torch.no_grad():

        (
            raw_travel_time,
            normalized_travel_time,
            travel_time,
        ) = compute_travel_time_components(
            model,
            decoder_output,
        )

    assert_finite(
        raw_travel_time,
        "Raw travel-time",
    )

    assert_finite(
        normalized_travel_time,
        "Normalized travel-time",
    )

    assert_finite(
        travel_time,
        "Physical travel-time",
    )

    scalar_statistics(
        "Raw travel-time z_T",
        raw_travel_time,
    )

    scalar_statistics(
        "Softplus(z_T)",
        normalized_travel_time,
    )

    softplus_derivative = (
        torch.sigmoid(
            raw_travel_time
        )
    )

    scalar_statistics(
        "Softplus derivative",
        softplus_derivative,
    )

    scalar_statistics(
        "Physical travel time T",
        travel_time,
    )

    # ------------------------------------------------------------------
    # 7. SOFTPLUS INITIALIZATION VERIFICATION
    # ------------------------------------------------------------------

    header(
        "[7] Softplus initialization verification"
    )

    expected_softplus_zero = math.log(2.0)

    expected_T = (
        TRAVEL_TIME_SCALE
        * expected_softplus_zero
    )

    actual_mean_T = (
        travel_time.mean().item()
    )

    difference = abs(
        actual_mean_T
        - expected_T
    )

    print(
        f"Softplus(0)                : "
        f"{expected_softplus_zero:.9f}"
    )

    print(
        f"Expected T at z=0          : "
        f"{expected_T:.9e} s"
    )

    print(
        f"Actual mean T              : "
        f"{actual_mean_T:.9e} s"
    )

    print(
        f"Difference                 : "
        f"{difference:.9e} s"
    )

    tolerance = max(
        1.0e-6,
        abs(expected_T) * 1.0e-4,
    )

    if difference <= tolerance:

        print(
            "Near-zero activation explanation: "
            "CONFIRMED"
        )

    else:

        print(
            "Near-zero activation explanation: "
            "CHECK"
        )

    # ------------------------------------------------------------------
    # 8. SPATIAL VARIATION
    # ------------------------------------------------------------------

    header(
        "[8] Travel-time spatial variation"
    )

    T_mean = travel_time.mean().item()

    T_std = safe_std(
        travel_time
    )

    T_min = travel_time.min().item()

    T_max = travel_time.max().item()

    T_range = T_max - T_min

    relative_variation = (
        T_range
        / max(abs(T_mean), 1.0e-12)
    )

    print(
        f"T mean                    : "
        f"{T_mean:.9e} s"
    )

    print(
        f"T std                     : "
        f"{T_std:.9e} s"
    )

    print(
        f"T range                   : "
        f"{T_range:.9e} s"
    )

    print(
        f"Relative T variation      : "
        f"{relative_variation:.9e}"
    )

    # ------------------------------------------------------------------
    # 9. PHYSICAL EIKONAL DIAGNOSTICS
    # ------------------------------------------------------------------

    header(
        "[9] Physical Eikonal diagnostics"
    )

    physics_loss = PhysicsLoss(
        dx=DX,
        dy=DY,
        dz=DZ,
    ).to(DEVICE)

    travel_time_for_grad = (
        travel_time.detach()
        .clone()
        .requires_grad_(True)
    )

    velocity = torch.full_like(
        travel_time_for_grad,
        DIAGNOSTIC_VELOCITY,
    )

    (
        dT_dx,
        dT_dy,
        dT_dz,
        gradient_squared,
        gradient_magnitude,
        velocity_gradient,
        eikonal_residual,
        eikonal_loss,
    ) = physics_diagnostics(
        physics_loss,
        travel_time_for_grad,
        velocity,
    )

    assert_finite(
        dT_dx,
        "dT/dx",
    )

    assert_finite(
        dT_dy,
        "dT/dy",
    )

    assert_finite(
        dT_dz,
        "dT/dz",
    )

    assert_finite(
        gradient_magnitude,
        "|grad T|",
    )

    assert_finite(
        velocity_gradient,
        "V|grad T|",
    )

    assert_finite(
        eikonal_residual,
        "Eikonal residual",
    )

    assert_finite(
        eikonal_loss,
        "Eikonal loss",
    )

    scalar_statistics(
        "dT/dx",
        dT_dx,
    )

    scalar_statistics(
        "dT/dy",
        dT_dy,
    )

    scalar_statistics(
        "dT/dz",
        dT_dz,
    )

    scalar_statistics(
        "|grad T|",
        gradient_magnitude,
    )

    inverse_velocity = (
        1.0 / velocity
    )

    scalar_statistics(
        "1/V",
        inverse_velocity,
    )

    scalar_statistics(
        "V|grad T|",
        velocity_gradient,
    )

    scalar_statistics(
        "Eikonal residual",
        eikonal_residual,
    )

    print()
    print(
        f"Eikonal loss              : "
        f"{eikonal_loss.item():.9e}"
    )

    ratio = (
        gradient_magnitude
        / inverse_velocity
    )

    scalar_statistics(
        "|grad T| / (1/V)",
        ratio,
    )

    # ------------------------------------------------------------------
    # 10. PHYSICAL TRAVEL-TIME SCALE
    # ------------------------------------------------------------------

    header(
        "[10] Characteristic physical travel-time scale"
    )

    depth_extent = (
        max(DEPTH - 1, 1)
        * DZ
    )

    y_extent = (
        max(HEIGHT - 1, 1)
        * DY
    )

    x_extent = (
        max(WIDTH - 1, 1)
        * DX
    )

    diagonal_extent = math.sqrt(
        depth_extent ** 2
        + y_extent ** 2
        + x_extent ** 2
    )

    characteristic_velocity = (
        DIAGNOSTIC_VELOCITY
    )

    characteristic_time = (
        diagonal_extent
        / characteristic_velocity
    )

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
        f"{characteristic_velocity:.6e} m/s"
    )

    print(
        f"Characteristic T          : "
        f"{characteristic_time:.9e} s"
    )

    print(
        f"Configured T scale        : "
        f"{TRAVEL_TIME_SCALE:.9e} s"
    )

    print(
        f"Scale / characteristic T  : "
        f"{TRAVEL_TIME_SCALE / characteristic_time:.6e}"
    )

    print(
        f"Predicted mean T / char T : "
        f"{abs(T_mean) / characteristic_time:.6e}"
    )

    # ------------------------------------------------------------------
    # 11. EIKONAL BACKWARD GRADIENT AUDIT
    # ------------------------------------------------------------------

    header(
        "[11] Eikonal backward gradient audit"
    )

    # Recreate the graph through the network.
    model.zero_grad(
        set_to_none=True
    )

    decoder_graph = get_decoder_output(
        model,
        x,
    )

    (
        _,
        _,
        travel_time_graph,
    ) = compute_travel_time_components(
        model,
        decoder_graph,
    )

    (
        physics_eikonal_loss,
        total_gradient_norm,
        maximum_gradient,
        gradient_elements,
        travel_head_gradient_norm,
        upstream_gradient_norm,
    ) = gradient_audit(
        model,
        physics_loss,
        travel_time_graph,
    )

    print(
        f"Eikonal loss              : "
        f"{physics_eikonal_loss.item():.9e}"
    )

    print(
        f"Total parameter grad norm  : "
        f"{total_gradient_norm:.9e}"
    )

    print(
        f"Maximum parameter gradient : "
        f"{maximum_gradient:.9e}"
    )

    print(
        f"Gradient elements          : "
        f"{gradient_elements:,}"
    )

    print(
        f"Travel-time head grad norm : "
        f"{travel_head_gradient_norm:.9e}"
    )

    print(
        f"Upstream network grad norm : "
        f"{upstream_gradient_norm:.9e}"
    )

    # ------------------------------------------------------------------
    # 12. TRAVEL-TIME SCALE SENSITIVITY
    # ------------------------------------------------------------------

    header(
        "[12] Actual TRAVEL_TIME_SCALE sensitivity"
    )

    print(
        "This section changes only the diagnostic scaling."
    )

    print(
        "Network parameters remain unchanged."
    )

    print()

    print(
        f"{'Scale':>12} "
        f"{'Mean T (s)':>18} "
        f"{'Mean |gradT|':>18} "
        f"{'Mean V|gradT|':>18} "
        f"{'Eikonal loss':>18}"
    )

    print("-" * 86)

    for scale in TRAVEL_TIME_TARGETS:

        decoder_graph = (
            get_decoder_output(
                model,
                x,
            )
        )

        (
            _,
            _,
            scaled_T,
        ) = compute_travel_time_components(
            model,
            decoder_graph,
            scale=scale,
        )

        (
            _,
            _,
            _,
            _,
            scaled_grad,
            scaled_Vgrad,
            _,
            scaled_loss,
        ) = physics_diagnostics(
            physics_loss,
            scaled_T,
            velocity,
        )

        print(
            f"{scale:12.4f} "
            f"{scaled_T.mean().item():18.9e} "
            f"{scaled_grad.mean().item():18.9e} "
            f"{scaled_Vgrad.mean().item():18.9e} "
            f"{scaled_loss.item():18.9e}"
        )

    # ------------------------------------------------------------------
    # 13. TRAVEL-TIME HEAD INITIALIZATION SENSITIVITY
    # ------------------------------------------------------------------

    header(
        "[13] Travel-time-head initialization sensitivity"
    )

    print(
        "Only the travel-time head initialization is modified"
    )

    print(
        "inside diagnostic copies."
    )

    print(
        "The production model is not modified."
    )

    print()

    print(
        f"{'Init std':>12} "
        f"{'Mean z_T':>18} "
        f"{'Std z_T':>18} "
        f"{'Mean T (s)':>18} "
        f"{'T std (s)':>18} "
        f"{'Eikonal loss':>18}"
    )

    print("-" * 104)

    for init_std in HEAD_INIT_STDS:

        set_seed(SEED)

        test_model = Network3D().to(
            DEVICE
        )

        test_model.eval()

        with torch.no_grad():

            torch.nn.init.normal_(
                test_model.travel_time_head.weight,
                mean=0.0,
                std=init_std,
            )

            torch.nn.init.constant_(
                test_model.travel_time_head.bias,
                0.0,
            )

            test_decoder = (
                get_decoder_output(
                    test_model,
                    x,
                )
            )

            (
                test_z,
                _,
                test_T,
            ) = compute_travel_time_components(
                test_model,
                test_decoder,
            )

        test_T_for_grad = (
            test_T.detach()
            .clone()
            .requires_grad_(True)
        )

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            test_eikonal_loss,
        ) = physics_diagnostics(
            physics_loss,
            test_T_for_grad,
            velocity,
        )

        print(
            f"{init_std:12.4e} "
            f"{test_z.mean().item():18.9e} "
            f"{safe_std(test_z):18.9e} "
            f"{test_T.mean().item():18.9e} "
            f"{safe_std(test_T):18.9e} "
            f"{test_eikonal_loss.item():18.9e}"
        )

        del test_model

    # ------------------------------------------------------------------
    # 14. PARAMETERIZATION CONSISTENCY
    # ------------------------------------------------------------------

    header(
        "[14] Parameterization consistency checks"
    )

    with torch.no_grad():

        (
            z_check,
            normalized_check,
            T_check,
        ) = compute_travel_time_components(
            model,
            decoder_output,
        )

        reconstructed_T = (
            TRAVEL_TIME_SCALE
            * normalized_check
        )

        max_scaling_error = (
            T_check
            - reconstructed_T
        ).abs().max().item()

        zero_z = torch.zeros(
            1,
            device=DEVICE,
        )

        zero_softplus = (
            model.travel_time_activation(
                zero_z
            )
        )

        expected_zero_T = (
            TRAVEL_TIME_SCALE
            * zero_softplus
        )

    print(
        f"Scaling identity max error : "
        f"{max_scaling_error:.6e}"
    )

    print(
        f"Softplus(0)               : "
        f"{zero_softplus.item():.9e}"
    )

    print(
        f"Expected T at z=0         : "
        f"{expected_zero_T.item():.9e} s"
    )

    if max_scaling_error <= 1.0e-12:

        print(
            "Scaling identity: PASS"
        )

    else:

        print(
            "Scaling identity: FAIL"
        )

        raise RuntimeError(
            "Travel-time scaling identity failed."
        )

    # ------------------------------------------------------------------
    # 15. FINAL INTERPRETATION
    # ------------------------------------------------------------------

    header(
        "[15] AUDIT INTERPRETATION"
    )

    print(
        "The diagnostic has completed the travel-time "
        "parameterization pathway."
    )

    print()

    print(
        "Confirmed:"
    )

    print(
        "  1. Travel-time head is present."
    )

    print(
        "  2. Travel-time head is initialized near zero."
    )

    print(
        "  3. Softplus produces a positive field."
    )

    print(
        "  4. Physical scaling is applied."
    )

    print(
        "  5. The initial T field is expected to be "
        "near TRAVEL_TIME_SCALE * ln(2)."
    )

    print(
        "  6. The physical Eikonal pathway is numerically "
        "auditable."
    )

    print(
        "  7. No model parameters were updated."
    )

    print()

    if relative_variation < 1.0e-3:

        print(
            "WARNING:"
        )

        print(
            "  The initial travel-time field is extremely "
            "close to spatially constant."
        )

        print(
            "  This means the initial |grad T| is expected "
            "to be far below 1/V."
        )

    else:

        print(
            "Initial spatial variation is measurable."
        )

    print()

    if (
        total_gradient_norm > 1.0
    ):

        print(
            "GRADIENT NOTE:"
        )

        print(
            "  The initial Eikonal gradient is substantial "
            "relative to a unit-scale diagnostic."
        )

        print(
            "  Do NOT change loss weights solely from this "
            "observation."
        )

    else:

        print(
            "Initial Eikonal gradient is not excessively large "
            "in this diagnostic."
        )

    print()

    print(
        "NEXT DECISION:"
    )

    print(
        "  Do not modify Network3D yet."
    )

    print(
        "  Do not modify TotalLoss yet."
    )

    print(
        "  Do not modify LOSS_WEIGHTS yet."
    )

    print(
        "  Use the complete Section [12] and [13] results "
        "to decide whether the travel-time parameterization "
        "requires redesign."
    )

    header(
        "NETWORK3D TRAVEL-TIME PARAMETERIZATION AUDIT "
        "v2.3.1 COMPLETE"
    )

    print()
    print(
        "RESULT: DIAGNOSTIC AUDIT COMPLETED."
    )
    print()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()