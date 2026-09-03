"""
===============================================================================
Travel-Time Gradient Initialization Audit v2.5
===============================================================================

Diagnostic-only audit for the travel-time branch of Network3D.

Purpose
-------
Investigates whether the very small initial Eikonal gradient is controlled by
travel-time-head initialization and/or the output time scale.

Production files are NOT modified.

Run from the project root:
    python -m test.test_travel_time_gradient_initialization_audit_v2_5
===============================================================================
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

from models.network import Network3D
from utils.config import DEVICE, DX, DY, DZ, TRAVEL_TIME_SCALE, VELOCITY_MIN, VELOCITY_MAX


# -----------------------------------------------------------------------------
# Audit configuration
# -----------------------------------------------------------------------------
TEST_SHAPE = (1, 1, 16, 32, 32)
SEED = 42
WEIGHT_STDS = [1e-4, 1e-3, 3e-3, 1e-2]
BIAS_VALUES = [-0.20, -0.10, 0.0, 0.10, 0.20]
DIAGNOSTIC_VELOCITY = 0.5 * (VELOCITY_MIN + VELOCITY_MAX)
EPS = 1e-12


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite(x):
    return bool(torch.isfinite(x).all().item())


def central_difference(t, spacing, dim):
    if spacing <= 0:
        raise ValueError("Spatial spacing must be positive.")
    n = t.shape[dim]
    if n < 2:
        raise ValueError("Spatial dimension must contain at least two samples.")
    d = torch.zeros_like(t)

    if dim == 2:
        d[:, :, 1:-1] = (t[:, :, 2:] - t[:, :, :-2]) / (2.0 * spacing)
        d[:, :, 0] = (t[:, :, 1] - t[:, :, 0]) / spacing
        d[:, :, -1] = (t[:, :, -1] - t[:, :, -2]) / spacing
    elif dim == 3:
        d[:, :, :, 1:-1] = (t[:, :, :, 2:] - t[:, :, :, :-2]) / (2.0 * spacing)
        d[:, :, :, 0] = (t[:, :, :, 1] - t[:, :, :, 0]) / spacing
        d[:, :, :, -1] = (t[:, :, :, -1] - t[:, :, :, -2]) / spacing
    elif dim == 4:
        d[:, :, :, :, 1:-1] = (t[:, :, :, :, 2:] - t[:, :, :, :, :-2]) / (2.0 * spacing)
        d[:, :, :, :, 0] = (t[:, :, :, :, 1] - t[:, :, :, :, 0]) / spacing
        d[:, :, :, :, -1] = (t[:, :, :, :, -1] - t[:, :, :, :, -2]) / spacing
    else:
        raise ValueError("Only dimensions 2, 3, and 4 are supported.")
    return d


def get_decoder_output(model, x):
    """Use exactly the production encoder -> bottleneck -> decoder pathway."""
    x1, x2, x3, x4, x5 = model.encoder(x)
    bottleneck_output = model.bottleneck(x5)
    decoder_output = model.decoder(x1, x2, x3, x4, bottleneck_output)
    if not isinstance(decoder_output, torch.Tensor) or decoder_output.ndim != 5:
        raise RuntimeError(f"Invalid decoder output: {type(decoder_output)} {getattr(decoder_output, 'shape', None)}")
    return decoder_output


def gradient_components(t):
    dT_dz = central_difference(t, DZ, 2)
    dT_dy = central_difference(t, DY, 3)
    dT_dx = central_difference(t, DX, 4)
    g2 = dT_dx.square() + dT_dy.square() + dT_dz.square()
    magnitude = torch.sqrt(g2 + EPS)  # matches production implementation
    return dT_dx, dT_dy, dT_dz, magnitude


def characteristic_time():
    depth = (TEST_SHAPE[2] - 1) * DZ
    y = (TEST_SHAPE[3] - 1) * DY
    x = (TEST_SHAPE[4] - 1) * DX
    diagonal = math.sqrt(depth * depth + y * y + x * x)
    return diagonal / DIAGNOSTIC_VELOCITY


def run_candidate(base_state, x, velocity, weight_std, bias, time_scale):
    model = Network3D().to(DEVICE)
    model.load_state_dict(base_state)
    model.eval()

    decoder_output = get_decoder_output(model, x)
    head = model.travel_time_head
    if not isinstance(head, nn.Conv3d):
        raise TypeError("travel_time_head must be nn.Conv3d")

    # Diagnostic-only mutation of this temporary model instance.
    nn.init.normal_(head.weight, mean=0.0, std=weight_std)
    nn.init.constant_(head.bias, bias)

    raw_z = head(decoder_output)
    activation = getattr(model, "travel_time_activation", nn.Softplus(beta=1.0, threshold=20.0))
    travel_time = time_scale * activation(raw_z)

    dT_dx, dT_dy, dT_dz, gradT = gradient_components(travel_time)
    residual = velocity * gradT - 1.0
    loss = residual.square().mean()

    grad_z = torch.autograd.grad(loss, raw_z, retain_graph=True)[0]
    head_grads = torch.autograd.grad(
        loss,
        tuple(head.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    decoder_grad = torch.autograd.grad(loss, decoder_output, retain_graph=False)[0]

    head_norm_sq = 0.0
    head_max = 0.0
    for g in head_grads:
        if g is None:
            continue
        if not finite(g):
            head_norm_sq = float("nan")
            head_max = float("nan")
            break
        head_norm_sq += float(g.square().sum().item())
        head_max = max(head_max, float(g.abs().max().item()))

    head_norm = math.sqrt(head_norm_sq) if math.isfinite(head_norm_sq) else float("nan")
    decoder_norm = float(torch.linalg.vector_norm(decoder_grad.detach()).item())
    t_char = characteristic_time()
    target = 1.0 / DIAGNOSTIC_VELOCITY

    return {
        "label": "",
        "weight_std": weight_std,
        "bias": bias,
        "time_scale": time_scale,
        "raw_z_mean": float(raw_z.detach().mean().item()),
        "raw_z_std": float(raw_z.detach().std(unbiased=False).item()),
        "T_mean": float(travel_time.detach().mean().item()),
        "T_std": float(travel_time.detach().std(unbiased=False).item()),
        "T_min": float(travel_time.detach().min().item()),
        "T_max": float(travel_time.detach().max().item()),
        "gradT_mean": float(gradT.detach().mean().item()),
        "Vgrad_mean": float((velocity * gradT).detach().mean().item()),
        "gradient_ratio": float(gradT.detach().mean().item() / target),
        "eikonal": float(loss.detach().item()),
        "residual_mean": float(residual.detach().mean().item()),
        "dL_dz_norm": float(torch.linalg.vector_norm(grad_z.detach()).item()),
        "dL_dz_max": float(grad_z.detach().abs().max().item()),
        "head_grad_norm": head_norm,
        "head_grad_max": head_max,
        "decoder_grad_norm": decoder_norm,
        "finite_T": finite(travel_time),
        "nonnegative_T": bool((travel_time.detach() >= 0).all().item()),
        "finite_grad": finite(grad_z) and math.isfinite(head_norm) and math.isfinite(decoder_norm),
        "T_over_Tchar": float(travel_time.detach().mean().item() / t_char),
        "T_relative_variation": float((travel_time.detach().std(unbiased=False) / (travel_time.detach().abs().mean() + 1e-12)).item()),
        "dTdx_mean": float(dT_dx.detach().mean().item()),
        "dTdy_mean": float(dT_dy.detach().mean().item()),
        "dTdz_mean": float(dT_dz.detach().mean().item()),
    }


def main():
    print("=" * 78)
    print("TRAVEL-TIME GRADIENT INITIALIZATION AUDIT v2.5")
    print("=" * 78)

    set_seed(SEED)

    print("\n[1] Configuration")
    print("-" * 78)
    print(f"Device                    : {DEVICE}")
    print(f"Test shape                : {TEST_SHAPE}")
    print(f"Seed                      : {SEED}")
    print(f"DX                        : {DX}")
    print(f"DY                        : {DY}")
    print(f"DZ                        : {DZ}")
    print(f"Configured T scale        : {TRAVEL_TIME_SCALE:.6e} s")
    print(f"Diagnostic velocity       : {DIAGNOSTIC_VELOCITY:.6e} m/s")
    print(f"Target |grad T|           : {1.0 / DIAGNOSTIC_VELOCITY:.6e} s/m")
    print("\nIMPORTANT: production configuration will NOT be modified.")

    x = torch.randn(TEST_SHAPE, device=DEVICE, dtype=torch.float32)
    velocity = torch.full(TEST_SHAPE, DIAGNOSTIC_VELOCITY, device=DEVICE, dtype=torch.float32)

    print("\n[2] Diagnostic input")
    print("-" * 78)
    print(f"Input finite              : {'PASS' if finite(x) else 'FAIL'}")

    print("\n[3] Creating baseline Network3D")
    print("-" * 78)
    base_model = Network3D().to(DEVICE)
    base_model.eval()
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    print("Network3D created successfully.")
    print("The same baseline state is restored for every candidate.")

    t_char = characteristic_time()
    results = []

    print("\n[4] Initialization sweep")
    print("-" * 78)
    print("Weight-std sweep: current bias=0, configured T scale")
    for std in WEIGHT_STDS:
        r = run_candidate(base_state, x, velocity, std, 0.0, TRAVEL_TIME_SCALE)
        r["label"] = f"WSTD_{std:.1e}_bias0_currentT"
        results.append(r)

    print("Bias sweep: current weight std=1e-3, configured T scale")
    for bias in BIAS_VALUES:
        r = run_candidate(base_state, x, velocity, 1e-3, bias, TRAVEL_TIME_SCALE)
        r["label"] = f"WSTD_1e-3_bias{bias:+.2f}_currentT"
        results.append(r)

    print("Characteristic-time sweep: bias=0")
    for std in WEIGHT_STDS:
        r = run_candidate(base_state, x, velocity, std, 0.0, t_char)
        r["label"] = f"WSTD_{std:.1e}_bias0_characteristicT"
        results.append(r)

    print("\n[5] Initialization results")
    print("-" * 78)
    print(f"{'Candidate':38s}{'Mean T':>12s}{'Std T':>12s}{'Mean |gradT|':>15s}{'V|gradT|':>12s}{'Eikonal':>12s}{'Grad ratio':>12s}")
    print("-" * 113)
    for r in results:
        print(f"{r['label'][:38]:38s}{r['T_mean']:12.6e}{r['T_std']:12.6e}{r['gradT_mean']:15.6e}{r['Vgrad_mean']:12.6e}{r['eikonal']:12.6e}{r['gradient_ratio']:12.6e}")

    print("\n[6] Gradient stability")
    print("-" * 78)
    print(f"{'Candidate':38s}{'dL/dz norm':>14s}{'dL/dz max':>14s}{'Head grad':>14s}{'Decoder grad':>15s}")
    print("-" * 95)
    for r in results:
        print(f"{r['label'][:38]:38s}{r['dL_dz_norm']:14.6e}{r['dL_dz_max']:14.6e}{r['head_grad_norm']:14.6e}{r['decoder_grad_norm']:15.6e}")

    print("\n[7] Positivity and finite-value audit")
    print("-" * 78)
    all_valid = True
    for r in results:
        ok = r["finite_T"] and r["nonnegative_T"] and r["finite_grad"]
        all_valid = all_valid and ok
        print(f"{r['label'][:38]:38s}finite T={'PASS' if r['finite_T'] else 'FAIL'} | non-negative T={'PASS' if r['nonnegative_T'] else 'FAIL'} | gradients={'PASS' if r['finite_grad'] else 'FAIL'}")

    baseline = next(r for r in results if r["label"] == "WSTD_1.0e-03_bias0_currentT")
    print("\n[8] Relative to current diagnostic baseline")
    print("-" * 78)
    for r in results:
        print(f"{r['label'][:38]:38s} | gradT x={r['gradT_mean']/baseline['gradT_mean']:8.4f} | Eikonal x={r['eikonal']/baseline['eikonal']:8.4f} | dL/dz x={r['dL_dz_norm']/baseline['dL_dz_norm']:8.4f}")

    print("\n[9] Characteristic physical time")
    print("-" * 78)
    depth = (TEST_SHAPE[2] - 1) * DZ
    y = (TEST_SHAPE[3] - 1) * DY
    x_extent = (TEST_SHAPE[4] - 1) * DX
    diagonal = math.sqrt(depth**2 + y**2 + x_extent**2)
    print(f"Depth extent              : {depth:.6e} m")
    print(f"Y extent                  : {y:.6e} m")
    print(f"X extent                  : {x_extent:.6e} m")
    print(f"Diagonal extent           : {diagonal:.6e} m")
    print(f"Characteristic velocity   : {DIAGNOSTIC_VELOCITY:.6e} m/s")
    print(f"Characteristic T          : {t_char:.6e} s")
    print(f"Configured T scale        : {TRAVEL_TIME_SCALE:.6e} s")
    print(f"Configured/characteristic: {TRAVEL_TIME_SCALE / t_char:.6f}")

    print("\n[10] Diagnostic ranking")
    print("-" * 78)
    valid = [r for r in results if r["finite_T"] and r["nonnegative_T"] and r["finite_grad"]]
    valid.sort(key=lambda r: r["gradient_ratio"], reverse=True)
    for i, r in enumerate(valid[:10], 1):
        print(f"{i:2d}. {r['label'][:38]:38s} ratio={r['gradient_ratio']:.6e} | Eikonal={r['eikonal']:.6e} | dL/dz={r['dL_dz_norm']:.6e}")

    print("\n[11] AUDIT INTERPRETATION")
    print("-" * 78)
    print("v2.5 tests whether travel-time-head initialization and output time scale")
    print("can improve the initial physical gradient without causing unstable gradients.")
    print("\nDo NOT select a production initialization solely from the lowest untrained")
    print("Eikonal loss. Consider positivity, physical time scale, spatial variation,")
    print("gradient proximity, gradient stability, and later source/boundary constraints.")
    print("\nIMPORTANT: v2.5 is diagnostic only.")
    print("Production files changed: NONE")

    print("\n[12] AUDIT STATUS")
    print("-" * 78)
    print("OVERALL NUMERICAL STABILITY: PASS" if all_valid else "OVERALL NUMERICAL STABILITY: REVIEW REQUIRED")
    print("This does NOT mean a production initialization has been selected.")
    print("=" * 78)
    print("TRAVEL-TIME GRADIENT INITIALIZATION AUDIT v2.5 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
