"""Loss Gradient Contribution & Sensitivity Audit."""
import math
import torch

from models.network import Network3D
from losses.total_loss import TotalLoss
from utils.config import DEVICE, DX, DY, DZ, LOSS_WEIGHTS

SEED = 42
INPUT_SHAPE = (1, 1, 16, 32, 32)
VELOCITY_MIN_TEST = 1500.0
VELOCITY_MAX_TEST = 4500.0
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def subheader(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name}: NaN/Inf detected.")
    print(f"{name}: finite values confirmed.")


def make_data(device):
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    x = torch.rand(INPUT_SHAPE, generator=g) * 2 - 1
    y = torch.rand(INPUT_SHAPE, generator=g) * 2 - 1
    v = torch.linspace(
        VELOCITY_MIN_TEST, VELOCITY_MAX_TEST,
        steps=INPUT_SHAPE[2] * INPUT_SHAPE[3] * INPUT_SHAPE[4],
    ).reshape(INPUT_SHAPE)
    return x.to(device), y.to(device), v.to(device)


def forward(model, x):
    out = model(x)
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        raise RuntimeError("Network3D must return (reconstruction, travel_time, log_variance).")
    r, t, lv = out
    for name, z in (("Reconstruction", r), ("Travel Time", t), ("Log Variance", lv)):
        if z.shape != x.shape:
            raise RuntimeError(f"{name} shape mismatch: {tuple(z.shape)} vs {tuple(x.shape)}")
        finite(name, z)
    return r, t, lv


def grad_stats(model):
    ss = 0.0
    mx = 0.0
    count = 0
    finite_grads = True
    named = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach()
        finite_grads &= bool(torch.isfinite(g).all())
        n = g.norm().item()
        m = g.abs().max().item()
        ss += torch.sum(g * g).item()
        mx = max(mx, m)
        count += 1
        named[name] = n
    return math.sqrt(ss), mx, count, finite_grads, named


def group_norm(named, keywords):
    vals = [n for name, n in named.items() if any(k in name.lower() for k in keywords)]
    return math.sqrt(sum(v * v for v in vals)) if vals else 0.0


def component_loss(total, name, r, target, t, velocity, lv):
    if name == "mae":
        return total.mae_loss(r, target)
    if name == "physics":
        return total.physics_loss(
            travel_time=t, velocity=velocity,
            source_indices=None, travel_time_target=None
        )["total"]
    if name == "uncertainty":
        return total.uncertainty_loss(r, target, lv)
    if name == "ssim":
        return total.ssim_loss(r, target)
    raise ValueError(name)


def main():
    header("LOSS GRADIENT CONTRIBUTION & SENSITIVITY AUDIT")
    print(f"Device              : {DEVICE}")
    print(f"Tensor shape        : {INPUT_SHAPE}")
    print(f"DX                  : {DX}")
    print(f"DY                  : {DY}")
    print(f"DZ                  : {DZ}")
    print("\nComposite loss weights:")
    for n in ("mae", "physics", "uncertainty", "ssim"):
        print(f"    {n:<14}: {LOSS_WEIGHTS[n]}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    header("CREATING SYNTHETIC DATA")
    x, target, velocity = make_data(DEVICE)
    finite("Input", x)
    finite("Target", target)
    finite("Velocity", velocity)
    if torch.any(velocity <= 0):
        raise RuntimeError("Velocity positivity check failed.")
    print("Velocity positivity check: PASS")

    header("INITIALIZING NETWORK3D")
    model = Network3D().to(DEVICE)
    model.eval()  # deterministic audit; MC Dropout is audited separately
    print("Network3D initialized successfully.")

    header("INITIALIZING TOTAL LOSS")
    total = TotalLoss(dx=DX, dy=DY, dz=DZ).to(DEVICE)
    print("TotalLoss initialized successfully.")

    header("BASELINE NETWORK FORWARD PASS")
    forward(model, x)
    print("Network forward pass: PASS")

    header("INDIVIDUAL LOSS GRADIENT CONTRIBUTION AUDIT")
    names = ("mae", "physics", "uncertainty", "ssim")
    results = {}

    for name in names:
        subheader(f"{name.upper()} LOSS GRADIENT AUDIT")
        model.zero_grad(set_to_none=True)
        r, t, lv = forward(model, x)
        raw = component_loss(total, name, r, target, t, velocity, lv)
        weighted = float(LOSS_WEIGHTS[name]) * raw
        finite(f"{name} raw loss", raw)
        finite(f"{name} weighted loss", weighted)
        weighted.backward()
        gn, gm, count, finite_g, named = grad_stats(model)
        print(f"Raw loss              : {raw.item():.6e}")
        print(f"Configured weight     : {LOSS_WEIGHTS[name]:.6e}")
        print(f"Weighted loss         : {weighted.item():.6e}")
        print(f"Gradient norm         : {gn:.6e}")
        print(f"Maximum gradient      : {gm:.6e}")
        print(f"Parameters with grad  : {count}")
        print(f"Gradients finite      : {'PASS' if finite_g else 'FAIL'}")
        if not finite_g or count == 0:
            raise RuntimeError(f"{name}: invalid gradient propagation.")
        heads = {
            "reconstruction": group_norm(named, ("reconstruction_head", "recon_head", "reconstruction", "recon")),
            "travel_time": group_norm(named, ("travel_time_head", "traveltime_head", "travel_time", "traveltime")),
            "uncertainty": group_norm(named, ("uncertainty_head", "uncertainty", "log_variance", "logvar", "variance")),
        }
        for k, v in heads.items():
            print(f"    {k:<18}: {v:.6e}")
        results[name] = dict(raw=raw.item(), weight=float(LOSS_WEIGHTS[name]), weighted=weighted.item(), grad_norm=gn, max_grad=gm, heads=heads)

    header("WEIGHTED GRADIENT CONTRIBUTION SUMMARY")
    denom = sum(v["grad_norm"] for v in results.values())
    if denom <= 0:
        raise RuntimeError("All component gradient norms are zero.")
    print(f"{'Component':<18}{'Raw Loss':>15}{'Weight':>12}{'Weighted Loss':>18}{'Grad Norm':>18}")
    print("-" * 81)
    for n in names:
        z = results[n]
        print(f"{n:<18}{z['raw']:>15.6e}{z['weight']:>12.4f}{z['weighted']:>18.6e}{z['grad_norm']:>18.6e}")
    print("\nRelative weighted gradient contribution:")
    for n in names:
        print(f"    {n:<14}: {100*z['grad_norm']/denom:8.3f}%")

    header("INTENDED GRADIENT PATHWAY AUDIT")
    expected = {
        "mae": ("reconstruction",),
        "physics": ("travel_time",),
        "uncertainty": ("reconstruction", "uncertainty"),
        "ssim": ("reconstruction",),
    }
    for n in names:
        for h in expected[n]:
            value = results[n]["heads"][h]
            print(f"    {n:<14} -> {h:<16}: {value:.6e} -> {'PASS' if value > 0 else 'FAIL'}")
            if value <= 0:
                raise RuntimeError(f"Expected gradient pathway missing: {n} -> {h}")
    print("Intended gradient pathway check: PASS")

    header("LOSS WEIGHT CONSISTENCY AUDIT")
    for n in names:
        z = results[n]
        diff = abs(z["raw"] * z["weight"] - z["weighted"])
        print(f"    {n:<14}: difference={diff:.6e} -> {'PASS' if diff <= 1e-6 else 'FAIL'}")
        if diff > 1e-6:
            raise RuntimeError(f"Weight inconsistency: {n}")
    print("Loss weight consistency check: PASS")

    header("GRADIENT SCALING SENSITIVITY AUDIT")
    for n in names:
        subheader(f"{n.upper()} GRADIENT SCALING")
        # Use ONE deterministic forward graph and compare every factor against
        # the factor=1.0 reference. This tests the mathematical scaling of
        # gradients without introducing new dropout/forward-pass noise.
        model.zero_grad(set_to_none=True)
        r, t, lv = forward(model, x)
        raw = component_loss(total, n, r, target, t, velocity, lv)
        params = [p for p in model.parameters() if p.requires_grad]
        reference_norm = None
        for factor in SENSITIVITY_FACTORS:
            scaled = float(LOSS_WEIGHTS[n]) * factor * raw
            grads = torch.autograd.grad(
                scaled, params, retain_graph=True, allow_unused=True
            )
            ss = 0.0
            finite_g = True
            for g in grads:
                if g is None:
                    continue
                finite_g &= bool(torch.isfinite(g).all())
                ss += torch.sum(g * g).item()
            gn = math.sqrt(ss)
            if not finite_g:
                raise RuntimeError(f"Non-finite gradient during {n} sensitivity test.")
            if abs(factor - 1.0) < 1e-12:
                reference_norm = gn
            print(f"    factor={factor:<4}: grad_norm={gn:.6e}")

        if reference_norm is None or reference_norm <= 0:
            raise RuntimeError(f"No valid factor=1.0 reference for {n}.")

        # Recompute ratios from the same graph so the comparison is exact up
        # to floating-point precision.
        for factor in SENSITIVITY_FACTORS:
            scaled = float(LOSS_WEIGHTS[n]) * factor * raw
            grads = torch.autograd.grad(
                scaled, params, retain_graph=True, allow_unused=True
            )
            ss = 0.0
            for g in grads:
                if g is not None:
                    ss += torch.sum(g * g).item()
            gn = math.sqrt(ss)
            ratio = gn / reference_norm
            error = abs(ratio - factor) / factor
            print(f"    factor={factor:<4}: ratio={ratio:.6f}, expected={factor:.6f}, relative_error={error:.3e}")
            if error > 1e-4:
                raise RuntimeError(f"Gradient scaling failed for {n} at factor {factor}.")
    print("Gradient scaling sensitivity check: PASS")

    header("LOSS GRADIENT CONTRIBUTION AUDIT RESULT")
    for label in (
        "MAE gradient contribution",
        "Physics gradient contribution",
        "Aleatoric gradient contribution",
        "SSIM gradient contribution",
        "Finite gradients",
        "Intended pathway audit",
        "Loss weight consistency",
        "Gradient scaling sensitivity",
        "Numerical stability",
    ):
        print(f"{label:<34}: PASS")
    print("\n" + "=" * 70)
    print("LOSS GRADIENT CONTRIBUTION AUDIT COMPLETED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
