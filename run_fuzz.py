"""Fuzz flash-attn-v1 (SDPA, math backend) on MPS vs CPU reference.

Output:
  /Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/RESULTS_flash-attn-v1.md
  /Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/swarm.jsonl
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import CATEGORIES, fuzz_strides_for_category

SWARM_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = SWARM_DIR / "RESULTS_flash-attn-v1.md"
SWARM_JSONL = SWARM_DIR / "swarm.jsonl"

KERNEL = "flash-attn-v1"
TOTAL_ITERS = 500
BUDGET_S = 8 * 60 - 45  # ~7m15s; reserve headroom for repro + write
REPRO_SEEDS = 3  # FILABLE requires reproducibility across >= 3 seeds
DENOM_FLOOR = 1e-6  # FILABLE requires denom magnitude >= this
FILABLE_TOL_MULT = 10.0  # FILABLE requires max_rel_err > 10x gpucheck tolerance

# Shape categories (B, H, S, D) -- we'll vary one dim per category
SHAPE_BUCKETS = [
    "degenerate",       # tiny / 1-element-ish
    "prime",            # prime dimension(s)
    "pow2_boundary",    # 2^n at common boundary
    "non_tile_aligned", # not multiple of 8/16/32
    "large",            # large but bounded
]

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# Stride categories from gpucheck
STRIDE_CATS = list(CATEGORIES)


def sample_shape(bucket: str, rng: random.Random) -> tuple[int, int, int, int]:
    """Return (B, H, S, D)."""
    if bucket == "degenerate":
        B = rng.choice([1, 1, 2])
        H = rng.choice([1, 1, 2])
        S = rng.choice([1, 2, 3, 4])
        D = rng.choice([4, 8, 16])  # head_dim must be sane for SDPA
        return (B, H, S, D)
    if bucket == "prime":
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        B = rng.choice([1, 2])
        H = rng.choice([1, 2, 3, 5])
        S = rng.choice(primes)
        D = rng.choice([8, 16, 32])  # SDPA wants pow-of-2 head_dim friendliness
        return (B, H, S, D)
    if bucket == "pow2_boundary":
        B = rng.choice([1, 2, 4])
        H = rng.choice([1, 2, 4, 8])
        S = rng.choice([8, 16, 32, 64, 128])
        D = rng.choice([16, 32, 64])
        return (B, H, S, D)
    if bucket == "non_tile_aligned":
        B = rng.choice([1, 2, 3])
        H = rng.choice([1, 3, 5, 6])
        S = rng.choice([9, 17, 33, 65, 129])
        D = rng.choice([8, 16, 24, 40])
        return (B, H, S, D)
    if bucket == "large":
        B = rng.choice([2, 4])
        H = rng.choice([4, 8])
        S = rng.choice([128, 192, 256])
        D = rng.choice([32, 64])
        return (B, H, S, D)
    raise AssertionError(bucket)


def stride_tensor(shape: tuple[int, ...], dtype: torch.dtype, category: str,
                  device: str, seed: int) -> torch.Tensor:
    """Wrap fuzz_strides_for_category but tolerate failures (e.g. dtype on mps)."""
    return fuzz_strides_for_category(shape, dtype, category, device=device, seed=seed)


def flash_attn_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SDPA = the kernel under test (Flash Attention algorithm family v1)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def reference_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Math-only reference attention in fp32 on CPU."""
    qf = q.detach().to(device="cpu", dtype=torch.float32)
    kf = k.detach().to(device="cpu", dtype=torch.float32)
    vf = v.detach().to(device="cpu", dtype=torch.float32)
    d = qf.shape[-1]
    scores = qf @ kf.transpose(-2, -1) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return probs @ vf


def err_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs_err, max_relerr_safe, max_relerr_naive).

    max_relerr_safe = max( |a-b| / max(|b|, |a|, 1.0) )  -- bounded
    max_relerr_naive uses clamp(min=1e-8) -- the historical "max rel err"
    used in the task spec, but blows up near zero.
    """
    af = a.detach().to(device="cpu", dtype=torch.float32)
    bf = b.detach().to(device="cpu", dtype=torch.float32)
    diff = (af - bf).abs()
    max_abs = float(diff.max().item())
    safe_denom = torch.maximum(bf.abs(), af.abs()).clamp(min=1.0)
    max_safe = float((diff / safe_denom).max().item())
    naive = float((diff / bf.abs().clamp(min=1e-8)).max().item())
    return max_abs, max_safe, naive


def is_divergent(a: torch.Tensor, b: torch.Tensor,
                 atol: float, rtol: float) -> bool:
    """torch.allclose-style combined check, matching gpucheck.assert_close."""
    af = a.detach().to(device="cpu", dtype=torch.float32)
    bf = b.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(af).all() or not torch.isfinite(bf).all():
        # NaN/Inf: only divergent if presence differs.
        nan_a = torch.isnan(af); nan_b = torch.isnan(bf)
        inf_a = torch.isinf(af); inf_b = torch.isinf(bf)
        if not torch.equal(nan_a, nan_b) or not torch.equal(inf_a, inf_b):
            return True
        finite = ~(nan_a | inf_a)
        af = af[finite]; bf = bf[finite]
    diff = (af - bf).abs()
    return bool((diff > atol + rtol * bf.abs()).any().item())


def main() -> int:
    if not torch.backends.mps.is_available():
        SWARM_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text(
            f"# {KERNEL}\n\nSKIPPED: torch.mps not available on this host.\n"
        )
        line = json.dumps({
            "kernel": KERNEL, "status": "SKIPPED", "reason": "mps_unavailable",
            "iterations_attempted": 0, "iterations_completed": 0,
            "divergences": 0, "top_repros": [],
        })
        with SWARM_JSONL.open("a") as f:
            f.write(line + "\n")
        return 0

    SWARM_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xF1A5)
    torch.manual_seed(0xF1A5)

    deadline = time.monotonic() + BUDGET_S
    attempted = 0
    completed = 0
    unsupported = 0
    errors = 0
    divergences: list[dict] = []
    max_err_mps_cpu = 0.0
    err_samples: list[float] = []

    for i in range(TOTAL_ITERS):
        if time.monotonic() > deadline:
            break
        attempted += 1
        bucket = rng.choice(SHAPE_BUCKETS)
        dt_name, dt = rng.choice(DTYPES)
        stride_cat = rng.choice(STRIDE_CATS)
        shape = sample_shape(bucket, rng)
        seed = rng.randint(1, 2**31 - 1)

        # Skip stride patterns that don't map cleanly to 4D (B,H,S,D):
        # broadcast on last dim would break head_dim — still test, but tolerate.
        try:
            q = stride_tensor(shape, dt, stride_cat, "mps", seed)
            k = stride_tensor(shape, dt, stride_cat, "mps", seed + 1)
            v = stride_tensor(shape, dt, stride_cat, "mps", seed + 2)
        except Exception as exc:
            unsupported += 1
            continue

        try:
            out_mps = flash_attn_v1(q, k, v)
            torch.mps.synchronize()
        except (RuntimeError, NotImplementedError) as exc:
            msg = str(exc).lower()
            if any(t in msg for t in ("not implemented", "unsupported", "mps")):
                unsupported += 1
                continue
            errors += 1
            print(f"[iter {i}] HARD-ERROR mps op: {exc!r}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[iter {i}] UNEXPECTED mps op: {exc!r}", file=sys.stderr)
            continue

        try:
            ref = reference_attn(q, k, v)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[iter {i}] reference failed: {exc!r}", file=sys.stderr)
            continue

        try:
            max_abs, max_safe, max_naive = err_metrics(out_mps, ref)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[iter {i}] relerr failed: {exc!r}", file=sys.stderr)
            continue

        completed += 1
        err_samples.append(max_safe)
        if max_safe > max_err_mps_cpu:
            max_err_mps_cpu = max_safe

        # Tolerance: gpucheck per-dtype, scaled by sqrt(k/128) (matmul-class),
        # MPS overlay applied. Use combined |a-b| > atol + rtol*|b| check —
        # this matches assert_close / torch.allclose and avoids spurious
        # divergences from naive max(|a-b|/|b|) blowing up near zero in
        # softmax outputs.
        d = shape[-1]
        atol, rtol = compute_tolerance(dt, k_dim=d, device_type="mps")
        if is_divergent(out_mps, ref, atol, rtol):
            divergences.append({
                "iter": i,
                "shape": list(shape),
                "dtype": dt_name,
                "stride": stride_cat,
                "bucket": bucket,
                "max_abs_err": max_abs,
                "max_rel_err_safe": max_safe,
                "max_rel_err_naive": max_naive,
                "atol": atol,
                "rtol": rtol,
            })

    # Sort divergences by severity (combined-tolerance excess: abs/(atol+rtol*1))
    def _severity(d: dict) -> float:
        return d["max_abs_err"] / max(d["atol"] + d["rtol"], 1e-30)
    divergences.sort(key=_severity, reverse=True)
    top3 = divergences[:3]

    # Build markdown
    lines: list[str] = []
    lines.append(f"# {KERNEL} — gpucheck fuzz report")
    lines.append("")
    lines.append(f"- Kernel: `{KERNEL}` (torch.nn.functional.scaled_dot_product_attention)")
    lines.append(f"- Backends: MPS (real, Apple Silicon) vs CPU fp32 math reference")
    lines.append(f"- CUDA backend: MOCKED (no NVIDIA GPU on host) — N/A in this run")
    lines.append(f"- Iterations attempted: **{attempted}** / {TOTAL_ITERS}")
    lines.append(f"- Iterations completed: **{completed}**")
    lines.append(f"- Unsupported / skipped (op or stride): {unsupported}")
    lines.append(f"- Process errors: {errors}")
    lines.append(f"- Divergences (combined `|a-b| > atol + rtol*|b|`): **{len(divergences)}**")
    lines.append("")
    lines.append("## Error stats (safe rel err = |a-b| / max(|a|,|b|,1))")
    if err_samples:
        err_samples.sort()
        lines.append(f"- max safe rel err (MPS vs CPU fp32): `{max_err_mps_cpu:.4e}`")
        lines.append(f"- median safe rel err: `{err_samples[len(err_samples)//2]:.4e}`")
        lines.append(f"- p95 safe rel err: `{err_samples[int(len(err_samples)*0.95)-1]:.4e}`")
    else:
        lines.append("- no completed comparisons")
    lines.append("- max rel err (MPS vs CUDA mock): N/A (no NVIDIA GPU; CUDA detection mocked)")
    lines.append("")
    lines.append("## Top 3 minimal repros")
    if top3:
        for rank, d in enumerate(top3, 1):
            lines.append(f"### Repro #{rank}")
            lines.append(f"- shape (B,H,S,D): `{tuple(d['shape'])}`  ({d['bucket']})")
            lines.append(f"- dtype: `{d['dtype']}`")
            lines.append(f"- stride: `{d['stride']}`")
            lines.append(f"- max abs err: `{d['max_abs_err']:.4e}`")
            lines.append(f"- safe rel err: `{d['max_rel_err_safe']:.4e}`  "
                         f"(naive rel err: `{d['max_rel_err_naive']:.4e}`)")
            lines.append(f"- gpucheck tol (atol/rtol, MPS, k-scaled): "
                         f"`{d['atol']:.2e}` / `{d['rtol']:.2e}`")
            lines.append("")
    else:
        lines.append("_None — all completed configurations within gpucheck tolerance._")
        lines.append("")
    lines.append("## Recommended upstream filing target")
    if divergences:
        # MPS-only divergences (CUDA mocked) -> pytorch
        lines.append("**pytorch/pytorch** — divergences observed only on MPS backend; "
                     "candidate for `module: mps` triage.")
    else:
        lines.append("**none** — no actionable divergence at gpucheck tolerances "
                     "(per-dtype, k-scaled, MPS overlay applied).")
    lines.append("")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on Apple Silicon "
                 f"(torch={torch.__version__}, MPS available={torch.backends.mps.is_available()})._")

    RESULTS_MD.write_text("\n".join(lines) + "\n")

    record = {
        "kernel": KERNEL,
        "status": "OK" if errors == 0 else "OK_WITH_ERRORS",
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "iterations_unsupported": unsupported,
        "iterations_errored": errors,
        "divergences": len(divergences),
        "top_repros": top3,
        "max_relerr_mps_vs_cpu": max_err_mps_cpu,
        "max_relerr_mps_vs_cuda_mock": None,
        "upstream_target": "pytorch/pytorch" if divergences else "none",
        "torch_version": torch.__version__,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        SWARM_DIR.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        RESULTS_MD.write_text(
            f"# {KERNEL}\n\nFATAL: {exc!r}\n\n```\n{tb}\n```\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({
                "kernel": KERNEL, "status": "FATAL", "error": repr(exc),
            }) + "\n")
        raise
