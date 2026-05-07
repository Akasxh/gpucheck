"""Fuzzer for conv-transpose-2d across MPS (real) and CPU reference.

Drives gpucheck's per-dtype tolerance + sqrt(k/128) scaling. CUDA path
is N/A on this Mac (no NVIDIA hardware) — we mark CUDA results UNSUPPORTED
and only validate the MPS-vs-CPU divergence channel.

FILABLE classification (per swarm v2 spec):
  - FILABLE:                max_rel_err > 10x tolerance AND
                            denom_magnitude >= 1e-6 AND
                            reproducible across >=3 seeds
  - TOLERANCE_RECALIBRATION: divergence but fails one of the FILABLE conditions
  - OK:                     within MPS-overlay tolerance
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-conv-transpose-2d"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "conv-transpose-2d"
N_ITERS = 500
BUDGET_S = 8 * 60 - 45  # leave ~45s for reproducibility re-runs + write-out

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_conv-transpose-2d.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice", "transpose_weight", "channels_last"]

# Spatial sizes
SPATIAL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SPATIAL_POW2 = [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]
SPATIAL_NON_TILE = [9, 18, 27, 36, 45, 54, 99, 100, 130]
SPATIAL_LARGE = [128, 192, 256, 320]
SPATIAL_DEGEN = [3, 4]  # smallest plausible spatial that fits a 3x3 kernel

CHANNEL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23]
CHANNEL_POW2 = [4, 8, 16, 32, 64]
CHANNEL_NON_TILE = [6, 12, 18, 24, 48]
CHANNEL_LARGE = [64, 96, 128]
CHANNEL_DEGEN = [1, 2]

KERNEL_SIZES = [1, 2, 3, 4, 5]
PADDINGS = [0, 1, 2]
STRIDES_OP = [1, 2, 3]


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    N: int
    C_in: int
    C_out: int
    H: int
    W: int
    kH: int
    kW: int
    op_stride: int
    op_padding: int
    op_output_padding: int
    groups: int
    dtype: str
    stride: str
    seed: int
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    classification: str = ""  # FILABLE | TOLERANCE_RECALIBRATION | OK | -
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    denom_at_max: float | None = None
    atol: float | None = None
    rtol: float | None = None
    rel_over_tol: float | None = None  # max_rel_err / rtol
    reproductions: int = 0
    note: str = ""


def pick_spatial(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(SPATIAL_DEGEN)
    if bucket == "prime":
        return rng.choice(SPATIAL_PRIMES)
    if bucket == "pow2_boundary":
        return rng.choice(SPATIAL_POW2)
    if bucket == "non_tile_aligned":
        return rng.choice(SPATIAL_NON_TILE)
    if bucket == "large":
        return rng.choice(SPATIAL_LARGE)
    raise ValueError(bucket)


def pick_channel(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(CHANNEL_DEGEN)
    if bucket == "prime":
        return rng.choice(CHANNEL_PRIMES)
    if bucket == "pow2_boundary":
        return rng.choice(CHANNEL_POW2)
    if bucket == "non_tile_aligned":
        return rng.choice(CHANNEL_NON_TILE)
    if bucket == "large":
        return rng.choice(CHANNEL_LARGE)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@dataclass
class Spec:
    bucket: str
    N: int
    C_in: int
    C_out: int
    H: int
    W: int
    kH: int
    kW: int
    op_stride: int
    op_padding: int
    op_output_padding: int
    groups: int
    dtype: str
    stride: str


def sample_spec(rng: random.Random) -> Spec:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)

    N = rng.choice([1, 2])
    groups = 1
    C_in = pick_channel(bucket, rng)
    C_out = pick_channel(bucket, rng)
    H = pick_spatial(bucket, rng)
    W = pick_spatial(bucket, rng)
    kH = rng.choice(KERNEL_SIZES)
    kW = rng.choice(KERNEL_SIZES)
    op_stride = rng.choice(STRIDES_OP)
    op_padding = rng.choice([p for p in PADDINGS if p < max(kH, kW)])
    # output_padding must be < stride and < dilation; we use dilation=1
    op_output_padding = rng.randrange(op_stride)
    return Spec(
        bucket=bucket, N=N, C_in=C_in, C_out=C_out, H=H, W=W, kH=kH, kW=kW,
        op_stride=op_stride, op_padding=op_padding,
        op_output_padding=op_output_padding, groups=groups,
        dtype=dtype, stride=stride,
    )


def make_inputs(spec: Spec, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input, weight, bias) on `device` with the requested stride pattern.

    For conv_transpose2d, weight shape is (C_in, C_out/groups, kH, kW).
    """
    dt = torch_dtype(spec.dtype)
    g = torch.Generator().manual_seed(seed)

    # Build everything in fp32 on CPU first, then cast/transfer for numerical determinism.
    if spec.stride == "contiguous":
        inp = torch.randn(spec.N, spec.C_in, spec.H, spec.W, generator=g, dtype=torch.float32)
    elif spec.stride == "slice":
        # over-allocate width and slice with stride 2
        inp_full = torch.randn(spec.N, spec.C_in, spec.H, spec.W * 2, generator=g, dtype=torch.float32)
        inp = inp_full[:, :, :, ::2].contiguous().clone()
        # Re-create as a non-contiguous slice by re-slicing on a fresh full tensor
        inp_full2 = torch.randn(spec.N, spec.C_in, spec.H, spec.W * 2, generator=g, dtype=torch.float32)
        inp = inp_full2[:, :, :, ::2]  # non-contiguous along W
    elif spec.stride == "transpose_weight":
        # We'll transpose H/W of the input as the stride trick
        inp_t = torch.randn(spec.N, spec.C_in, spec.W, spec.H, generator=g, dtype=torch.float32)
        inp = inp_t.transpose(-1, -2)  # non-contiguous (H,W) view
    elif spec.stride == "channels_last":
        inp_c = torch.randn(spec.N, spec.C_in, spec.H, spec.W, generator=g, dtype=torch.float32)
        inp = inp_c.contiguous(memory_format=torch.channels_last)
    else:
        raise ValueError(spec.stride)

    g2 = torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)
    weight_full = torch.randn(spec.C_in, spec.C_out // spec.groups, spec.kH, spec.kW, generator=g2, dtype=torch.float32)
    bias_full = torch.randn(spec.C_out, generator=torch.Generator().manual_seed(seed ^ 0x12345678), dtype=torch.float32)

    inp_d = inp.to(device=device, dtype=dt)
    weight_d = weight_full.to(device=device, dtype=dt)
    bias_d = bias_full.to(device=device, dtype=dt)
    return inp_d, weight_d, bias_d


def run_op(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, spec: Spec) -> torch.Tensor:
    return F.conv_transpose2d(
        inp, weight, bias=bias,
        stride=spec.op_stride, padding=spec.op_padding,
        output_padding=spec.op_output_padding, groups=spec.groups, dilation=1,
    )


def compare_one(spec: Spec, seed: int) -> dict:
    """Run conv_transpose_2d on CPU and MPS at the given seed, return error metrics."""
    cpu_dev = torch.device("cpu")
    mps_dev = torch.device("mps")

    inp_cpu, w_cpu, b_cpu = make_inputs(spec, seed, cpu_dev)
    inp_mps, w_mps, b_mps = make_inputs(spec, seed, mps_dev)

    # Reference: fp32 (or fp64 for fp32 dtype) on CPU using upcasted inputs to model "infinite precision" baseline
    ref_dtype = torch.float64 if spec.dtype == "float32" else torch.float32
    ref_inp = inp_cpu.to(ref_dtype)
    ref_w = w_cpu.to(ref_dtype)
    ref_b = b_cpu.to(ref_dtype)
    Y_ref = run_op(ref_inp, ref_w, ref_b, spec)

    Y_cpu = run_op(inp_cpu, w_cpu, b_cpu, spec).to(ref_dtype)
    Y_mps = run_op(inp_mps, w_mps, b_mps, spec).to(cpu_dev).to(ref_dtype)

    # k_dim for sqrt scaling = number of accumulations per output element
    # = (C_in // groups) * kH * kW for conv_transpose2d
    k_dim = (spec.C_in // spec.groups) * spec.kH * spec.kW
    atol_mps, rtol_mps = compute_tolerance(spec.dtype, k_dim=k_dim, device_type="mps")
    atol_cpu, rtol_cpu = compute_tolerance(spec.dtype, k_dim=k_dim, device_type="cpu")
    atol, rtol = atol_mps, rtol_mps

    diff = (Y_mps - Y_cpu).abs()
    if diff.numel() == 0:
        return {
            "max_abs_err": 0.0, "max_rel_err": 0.0, "denom_at_max": 0.0,
            "atol": atol, "rtol": rtol, "k_dim": k_dim,
            "atol_cpu": atol_cpu, "rtol_cpu": rtol_cpu,
            "ok": True,
            "ref_drift": 0.0, "cpu_ref_drift": 0.0,
        }

    flat_diff = diff.flatten()
    max_idx = int(flat_diff.argmax().item())
    max_abs = float(flat_diff[max_idx].item())
    denom_full = Y_cpu.abs().flatten()
    # Relative error vs CPU magnitude (avoid div-by-zero).
    safe_denom = denom_full.clamp_min(1e-12)
    rel_full = flat_diff / safe_denom
    max_rel_idx = int(rel_full.argmax().item())
    max_rel = float(rel_full[max_rel_idx].item())
    denom_at_max = float(denom_full[max_rel_idx].item())

    ok = bool(((diff - (atol + rtol * Y_cpu.abs())) <= 0).all().item())

    ref_drift = float((Y_mps - Y_ref).abs().max().item())
    cpu_ref_drift = float((Y_cpu - Y_ref).abs().max().item())

    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "denom_at_max": denom_at_max,
        "atol": atol,
        "rtol": rtol,
        "k_dim": k_dim,
        "atol_cpu": atol_cpu,
        "rtol_cpu": rtol_cpu,
        "ok": ok,
        "ref_drift": ref_drift,
        "cpu_ref_drift": cpu_ref_drift,
    }


def classify(metrics: dict, reproductions: int) -> str:
    """Apply FILABLE filter."""
    rel = metrics["max_rel_err"] or 0.0
    rtol = metrics["rtol"] or 0.0
    denom = metrics["denom_at_max"] or 0.0
    if metrics["ok"]:
        return "OK"
    # 10x tolerance => max_rel_err > 10 * rtol
    if rtol > 0 and rel > 10.0 * rtol and denom >= 1e-6 and reproductions >= 3:
        return "FILABLE"
    return "TOLERANCE_RECALIBRATION"


def run_one(idx: int, rng: random.Random) -> IterResult:
    spec = sample_spec(rng)
    seed = rng.randrange(1 << 30)
    base = IterResult(
        idx=idx, shape_bucket=spec.bucket, N=spec.N,
        C_in=spec.C_in, C_out=spec.C_out, H=spec.H, W=spec.W,
        kH=spec.kH, kW=spec.kW, op_stride=spec.op_stride,
        op_padding=spec.op_padding, op_output_padding=spec.op_output_padding,
        groups=spec.groups, dtype=spec.dtype, stride=spec.stride, seed=seed,
        status="ERROR",
    )
    try:
        m = compare_one(spec, seed)
        base.max_abs_err = m["max_abs_err"]
        base.max_rel_err = m["max_rel_err"]
        base.denom_at_max = m["denom_at_max"]
        base.atol = m["atol"]
        base.rtol = m["rtol"]
        base.rel_over_tol = (m["max_rel_err"] / m["rtol"]) if m["rtol"] else None
        base.status = "OK" if m["ok"] else "DIVERGENCE"
        if not m["ok"]:
            base.note = (
                f"k_dim={m['k_dim']} max_abs={m['max_abs_err']:.3e} "
                f"denom_at_max={m['denom_at_max']:.3e} "
                f"mps-vs-ref={m['ref_drift']:.3e} cpu-vs-ref={m['cpu_ref_drift']:.3e}"
            )
        # If divergence: re-run on 2 additional seeds for reproducibility
        if not m["ok"]:
            reproductions = 1  # the first run counts
            for off in (0xC0FFEE, 0xDEADBEEF):
                m2 = compare_one(spec, seed ^ off)
                if not m2["ok"]:
                    reproductions += 1
            base.reproductions = reproductions
        else:
            base.reproductions = 0
        base.classification = classify(m, base.reproductions)
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {e}"
        base.classification = "-"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported", "MPS")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
        base.classification = "-"
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        base.classification = "-"
    return base


def main() -> int:
    rng = random.Random(0xC047_2D)
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write(f"# {KERNEL_NAME} fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters", flush=True)
            break
        attempted = i + 1
        r = run_one(i, rng)
        results.append(r)
        completed += 1
        if r.status == "DIVERGENCE":
            print(f"[{i:03d}] {r.classification} {r.dtype} {r.stride} N{r.N} "
                  f"C{r.C_in}->{r.C_out} {r.H}x{r.W} k{r.kH}x{r.kW} "
                  f"s{r.op_stride}p{r.op_padding}op{r.op_output_padding} "
                  f"max_rel={r.max_rel_err:.3e} rtol={r.rtol:.3e} repro={r.reproductions}/3",
                  flush=True)
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.note}", flush=True)
        elif r.status == "UNSUPPORTED":
            print(f"[{i:03d}] UNSUPPORTED {r.dtype} {r.stride} {r.note}", flush=True)

    # Aggregate
    divergences = [r for r in results if r.status == "DIVERGENCE"]
    filable = [r for r in divergences if r.classification == "FILABLE"]
    recal = [r for r in divergences if r.classification == "TOLERANCE_RECALIBRATION"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    ok = [r for r in results if r.status == "OK"]
    skipped = [r for r in results if r.status == "SKIP"]

    # Top FILABLE / TOLERANCE_RECAL by rel_over_tol
    def rotk(r): return r.rel_over_tol or 0.0
    filable.sort(key=rotk, reverse=True)
    recal.sort(key=rotk, reverse=True)
    top_filable = filable[:3]
    top_recal = recal[:3]

    overall_max_rel = max((r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0)
    overall_max_abs = max((r.max_abs_err for r in results if r.max_abs_err is not None), default=0.0)

    if filable:
        target = "pytorch/pytorch"
    elif recal:
        target = "gpucheck (tolerance recalibration)"
    else:
        target = "none"

    md: list[str] = []
    md.append(f"# {KERNEL_NAME} fuzz results\n")
    md.append(f"- **kernel:** `torch.nn.functional.conv_transpose2d`")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **OK:** {len(ok)}")
    md.append(f"- **DIVERGENCE total:** {len(divergences)}")
    md.append(f"  - **FILABLE:** {len(filable)}")
    md.append(f"  - **TOLERANCE_RECALIBRATION:** {len(recal)}")
    md.append(f"- **UNSUPPORTED:** {len(unsupported)}")
    md.append(f"- **ERROR:** {len(errors)}")
    md.append(f"- **SKIP:** {len(skipped)}")
    md.append(f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}")
    md.append(f"- **MPS-vs-CPU max absolute error (overall):** {overall_max_abs:.3e}")
    md.append(f"- **MPS-vs-CUDA:** N/A (no NVIDIA hardware on host; CUDA detection mockable but kernel unrunnable)")
    md.append(f"- **recommended upstream target:** {target}")
    md.append(f"- **elapsed:** {time.monotonic() - t0:.1f}s")
    md.append(f"- **torch:** {torch.__version__}\n")

    md.append("## FILABLE filter (per swarm v2 spec)\n")
    md.append("- max_rel_err > 10× rtol (gpucheck MPS overlay)")
    md.append("- denom_magnitude (|y_cpu| at the max-rel element) >= 1e-6")
    md.append("- reproducible across >= 3 seeds (orig + 2 perturbations)")
    md.append("- otherwise classified as TOLERANCE_RECALIBRATION (within 10× tol or low-denom artifact or non-reproducible)\n")

    md.append(f"## Top FILABLE repros ({len(top_filable)})\n")
    if top_filable:
        md.append("| # | dtype | stride | N | C_in→C_out | H×W | k | op_s | op_p | op_op | k_dim | max_abs | max_rel | rtol | rel/rtol | denom@max | repro |")
        md.append("|---|-------|--------|---|------------|-----|---|------|------|-------|-------|---------|---------|------|----------|-----------|-------|")
        for i, r in enumerate(top_filable, 1):
            k_dim_val = (r.C_in // r.groups) * r.kH * r.kW
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.N} | {r.C_in}→{r.C_out} | "
                f"{r.H}×{r.W} | {r.kH}×{r.kW} | {r.op_stride} | {r.op_padding} | "
                f"{r.op_output_padding} | {k_dim_val} | "
                f"{r.max_abs_err:.3e} | {r.max_rel_err:.3e} | {r.rtol:.3e} | "
                f"{(r.rel_over_tol or 0):.1f}× | {(r.denom_at_max or 0):.3e} | {r.reproductions}/3 |"
            )
    else:
        md.append("_No FILABLE divergences found._")
    md.append("")

    md.append(f"## Top TOLERANCE_RECALIBRATION repros ({len(top_recal)})\n")
    if top_recal:
        md.append("| # | dtype | stride | N | C_in→C_out | H×W | k | op_s | op_p | op_op | k_dim | max_abs | max_rel | rtol | rel/rtol | denom@max | repro |")
        md.append("|---|-------|--------|---|------------|-----|---|------|------|-------|-------|---------|---------|------|----------|-----------|-------|")
        for i, r in enumerate(top_recal, 1):
            k_dim_val = (r.C_in // r.groups) * r.kH * r.kW
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.N} | {r.C_in}→{r.C_out} | "
                f"{r.H}×{r.W} | {r.kH}×{r.kW} | {r.op_stride} | {r.op_padding} | "
                f"{r.op_output_padding} | {k_dim_val} | "
                f"{r.max_abs_err:.3e} | {r.max_rel_err:.3e} | {r.rtol:.3e} | "
                f"{(r.rel_over_tol or 0):.1f}× | {(r.denom_at_max or 0):.3e} | {r.reproductions}/3 |"
            )
    else:
        md.append("_No TOLERANCE_RECALIBRATION cases._")
    md.append("")

    if divergences:
        # Aggregate stride-pattern × dtype heatmap of divergence counts
        from collections import Counter
        c = Counter((r.stride, r.dtype) for r in divergences)
        md.append("## Divergence breakdown (stride × dtype)\n")
        md.append("| stride | dtype | count |")
        md.append("|--------|-------|-------|")
        for (st, dt), n in sorted(c.items(), key=lambda x: -x[1]):
            md.append(f"| {st} | {dt} | {n} |")
        md.append("")

    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})\n")
        from collections import Counter
        c = Counter((r.dtype, r.stride) for r in unsupported)
        for (dt, st), n in sorted(c.items(), key=lambda x: -x[1])[:10]:
            ex = next(r for r in unsupported if r.dtype == dt and r.stride == st)
            md.append(f"- {dt} / {st}: {n} cases — first: `{ex.note}`")
        md.append("")

    if errors:
        md.append(f"## ERROR summary ({len(errors)})\n")
        for r in errors[:10]:
            md.append(f"- iter {r.idx} {r.dtype}/{r.stride}/{r.shape_bucket}: {r.note}")
        md.append("")

    md.append("## Methodology notes\n")
    md.append("- Op: `torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups=1, dilation=1)`")
    md.append("- Inputs synthesized in fp32 on CPU then cast/transferred to MPS so both devices see numerically identical source data.")
    md.append("- Reference: fp64 (for fp32 input) or fp32 (for fp16/bf16 input) on CPU with upcasted operands.")
    md.append("- Pass condition: `|Y_mps - Y_cpu| <= atol + rtol·|Y_cpu|` element-wise.")
    md.append("- Tolerance: `compute_tolerance(dtype, k_dim=(C_in/g)*kH*kW, device_type='mps')` — base + sqrt(k/128) scaling + MPS overlay (2× provisional).")
    md.append("- Stride/contiguity patterns: contiguous, slice (W stride-2 view), transpose (H↔W view), channels_last (NHWC memory format).")
    md.append("- Reproducibility check: each divergent case is replayed with `seed ^ 0xC0FFEE` and `seed ^ 0xDEADBEEF`; case is FILABLE only if all three reproduce.")
    md.append("- CUDA channel: not exercised — host has no NVIDIA GPU; gpucheck mocks detection but cannot run kernels.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "divergences": len(divergences),
        "filable": len(filable),
        "tolerance_recalibration": len(recal),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cpu_max_abs_err": overall_max_abs,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_no_nvidia_hardware",
        "top_filable": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride,
                "N": r.N, "C_in": r.C_in, "C_out": r.C_out,
                "H": r.H, "W": r.W, "kH": r.kH, "kW": r.kW,
                "op_stride": r.op_stride, "op_padding": r.op_padding,
                "op_output_padding": r.op_output_padding, "groups": r.groups,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err, "max_abs_err": r.max_abs_err,
                "rtol": r.rtol, "atol": r.atol,
                "rel_over_tol": r.rel_over_tol,
                "denom_at_max": r.denom_at_max,
                "reproductions": r.reproductions,
                "seed": r.seed,
                "note": r.note,
            }
            for i, r in enumerate(top_filable)
        ],
        "top_tolerance_recalibration": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride,
                "N": r.N, "C_in": r.C_in, "C_out": r.C_out,
                "H": r.H, "W": r.W, "kH": r.kH, "kW": r.kW,
                "op_stride": r.op_stride, "op_padding": r.op_padding,
                "op_output_padding": r.op_output_padding, "groups": r.groups,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err, "max_abs_err": r.max_abs_err,
                "rtol": r.rtol, "atol": r.atol,
                "rel_over_tol": r.rel_over_tol,
                "denom_at_max": r.denom_at_max,
                "reproductions": r.reproductions,
                "seed": r.seed,
                "note": r.note,
            }
            for i, r in enumerate(top_recal)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": time.monotonic() - t0,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
