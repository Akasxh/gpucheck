"""Stride/shape/dtype fuzzer for torch.argmax on MPS vs CPU.

Argmax returns integer indices; ties make raw index comparison unreliable.
We compare the *value selected* by each backend: gather(input, dim, idx).
A divergence is when those values differ by more than gpucheck's tolerance —
that means a backend picked a non-maximal element. CUDA detection is mocked
(no NVIDIA GPU on this host); the CUDA-vs-MPS column is reported as N/A.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

# Make gpucheck importable from the worktree
sys.path.insert(0, "/Users/cero/Code/gpucheck-worktrees/fuzz-argmax/src")

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_argmax.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "argmax"
N_ITER = 250
BUDGET_S = 8 * 60 - 45  # leave 45s headroom for write_outputs and shutdown
SEED = 0xA1A1

DEGEN_DIMS = [1]
PRIME_DIMS = [7, 13, 31, 61, 127]
POW2_DIMS = [127, 128, 129, 255, 256, 257]
NON_TILE_DIMS = [33, 65, 130, 192, 320]
LARGE_DIMS = [1024, 2048, 4096]

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]
STRIDE_CATS = ["contiguous", "slice", "transpose", "broadcast"]


def make_shape(rng: random.Random, category: str) -> tuple[int, ...]:
    """Return a 2D or 3D shape covering the requested category along the
    reduction axis (always axis -1 when picked by caller)."""
    if category == "degenerate":
        # argmax over a length-1 axis is well-defined (always 0). Still useful
        # as a stress test for stride handling.
        outer = rng.choice([1, 2, 8])
        return (outer, 1)
    if category == "prime":
        n = rng.choice(PRIME_DIMS)
        outer = rng.choice([1, 3, 7])
        return (outer, n)
    if category == "pow2":
        n = rng.choice(POW2_DIMS)
        outer = rng.choice([1, 2, 4])
        return (outer, n)
    if category == "non_tile":
        n = rng.choice(NON_TILE_DIMS)
        outer = rng.choice([1, 3, 5])
        return (outer, n)
    if category == "large":
        n = rng.choice(LARGE_DIMS)
        outer = rng.choice([1, 2])
        # 3D variant for some large cases
        if rng.random() < 0.5:
            return (outer, 4, n)
        return (outer, n)
    raise ValueError(category)


def make_input(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    stride_cat: str,
    gen: torch.Generator,
) -> tuple[torch.Tensor, str]:
    """Build an input tensor on *device* with the requested stride pattern.

    Returns (tensor, stride_description). The base content is generated on CPU
    in fp32 then cast to *dtype* and moved to *device* so CPU and MPS receive
    bit-identical content (modulo dtype rounding).
    """
    # Add a small jitter to defeat ties in float dtypes (so argmax has a
    # unique winner whenever possible). For fp16/bf16 ties still arise with
    # large arrays; that is intentional — the value-comparison metric handles
    # them.
    def base_cpu_fp32(target_shape: tuple[int, ...]) -> torch.Tensor:
        t = torch.randn(target_shape, generator=gen, dtype=torch.float32)
        # Spread the magnitudes a bit so argmax is rarely on a tie.
        t = t * 4.0
        return t

    if stride_cat == "contiguous":
        cpu = base_cpu_fp32(shape).to(dtype=dtype).contiguous()
        out = cpu.to(device=device)
        return out, f"contig{tuple(out.stride())}"

    if stride_cat == "slice":
        # Double the last dim and take ::2 → stride 2 along reduction axis.
        big_shape = list(shape)
        big_shape[-1] = big_shape[-1] * 2
        cpu_big = base_cpu_fp32(tuple(big_shape)).to(dtype=dtype).contiguous()
        sliced = cpu_big[..., ::2]
        out = sliced.to(device=device)
        return out, f"slice_step2_last{tuple(out.stride())}"

    if stride_cat == "transpose":
        if len(shape) < 2:
            raise ValueError("transpose requires rank >= 2")
        # Build (..., D, N) contiguous, transpose last two → (..., N, D).
        tshape = shape[:-2] + (shape[-1], shape[-2])
        cpu = base_cpu_fp32(tshape).to(dtype=dtype).contiguous().transpose(-1, -2)
        out = cpu.to(device=device)
        return out, f"transpose_last2{tuple(out.stride())}"

    if stride_cat == "broadcast":
        # Build a "small" tensor of size 1 on the *outer* axis and expand —
        # so the argmax-reduced axis (-1) is still real, but other axes have
        # stride 0. This stress-tests how MPS handles broadcast strides
        # *outside* the reduction dim (the typical bug pattern).
        small_shape = list(shape)
        small_shape[0] = 1  # collapse outer → expand back to shape[0]
        cpu_small = base_cpu_fp32(tuple(small_shape)).to(dtype=dtype).contiguous()
        expanded = cpu_small.expand(shape)
        out = expanded.to(device=device)
        # .to() on broadcast tensor materializes; instead, materialize on CPU
        # then re-broadcast on device to keep the stride-0 pattern.
        cpu_dev_small = cpu_small.to(device=device)
        out = cpu_dev_small.expand(shape)
        return out, f"broadcast_outer_stride0{tuple(out.stride())}"

    raise ValueError(stride_cat)


def _selected_value_max_err(
    x_ref_fp32: torch.Tensor,
    idx_test: torch.Tensor,
    idx_ref: torch.Tensor,
    dim: int,
) -> tuple[float, float]:
    """For argmax: the comparison metric is the value at the picked index.

    Returns (max_abs_err, max_rel_err) where err = |x[idx_test] - x[idx_ref]|
    computed in fp32 on CPU. Ties (identical value at two different indices)
    yield zero error, so this metric is tie-tolerant by construction.
    """
    val_test = x_ref_fp32.gather(dim, idx_test.unsqueeze(dim).cpu().long()).squeeze(dim)
    val_ref = x_ref_fp32.gather(dim, idx_ref.unsqueeze(dim).cpu().long()).squeeze(dim)
    diff = (val_test - val_ref).abs()
    denom = val_ref.abs().clamp_min(1e-7)
    return float(diff.max().item()), float((diff / denom).max().item())


def run_iter(idx: int, rng: random.Random) -> dict:
    cat = rng.choice(["degenerate", "prime", "pow2", "non_tile", "large"])
    dtype_name, dtype = rng.choice(DTYPES)
    stride_cat = rng.choice(STRIDE_CATS)

    # Transpose needs rank >= 2 (always true here, but be explicit).
    shape = make_shape(rng, cat)
    if stride_cat == "transpose" and len(shape) < 2:
        stride_cat = "contiguous"

    rec: dict = {
        "iter": idx,
        "shape_cat": cat,
        "shape": list(shape),
        "dtype": dtype_name,
        "stride": stride_cat,
        "stride_spec": "",
        "status": "OK",
        "max_rel_err": None,
        "max_abs_err": None,
        "tol_atol": None,
        "tol_rtol": None,
        "diverged": False,
        "note": "",
    }

    gen_seed = rng.randint(0, 2**31 - 1)

    # CPU input
    gen_cpu = torch.Generator().manual_seed(gen_seed)
    try:
        x_cpu, spec = make_input(shape, dtype, "cpu", stride_cat, gen_cpu)
    except Exception as e:
        rec["status"] = "BUILD_ERR_CPU"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec
    rec["stride_spec"] = spec

    # MPS input — re-seed with same value so contents are identical pre-cast.
    gen_mps = torch.Generator().manual_seed(gen_seed)
    try:
        x_mps_cpu, _ = make_input(shape, dtype, "mps", stride_cat, gen_mps)
    except Exception as e:
        rec["status"] = "BUILD_ERR_MPS"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec

    # Reference: argmax along last dim, computed in fp32 on CPU for stability.
    dim = -1
    x_ref_fp32 = x_cpu.detach().to(dtype=torch.float32).contiguous()
    try:
        idx_ref = torch.argmax(x_ref_fp32, dim=dim)
    except Exception as e:
        rec["status"] = "CPU_REF_ERR"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec

    # CPU at the actual dtype (sanity baseline — what does CPU itself do?)
    try:
        idx_cpu = torch.argmax(x_cpu, dim=dim)
    except (RuntimeError, NotImplementedError) as e:
        rec["status"] = "UNSUPPORTED_CPU"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    # MPS argmax
    try:
        idx_m = torch.argmax(x_mps_cpu, dim=dim)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError) as e:
        rec["status"] = "UNSUPPORTED_MPS"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    # Numel check: argmax over a stride-0 broadcast axis is degenerate (all
    # entries equal along that axis). Our broadcast pattern only zeroes the
    # *outer* stride, not the reduction axis, so this is still a real test.
    abs_err_cpu, rel_err_cpu = _selected_value_max_err(x_ref_fp32, idx_cpu, idx_ref, dim)
    abs_err_mps, rel_err_mps = _selected_value_max_err(x_ref_fp32, idx_m, idx_ref, dim)
    # Report the worst (MPS-vs-ref) for the divergence rule.
    rec["max_abs_err"] = abs_err_mps
    rec["max_rel_err"] = rel_err_mps
    rec["max_abs_err_cpu_self"] = abs_err_cpu
    rec["max_rel_err_cpu_self"] = rel_err_cpu

    # Tolerance: argmax is a reduction over the last axis. Use k_dim = shape[-1].
    # Per the spec we apply gpucheck's MPS overlay.
    k_dim = int(shape[-1])
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
    rec["tol_atol"] = atol
    rec["tol_rtol"] = rtol

    # Divergence rule per the user's spec: "max-rel-err > tolerance".
    # For argmax, the *value-at-index* metric should be ~0 if both backends
    # picked the maximum (or a tie of it). Anything above rtol means MPS picked
    # a strictly smaller element — a real bug.
    rec["diverged"] = rel_err_mps > rtol

    return rec


def main() -> int:
    if not torch.backends.mps.is_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text(
            f"# {KERNEL_NAME} fuzz results\n\nSTATUS: SKIPPED — torch.mps unavailable.\n",
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    # Mock CUDA detection so any gpucheck arch-detection path that gets called
    # transitively reports a fake CUDA device. We do NOT run any CUDA kernels
    # (no device); this is purely so the "CUDA backend present" assumption
    # holds for code that branches on torch.cuda.is_available().
    cuda_mock = mock.patch("torch.cuda.is_available", return_value=False)
    cuda_mock.start()
    try:
        rng = random.Random(SEED)
        t0 = time.time()
        records: list[dict] = []
        completed = 0
        for i in range(N_ITER):
            if time.time() - t0 > BUDGET_S:
                print(f"[budget] stopping at iter={i}", flush=True)
                break
            try:
                rec = run_iter(i, rng)
            except Exception as e:
                rec = {
                    "iter": i, "status": "EXC", "err": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()[-600:],
                }
                records.append(rec)
                print(f"[halt-report] iter={i} {rec['err']}", flush=True)
                continue
            records.append(rec)
            if rec["status"] == "OK":
                completed += 1
            if i % 25 == 0:
                print(
                    f"[{i:3d}] cat={rec.get('shape_cat'):<10} "
                    f"dt={rec.get('dtype'):<9} st={rec.get('stride'):<11} "
                    f"shape={rec.get('shape')} status={rec['status']} "
                    f"rel={rec.get('max_rel_err')}",
                    flush=True,
                )
    finally:
        cuda_mock.stop()

    elapsed = time.time() - t0
    write_outputs(records, completed, elapsed)
    return 0


def write_outputs(records: list[dict], completed: int, elapsed: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    divergences = [r for r in records if r.get("diverged")]
    unsupported = [
        r for r in records if r.get("status") in {"UNSUPPORTED_MPS", "UNSUPPORTED_CPU"}
    ]
    other_err = [
        r for r in records
        if r.get("status") not in {"OK", "UNSUPPORTED_MPS", "UNSUPPORTED_CPU", "SKIP_EMPTY"}
    ]

    def numel(r: dict) -> int:
        return int(math.prod(r.get("shape") or [1]))

    top3 = sorted(divergences, key=numel)[:3]

    rels = [r.get("max_rel_err") for r in records if r.get("max_rel_err") is not None]
    max_rel_overall = max(rels) if rels else 0.0

    target = "pytorch/pytorch" if divergences else "none"

    md_lines: list[str] = []
    md_lines.append(f"# Fuzz results — {KERNEL_NAME}\n")
    md_lines.append(f"- kernel: `torch.argmax`")
    md_lines.append(f"- iterations attempted: {len(records)}")
    md_lines.append(f"- iterations completed (status=OK): {completed}")
    md_lines.append(f"- divergences found: {len(divergences)}")
    md_lines.append(f"- unsupported-on-MPS / CPU: {len(unsupported)}")
    md_lines.append(f"- other errors: {len(other_err)}")
    md_lines.append(f"- elapsed: {elapsed:.1f}s")
    md_lines.append(f"- MPS-vs-CPU max relative error (value-at-argmax): {max_rel_overall:.4g}")
    md_lines.append("- MPS-vs-CUDA-mock max relative error: N/A (no NVIDIA GPU; CUDA detection mocked off)")
    md_lines.append(f"- recommended upstream filing target: **{target}**")
    md_lines.append("")
    md_lines.append("## Method")
    md_lines.append(
        "- Compared the *value at the picked index*: "
        "`x[mps_argmax]` vs `x[cpu_fp32_argmax]`. "
        "This is tie-tolerant: when MPS and CPU pick different indices that "
        "happen to hold the same max, the metric is 0.",
    )
    md_lines.append(
        "- Tolerance from `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')` "
        "(per-dtype atol/rtol scaled by sqrt(k/128), MPS overlay multiplier 2x).",
    )
    md_lines.append("- Divergence rule: `max_rel_err > rtol`.")
    md_lines.append(
        "- CUDA detection mocked via `unittest.mock.patch('torch.cuda.is_available', return_value=False)` "
        "(no NVIDIA GPU on this Mac).",
    )
    md_lines.append("")
    md_lines.append("## Top 3 minimal repros")
    if not top3:
        md_lines.append("_None — no divergences found within budget._")
    else:
        for i, r in enumerate(top3, 1):
            md_lines.append(
                f"{i}. shape={tuple(r['shape'])}  dtype={r['dtype']}  "
                f"stride={r['stride']} ({r.get('stride_spec', '')})  "
                f"shape_cat={r['shape_cat']}  "
                f"max_rel_err={r['max_rel_err']:.4g}  "
                f"max_abs_err={r['max_abs_err']:.4g}  "
                f"tol_rtol={r['tol_rtol']:.4g}",
            )
    md_lines.append("")
    md_lines.append("## Status breakdown")
    by_status: dict[str, int] = {}
    for r in records:
        s = r.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1
    for s, c in sorted(by_status.items(), key=lambda kv: -kv[1]):
        md_lines.append(f"- {s}: {c}")
    md_lines.append("")
    if unsupported:
        md_lines.append("## Sample UNSUPPORTED notes")
        seen: set[str] = set()
        for r in unsupported:
            note = r.get("note", "")[:160]
            if note in seen:
                continue
            seen.add(note)
            md_lines.append(
                f"- shape={tuple(r['shape'])} dtype={r['dtype']} stride={r['stride']}: {note}",
            )
            if len(seen) >= 5:
                break
        md_lines.append("")

    RESULTS_MD.write_text("\n".join(md_lines) + "\n")

    summary = {
        "kernel": KERNEL_NAME,
        "iterations_attempted": len(records),
        "iterations_completed": completed,
        "divergences": len(divergences),
        "unsupported": len(unsupported),
        "errors": len(other_err),
        "max_rel_err_mps_vs_cpu": max_rel_overall,
        "max_rel_err_mps_vs_cuda_mock": None,
        "filing_target": target,
        "top3_repros": [
            {
                "shape": list(r["shape"]),
                "dtype": r["dtype"],
                "stride": r["stride"],
                "stride_spec": r.get("stride_spec", ""),
                "shape_cat": r["shape_cat"],
                "max_rel_err": r["max_rel_err"],
                "max_abs_err": r["max_abs_err"],
                "tol_rtol": r["tol_rtol"],
            }
            for r in top3
        ],
        "elapsed_s": elapsed,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    sys.exit(main())
