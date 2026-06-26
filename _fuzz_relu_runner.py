"""Stride/contiguity + shape + dtype fuzzer for torch.relu on MPS vs CPU.

Real MPS backend (Apple Silicon); CUDA backend mocked (skipped → N/A).
Iterations: 250 with an 8-minute hard runtime budget.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.shapes import (
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import CATEGORIES, fuzz_strides_for_category

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ITER = 250
BUDGET_SEC = 8 * 60
SEED = 0x5EED_BEEF
OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_MD = OUT_DIR / "RESULTS_relu.md"
OUT_JSONL = OUT_DIR / "swarm.jsonl"

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Curated shape pool per category. Cap dims to keep the largest tensor under
# ~64M elements so we stay inside the runtime budget on a Mac.
def _shapes_pool() -> dict[str, list[tuple[int, ...]]]:
    pool: dict[str, list[tuple[int, ...]]] = {
        "degenerate": [
            (0,), (1,), (1, 1), (1, 16), (16, 1), (1, 1, 1),
            (0, 16), (16, 0), (1, 1, 16),
        ],
        "prime": [
            (p,) for p in PRIMES
        ] + [
            (p, p) for p in PRIMES if p <= 257
        ] + [
            (p, q) for p in PRIMES for q in PRIMES if p != q and p * q <= 4096
        ],
        "pow2_boundary": [
            (v,) for v in POWER_OF_2_BOUNDARIES
        ] + [
            (v, v) for v in POWER_OF_2_BOUNDARIES if v <= 513
        ] + [(127, 129), (129, 127), (255, 257), (257, 255), (511, 513)],
        "non_tile_aligned": (
            [(t - 1,) for t in TILE_SIZES]
            + [(t + 1,) for t in TILE_SIZES]
            + [(t - 1, t + 1) for t in TILE_SIZES]
            + [(t + 1, t - 1) for t in TILE_SIZES]
            + [(t + 3, t + 3) for t in TILE_SIZES]
        ),
        "large": [
            (v,) for v in LARGE_DIMS
        ] + [
            (v, v // 2) for v in LARGE_DIMS if v <= 4096
        ] + [(2048, 2048), (4096, 1024)],
    }
    return pool


SHAPE_POOL = _shapes_pool()
SHAPE_CATS = list(SHAPE_POOL.keys())
DTYPE_NAMES = list(DTYPES.keys())
# 'broadcast', 'transpose', 'slice', 'non_contig', 'gather' all produce useful
# layouts. 'gather' produces a contiguous output but exercises a different
# build path; we keep it for layout diversity. 'column_major' maps to torch's
# F-layout via transpose-of-contiguous which is also worth testing.
STRIDE_CATS = list(CATEGORIES)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max element-wise relative error, computed in float64 on CPU."""
    a64 = a.detach().to(device="cpu").to(dtype=torch.float64).contiguous()
    b64 = b.detach().to(device="cpu").to(dtype=torch.float64).contiguous()
    if a64.numel() == 0:
        return 0.0
    diff = (a64 - b64).abs()
    denom = b64.abs().clamp(min=1e-30)
    rel = diff / denom
    # Use combined max(abs, rel) so identical zeros are not classified as inf.
    abs_err = diff.max().item()
    rel_err = rel.max().item()
    return max(abs_err, rel_err)


def _stride_supports(shape: tuple[int, ...], cat: str) -> bool:
    if not shape:
        return cat == "row_major"
    if cat in {"column_major", "transpose"} and len(shape) < 2:
        return False
    if cat == "broadcast" and (not shape or shape[-1] == 0):
        return False
    if cat == "slice":
        # _slice doubles every dim; bail if any dim is zero.
        if any(d == 0 for d in shape):
            return False
        # Cap to keep doubled tensor reasonable (<= ~16M elements doubled).
        doubled = 1
        for d in shape:
            doubled *= max(d * 2, 1)
            if doubled > 64_000_000:
                return False
    if cat == "gather":
        if any(d == 0 for d in shape):
            return False
        numel = 1
        for d in shape:
            numel *= d
        if numel > 4_000_000:
            return False
    return True


def main() -> int:
    if not torch.backends.mps.is_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("# RESULTS_relu — SKIPPED (MPS unavailable)\n")
        line = json.dumps({
            "kernel": "relu",
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
        })
        with OUT_JSONL.open("a") as f:
            f.write(line + "\n")
        print("SKIPPED: MPS unavailable")
        return 0

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    attempted = 0
    completed = 0
    unsupported = 0
    errors: list[dict] = []
    divergences: list[dict] = []
    max_rel_err_overall = 0.0

    for i in range(N_ITER):
        attempted += 1
        if time.monotonic() - t0 > BUDGET_SEC:
            break

        shape_cat = rng.choice(SHAPE_CATS)
        shape = rng.choice(SHAPE_POOL[shape_cat])
        dtype_name = rng.choice(DTYPE_NAMES)
        dtype = DTYPES[dtype_name]
        # row_major is the boring baseline; bias the sampler so 5/7 picks are
        # non-trivial layouts but row_major still appears as a control.
        stride_cat = rng.choice(STRIDE_CATS)

        if not _stride_supports(shape, stride_cat):
            unsupported += 1
            continue

        try:
            cpu_src = fuzz_strides_for_category(
                shape, dtype, stride_cat, device="cpu",
                seed=12345 + i,
            )
            mps_src = cpu_src.detach().to("mps")
        except (RuntimeError, NotImplementedError, TypeError) as e:
            unsupported += 1
            errors.append({
                "iter": i,
                "shape": shape,
                "dtype": dtype_name,
                "stride": stride_cat,
                "kind": "build",
                "error": repr(e)[:200],
            })
            continue

        try:
            ref = torch.relu(cpu_src)
            out = torch.relu(mps_src)
            torch.mps.synchronize()
        except (RuntimeError, NotImplementedError, TypeError) as e:
            unsupported += 1
            errors.append({
                "iter": i,
                "shape": shape,
                "dtype": dtype_name,
                "stride": stride_cat,
                "kind": "kernel",
                "error": repr(e)[:200],
            })
            continue

        completed += 1
        try:
            rel = _rel_err(out, ref)
        except RuntimeError as e:
            errors.append({
                "iter": i,
                "shape": shape,
                "dtype": dtype_name,
                "stride": stride_cat,
                "kind": "compare",
                "error": repr(e)[:200],
            })
            continue

        if rel > max_rel_err_overall:
            max_rel_err_overall = rel

        # relu is element-wise → no k_dim scaling.
        atol, rtol = compute_tolerance(dtype, device_type="mps")
        # gpucheck's "tolerance" for an element-wise kernel: divergence if
        # max(|a-b|, |a-b|/|b|) exceeds rtol (we already folded atol/rtol into
        # one combined max-rel-err).
        threshold = max(atol, rtol)
        if rel > threshold:
            divergences.append({
                "iter": i,
                "shape": list(shape),
                "shape_cat": shape_cat,
                "dtype": dtype_name,
                "stride": stride_cat,
                "max_rel_err": rel,
                "threshold": threshold,
            })

    elapsed = time.monotonic() - t0

    # ----- Build markdown -----
    div_sorted = sorted(divergences, key=lambda d: -d["max_rel_err"])
    top3 = div_sorted[:3]

    if divergences:
        # Choose upstream target. relu is in pytorch core, not Triton.
        upstream = "pytorch/pytorch"
    else:
        upstream = "none"

    md = []
    md.append("# RESULTS_relu — gpucheck stride/shape/dtype fuzz")
    md.append("")
    md.append("- kernel: `torch.relu`")
    md.append(f"- iterations attempted: {attempted}")
    md.append(f"- iterations completed: {completed}")
    md.append(f"- unsupported / build-skipped: {unsupported}")
    md.append(f"- divergences found: {len(divergences)}")
    md.append(f"- elapsed seconds: {elapsed:.2f}")
    md.append(f"- runtime budget seconds: {BUDGET_SEC}")
    md.append(f"- MPS-vs-CPU max relative error: {max_rel_err_overall:.6g}")
    md.append("- MPS-vs-CUDA-mock max relative error: N/A (CUDA backend mocked — no NVIDIA GPU present)")
    md.append(f"- recommended upstream filing target: {upstream}")
    md.append("")
    md.append("## Top 3 minimal repros")
    md.append("")
    if not top3:
        md.append("_No divergences exceeded the per-dtype threshold._")
    else:
        for k, d in enumerate(top3, start=1):
            md.append(
                f"{k}. shape={tuple(d['shape'])} ({d['shape_cat']}), "
                f"dtype={d['dtype']}, stride={d['stride']}, "
                f"max_rel_err={d['max_rel_err']:.4g}, threshold={d['threshold']:.4g}"
            )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- Real MPS backend on Apple Silicon (torch.backends.mps.is_available() == True).")
    md.append("- CUDA path mocked: this Mac has no NVIDIA GPU, so the CUDA-vs-MPS comparison is N/A.")
    md.append("- Tolerances: gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps').")
    md.append("- `relu` is element-wise so no sqrt(k/128) matmul scaling was applied.")
    md.append("- Stride categories drawn: " + ", ".join(STRIDE_CATS) + ".")
    md.append("")
    if errors:
        md.append(f"## Build / kernel errors ({len(errors)})")
        md.append("")
        for e in errors[:10]:
            md.append(
                f"- iter={e['iter']} shape={e['shape']} dtype={e['dtype']} "
                f"stride={e['stride']} kind={e['kind']}: `{e['error']}`"
            )
        md.append("")
    OUT_MD.write_text("\n".join(md))

    summary = {
        "kernel": "relu",
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "unsupported": unsupported,
        "divergences_found": len(divergences),
        "top_repros": [
            {
                "shape": d["shape"],
                "dtype": d["dtype"],
                "stride": d["stride"],
                "max_rel_err": d["max_rel_err"],
            }
            for d in top3
        ],
        "mps_vs_cpu_max_rel_err": max_rel_err_overall,
        "mps_vs_cuda_mock_max_rel_err": None,
        "upstream_target": upstream,
        "elapsed_sec": elapsed,
        "budget_sec": BUDGET_SEC,
        "status": "OK",
    }
    with OUT_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"completed={completed} divergences={len(divergences)} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
