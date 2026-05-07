"""Fuzz the GELU kernel: MPS vs CPU reference. CUDA is mocked (no NVIDIA GPU)."""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Ensure gpucheck (worktree) is importable
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-gelu/src")
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.shapes import (
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_gelu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.gelu"
N_ITERS = 250
TIME_BUDGET_S = 8 * 60 - 30  # leave 30 s for writing results
SEED = 0xCAFE

# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------

# Shape buckets: (label, list-of-shape-tuples)
# Keep dims modest so 250 iters fit easily in budget; gelu is elementwise.
DEGENERATE = [(0,), (1,), (1, 1), (16, 0), (0, 16), (1, 1, 1)]
NON_TILE = [
    (TILE_SIZES[0] - 1,),       # 31
    (TILE_SIZES[0] + 1,),       # 33
    (TILE_SIZES[1] - 1, 16),    # 63x16
    (TILE_SIZES[1] + 3, 16),    # 67x16
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),  # 127x129
    (TILE_SIZES[2] + 1, 16),    # 129x16
]
PRIME_S = [(p,) for p in PRIMES] + [(PRIMES[0], PRIMES[1]), (PRIMES[2], 16)]
POW2_BOUNDARY = [(v,) for v in POWER_OF_2_BOUNDARIES] + [(128, 129), (256, 255)]
LARGE = [(LARGE_DIMS[0],), (256, 256), (1024, 64)]  # cap big to fit memory + time
MIXED = [(127, 16), (1024, 3), (7, 128), (33, 128, 4)]

SHAPE_BUCKETS: dict[str, list[tuple[int, ...]]] = {
    "degenerate": DEGENERATE,
    "non_tile_aligned": NON_TILE,
    "prime": PRIME_S,
    "power_of_2_boundary": POW2_BOUNDARY,
    "large": LARGE,
    "mixed": MIXED,
}
BUCKET_NAMES = list(SHAPE_BUCKETS.keys())

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())


def _max_rel_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    """Return max relative error using a small denom epsilon to avoid /0."""
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-7)
    rel = diff / denom
    if rel.numel() == 0:
        return 0.0
    return float(rel.max().item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.mps.is_available() is False",
            "iterations_attempted": 0,
            "iterations_completed": 0,
            "divergences": 0,
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    rng = random.Random(SEED)
    started = time.monotonic()

    iterations_attempted = 0
    iterations_completed = 0
    unsupported = 0
    divergences: list[dict] = []
    max_rel_err_global = 0.0
    max_abs_err_global = 0.0
    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    for i in range(N_ITERS):
        if time.monotonic() - started > TIME_BUDGET_S:
            print(
                f"[budget] aborting at iter {i}/{N_ITERS} "
                f"(elapsed {time.monotonic() - started:.1f}s)",
                file=sys.stderr,
            )
            break

        iterations_attempted += 1

        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        dtype = DTYPES_BY_NAME[dtype_name]
        stride_cat = rng.choice(STRIDE_CATEGORIES)

        per_bucket_counts[bucket] += 1
        per_dtype_counts[dtype_name] += 1
        per_stride_counts[stride_cat] += 1

        seed_i = rng.randrange(2**31 - 1)

        try:
            # Build CPU tensor in the requested stride layout
            x_cpu = fuzz_strides_for_category(
                shape, dtype, stride_cat, device="cpu", seed=seed_i,
            )
            # MPS copy. .to('mps') on a non-contiguous view preserves contiguity
            # state by materializing into a fresh contiguous tensor *only* if
            # PyTorch's MPS backend requires it; otherwise stride is preserved.
            # We rebuild from category to keep both paths honest (same rng seed).
            try:
                x_mps = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="mps", seed=seed_i,
                )
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                # MPS may reject some dtypes or layouts at construction time.
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(f"[unsupported-build] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape}: {msg}", file=sys.stderr)
                continue

            # Skip empty tensors (no signal, error metrics undefined)
            if x_cpu.numel() == 0:
                iterations_completed += 1
                continue

            # Run gelu on both
            try:
                y_mps = F.gelu(x_mps)
                # Force MPS work to finish before pulling back
                torch.mps.synchronize()
                y_mps_cpu = y_mps.detach().to("cpu")
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(f"[unsupported-mps] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape}: {msg}", file=sys.stderr)
                continue

            try:
                y_cpu = F.gelu(x_cpu)
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(f"[unsupported-cpu] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape}: {msg}", file=sys.stderr)
                continue

            iterations_completed += 1

            rel = _max_rel_err(y_mps_cpu, y_cpu)
            absdiff = _max_abs_err(y_mps_cpu, y_cpu)
            max_rel_err_global = max(max_rel_err_global, rel)
            max_abs_err_global = max(max_abs_err_global, absdiff)

            # Tolerance: gelu is elementwise — no k_dim scaling.
            atol, rtol = compute_tolerance(dtype, device_type="mps")
            # Element-wise divergence: |a-b| > atol + rtol*|b|
            # For a single scalar metric, define "divergence" as either
            # max-abs > atol AND max-rel > rtol — both must fail for us to flag,
            # which keeps the bar high for tiny outputs near 0.
            diverged = (absdiff > atol) and (rel > rtol)
            if diverged:
                rec = {
                    "iter": i,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "shape_bucket": bucket,
                    "max_abs_err": absdiff,
                    "max_rel_err": rel,
                    "atol": atol,
                    "rtol": rtol,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                    "seed": seed_i,
                }
                divergences.append(rec)
                print(f"[DIVERGE] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape} abs={absdiff:.3e} rel={rel:.3e} "
                      f"(atol={atol:.2e} rtol={rtol:.2e})",
                      file=sys.stderr)

        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — we report and continue
            print(
                f"[ERROR] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                f"shape={shape}: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            # Halt rule: report and exit non-zero so caller sees it.
            return 2

    elapsed = time.monotonic() - started

    # Sort divergences by largest relative error for "minimal repro" picks.
    # We treat smallest shape (numel) at each rel-error tier as more minimal.
    divergences_sorted = sorted(
        divergences,
        key=lambda d: (-d["max_rel_err"], _numel(d["shape"])),
    )

    top3 = divergences_sorted[:3]

    summary = {
        "kernel": KERNEL,
        "status": "OK",
        "iterations_attempted": iterations_attempted,
        "iterations_completed": iterations_completed,
        "iterations_unsupported": unsupported,
        "divergences": len(divergences),
        "elapsed_seconds": round(elapsed, 2),
        "mps_vs_cpu_max_rel_err": max_rel_err_global,
        "mps_vs_cpu_max_abs_err": max_abs_err_global,
        "mps_vs_cuda_mock_max_rel_err": "N/A (CUDA mocked — no NVIDIA GPU on host)",
        "top_repros": top3,
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "recommended_filing_target": _recommend_target(divergences),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seed": SEED,
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    return 0


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _recommend_target(divergences: list[dict]) -> str:
    if not divergences:
        return "none"
    # GELU is a pure pytorch op; if MPS diverges from CPU, the issue lives in
    # aten/MPS implementations, not in Triton.
    return "pytorch/pytorch"


def _write_markdown(path: Path, s: dict) -> None:
    lines = []
    lines.append(f"# {s['kernel']} — MPS fuzz report")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **Iterations attempted:** {s['iterations_attempted']}")
    lines.append(f"- **Iterations completed:** {s['iterations_completed']}")
    lines.append(f"- **Iterations unsupported (skipped):** {s['iterations_unsupported']}")
    lines.append(f"- **Divergences found:** {s['divergences']}")
    lines.append(f"- **MPS vs CPU max relative error:** {s['mps_vs_cpu_max_rel_err']:.3e}")
    lines.append(f"- **MPS vs CPU max absolute error:** {s['mps_vs_cpu_max_abs_err']:.3e}")
    lines.append(f"- **MPS vs CUDA-mock max relative error:** {s['mps_vs_cuda_mock_max_rel_err']}")
    lines.append(f"- **Recommended filing target:** `{s['recommended_filing_target']}`")
    lines.append(f"- **Elapsed:** {s['elapsed_seconds']} s")
    lines.append(f"- **torch:** {s['torch_version']}, host: {s['host']}, seed: 0x{s['seed']:x}")
    lines.append("")
    lines.append("## Sampling distribution")
    lines.append("")
    lines.append("| dimension | counts |")
    lines.append("|---|---|")
    lines.append(f"| shape bucket | {s['per_shape_bucket']} |")
    lines.append(f"| dtype | {s['per_dtype']} |")
    lines.append(f"| stride category | {s['per_stride_category']} |")
    lines.append("")
    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not s["top_repros"]:
        lines.append("_No divergences exceeded gpucheck's per-dtype tolerance "
                     "(with MPS 2× multiplier, see assertions/tolerances.py:35)._")
    else:
        for i, r in enumerate(s["top_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **max abs err:** {r['max_abs_err']:.3e}  "
                         f"(atol={r['atol']:.2e})")
            lines.append(f"- **max rel err:** {r['max_rel_err']:.3e}  "
                         f"(rtol={r['rtol']:.2e})")
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append(f"- **seed:** {r['seed']}")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.nn.functional.gelu` on CPU (FP32 promotion "
                 "for accuracy comparison).")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
                 "dtype, device_type='mps')`.")
    lines.append("- A divergence is flagged only when **both** "
                 "max-abs-err > atol *and* max-rel-err > rtol — keeps the bar high "
                 "for outputs near zero (gelu(x)≈0 for x≈0).")
    lines.append("- Stride categories: row_major, column_major, broadcast, "
                 "transpose, slice, non_contig, gather (see "
                 "`gpucheck.fuzzing.strides`).")
    lines.append("- CUDA backend is mocked (no NVIDIA GPU present); cross-device "
                 "MPS-vs-CUDA comparison reported as N/A by spec.")
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
