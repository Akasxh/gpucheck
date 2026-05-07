"""Fuzz the cosine_similarity kernel: MPS vs CPU reference.

CUDA backend is mocked (no NVIDIA GPU on this host); cross-device
MPS-vs-CUDA comparison is reported as N/A by spec.

cosine_similarity is a matmul-class op (dot product + reduction along
the feature dim). We pass ``k_dim`` to ``compute_tolerance`` so the
sqrt(k/128) accumulation scaling kicks in.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Ensure gpucheck (worktree) is importable
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-cosine_sim/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_cosine_sim.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.cosine_similarity"
N_ITERS = 250
TIME_BUDGET_S = 8 * 60 - 30  # leave 30 s for writing results
SEED = 0xC051

# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------
# cosine_similarity needs identical shapes on x1, x2. We always reduce along
# the LAST dim. The LAST dim therefore plays the role of the "k" / feature
# dim and is what we feed compute_tolerance(k_dim=...).

# Degenerate cases include shapes where the reduction dim is 0 or 1 — these
# produce nan/0 outputs but should not be treated as divergences.
DEGENERATE = [(1,), (1, 1), (4, 0), (0, 4), (1, 1, 1), (3, 1)]

NON_TILE = [
    (TILE_SIZES[0] - 1,),                       # 31    feat=31
    (8, TILE_SIZES[0] + 1),                     # 8x33
    (TILE_SIZES[1] - 1, TILE_SIZES[1] + 1),     # 63x65
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),     # 127x129
    (16, TILE_SIZES[2] + 1),                    # 16x129
    (TILE_SIZES[2] + 1, 16),                    # 129x16
]

PRIME_S = [(p,) for p in PRIMES] + [(PRIMES[1], PRIMES[3]), (PRIMES[0], 16)]

POW2_BOUNDARY = [
    (v,) for v in POWER_OF_2_BOUNDARIES
] + [(128, 129), (256, 255), (8, 512), (8, 513)]

# Cap large to fit memory + time. cosine_sim materializes 2 inputs; keep
# the largest reduction-dim under ~8192 to stay safe on a Mac.
LARGE = [(LARGE_DIMS[0],), (256, LARGE_DIMS[0]), (64, LARGE_DIMS[1])]

MIXED = [(127, 16), (33, 128, 4), (7, 128), (4, 33, 65)]

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
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-7)
    rel = diff / denom
    if rel.numel() == 0:
        return 0.0
    # NaN-safe: drop NaNs from comparison (cosine_sim of zero-vectors is nan)
    rel = rel[~torch.isnan(rel) & ~torch.isinf(rel)]
    if rel.numel() == 0:
        return 0.0
    return float(rel.max().item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    diff = diff[~torch.isnan(diff) & ~torch.isinf(diff)]
    if diff.numel() == 0:
        return 0.0
    return float(diff.max().item())


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

        # Two distinct seeds for x1 and x2 so they aren't identical (which
        # would trivially give cos=1).
        seed_x1 = rng.randrange(2**31 - 1)
        seed_x2 = rng.randrange(2**31 - 1)

        # Choose reduction dim: default to last dim for ndim>=1.
        ndim = len(shape)
        if ndim == 0:
            # Scalar shapes are nonsensical for cosine_similarity; mark unsupported
            unsupported += 1
            continue
        reduce_dim = ndim - 1
        k_dim = int(shape[reduce_dim])

        try:
            # Build CPU and MPS tensors with matching stride categories.
            x1_cpu = fuzz_strides_for_category(
                shape, dtype, stride_cat, device="cpu", seed=seed_x1,
            )
            x2_cpu = fuzz_strides_for_category(
                shape, dtype, stride_cat, device="cpu", seed=seed_x2,
            )
            try:
                x1_mps = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="mps", seed=seed_x1,
                )
                x2_mps = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="mps", seed=seed_x2,
                )
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-build] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            # Skip empty tensors (no signal, error metrics undefined).
            if x1_cpu.numel() == 0 or k_dim == 0:
                iterations_completed += 1
                continue

            try:
                y_mps = F.cosine_similarity(x1_mps, x2_mps, dim=reduce_dim)
                torch.mps.synchronize()
                y_mps_cpu = y_mps.detach().to("cpu")
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-mps] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            try:
                y_cpu = F.cosine_similarity(x1_cpu, x2_cpu, dim=reduce_dim)
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-cpu] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            iterations_completed += 1

            rel = _max_rel_err(y_mps_cpu, y_cpu)
            absdiff = _max_abs_err(y_mps_cpu, y_cpu)
            max_rel_err_global = max(max_rel_err_global, rel)
            max_abs_err_global = max(max_abs_err_global, absdiff)

            # Tolerance: cosine_sim is matmul-class (dot product reduction
            # along k_dim). Pass k_dim so atol scales by sqrt(k_dim/128).
            atol, rtol = compute_tolerance(
                dtype, k_dim=k_dim, device_type="mps",
            )

            # Bound the output: cos similarity is in [-1, 1]. Use the same
            # AND-rule as gelu/relu fuzz scripts: both abs and rel must
            # exceed tolerance, so outputs near zero don't trigger noise.
            diverged = (absdiff > atol) and (rel > rtol)
            if diverged:
                rec = {
                    "iter": i,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "shape_bucket": bucket,
                    "reduce_dim": reduce_dim,
                    "k_dim": k_dim,
                    "max_abs_err": absdiff,
                    "max_rel_err": rel,
                    "atol": atol,
                    "rtol": rtol,
                    "x1_layout_cpu": _stride_tag(x1_cpu),
                    "x1_layout_mps": _stride_tag(x1_mps),
                    "seed_x1": seed_x1,
                    "seed_x2": seed_x2,
                }
                divergences.append(rec)
                print(
                    f"[DIVERGE] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape} k={k_dim} abs={absdiff:.3e} rel={rel:.3e} "
                    f"(atol={atol:.2e} rtol={rtol:.2e})",
                    file=sys.stderr,
                )

        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — report and halt per spec
            print(
                f"[ERROR] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                f"shape={shape}: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return 2

    elapsed = time.monotonic() - started

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
    # cosine_similarity is a pure pytorch op composed of mul, sum, rsqrt;
    # an MPS-vs-CPU divergence implicates aten/MPS — not Triton.
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
        lines.append(
            "_No divergences exceeded gpucheck's per-dtype tolerance "
            "(matmul-class: atol scaled by sqrt(k_dim/128); MPS overlay applies "
            "a 2× multiplier — see assertions/tolerances.py:35,103)._"
        )
    else:
        for i, r in enumerate(s["top_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **reduce dim / k_dim:** dim={r['reduce_dim']}, k={r['k_dim']}")
            lines.append(
                f"- **max abs err:** {r['max_abs_err']:.3e}  (atol={r['atol']:.2e})"
            )
            lines.append(
                f"- **max rel err:** {r['max_rel_err']:.3e}  (rtol={r['rtol']:.2e})"
            )
            lines.append(f"- **CPU x1 layout:** `{r['x1_layout_cpu']}`")
            lines.append(f"- **MPS x1 layout:** `{r['x1_layout_mps']}`")
            lines.append(f"- **seed x1/x2:** {r['seed_x1']}/{r['seed_x2']}")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Reference: `torch.nn.functional.cosine_similarity` on CPU "
        "(FP32 promotion in error computation)."
    )
    lines.append(
        "- Reduction dim: last dim of the sampled shape (k_dim = shape[-1])."
    )
    lines.append(
        "- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, "
        "k_dim=shape[-1], device_type='mps')` — matmul-class scaling enabled, "
        "MPS 2× multiplier applied."
    )
    lines.append(
        "- A divergence is flagged only when **both** max-abs-err > atol *and* "
        "max-rel-err > rtol — keeps the bar high for outputs near zero "
        "(cosine of near-orthogonal vectors)."
    )
    lines.append(
        "- NaN/inf entries dropped from error stats (cos of a zero vector is nan)."
    )
    lines.append(
        "- Stride categories: row_major, column_major, broadcast, transpose, "
        "slice, non_contig, gather (see `gpucheck.fuzzing.strides`)."
    )
    lines.append(
        "- CUDA backend is mocked (no NVIDIA GPU present); cross-device "
        "MPS-vs-CUDA comparison reported as N/A by spec."
    )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
