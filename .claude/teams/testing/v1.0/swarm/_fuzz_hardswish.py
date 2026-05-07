"""V2 fuzz of torch.nn.functional.hardswish: MPS vs CPU reference.

Differences from v1 drivers:
- Each (shape, dtype, stride_category) config is run across 5 seeds (0..4).
- Divergence taxonomy uses the v2 rubric:
    * max_abs_err > 10x tol  -> FILABLE_CANDIDATE (always counts).
    * max_rel_err > 10x tol AND denom_magnitude >= 1e-6
                              -> FILABLE_CANDIDATE.
    * 1x..5x tol on either   -> RECALIBRATION.
    * < 1x tol               -> OK.
  A config is FILABLE only if >=3 of the 5 seeds hit the FILABLE_CANDIDATE bar.
- denom_magnitude is |b| at the element where rel-err peaks, so we don't
  flag near-zero-denominator artifacts as bugs.

CUDA backend is mocked: this Mac has no NVIDIA GPU; CUDA-vs-MPS is N/A by spec.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Import gpucheck from the merged release/v1.0 worktree.
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-hardswish/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_hardswish.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.hardswish"
N_CONFIGS = 1000
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 12 * 60 - 45  # leave 45s headroom for writing results
MASTER_SEED = 0xCAFE

# ---------------------------------------------------------------------------
# Sampling space (mirrors _fuzz_gelu.py; hardswish is also elementwise)
# ---------------------------------------------------------------------------

DEGENERATE = [(0,), (1,), (1, 1), (16, 0), (0, 16), (1, 1, 1)]
NON_TILE = [
    (TILE_SIZES[0] - 1,),
    (TILE_SIZES[0] + 1,),
    (TILE_SIZES[1] - 1, 16),
    (TILE_SIZES[1] + 3, 16),
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),
    (TILE_SIZES[2] + 1, 16),
]
PRIME_S = [(p,) for p in PRIMES] + [(PRIMES[0], PRIMES[1]), (PRIMES[2], 16)]
POW2_BOUNDARY = [(v,) for v in POWER_OF_2_BOUNDARIES] + [(128, 129), (256, 255)]
LARGE = [(LARGE_DIMS[0],), (256, 256), (1024, 64)]
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


def _err_metrics(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_at_peak_rel).

    denom_at_peak_rel is |b| at the element where rel-err peaks. A small value
    (<1e-6) means the rel-err blow-up is a near-zero-denominator artifact.
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a - b).abs()
    abs_err = float(diff.max().item())

    denom = b.abs()
    eps = 1e-12
    rel = diff / denom.clamp_min(eps)
    rel_err = float(rel.max().item())

    rel_flat = rel.reshape(-1)
    denom_flat = denom.reshape(-1)
    flat_idx = int(torch.argmax(rel_flat).item())
    denom_at_peak = float(denom_flat[flat_idx].item())
    return abs_err, rel_err, denom_at_peak


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(
    abs_err: float,
    rel_err: float,
    denom_at_peak: float,
    atol: float,
    rtol: float,
) -> str:
    """v2 per-seed verdict: 'OK' | 'RECAL' | 'DIVERG'."""
    abs_div = abs_err > 10.0 * atol
    rel_div = (rel_err > 10.0 * rtol) and (denom_at_peak >= 1e-6)
    if abs_div or rel_div:
        return "DIVERG"

    abs_recal = abs_err > 1.0 * atol
    rel_recal = (rel_err > 1.0 * rtol) and (denom_at_peak >= 1e-6)
    if abs_recal or rel_recal:
        return "RECAL"
    return "OK"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "top_3_repros": [],
            "torch_version": torch.__version__,
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    rng = random.Random(MASTER_SEED)
    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    # Per-config results across 5 seeds
    filable: list[dict] = []
    recalibration: list[dict] = []

    for config_idx in range(N_CONFIGS):
        if time.monotonic() - started > WALL_BUDGET_S:
            print(
                f"[budget] aborting at config {config_idx}/{N_CONFIGS} "
                f"(elapsed {time.monotonic() - started:.1f}s)",
                file=sys.stderr,
            )
            break

        iters_attempted += 1

        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        dtype = DTYPES_BY_NAME[dtype_name]
        stride_cat = rng.choice(STRIDE_CATEGORIES)

        per_bucket_counts[bucket] += 1
        per_dtype_counts[dtype_name] += 1
        per_stride_counts[stride_cat] += 1

        atol, rtol = compute_tolerance(dtype, device_type="mps")

        per_seed_results: list[dict] = []
        config_unsupported = False

        for seed in SEEDS:
            try:
                x_cpu = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="cpu", seed=seed,
                )
                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:160]
                    print(
                        f"[unsupported-build] cfg={config_idx} seed={seed} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    config_unsupported = True
                    break

                if x_cpu.numel() == 0:
                    per_seed_results.append({
                        "seed": seed,
                        "verdict": "OK",
                        "max_abs_err": 0.0,
                        "max_rel_err": 0.0,
                        "denom_at_peak_rel": 0.0,
                        "empty": True,
                        "cpu_layout": _stride_tag(x_cpu),
                        "mps_layout": _stride_tag(x_mps),
                    })
                    continue

                try:
                    y_mps = F.hardswish(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:160]
                    print(
                        f"[unsupported-mps] cfg={config_idx} seed={seed} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    config_unsupported = True
                    break

                try:
                    y_cpu = F.hardswish(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:160]
                    print(
                        f"[unsupported-cpu] cfg={config_idx} seed={seed} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    config_unsupported = True
                    break

                abs_err, rel_err, denom_peak = _err_metrics(y_mps_cpu, y_cpu)
                verdict = _classify(abs_err, rel_err, denom_peak, atol, rtol)

                max_abs_err_global = max(max_abs_err_global, abs_err)
                max_rel_err_global = max(max_rel_err_global, rel_err)

                per_seed_results.append({
                    "seed": seed,
                    "verdict": verdict,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "denom_at_peak_rel": denom_peak,
                    "cpu_layout": _stride_tag(x_cpu),
                    "mps_layout": _stride_tag(x_mps),
                })

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Halt rule: report and exit non-zero so the caller sees it.
                print(
                    f"[ERROR] cfg={config_idx} seed={seed} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                return 2

        if config_unsupported:
            iters_unsupported += 1
            continue
        if not per_seed_results:
            iters_unsupported += 1
            continue

        iters_completed += 1

        diverg_seeds = [r for r in per_seed_results if r["verdict"] == "DIVERG"]
        recal_seeds = [r for r in per_seed_results if r["verdict"] == "RECAL"]

        if len(diverg_seeds) >= 3:
            worst = max(per_seed_results, key=lambda r: (r["max_rel_err"], r["max_abs_err"]))
            filable.append({
                "shape": list(shape),
                "shape_bucket": bucket,
                "dtype": dtype_name,
                "stride_category": stride_cat,
                "atol": atol,
                "rtol": rtol,
                "n_seeds_total": len(per_seed_results),
                "n_seeds_diverg": len(diverg_seeds),
                "n_seeds_recal": len(recal_seeds),
                "worst_seed": worst["seed"],
                "max_abs_err": worst["max_abs_err"],
                "max_rel_err": worst["max_rel_err"],
                "denom_at_peak_rel": worst["denom_at_peak_rel"],
                "cpu_layout": worst["cpu_layout"],
                "mps_layout": worst["mps_layout"],
                "per_seed": per_seed_results,
            })
            print(
                f"[FILABLE] cfg={config_idx} {bucket}/{dtype_name}/{stride_cat} "
                f"shape={shape} diverg_seeds={len(diverg_seeds)}/5 "
                f"abs={worst['max_abs_err']:.3e} rel={worst['max_rel_err']:.3e}",
                file=sys.stderr,
            )
        elif diverg_seeds or recal_seeds:
            worst = max(per_seed_results, key=lambda r: (r["max_rel_err"], r["max_abs_err"]))
            recalibration.append({
                "shape": list(shape),
                "shape_bucket": bucket,
                "dtype": dtype_name,
                "stride_category": stride_cat,
                "atol": atol,
                "rtol": rtol,
                "n_seeds_total": len(per_seed_results),
                "n_seeds_diverg": len(diverg_seeds),
                "n_seeds_recal": len(recal_seeds),
                "worst_seed": worst["seed"],
                "max_abs_err": worst["max_abs_err"],
                "max_rel_err": worst["max_rel_err"],
                "denom_at_peak_rel": worst["denom_at_peak_rel"],
                "cpu_layout": worst["cpu_layout"],
                "mps_layout": worst["mps_layout"],
            })

    elapsed = time.monotonic() - started

    # Top-3 repros: prefer FILABLE; fall back to RECALIBRATION; rank by
    # (largest rel-err, then smallest numel for "minimal repro").
    pool = filable + recalibration
    pool_sorted = sorted(
        pool,
        key=lambda d: (
            0 if d in filable else 1,    # filable first
            -d["max_rel_err"],
            _numel(d["shape"]),
        ),
    )
    top3 = pool_sorted[:3]

    summary = {
        "kernel": KERNEL,
        "status": "OK",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": top3,
        "elapsed_seconds": round(elapsed, 2),
        "wall_budget_seconds": WALL_BUDGET_S,
        "n_configs_target": N_CONFIGS,
        "seeds": list(SEEDS),
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "master_seed": MASTER_SEED,
        "cuda_vs_mps": "N/A (no NVIDIA GPU on host; CUDA backend is mocked)",
        "recommended_filing_target": _recommend_target(filable),
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


def _recommend_target(filable: list[dict]) -> str:
    if not filable:
        return "none"
    # hardswish is a pure pytorch op; if MPS diverges from CPU on >=3 seeds
    # the bug lives in aten/MPS, not Triton.
    return "pytorch/pytorch"


def _write_markdown(path: Path, s: dict) -> None:
    lines: list[str] = []
    lines.append(f"# {s['kernel']} - MPS v2 fuzz report")
    lines.append("")
    lines.append(f"- **kernel:** `{s['kernel']}`")
    lines.append(f"- **status:** {s['status']}")
    lines.append(f"- **iters_attempted:** {s['iters_attempted']}")
    lines.append(f"- **iters_completed:** {s['iters_completed']}")
    lines.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    lines.append(f"- **divergences_filable:** {s['divergences_filable']}")
    lines.append(f"- **divergences_recalibration:** {s['divergences_recalibration']}")
    lines.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    lines.append(f"- **max_rel_err:** {s['max_rel_err']:.3e}")
    lines.append(f"- **elapsed_seconds:** {s['elapsed_seconds']} (budget {s['wall_budget_seconds']}s)")
    lines.append(f"- **seeds:** {s['seeds']}")
    lines.append(f"- **n_configs_target:** {s['n_configs_target']}")
    lines.append(f"- **torch:** {s['torch_version']} on {s['host']}")
    lines.append(f"- **CUDA vs MPS:** {s['cuda_vs_mps']}")
    lines.append(f"- **recommended_filing_target:** `{s['recommended_filing_target']}`")
    lines.append("")
    lines.append("## v2 divergence rubric")
    lines.append("")
    lines.append("- `max_abs_err > 10x atol` -> FILABLE_CANDIDATE (always counts).")
    lines.append("- `max_rel_err > 10x rtol AND denom_at_peak_rel >= 1e-6` -> FILABLE_CANDIDATE.")
    lines.append("- `1x..5x tol` -> RECALIBRATION (recommend xfail entry).")
    lines.append("- `< 1x tol` -> OK.")
    lines.append("- A config is FILABLE only when >=3 of the 5 seeds hit the FILABLE_CANDIDATE bar.")
    lines.append("")
    lines.append("## Sampling distribution")
    lines.append("")
    lines.append("| dimension | counts |")
    lines.append("|---|---|")
    lines.append(f"| shape bucket | {s['per_shape_bucket']} |")
    lines.append(f"| dtype | {s['per_dtype']} |")
    lines.append(f"| stride category | {s['per_stride_category']} |")
    lines.append("")
    lines.append("## Top 3 repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append("_No divergences exceeded the v2 thresholds (atol/rtol from "
                     "`gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`)._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            tier = "FILABLE" if r.get("n_seeds_diverg", 0) >= 3 else "RECALIBRATION"
            lines.append(f"### Repro #{i} - {tier}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}` (bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **seeds_diverg / total:** {r['n_seeds_diverg']} / {r['n_seeds_total']}")
            lines.append(f"- **seeds_recal / total:** {r['n_seeds_recal']} / {r['n_seeds_total']}")
            lines.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} (atol={r['atol']:.2e})")
            lines.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} (rtol={r['rtol']:.2e})")
            lines.append(f"- **denom_at_peak_rel:** {r['denom_at_peak_rel']:.3e} (filtered if <1e-6)")
            lines.append(f"- **worst seed:** {r['worst_seed']}")
            lines.append(f"- **CPU layout:** `{r['cpu_layout']}`")
            lines.append(f"- **MPS layout:** `{r['mps_layout']}`")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.nn.functional.hardswish` on CPU (FP32 promotion "
                 "for accuracy comparison).")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
                 "dtype, device_type='mps')`.")
    lines.append("- `denom_at_peak_rel` is `|y_cpu_fp32|` at the element where the "
                 "MPS-vs-CPU rel-err peaks. Values below `1e-6` are treated as "
                 "near-zero-denominator artifacts and excluded from divergence "
                 "(per v2 spec).")
    lines.append("- Stride categories: row_major, column_major, broadcast, transpose, "
                 "slice, non_contig, gather (see `gpucheck.fuzzing.strides`).")
    lines.append("- CUDA backend is mocked (no NVIDIA GPU on host); cross-device "
                 "MPS-vs-CUDA comparison reported as N/A by spec.")
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
