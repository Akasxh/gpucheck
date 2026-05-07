"""Fuzz the softplus kernel: MPS vs CPU reference (v2 spec).

v2 changes vs v1:
  * 1000 iterations spread across seeds {0,1,2,3,4}
  * dtypes fp32/fp16/bf16 (bf16 best-effort; falls back to UNSUPPORTED on op-level error)
  * Divergence buckets:
      - max_abs_err > 10× tol  -> divergence candidate (always counts)
      - max_rel_err > 10× tol  -> divergence candidate ONLY if denom_magnitude >= 1e-6
      - 1× < ratio <= 5× tol   -> TOLERANCE_RECALIBRATION
      - ratio <= 1×            -> OK
  * FILABLE = reproducible across >=3 of the 5 seeds at same (shape,dtype,stride)
  * Hard wall: 12 min; if exceeded, write what we have and exit cleanly.
  * UNSUPPORTED if torch.mps lacks the op (do not invent divergences).
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Use the merged release/v1.0 (current worktree at fuzz-softplus)
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-softplus/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_softplus.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.softplus"
N_ITERS_TOTAL = 1000
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 12 * 60 - 30  # leave 30 s for writing results

# Sampling space: softplus is elementwise so we keep dims modest.
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

# softplus parameter knobs — defaults are beta=1, threshold=20.
# Vary these too: at threshold the implementation switches to identity, which
# is a different code path on both CPU and MPS.
BETA_CHOICES: tuple[float, ...] = (1.0, 0.5, 2.0)
THRESHOLD_CHOICES: tuple[float, ...] = (20.0, 5.0, 50.0)


def _cmp_errors(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_magnitude) on float32-promoted tensors.

    `denom_magnitude` is the reference-value magnitude at the location of the
    worst relative error — i.e., |b[argmax_rel]|. Used to filter near-zero
    denominator artifacts per v2 spec.
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a - b).abs()
    abs_err = float(diff.max().item())
    denom = b.abs().clamp_min(1e-7)
    rel = diff / denom
    rel_max_idx = int(rel.argmax().item())
    rel_err = float(rel.flatten()[rel_max_idx].item())
    denom_mag = float(b.abs().flatten()[rel_max_idx].item())
    return abs_err, rel_err, denom_mag


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(
    abs_err: float,
    rel_err: float,
    denom_mag: float,
    atol: float,
    rtol: float,
) -> tuple[str, float]:
    """Return (bucket, ratio) where bucket is one of:
        OK | RECALIBRATION | DIVERGENCE
    `ratio` is the worst-case multiple-of-tolerance.
    """
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = (rel_err / rtol) if (rtol > 0 and denom_mag >= 1e-6) else 0.0
    ratio = max(abs_ratio, rel_ratio)

    if abs_err > 10 * atol:
        return "DIVERGENCE", ratio
    if rel_err > 10 * rtol and denom_mag >= 1e-6:
        return "DIVERGENCE", ratio
    if ratio > 5.0:
        return "RECALIBRATION", ratio
    if ratio > 1.0:
        return "RECALIBRATION", ratio
    return "OK", ratio


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": "softplus",
            "op_path": KERNEL,
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
            "schema": "swarm-v2",
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_errored = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    max_rel_err_with_denom_global = 0.0  # only when denom_magnitude >= 1e-6

    # Group results by (shape, dtype, stride_category) for cross-seed repro tracking
    # Each entry tracks per-seed classification and worst error metrics.
    config_records: dict[
        tuple[tuple[int, ...], str, str, float, float],
        dict,
    ] = {}

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    # Plan iter distribution: 1000 iters / 5 seeds = 200 per seed.
    per_seed_iters = N_ITERS_TOTAL // len(SEEDS)
    aborted_early = False

    for seed in SEEDS:
        rng = random.Random(seed)
        for k in range(per_seed_iters):
            if time.monotonic() - started > WALL_BUDGET_S:
                aborted_early = True
                print(
                    f"[budget] aborting at seed={seed} iter {k}/{per_seed_iters} "
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
            beta = rng.choice(BETA_CHOICES)
            threshold = rng.choice(THRESHOLD_CHOICES)

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            # Use a per-iter sub-seed derived from the seed for the rng inside the
            # stride builder. Each (seed, k) pair => a unique sub-seed.
            sub_seed = rng.randrange(2**31 - 1)

            try:
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=sub_seed,
                    )
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=sub_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build] seed={seed} k={k} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.softplus(x_mps, beta=beta, threshold=threshold)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] seed={seed} k={k} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.softplus(x_cpu, beta=beta, threshold=threshold)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] seed={seed} k={k} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                abs_err, rel_err, denom_mag = _cmp_errors(y_mps_cpu, y_cpu)
                max_abs_err_global = max(max_abs_err_global, abs_err)
                max_rel_err_global = max(max_rel_err_global, rel_err)
                if denom_mag >= 1e-6:
                    max_rel_err_with_denom_global = max(
                        max_rel_err_with_denom_global, rel_err,
                    )

                # softplus is elementwise: no k_dim scaling.
                atol, rtol = compute_tolerance(dtype, device_type="mps")
                bucket_class, ratio = _classify(
                    abs_err, rel_err, denom_mag, atol, rtol,
                )

                config_key = (
                    tuple(shape), dtype_name, stride_cat, float(beta), float(threshold),
                )
                rec = config_records.setdefault(
                    config_key,
                    {
                        "shape": list(shape),
                        "shape_bucket": bucket,
                        "dtype": dtype_name,
                        "stride_category": stride_cat,
                        "beta": float(beta),
                        "threshold": float(threshold),
                        "atol": atol,
                        "rtol": rtol,
                        "seeds_seen": set(),
                        "seeds_diverged": set(),
                        "seeds_recalibration": set(),
                        "max_abs_err": 0.0,
                        "max_rel_err": 0.0,
                        "max_denom_mag_at_rel_max": 0.0,
                        "max_ratio": 0.0,
                        "x_layout_cpu": _stride_tag(x_cpu),
                        "x_layout_mps": _stride_tag(x_mps),
                        "first_sub_seed": sub_seed,
                    },
                )
                rec["seeds_seen"].add(seed)
                if bucket_class == "DIVERGENCE":
                    rec["seeds_diverged"].add(seed)
                elif bucket_class == "RECALIBRATION":
                    rec["seeds_recalibration"].add(seed)
                if abs_err > rec["max_abs_err"]:
                    rec["max_abs_err"] = abs_err
                    rec["x_layout_cpu"] = _stride_tag(x_cpu)
                    rec["x_layout_mps"] = _stride_tag(x_mps)
                if rel_err > rec["max_rel_err"]:
                    rec["max_rel_err"] = rel_err
                    rec["max_denom_mag_at_rel_max"] = denom_mag
                if ratio > rec["max_ratio"]:
                    rec["max_ratio"] = ratio

                if bucket_class == "DIVERGENCE":
                    print(
                        f"[DIVERGE] seed={seed} k={k} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                        f"beta={beta} thr={threshold} "
                        f"abs={abs_err:.3e} rel={rel_err:.3e} "
                        f"denom={denom_mag:.3e} (atol={atol:.2e} rtol={rtol:.2e})",
                        file=sys.stderr,
                    )

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                iters_errored += 1
                print(
                    f"[ERROR] seed={seed} k={k} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                # Hard rule: halt and report.
                _emit_results(
                    started=started,
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported=iters_unsupported,
                    iters_errored=iters_errored,
                    config_records=config_records,
                    max_abs_err_global=max_abs_err_global,
                    max_rel_err_global=max_rel_err_global,
                    max_rel_err_with_denom_global=max_rel_err_with_denom_global,
                    per_bucket_counts=per_bucket_counts,
                    per_dtype_counts=per_dtype_counts,
                    per_stride_counts=per_stride_counts,
                    aborted_early=True,
                    halt_reason=f"runtime error in iter loop: {exc!r}",
                )
                return 2

        if aborted_early:
            break

    _emit_results(
        started=started,
        iters_attempted=iters_attempted,
        iters_completed=iters_completed,
        iters_unsupported=iters_unsupported,
        iters_errored=iters_errored,
        config_records=config_records,
        max_abs_err_global=max_abs_err_global,
        max_rel_err_global=max_rel_err_global,
        max_rel_err_with_denom_global=max_rel_err_with_denom_global,
        per_bucket_counts=per_bucket_counts,
        per_dtype_counts=per_dtype_counts,
        per_stride_counts=per_stride_counts,
        aborted_early=aborted_early,
        halt_reason=None,
    )
    return 0


def _emit_results(  # noqa: PLR0913 — reporter aggregator
    *,
    started: float,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_errored: int,
    config_records: dict,
    max_abs_err_global: float,
    max_rel_err_global: float,
    max_rel_err_with_denom_global: float,
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    aborted_early: bool,
    halt_reason: str | None,
) -> None:
    elapsed = time.monotonic() - started

    # FILABLE = >=3 of the 5 seeds DIVERGED at the same config.
    filable = []
    recalibration = []
    other_diverged_not_filable = []
    for cfg_key, rec in config_records.items():
        n_div_seeds = len(rec["seeds_diverged"])
        n_rcal_seeds = len(rec["seeds_recalibration"])
        if n_div_seeds >= 3:
            filable.append(rec)
        elif n_div_seeds > 0:
            other_diverged_not_filable.append(rec)
        elif n_rcal_seeds >= 3:
            recalibration.append(rec)

    # Sort filable by max_ratio desc (most severe first).
    filable_sorted = sorted(filable, key=lambda r: -r["max_ratio"])
    recalibration_sorted = sorted(recalibration, key=lambda r: -r["max_ratio"])

    # Top-3 repros: prefer filable; if none, fall back to recalibration; else
    # the worst-ratio configs encountered (reported but classified OK).
    if filable_sorted:
        top3_pool = filable_sorted
    elif recalibration_sorted:
        top3_pool = recalibration_sorted
    else:
        top3_pool = sorted(
            config_records.values(), key=lambda r: -r["max_ratio"],
        )[:3]
    top3 = top3_pool[:3]

    # Recommend filing target only if FILABLE.
    if filable_sorted:
        filing_target = "pytorch/pytorch"
    else:
        filing_target = "none"

    summary = {
        "kernel": "softplus",
        "op_path": KERNEL,
        "schema": "swarm-v2",
        "status": "OK" if not halt_reason else "HALTED",
        "halt_reason": halt_reason,
        "aborted_early": aborted_early,
        "backend_mps": True,
        "backend_cuda": "mocked (no NVIDIA GPU on host)",
        "torch_version": torch.__version__,
        "seeds": list(SEEDS),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_errored": iters_errored,
        "divergences_filable": len(filable_sorted),
        "divergences_recalibration": len(recalibration_sorted),
        "divergences_other_not_filable": len(other_diverged_not_filable),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "max_rel_err_denom_filtered": max_rel_err_with_denom_global,
        "top_3_repros": [_serialize_rec(r) for r in top3],
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "recommended_filing_target": filing_target,
        "elapsed_s": round(elapsed, 2),
        "host": "darwin/arm64 (Apple Silicon)",
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _serialize_rec(rec: dict) -> dict:
    out = dict(rec)
    out["seeds_seen"] = sorted(rec["seeds_seen"])
    out["seeds_diverged"] = sorted(rec["seeds_diverged"])
    out["seeds_recalibration"] = sorted(rec["seeds_recalibration"])
    return out


def _write_markdown(path: Path, s: dict) -> None:
    lines = []
    lines.append(f"# {s['op_path']} — MPS fuzz report (v2)")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}` (`{s['op_path']}`)")
    lines.append(f"- **Status:** {s['status']}")
    if s.get("halt_reason"):
        lines.append(f"- **Halt reason:** {s['halt_reason']}")
    lines.append(f"- **Aborted early:** {s['aborted_early']}")
    lines.append(f"- **Iterations attempted:** {s['iters_attempted']}")
    lines.append(f"- **Iterations completed:** {s['iters_completed']}")
    lines.append(f"- **Iterations unsupported (skipped):** {s['iters_unsupported']}")
    lines.append(f"- **Iterations errored:** {s['iters_errored']}")
    lines.append(f"- **Divergences (FILABLE, >=3 seeds):** {s['divergences_filable']}")
    lines.append(f"- **Divergences (RECALIBRATION):** {s['divergences_recalibration']}")
    lines.append(
        f"- **Divergences (other, <3 seeds, not filable):** "
        f"{s['divergences_other_not_filable']}"
    )
    lines.append(f"- **Max abs err (global):** {s['max_abs_err']:.3e}")
    lines.append(f"- **Max rel err (global, raw):** {s['max_rel_err']:.3e}")
    lines.append(
        f"- **Max rel err (denom>=1e-6 filter):** "
        f"{s['max_rel_err_denom_filtered']:.3e}"
    )
    lines.append(f"- **Recommended filing target:** `{s['recommended_filing_target']}`")
    lines.append(f"- **Elapsed:** {s['elapsed_s']} s")
    lines.append(
        f"- **torch:** {s['torch_version']}, host: {s['host']}, "
        f"seeds: {s['seeds']}"
    )
    lines.append("")
    lines.append("## v2 divergence rules")
    lines.append("")
    lines.append("- `max_abs_err > 10×atol` always counts as a divergence candidate.")
    lines.append(
        "- `max_rel_err > 10×rtol` counts ONLY if `denom_magnitude >= 1e-6` "
        "(filters near-zero artifacts where `softplus(x)≈0` for very negative `x`)."
    )
    lines.append(
        "- A config is FILABLE only if the divergence reproduces across "
        "≥3 of the 5 seeds {0,1,2,3,4}."
    )
    lines.append("- 1×–5× tolerance ⇒ TOLERANCE_RECALIBRATION (xfail or tighten).")
    lines.append("- ≤1× tolerance ⇒ OK.")
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
        lines.append(
            "_No (shape,dtype,stride) configs observed — see iters_completed._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}`  "
                         f"(bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **softplus params:** beta={r['beta']}, "
                         f"threshold={r['threshold']}")
            lines.append(f"- **seeds seen:** {r['seeds_seen']}")
            lines.append(f"- **seeds with DIVERGENCE:** {r['seeds_diverged']}")
            lines.append(
                f"- **seeds with RECALIBRATION:** {r['seeds_recalibration']}"
            )
            lines.append(f"- **max abs err:** {r['max_abs_err']:.3e}  "
                         f"(atol={r['atol']:.2e})")
            lines.append(
                f"- **max rel err:** {r['max_rel_err']:.3e}  "
                f"(rtol={r['rtol']:.2e}, "
                f"denom_mag={r['max_denom_mag_at_rel_max']:.3e})"
            )
            lines.append(f"- **max ratio (×tol):** {r['max_ratio']:.2f}")
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.nn.functional.softplus` on CPU "
                 "(comparison against MPS, both promoted to FP32 for error metrics).")
    lines.append("- Tolerance: "
                 "`gpucheck.assertions.tolerances.compute_tolerance(dtype, "
                 "device_type='mps')`. softplus is elementwise — no `k_dim` scaling.")
    lines.append(
        "- softplus(x) = log(1 + exp(beta*x)) / beta with `x>threshold` returning "
        "x as-is. Both branches are exercised via `BETA_CHOICES` and "
        "`THRESHOLD_CHOICES`."
    )
    lines.append(
        "- denom_magnitude is the |reference| value at the worst-rel-err location, "
        "used to discard near-zero-denominator artifacts (softplus(x)→0 as "
        "x→-∞)."
    )
    lines.append(
        "- CUDA backend is mocked (no NVIDIA GPU present); MPS-vs-CUDA is N/A."
    )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
