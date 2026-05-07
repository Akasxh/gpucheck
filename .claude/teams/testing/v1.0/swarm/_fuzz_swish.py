"""Fuzz the swish (== silu) kernel: MPS vs CPU reference. CUDA is mocked.

Spec (v2 swarm):
  - 1000 iterations across seeds {0, 1, 2, 3, 4}  -> 200 iters/seed.
  - dtypes fp32/fp16/bf16 (swish supports all three on MPS — verified).
  - Divergence filtering:
      * max_rel_err > 10x tolerance counts only if max(|y_cpu|) >= 1e-6
        (i.e. not a near-zero-denominator artifact).
      * max_abs_err > 10x tolerance always counts.
      * Both must reproduce across >= 3 seeds at the same config to be FILABLE.
      * 1x..5x tolerance band -> TOLERANCE_RECALIBRATION (xfail candidate).
      * Below 1x tolerance -> OK.
  - 12-minute wall budget: write what we have and exit cleanly.
  - UNSUPPORTED if torch.mps lacks the op for a dtype/layout.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Use the swish worktree as the source of truth for gpucheck.
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-swish/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_swish.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.silu  # swish == x * sigmoid(x)"
SEEDS = (0, 1, 2, 3, 4)
ITERS_PER_SEED = 200
N_ITERS_TOTAL = ITERS_PER_SEED * len(SEEDS)
TIME_BUDGET_S = 12 * 60 - 30  # leave 30s for writing results
CONFIG_RNG_SEED = 0xC0FFEE_5417  # config-space rng seed (deterministic across runs)

# Reproducibility thresholds
FILABLE_MULT = 10.0
RECALIB_LO_MULT = 1.0
RECALIB_HI_MULT = 5.0
DENOM_FLOOR = 1.0e-6  # near-zero denom guard for relative error
MIN_REPRO_SEEDS = 3


# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------

DEGENERATE = [(0,), (1,), (1, 1), (16, 0), (0, 16), (1, 1, 1), (8, 1)]
NON_TILE = [
    (TILE_SIZES[0] - 1,),                        # 31
    (TILE_SIZES[0] + 1,),                        # 33
    (TILE_SIZES[1] - 1, 16),                     # 63 x 16
    (TILE_SIZES[1] + 3, 16),                     # 67 x 16
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),      # 127 x 129
    (TILE_SIZES[2] + 1, 16),                     # 129 x 16
    (TILE_SIZES[1] + 1, TILE_SIZES[1] + 1),      # 65 x 65
]
PRIME_S = [(p,) for p in PRIMES] + [
    (PRIMES[0], PRIMES[1]),                      # 7 x 13
    (PRIMES[2], 16),                             # 31 x 16
    (PRIMES[3], 8),                              # 127 x 8
]
POW2_BOUNDARY = [(v,) for v in POWER_OF_2_BOUNDARIES] + [
    (128, 129),
    (256, 255),
    (512, 1),
]
LARGE = [(LARGE_DIMS[0],), (256, 256), (1024, 64), (4096, 16)]
MIXED = [(127, 16), (1024, 3), (7, 128), (33, 128, 4), (3, 5, 7), (8, 9, 10)]

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_rel_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-7)
    return float((diff / denom).max().item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _denom_magnitude(b_cpu: torch.Tensor) -> float:
    """max(|b|) — the natural scale of the reference output."""
    if b_cpu.numel() == 0:
        return 0.0
    return float(b_cpu.to(torch.float32).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _build_configs(n: int) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Deterministic list of n (bucket, shape, dtype_name, stride_cat) configs.

    Same configs across all seeds, so per-config divergence reproducibility
    is well-defined.
    """
    rng = random.Random(CONFIG_RNG_SEED)
    configs: list[tuple[str, tuple[int, ...], str, str]] = []
    for _ in range(n):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(list(STRIDE_CATEGORIES))
        configs.append((bucket, tuple(shape), dtype_name, stride_cat))
    return configs


def _classify(
    abs_err: float, rel_err: float, denom_mag: float, atol: float, rtol: float,
) -> str:
    """Return one of: 'filable_candidate', 'recalibration', 'ok', 'noise'."""
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = rel_err / rtol if rtol > 0 else 0.0

    abs_filable = abs_ratio > FILABLE_MULT
    rel_filable = (rel_ratio > FILABLE_MULT) and (denom_mag >= DENOM_FLOOR)
    if abs_filable or rel_filable:
        return "filable_candidate"

    in_band = (
        (RECALIB_LO_MULT < abs_ratio <= RECALIB_HI_MULT)
        or (RECALIB_LO_MULT < rel_ratio <= RECALIB_HI_MULT and denom_mag >= DENOM_FLOOR)
    )
    if in_band:
        return "recalibration"

    return "ok"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": "swish",
            "kernel_op": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    configs = _build_configs(ITERS_PER_SEED)

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0

    # config_idx -> {seed -> per-run record}
    per_config: dict[int, dict[int, dict]] = {}

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    aborted = False

    for seed in SEEDS:
        if aborted:
            break
        for cfg_idx, (bucket, shape, dtype_name, stride_cat) in enumerate(configs):
            elapsed = time.monotonic() - started
            if elapsed > TIME_BUDGET_S:
                print(
                    f"[budget] aborting at seed={seed} cfg_idx={cfg_idx} "
                    f"(elapsed {elapsed:.1f}s of {TIME_BUDGET_S}s)",
                    file=sys.stderr,
                )
                aborted = True
                break

            iters_attempted += 1
            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            dtype = DTYPES_BY_NAME[dtype_name]

            try:
                # Identical seed for both CPU and MPS so values match.
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build-cpu] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build-mps] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    # Empty tensor: silu trivially reproduces; nothing to compare.
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.silu(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.silu(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                abs_err = _max_abs_err(y_mps_cpu, y_cpu)
                rel_err = _max_rel_err(y_mps_cpu, y_cpu)
                denom_mag = _denom_magnitude(y_cpu)
                max_abs_err_global = max(max_abs_err_global, abs_err)
                max_rel_err_global = max(max_rel_err_global, rel_err)

                # silu is elementwise — no sqrt(k/128) matmul scaling.
                atol, rtol = compute_tolerance(dtype, device_type="mps")
                klass = _classify(abs_err, rel_err, denom_mag, atol, rtol)

                rec = {
                    "seed": seed,
                    "cfg_idx": cfg_idx,
                    "shape": list(shape),
                    "shape_bucket": bucket,
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "denom_magnitude": denom_mag,
                    "atol": atol,
                    "rtol": rtol,
                    "abs_ratio": abs_err / atol if atol > 0 else 0.0,
                    "rel_ratio": rel_err / rtol if rtol > 0 else 0.0,
                    "class": klass,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                }
                per_config.setdefault(cfg_idx, {})[seed] = rec

                if klass == "filable_candidate":
                    print(
                        f"[FILABLE?] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                        f"abs={abs_err:.3e} (atol={atol:.2e}) "
                        f"rel={rel_err:.3e} (rtol={rtol:.2e}) "
                        f"denom={denom_mag:.3e}",
                        file=sys.stderr,
                    )

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Halt rule: report and exit non-zero on unexpected error.
                print(
                    f"[ERROR] seed={seed} cfg={cfg_idx} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                # Still try to write whatever we have so the swarm has a row.
                _flush(
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported=iters_unsupported,
                    per_config=per_config,
                    per_bucket_counts=per_bucket_counts,
                    per_dtype_counts=per_dtype_counts,
                    per_stride_counts=per_stride_counts,
                    max_abs_err_global=max_abs_err_global,
                    max_rel_err_global=max_rel_err_global,
                    elapsed=time.monotonic() - started,
                    aborted=True,
                    error=str(exc),
                )
                return 2

    elapsed = time.monotonic() - started
    _flush(
        iters_attempted=iters_attempted,
        iters_completed=iters_completed,
        iters_unsupported=iters_unsupported,
        per_config=per_config,
        per_bucket_counts=per_bucket_counts,
        per_dtype_counts=per_dtype_counts,
        per_stride_counts=per_stride_counts,
        max_abs_err_global=max_abs_err_global,
        max_rel_err_global=max_rel_err_global,
        elapsed=elapsed,
        aborted=aborted,
    )
    return 0


# ---------------------------------------------------------------------------
# Reproducibility classification + output
# ---------------------------------------------------------------------------

def _classify_per_config(
    per_config: dict[int, dict[int, dict]],
) -> tuple[list[dict], list[dict], int, int]:
    """Group per-config across seeds and decide filable vs recalibration."""
    filable: list[dict] = []
    recalibration: list[dict] = []

    for cfg_idx, per_seed in per_config.items():
        seed_runs = list(per_seed.values())
        if not seed_runs:
            continue

        n_filable = sum(1 for r in seed_runs if r["class"] == "filable_candidate")
        n_recalib = sum(1 for r in seed_runs if r["class"] == "recalibration")

        max_abs = max(r["max_abs_err"] for r in seed_runs)
        max_rel = max(r["max_rel_err"] for r in seed_runs)
        max_abs_ratio = max(r["abs_ratio"] for r in seed_runs)
        max_rel_ratio = max(r["rel_ratio"] for r in seed_runs)
        max_denom = max(r["denom_magnitude"] for r in seed_runs)

        sample = seed_runs[0]
        agg = {
            "cfg_idx": cfg_idx,
            "shape": sample["shape"],
            "shape_bucket": sample["shape_bucket"],
            "dtype": sample["dtype"],
            "stride_category": sample["stride_category"],
            "atol": sample["atol"],
            "rtol": sample["rtol"],
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "max_abs_ratio": max_abs_ratio,
            "max_rel_ratio": max_rel_ratio,
            "max_denom_magnitude": max_denom,
            "n_seeds_run": len(seed_runs),
            "n_seeds_filable": n_filable,
            "n_seeds_recalibration": n_recalib,
            "x_layout_cpu": sample["x_layout_cpu"],
            "x_layout_mps": sample["x_layout_mps"],
        }

        if n_filable >= MIN_REPRO_SEEDS:
            filable.append(agg)
        elif n_recalib >= MIN_REPRO_SEEDS or (n_filable + n_recalib) >= MIN_REPRO_SEEDS:
            recalibration.append(agg)

    # Counts of single-seed-only divergences (informational)
    single_seed_filable = sum(
        1
        for cfg_idx, per_seed in per_config.items()
        if 0 < sum(1 for r in per_seed.values() if r["class"] == "filable_candidate") < MIN_REPRO_SEEDS
    )
    single_seed_recalib = sum(
        1
        for cfg_idx, per_seed in per_config.items()
        if 0 < sum(1 for r in per_seed.values() if r["class"] == "recalibration") < MIN_REPRO_SEEDS
    )

    return filable, recalibration, single_seed_filable, single_seed_recalib


def _flush(
    *,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    per_config: dict[int, dict[int, dict]],
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    max_abs_err_global: float,
    max_rel_err_global: float,
    elapsed: float,
    aborted: bool,
    error: str | None = None,
) -> None:
    filable, recalibration, single_filable, single_recalib = _classify_per_config(per_config)

    # Sort filable + recalibration by severity (worst rel ratio, then abs ratio, then small numel)
    filable_sorted = sorted(
        filable,
        key=lambda d: (
            -max(d["max_rel_ratio"], d["max_abs_ratio"]),
            _numel(d["shape"]),
        ),
    )
    recalibration_sorted = sorted(
        recalibration,
        key=lambda d: (
            -max(d["max_rel_ratio"], d["max_abs_ratio"]),
            _numel(d["shape"]),
        ),
    )

    # Top 3 minimal repros: prefer filable; fill from recalibration if short.
    top3_pool = filable_sorted + recalibration_sorted
    top3 = top3_pool[:3]

    summary = {
        "kernel": "swish",
        "kernel_op": KERNEL,
        "status": "ABORTED_BUDGET" if aborted and not error else ("ERROR" if error else "OK"),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "single_seed_filable_only": single_filable,
        "single_seed_recalibration_only": single_recalib,
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "elapsed_seconds": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "iters_per_seed": ITERS_PER_SEED,
        "filable_threshold_mult": FILABLE_MULT,
        "denom_floor": DENOM_FLOOR,
        "min_repro_seeds": MIN_REPRO_SEEDS,
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "top_3_repros": top3,
        "recommended_filing_target": "pytorch/pytorch" if filable else "none",
    }
    if error:
        summary["error"] = error

    _write_markdown(RESULTS_MD, summary, filable_sorted, recalibration_sorted)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(
    path: Path,
    s: dict,
    filable_sorted: list[dict],
    recalibration_sorted: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# swish — MPS fuzz report")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel_op']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **iters_attempted:** {s['iters_attempted']}")
    lines.append(f"- **iters_completed:** {s['iters_completed']}")
    lines.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    lines.append(f"- **divergences_filable:** {s['divergences_filable']}")
    lines.append(f"- **divergences_recalibration:** {s['divergences_recalibration']}")
    lines.append(
        "- **single-seed-only filable (not reproducible across "
        f"{s['min_repro_seeds']}+ seeds):** {s['single_seed_filable_only']}"
    )
    lines.append(
        f"- **single-seed-only recalibration:** {s['single_seed_recalibration_only']}"
    )
    lines.append(f"- **max_abs_err (global):** {s['max_abs_err']:.3e}")
    lines.append(f"- **max_rel_err (global):** {s['max_rel_err']:.3e}")
    lines.append(f"- **elapsed_seconds:** {s['elapsed_seconds']}")
    lines.append(f"- **seeds:** {s['seeds']}  (iters_per_seed={s['iters_per_seed']})")
    lines.append(
        f"- **filable threshold:** > {s['filable_threshold_mult']}× tolerance "
        f"AND (abs OR (rel AND |y_cpu|≥{s['denom_floor']:.0e})), "
        f"reproduced ≥{s['min_repro_seeds']} seeds"
    )
    lines.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    lines.append(f"- **MPS vs CUDA-mock:** N/A (no NVIDIA GPU on host)")
    lines.append(f"- **Recommended filing target:** `{s['recommended_filing_target']}`")
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
    if not s["top_3_repros"]:
        lines.append(
            "_No configuration exceeded the per-dtype tolerance band on ≥3 seeds. "
            "swish (== silu) on MPS appears numerically faithful to CPU within "
            "gpucheck's MPS-overlay tolerance._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(
                f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)"
            )
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(
                f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                f"(atol={r['atol']:.2e}, ratio={r['max_abs_ratio']:.2f}×)"
            )
            lines.append(
                f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                f"(rtol={r['rtol']:.2e}, ratio={r['max_rel_ratio']:.2f}×)"
            )
            lines.append(
                f"- **denom magnitude (max |y_cpu|):** {r['max_denom_magnitude']:.3e}"
            )
            lines.append(
                f"- **reproducibility:** {r['n_seeds_filable']}/{r['n_seeds_run']} "
                f"seeds filable, {r['n_seeds_recalibration']}/{r['n_seeds_run']} "
                f"seeds in recalibration band"
            )
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append("")

    if filable_sorted:
        lines.append("## All filable divergences")
        lines.append("")
        lines.append("| dtype | shape | stride | abs_ratio× | rel_ratio× | denom | seeds_filable |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in filable_sorted:
            lines.append(
                f"| {r['dtype']} | {tuple(r['shape'])} | {r['stride_category']} | "
                f"{r['max_abs_ratio']:.2f} | {r['max_rel_ratio']:.2f} | "
                f"{r['max_denom_magnitude']:.2e} | "
                f"{r['n_seeds_filable']}/{r['n_seeds_run']} |"
            )
        lines.append("")

    if recalibration_sorted:
        lines.append("## Tolerance-recalibration candidates (xfail entries)")
        lines.append("")
        lines.append("| dtype | shape | stride | abs_ratio× | rel_ratio× | denom | seeds_in_band |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in recalibration_sorted:
            lines.append(
                f"| {r['dtype']} | {tuple(r['shape'])} | {r['stride_category']} | "
                f"{r['max_abs_ratio']:.2f} | {r['max_rel_ratio']:.2f} | "
                f"{r['max_denom_magnitude']:.2e} | "
                f"{r['n_seeds_recalibration']}/{r['n_seeds_run']} |"
            )
        lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- swish == silu == x · sigmoid(x) (the v2 swarm spec lists swish and "
        "silu separately; this run uses `torch.nn.functional.silu`)."
    )
    lines.append(
        "- Reference: `F.silu` on CPU at the input dtype, comparison done in fp32."
    )
    lines.append(
        "- Tolerance: "
        "`gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` "
        "(MPS 2× overlay applied). Elementwise op — no matmul √(k/128) scaling."
    )
    lines.append(
        "- Filable: max_abs > 10×atol, OR (max_rel > 10×rtol AND max|y_cpu| ≥ "
        f"{DENOM_FLOOR:.0e}); reproduced across ≥{MIN_REPRO_SEEDS} of "
        f"{len(SEEDS)} seeds."
    )
    lines.append(
        "- Recalibration: 1×–5× tolerance with same denom guard; suggests an "
        "xfail registry entry rather than a torch-issue file."
    )
    lines.append(
        "- Stride categories: row_major, column_major, broadcast, transpose, "
        "slice, non_contig, gather (gpucheck.fuzzing.strides)."
    )
    lines.append(
        "- CUDA backend mocked (no NVIDIA GPU). MPS-vs-CUDA reported as N/A."
    )
    lines.append(
        "- Configs are deterministic across seeds: index `i` always maps to the "
        "same (shape, dtype, stride) tuple, so per-seed runs are directly "
        "comparable for reproducibility scoring."
    )
    lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
