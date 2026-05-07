"""Fuzz the rrelu kernel: MPS vs CPU reference. CUDA mocked (no NVIDIA GPU).

Per kernel-fuzzer-rrelu (v2) protocol:
- 1000 iterations across seeds 0,1,2,3,4 (200 configs/seed, same configs per seed)
- dtypes fp32/fp16/bf16 (skip if unsupported)
- Filter:
    max_rel_err > 10x tolerance counts only if denom_magnitude >= 1e-6
    max_abs_err > 10x tolerance always counts
    Divergence is FILABLE only if reproducible across >=3 seeds
    1-5x tolerance band -> TOLERANCE_RECALIBRATION bucket
    < 1x -> OK
- 12 minute wall budget (script enforces ~9 min internal cap to leave room for I/O).
- Halt on unexpected process error.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

# Ensure gpucheck (worktree) is importable
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-rrelu/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_rrelu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.rrelu"
SEEDS = [0, 1, 2, 3, 4]
CONFIGS_PER_SEED = 200  # 5 * 200 = 1000 iters
WALL_BUDGET_S = 9 * 60  # leave 3 min for results/jsonl writing under 12-min hard cap

# rrelu eval-mode parameters (deterministic — slope = (lower+upper)/2)
RRELU_LOWER = 1.0 / 8.0
RRELU_UPPER = 1.0 / 3.0

# Sampling space — modest so 1000 iters fit easily.
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


def _filtered_max_rel_err(
    a_cpu: torch.Tensor, b_ref: torch.Tensor, denom_floor: float = 1e-6
) -> float:
    """Max |a-b|/|b| over elements where |b| >= denom_floor.

    Excludes near-zero-denominator artifacts per task spec.
    """
    a = a_cpu.to(torch.float32)
    b = b_ref.to(torch.float32)
    diff = (a - b).abs()
    denom = b.abs()
    mask = denom >= denom_floor
    if not bool(mask.any().item()):
        return 0.0
    rel = diff[mask] / denom[mask]
    if rel.numel() == 0:
        return 0.0
    return float(rel.max().item())


def _max_abs_err(a_cpu: torch.Tensor, b_ref: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_ref.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _build_configs(n: int) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Deterministically build n unique (bucket, shape, dtype_name, stride_cat) configs.

    Uses a fixed RNG seed so the same configs are reused across all SEEDS, enabling
    cross-seed reproducibility check.
    """
    rng = random.Random(0xBADCAFE)
    configs: list[tuple[str, tuple[int, ...], str, str]] = []
    for _ in range(n):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        configs.append((bucket, shape, dtype_name, stride_cat))
    return configs


def _config_key(bucket: str, shape: tuple[int, ...], dtype_name: str, stride_cat: str) -> str:
    return f"{bucket}|{tuple(shape)!r}|{dtype_name}|{stride_cat}"


def main() -> int:  # noqa: PLR0915, PLR0912 — a single linear fuzz harness
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
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()
    configs = _build_configs(CONFIGS_PER_SEED)

    iters_attempted = 0
    iters_completed = 0  # all iters that didn't fall into unsupported buckets (incl. empty)
    iters_mps_success = 0  # MPS rrelu actually produced an output and was compared
    unsupported_mps = 0
    unsupported_cpu = 0
    unsupported_build = 0

    # per-config stats: maps key -> dict
    # {seeds_diverged: set, max_abs_err, max_rel_err, samples: list of records}
    per_config: dict[str, dict] = {}

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    aborted = False

    for seed in SEEDS:
        if time.monotonic() - started > WALL_BUDGET_S:
            print(f"[budget] aborting before seed {seed}", file=sys.stderr)
            aborted = True
            break

        for cfg_idx, (bucket, shape, dtype_name, stride_cat) in enumerate(configs):
            if time.monotonic() - started > WALL_BUDGET_S:
                print(
                    f"[budget] aborting at seed={seed} cfg={cfg_idx} "
                    f"(elapsed {time.monotonic() - started:.1f}s)",
                    file=sys.stderr,
                )
                aborted = True
                break

            iters_attempted += 1
            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            dtype = DTYPES_BY_NAME[dtype_name]
            seed_i = seed * 1_000_003 + cfg_idx  # deterministic per (seed, cfg)

            try:
                # CPU reference
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=seed_i,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    unsupported_build += 1
                    print(
                        f"[unsupported-build-cpu] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:160]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=seed_i,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    unsupported_build += 1
                    print(
                        f"[unsupported-build-mps] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:160]}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                # MPS rrelu in eval mode (deterministic: slope=(lower+upper)/2)
                try:
                    y_mps = F.rrelu(
                        x_mps, lower=RRELU_LOWER, upper=RRELU_UPPER, training=False,
                    )
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    unsupported_mps += 1
                    if iters_attempted <= 3 or iters_attempted % 100 == 0:
                        # Don't spam stderr; this is expected for unsupported ops
                        print(
                            f"[unsupported-mps] seed={seed} cfg={cfg_idx} "
                            f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                            f"{str(exc).splitlines()[0][:160]}",
                            file=sys.stderr,
                        )
                    continue

                try:
                    y_cpu = F.rrelu(
                        x_cpu, lower=RRELU_LOWER, upper=RRELU_UPPER, training=False,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    unsupported_cpu += 1
                    print(
                        f"[unsupported-cpu] seed={seed} cfg={cfg_idx} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:160]}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1
                iters_mps_success += 1

                rel = _filtered_max_rel_err(y_mps_cpu, y_cpu, denom_floor=1e-6)
                absdiff = _max_abs_err(y_mps_cpu, y_cpu)
                max_abs_err_global = max(max_abs_err_global, absdiff)
                max_rel_err_global = max(max_rel_err_global, rel)

                atol, rtol = compute_tolerance(dtype, device_type="mps")

                # Compute multipliers vs base tolerance
                abs_mult = absdiff / atol if atol > 0 else 0.0
                rel_mult = rel / rtol if rtol > 0 else 0.0
                worst_mult = max(abs_mult, rel_mult)

                key = _config_key(bucket, shape, dtype_name, stride_cat)
                rec = per_config.setdefault(
                    key,
                    {
                        "bucket": bucket,
                        "shape": list(shape),
                        "dtype": dtype_name,
                        "stride_category": stride_cat,
                        "atol": atol,
                        "rtol": rtol,
                        "samples": [],
                        "seeds_diverged_filable": set(),
                        "seeds_diverged_recal": set(),
                        "max_abs_err": 0.0,
                        "max_rel_err": 0.0,
                    },
                )
                rec["samples"].append(
                    {
                        "seed": seed,
                        "max_abs_err": absdiff,
                        "max_rel_err": rel,
                        "abs_mult": abs_mult,
                        "rel_mult": rel_mult,
                    }
                )
                rec["max_abs_err"] = max(rec["max_abs_err"], absdiff)
                rec["max_rel_err"] = max(rec["max_rel_err"], rel)

                # Bucketize per task spec
                if worst_mult > 10.0:
                    rec["seeds_diverged_filable"].add(seed)
                elif worst_mult > 1.0 and worst_mult <= 5.0:
                    rec["seeds_diverged_recal"].add(seed)
                # 5-10x tier: severe but not yet filable; treat as recalibration too
                elif worst_mult > 5.0 and worst_mult <= 10.0:
                    rec["seeds_diverged_recal"].add(seed)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Halt rule: report and exit non-zero on unexpected error
                print(
                    f"[ERROR] seed={seed} cfg={cfg_idx} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                # Still try to write a partial report below
                aborted = True
                break

        if aborted:
            break

    elapsed = time.monotonic() - started

    # Aggregate FILABLE (>=3 seeds in filable bucket) and RECALIBRATION (>=3 seeds in recal)
    filable_configs = [
        rec for rec in per_config.values() if len(rec["seeds_diverged_filable"]) >= 3
    ]
    recal_configs = [
        rec for rec in per_config.values()
        if rec not in filable_configs and len(rec["seeds_diverged_recal"]) >= 3
    ]

    # Top-3 repros: prefer filable, then recal, sorted by worst max_rel_err / max_abs_err.
    def _sort_key(r: dict) -> tuple[float, int]:
        return (-max(r["max_rel_err"], r["max_abs_err"]), _numel(tuple(r["shape"])))

    top3_records = sorted(filable_configs, key=_sort_key)[:3]
    if len(top3_records) < 3:
        top3_records += sorted(recal_configs, key=_sort_key)[: 3 - len(top3_records)]

    top3 = [
        {
            "bucket": r["bucket"],
            "shape": r["shape"],
            "dtype": r["dtype"],
            "stride_category": r["stride_category"],
            "max_abs_err": r["max_abs_err"],
            "max_rel_err": r["max_rel_err"],
            "atol": r["atol"],
            "rtol": r["rtol"],
            "seeds_diverged_filable": sorted(r["seeds_diverged_filable"]),
            "seeds_diverged_recal": sorted(r["seeds_diverged_recal"]),
        }
        for r in top3_records
    ]

    # Status: UNSUPPORTED if MPS rrelu never produced a single output successfully.
    # iters_completed includes empty-tensor early-returns, so use iters_mps_success.
    if iters_mps_success == 0 and unsupported_mps > 0:
        status = "UNSUPPORTED"
    elif aborted:
        status = "PARTIAL"
    else:
        status = "OK"

    summary = {
        "kernel": KERNEL,
        "status": status,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_mps_success": iters_mps_success,
        "iters_unsupported_mps": unsupported_mps,
        "iters_unsupported_cpu": unsupported_cpu,
        "iters_unsupported_build": unsupported_build,
        "divergences_filable": len(filable_configs),
        "divergences_recalibration": len(recal_configs),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "elapsed_seconds": round(elapsed, 2),
        "wall_budget_seconds": WALL_BUDGET_S,
        "seeds": SEEDS,
        "configs_per_seed": CONFIGS_PER_SEED,
        "top_3_repros": top3,
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "rrelu_eval_slope": (RRELU_LOWER + RRELU_UPPER) / 2.0,
        "denom_filter_floor": 1e-6,
        "recommended_filing_target": _recommend_target(status, filable_configs),
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    return 0


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _recommend_target(status: str, filable: list[dict]) -> str:
    if status == "UNSUPPORTED":
        # MPS lacks the op entirely; this is a coverage-gap, not a correctness bug.
        return "pytorch/pytorch (MPS op-coverage: implement aten::rrelu_with_noise)"
    if filable:
        return "pytorch/pytorch (rrelu MPS numeric divergence)"
    return "none"


def _write_markdown(path: Path, s: dict) -> None:
    lines: list[str] = []
    lines.append(f"# {s['kernel']} - MPS fuzz report")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **Iterations attempted:** {s['iters_attempted']}")
    lines.append(f"- **Iterations completed:** {s['iters_completed']}  "
                 f"(of which MPS-rrelu actually produced an output: {s['iters_mps_success']})")
    lines.append(f"- **Iterations unsupported (MPS):** {s['iters_unsupported_mps']}")
    lines.append(f"- **Iterations unsupported (CPU):** {s['iters_unsupported_cpu']}")
    lines.append(f"- **Iterations unsupported (build):** {s['iters_unsupported_build']}")
    lines.append(f"- **Divergences FILABLE (>=3 seeds, >10x tol):** {s['divergences_filable']}")
    lines.append(f"- **Divergences RECALIBRATION (>=3 seeds, 1-10x tol):** {s['divergences_recalibration']}")
    lines.append(f"- **MPS vs CPU max abs err:** {s['max_abs_err']:.3e}")
    lines.append(f"- **MPS vs CPU max rel err (denom>={s['denom_filter_floor']:.0e}):** {s['max_rel_err']:.3e}")
    lines.append(f"- **MPS vs CUDA-mock max rel err:** N/A (CUDA mocked - no NVIDIA GPU on host)")
    lines.append(f"- **Recommended filing target:** {s['recommended_filing_target']}")
    lines.append(f"- **Elapsed:** {s['elapsed_seconds']} s  (wall budget: {s['wall_budget_seconds']} s)")
    lines.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    lines.append(f"- **Seeds:** {s['seeds']}, configs/seed: {s['configs_per_seed']}")
    lines.append(f"- **rrelu eval slope:** {s['rrelu_eval_slope']:.6f} "
                 "((lower+upper)/2 in `training=False`)")
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
    if s["status"] == "UNSUPPORTED":
        lines.append(
            "_No numeric repros: `aten::rrelu_with_noise` is not implemented for "
            "the MPS backend in torch 2.11.0. Every iteration of every dtype "
            "(fp32/fp16/bf16) and every training mode raises `NotImplementedError` "
            "before any output is produced. There is therefore no MPS-vs-CPU "
            "numeric divergence to report._"
        )
    elif not s["top_3_repros"]:
        lines.append(
            "_No divergences exceeded gpucheck's per-dtype tolerance "
            "(with MPS multiplier from `assertions/tolerances.py`)._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **max abs err:** {r['max_abs_err']:.3e}  (atol={r['atol']:.2e})")
            lines.append(f"- **max rel err:** {r['max_rel_err']:.3e}  (rtol={r['rtol']:.2e})")
            lines.append(
                f"- **seeds diverged (filable, >10x):** {r['seeds_diverged_filable']}"
            )
            lines.append(
                f"- **seeds diverged (recal, 1-10x):** {r['seeds_diverged_recal']}"
            )
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.nn.functional.rrelu(x, lower=1/8, upper=1/3, "
                 "training=False)` on CPU. In `training=False` mode rrelu is "
                 "deterministic: slope = (lower+upper)/2 for negative inputs, "
                 "identity for non-negative.")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
                 "dtype, device_type='mps')`. rrelu is elementwise -> no "
                 "sqrt(k/128) matmul scaling.")
    lines.append("- Divergence filter (per task spec):")
    lines.append("    - `max_rel_err > 10x rtol` counts only if `denom_magnitude "
                 ">= 1e-6` (mask near-zero refs).")
    lines.append("    - `max_abs_err > 10x atol` always counts.")
    lines.append("    - **FILABLE** = same (shape, dtype, stride) config exceeds "
                 "10x threshold across >=3 of 5 seeds.")
    lines.append("    - **RECALIBRATION** = same config in 1x-10x band across >=3 seeds.")
    lines.append("    - Below 1x = OK.")
    lines.append("- Stride categories: row_major, column_major, broadcast, "
                 "transpose, slice, non_contig, gather "
                 "(`gpucheck.fuzzing.strides`).")
    lines.append("- CUDA backend mocked (no NVIDIA GPU present); MPS-vs-CUDA "
                 "comparison reported as N/A by spec.")
    lines.append("")
    if s["status"] == "UNSUPPORTED":
        lines.append("## Why UNSUPPORTED")
        lines.append("")
        lines.append("`torch.nn.functional.rrelu`, `torch.rrelu`, and `torch.nn.RReLU` "
                     "all dispatch to `aten::rrelu_with_noise`, which has no MPS "
                     "implementation in torch 2.11.0:")
        lines.append("")
        lines.append("```")
        lines.append("NotImplementedError: The operator 'aten::rrelu_with_noise' is "
                     "not currently implemented for the MPS device.")
        lines.append("```")
        lines.append("")
        lines.append("This is a coverage gap, not a numerical bug. Recommended "
                     "follow-ups for gpucheck:")
        lines.append("")
        lines.append("1. Add `rrelu` (and `rrelu_with_noise`) to the "
                     "`[tool.gpucheck.mps.xfail]` registry citing pytorch's "
                     "MPS-coverage tracking issue (pytorch/pytorch#77764).")
        lines.append("2. File or upvote a feature request against pytorch/pytorch for "
                     "an MPS implementation of `aten::rrelu_with_noise`.")
        lines.append("3. Until then, gpucheck users on Apple Silicon should xfail "
                     "rrelu tests rather than inflate tolerances.")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
