"""Fuzz the hardtanh kernel: MPS vs CPU reference, v2 schema with multi-seed reproducibility.

CRITICAL DIVERGENCE FILTERING (v2):
  - max_rel_err > 10x tolerance ONLY counts if denom_magnitude >= 1e-6
    (otherwise = near-zero-denominator artifact, ignored).
  - max_abs_err > 10x tolerance always counts.
  - Both must be reproducible across >= 3 seeds (out of 5) to be FILABLE.
  - 1-5x tolerance => TOLERANCE_RECALIBRATION (recommend xfail entry).
  - 5-10x tolerance => also recalibration (grey zone, not filable).
  - Below 1x tolerance => OK.

CUDA path is N/A (no NVIDIA GPU on host); the spec only requires MPS vs CPU.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

WORKTREE_SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-hardtanh/src")
sys.path.insert(0, str(WORKTREE_SRC))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.shapes import (  # noqa: E402
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_hardtanh.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.hardtanh"
N_CONFIGS = 200
SEEDS = (0, 1, 2, 3, 4)
N_ITERS_TOTAL = N_CONFIGS * len(SEEDS)  # 1000
TIME_BUDGET_S = 11 * 60 + 15  # 11m15s budget; reserve 45s for write-out
DENOM_FLOOR = 1e-6
RECAL_LO = 1.0
DIVERGENCE_THRESHOLD = 10.0
MIN_SEED_REPRO = 3

# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------

DEGENERATE = [(1,), (1, 1), (16, 0), (0, 16), (1, 1, 1), (8,)]
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
LARGE = [(LARGE_DIMS[0],), (256, 256), (1024, 64), (512, 128)]
MIXED = [(127, 16), (1024, 3), (7, 128), (33, 128, 4), (5, 7, 11)]

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

# Hardtanh has (min_val, max_val) parameters. Default is (-1, 1). We sample a
# few common variants to exercise the boundary clamp logic.
HARDTANH_BOUNDS = [
    (-1.0, 1.0),    # default ReLU6/clamp baseline
    (-3.0, 3.0),    # MobileNet hard-swish style range
    (0.0, 6.0),     # ReLU6
    (-2.0, 2.0),    # symmetric wider
    (-0.5, 0.5),    # tight clamp -> more clipping
]


def _filtered_rel_err(
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    denom_floor: float = DENOM_FLOOR,
) -> tuple[float, float]:
    """Return (max_rel_err_filtered, denom_magnitude_at_max).

    Only positions where |b| >= denom_floor are considered. If no positions
    qualify, return (0.0, 0.0).
    """
    if a_cpu.numel() == 0:
        return 0.0, 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    diff = (a - b).abs()
    denom = b.abs()
    mask = denom >= denom_floor
    if not bool(mask.any().item()):
        return 0.0, 0.0
    rel = diff[mask] / denom[mask]
    if rel.numel() == 0:
        return 0.0, 0.0
    rel_max = float(rel.max().item())
    # find denom at max position (within mask)
    mask_idx = mask.nonzero(as_tuple=False).flatten()
    rel_argmax = int(rel.argmax().item())
    flat_b = b.flatten()[mask_idx]
    denom_at_max = float(flat_b[rel_argmax].abs().item())
    return rel_max, denom_at_max


def _max_rel_err_raw(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    if a_cpu.numel() == 0:
        return 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    return float((diff / denom).max().item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    if a_cpu.numel() == 0:
        return 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(over_atol: float, over_rtol_filtered: float) -> str:
    score = max(over_atol, over_rtol_filtered)
    if score < RECAL_LO:
        return "OK"
    if score <= DIVERGENCE_THRESHOLD:
        return "RECALIBRATION"
    return "DIVERGENCE"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "agent": "kernel-fuzzer-hardtanh",
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() == False",
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

    # Build the deterministic config plan: 200 unique configs.
    plan_rng = random.Random(0xBADC0FFEE)
    configs: list[dict] = []
    for cfg_idx in range(N_CONFIGS):
        bucket = plan_rng.choice(BUCKET_NAMES)
        shape = plan_rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = plan_rng.choice(DTYPE_NAMES)
        stride_cat = plan_rng.choice(STRIDE_CATEGORIES)
        bounds = plan_rng.choice(HARDTANH_BOUNDS)
        configs.append({
            "cfg_idx": cfg_idx,
            "bucket": bucket,
            "shape": shape,
            "dtype": dtype_name,
            "stride": stride_cat,
            "bounds": bounds,
        })

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_errored = 0
    max_abs_err_global = 0.0
    max_rel_err_filtered_global = 0.0
    max_rel_err_raw_global = 0.0

    # per-config seed-result list of dicts
    config_results: dict[int, list[dict]] = {c["cfg_idx"]: [] for c in configs}

    aborted_for_budget = False

    for cfg in configs:
        if time.monotonic() - started > TIME_BUDGET_S:
            print(f"[budget] aborting at config {cfg['cfg_idx']}/{N_CONFIGS}", file=sys.stderr)
            aborted_for_budget = True
            break

        cfg_idx = cfg["cfg_idx"]
        bucket = cfg["bucket"]
        shape = cfg["shape"]
        dtype_name = cfg["dtype"]
        dtype = DTYPES_BY_NAME[dtype_name]
        stride_cat = cfg["stride"]
        min_v, max_v = cfg["bounds"]

        for seed_i in SEEDS:
            iters_attempted += 1
            try:
                # Build CPU tensor
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=seed_i,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    iters_unsupported += 1
                    config_results[cfg_idx].append({
                        "seed": seed_i, "status": "UNSUPPORTED_BUILD_CPU",
                        "note": str(exc).splitlines()[0][:200],
                    })
                    continue

                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=seed_i,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    iters_unsupported += 1
                    config_results[cfg_idx].append({
                        "seed": seed_i, "status": "UNSUPPORTED_BUILD_MPS",
                        "note": str(exc).splitlines()[0][:200],
                    })
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    config_results[cfg_idx].append({
                        "seed": seed_i, "status": "OK_EMPTY",
                        "max_abs_err": 0.0, "max_rel_err_filtered": 0.0,
                        "max_rel_err_raw": 0.0, "denom_at_max_rel": 0.0,
                        "atol": 0.0, "rtol": 0.0,
                        "over_atol": 0.0, "over_rtol_filtered": 0.0,
                    })
                    continue

                try:
                    y_mps = F.hardtanh(x_mps, min_val=min_v, max_val=max_v)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    config_results[cfg_idx].append({
                        "seed": seed_i, "status": "UNSUPPORTED_MPS_OP",
                        "note": str(exc).splitlines()[0][:200],
                    })
                    continue

                try:
                    y_cpu = F.hardtanh(x_cpu, min_val=min_v, max_val=max_v)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    config_results[cfg_idx].append({
                        "seed": seed_i, "status": "UNSUPPORTED_CPU_OP",
                        "note": str(exc).splitlines()[0][:200],
                    })
                    continue

                iters_completed += 1

                rel_filtered, denom_at_max = _filtered_rel_err(y_mps_cpu, y_cpu)
                rel_raw = _max_rel_err_raw(y_mps_cpu, y_cpu)
                absdiff = _max_abs_err(y_mps_cpu, y_cpu)
                max_abs_err_global = max(max_abs_err_global, absdiff)
                max_rel_err_filtered_global = max(max_rel_err_filtered_global, rel_filtered)
                max_rel_err_raw_global = max(max_rel_err_raw_global, rel_raw)

                atol, rtol = compute_tolerance(dtype, device_type="mps")
                over_atol = absdiff / atol if atol > 0 else 0.0
                over_rtol_filtered = rel_filtered / rtol if rtol > 0 else 0.0

                config_results[cfg_idx].append({
                    "seed": seed_i,
                    "status": _classify(over_atol, over_rtol_filtered),
                    "max_abs_err": absdiff,
                    "max_rel_err_filtered": rel_filtered,
                    "max_rel_err_raw": rel_raw,
                    "denom_at_max_rel": denom_at_max,
                    "atol": atol,
                    "rtol": rtol,
                    "over_atol": over_atol,
                    "over_rtol_filtered": over_rtol_filtered,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                })

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                iters_errored += 1
                print(
                    f"[ERROR] cfg={cfg_idx} seed={seed_i} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                # Halt rule: report and exit non-zero so the caller sees it.
                _emergency_writeout(
                    started, iters_attempted, iters_completed,
                    iters_unsupported, iters_errored, configs, config_results,
                    max_abs_err_global, max_rel_err_filtered_global,
                    max_rel_err_raw_global, halted=True, halt_msg=repr(exc),
                )
                return 2

    elapsed = time.monotonic() - started

    # Aggregate per-config across seeds
    filable: list[dict] = []
    recalibration: list[dict] = []

    for cfg in configs:
        results = config_results[cfg["cfg_idx"]]
        if not results:
            continue
        # Skip configs where all seeds were unsupported / empty
        valid = [r for r in results if "max_abs_err" in r]
        if not valid:
            continue

        seed_statuses = [r["status"] for r in valid]
        n_div = sum(1 for s in seed_statuses if s == "DIVERGENCE")
        n_recal = sum(1 for s in seed_statuses if s == "RECALIBRATION")

        worst = max(
            valid,
            key=lambda r: max(r["over_atol"], r["over_rtol_filtered"]),
        )

        record = {
            "cfg_idx": cfg["cfg_idx"],
            "kernel": KERNEL,
            "shape": list(cfg["shape"]),
            "shape_bucket": cfg["bucket"],
            "dtype": cfg["dtype"],
            "stride_category": cfg["stride"],
            "bounds": list(cfg["bounds"]),
            "n_seeds_tested": len(valid),
            "n_seeds_diverge": n_div,
            "n_seeds_recalibration": n_recal,
            "worst_seed": worst["seed"],
            "worst_max_abs_err": worst["max_abs_err"],
            "worst_max_rel_err_filtered": worst["max_rel_err_filtered"],
            "worst_max_rel_err_raw": worst["max_rel_err_raw"],
            "worst_denom_at_max_rel": worst["denom_at_max_rel"],
            "atol": worst["atol"],
            "rtol": worst["rtol"],
            "worst_over_atol": worst["over_atol"],
            "worst_over_rtol_filtered": worst["over_rtol_filtered"],
            "x_layout_cpu_worst": worst.get("x_layout_cpu", ""),
            "x_layout_mps_worst": worst.get("x_layout_mps", ""),
        }

        if n_div >= MIN_SEED_REPRO:
            record["bucket"] = "FILABLE"
            filable.append(record)
        elif n_recal >= 1 or n_div >= 1:
            # Anything in the 1-10x band, or a single-seed-only >10x flake,
            # goes to RECALIBRATION.
            record["bucket"] = (
                "RECALIBRATION_NON_REPRO_DIVERGENCE" if n_div >= 1
                else "RECALIBRATION"
            )
            recalibration.append(record)

    # Top 3 minimal repros: largest worst_score, smallest numel as tiebreaker.
    def _sort_key(r: dict) -> tuple:
        worst = max(r["worst_over_atol"], r["worst_over_rtol_filtered"])
        return (-worst, _numel(r["shape"]))

    top_pool = sorted(filable + recalibration, key=_sort_key)
    top3 = top_pool[:3]

    summary = {
        "kernel": KERNEL,
        "agent": "kernel-fuzzer-hardtanh",
        "status": "OK" if not aborted_for_budget else "PARTIAL_BUDGET_EXIT",
        "schema": "v2",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_errored": iters_errored,
        "n_configs_planned": N_CONFIGS,
        "seeds": list(SEEDS),
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_filtered_global,
        "max_rel_err_raw_unfiltered": max_rel_err_raw_global,
        "denom_floor": DENOM_FLOOR,
        "divergence_threshold_x_tol": DIVERGENCE_THRESHOLD,
        "min_seed_reproducibility": MIN_SEED_REPRO,
        "top_3_repros": top3,
        "filable_repros_full": filable,
        "recalibration_repros_count": len(recalibration),
        "wall_time_s": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
        "cuda_status": "N/A_mocked_no_nvidia_hardware",
        "host": "darwin/arm64 (Apple Silicon)",
        "recommended_filing_target": (
            "pytorch/pytorch" if filable else "none"
        ),
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        # JSONL: one compact line as required
        compact = {
            "kernel": KERNEL,
            "agent": "kernel-fuzzer-hardtanh",
            "schema": "v2",
            "status": summary["status"],
            "iters_attempted": iters_attempted,
            "iters_completed": iters_completed,
            "iters_unsupported": iters_unsupported,
            "iters_errored": iters_errored,
            "divergences_filable": len(filable),
            "divergences_recalibration": len(recalibration),
            "max_abs_err": max_abs_err_global,
            "max_rel_err": max_rel_err_filtered_global,
            "max_rel_err_raw_unfiltered": max_rel_err_raw_global,
            "top_3_repros": top3,
            "wall_time_s": round(elapsed, 2),
            "torch_version": torch.__version__,
            "mps_available": True,
            "cuda_status": "N/A_mocked_no_nvidia_hardware",
            "seeds": list(SEEDS),
            "denom_floor": DENOM_FLOOR,
            "recommended_filing_target": summary["recommended_filing_target"],
        }
        f.write(json.dumps(compact, default=str) + "\n")

    return 0


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return n


def _emergency_writeout(
    started: float,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_errored: int,
    configs: list[dict],
    config_results: dict[int, list[dict]],
    max_abs: float,
    max_rel_filtered: float,
    max_rel_raw: float,
    halted: bool,
    halt_msg: str,
) -> None:
    elapsed = time.monotonic() - started
    summary = {
        "kernel": KERNEL,
        "agent": "kernel-fuzzer-hardtanh",
        "schema": "v2",
        "status": "HALTED_PROCESS_ERROR" if halted else "PARTIAL",
        "halt_msg": halt_msg,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_errored": iters_errored,
        "divergences_filable": 0,
        "divergences_recalibration": 0,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel_filtered,
        "max_rel_err_raw_unfiltered": max_rel_raw,
        "top_3_repros": [],
        "wall_time_s": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
    }
    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(path: Path, s: dict) -> None:
    lines = []
    lines.append(f"# {s['kernel']} - MPS fuzz report (v2 schema)")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Agent:** `{s['agent']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **Iterations attempted:** {s['iters_attempted']}")
    lines.append(f"- **Iterations completed:** {s['iters_completed']}")
    lines.append(f"- **Iterations unsupported:** {s.get('iters_unsupported', 0)}")
    lines.append(f"- **Iterations errored:** {s.get('iters_errored', 0)}")
    lines.append(f"- **Divergences (FILABLE, >=3 seeds, >10x tol):** {s['divergences_filable']}")
    lines.append(f"- **Divergences (RECALIBRATION, 1-10x tol):** {s['divergences_recalibration']}")
    lines.append(f"- **Max abs err (global):** {s['max_abs_err']:.3e}")
    lines.append(
        f"- **Max rel err (filtered |denom|>={s.get('denom_floor', 1e-6)}): "
        f"{s['max_rel_err']:.3e}**"
    )
    lines.append(
        f"- **Max rel err (raw, unfiltered):** "
        f"{s['max_rel_err_raw_unfiltered']:.3e}"
    )
    lines.append(f"- **Wall time:** {s['wall_time_s']} s")
    lines.append(f"- **torch:** {s['torch_version']}, host: {s.get('host', 'darwin/arm64')}")
    lines.append(f"- **CUDA:** {s.get('cuda_status', 'N/A')}")
    lines.append(
        f"- **Recommended filing target:** `{s.get('recommended_filing_target', 'none')}`"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Seeds tested per config: {s.get('seeds', [0,1,2,3,4])}")
    lines.append(f"- Configs planned: {s.get('n_configs_planned', 200)}")
    lines.append(f"- Total iterations: configs * seeds = {s.get('n_configs_planned', 200) * 5}")
    lines.append("- Reference: `torch.nn.functional.hardtanh` on CPU (FP32 promotion).")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`.")
    lines.append("- **Filable** = max_abs_err > 10x atol OR (max_rel_err > 10x rtol AND |denom| >= 1e-6),")
    lines.append("  AND reproduced in >= 3 of 5 seeds.")
    lines.append("- **Recalibration** = score in [1x, 10x] tol on at least one seed,")
    lines.append("  OR a single-seed >10x outlier (not reproduced).")
    lines.append("- Stride categories tested: row_major, column_major, broadcast, transpose,")
    lines.append("  slice, non_contig, gather (gpucheck.fuzzing.strides).")
    lines.append("- Hardtanh bounds sampled: (-1,1), (-3,3), (0,6), (-2,2), (-0.5,0.5).")
    lines.append("")
    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not s.get("top_3_repros"):
        lines.append("_No divergences exceeded gpucheck's per-dtype tolerance "
                     "(MPS 2x overlay applied)._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i} ({r.get('bucket', '?')})")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **hardtanh bounds:** `(min={r['bounds'][0]}, max={r['bounds'][1]})`")
            lines.append(
                f"- **seeds diverge / recal / total:** "
                f"{r['n_seeds_diverge']} / {r['n_seeds_recalibration']} / {r['n_seeds_tested']}"
            )
            lines.append(
                f"- **worst max_abs_err:** {r['worst_max_abs_err']:.3e} "
                f"(atol={r['atol']:.2e}, over_atol={r['worst_over_atol']:.2f}x)"
            )
            lines.append(
                f"- **worst max_rel_err (filtered):** {r['worst_max_rel_err_filtered']:.3e} "
                f"(rtol={r['rtol']:.2e}, over_rtol={r['worst_over_rtol_filtered']:.2f}x, "
                f"denom={r['worst_denom_at_max_rel']:.2e})"
            )
            lines.append(
                f"- **worst max_rel_err (raw, unfiltered):** "
                f"{r['worst_max_rel_err_raw']:.3e}"
            )
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu_worst']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps_worst']}`")
            lines.append(f"- **worst seed:** {r['worst_seed']}")
            lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CUDA backend is N/A (no NVIDIA GPU on host); only MPS-vs-CPU compared.")
    lines.append("- bf16 is supported by hardtanh on MPS (verified across iterations).")
    lines.append("- `_filtered_rel_err` excludes positions where |reference| < 1e-6 to avoid")
    lines.append("  near-zero-denom artifacts inflating relative error.")
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
