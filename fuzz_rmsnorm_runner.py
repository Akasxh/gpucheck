"""Fuzzer v2 for RMSNorm — MPS vs CPU with multi-seed reproducibility filter.

Generates ~200 unique (shape, dtype, stride_category) configurations, then
re-runs each across seeds {0,1,2,3,4} for a target of 1000 total iterations.
Divergence filtering (per kernel-fuzzer-rmsnorm v2 spec):

  * abs_err > 10x atol  → ALWAYS counts as candidate divergence
  * rel_err > 10x rtol  → counts ONLY if denom_magnitude (= max |y_cpu|) >= 1e-6
  * candidate is FILABLE only if it reproduces on >= 3 distinct seeds
    with the same (shape, dtype, stride_cat) config
  * 1x..10x tolerance   → TOLERANCE_RECALIBRATION (xfail recommendation)
  * < 1x tolerance      → OK
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Any

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.backends.mps import MPSBackend
from gpucheck.fuzzing.strides import CATEGORIES, fuzz_strides_for_category

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2, 3, 4)
TARGET_ITERATIONS = 1000           # total across all seeds
N_CONFIGS = TARGET_ITERATIONS // len(SEEDS)  # 200 unique configs
CONFIG_SAMPLER_SEED = 0xC0DE
BUDGET_SECONDS = 11 * 60           # 11 min, leave 1 min for write-out
DENOM_FLOOR = 1e-6                 # below this, rel_err is a near-zero artifact

# Shape buckets per CLAUDE.md priority ordering (degenerate > non-tile-aligned >
# prime > pow2_boundary > large > mixed). Mixed = sampled combinations.
SHAPES_DEGENERATE = [
    (1,), (1, 1), (1, 1, 1), (1, 4), (4, 1),
]
SHAPES_PRIME = [
    (7,), (13, 17), (29, 31), (3, 5, 7), (11, 13, 17), (5, 23),
]
SHAPES_POW2_BOUNDARY = [
    (32,), (33,), (63,), (64,), (65,), (127,), (128,), (129,),
    (32, 64), (64, 65), (128, 127), (16, 16, 32), (8, 32, 33),
]
SHAPES_NON_TILE_ALIGNED = [
    (15, 50), (33, 70), (100, 100), (3, 99), (1023,), (513, 7), (7, 257),
]
SHAPES_LARGE = [
    (1024,), (256, 1024), (4, 64, 1024), (8, 8, 1024), (2, 4, 2048),
]

SHAPE_BUCKETS: dict[str, list[tuple[int, ...]]] = {
    "degenerate": SHAPES_DEGENERATE,
    "prime": SHAPES_PRIME,
    "pow2_boundary": SHAPES_POW2_BOUNDARY,
    "non_tile_aligned": SHAPES_NON_TILE_ALIGNED,
    "large": SHAPES_LARGE,
}

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# ---------------------------------------------------------------------------
# RMSNorm reference
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LLaMA-style RMSNorm along last dim, fp32-upcast reduction."""
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    return y.to(orig_dtype)


def err_metrics(y_mps: torch.Tensor, y_cpu: torch.Tensor) -> tuple[float, float, float]:
    """Return (abs_err, rel_err, denom_magnitude) computed in float64 on CPU.

    denom_magnitude = max |y_cpu|, used to detect near-zero-denominator
    inflation in rel_err.
    """
    a64 = y_mps.detach().cpu().to(torch.float64)
    b64 = y_cpu.detach().cpu().to(torch.float64)
    if a64.numel() == 0 or b64.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a64 - b64).abs()
    abs_err = float(diff.max().item())
    denom_mag = float(b64.abs().max().item())
    denom = b64.abs().clamp_min(1e-30)
    rel_err = float((diff / denom).max().item())
    if not math.isfinite(abs_err):
        abs_err = float("inf")
    if not math.isfinite(rel_err):
        rel_err = float("inf")
    return abs_err, rel_err, denom_mag


# ---------------------------------------------------------------------------
# Configuration sampling
# ---------------------------------------------------------------------------

def build_configs(n_configs: int, sampler_seed: int) -> list[dict[str, Any]]:
    """Deterministically sample `n_configs` unique (shape, dtype, stride) tuples."""
    rng = random.Random(sampler_seed)
    configs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    bucket_names = list(SHAPE_BUCKETS.keys())
    # Distribute roughly evenly across buckets/dtypes/strides for coverage.
    attempts = 0
    while len(configs) < n_configs and attempts < n_configs * 20:
        attempts += 1
        bucket = rng.choice(bucket_names)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name, dtype = rng.choice(DTYPES)
        cat = rng.choice(CATEGORIES)
        key = (shape, dtype_name, cat)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "bucket": bucket,
            "shape": shape,
            "dtype_name": dtype_name,
            "dtype": dtype,
            "stride_cat": cat,
        })
    # If we couldn't fill (small Cartesian product corner) just allow dupes.
    while len(configs) < n_configs:
        bucket = rng.choice(bucket_names)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name, dtype = rng.choice(DTYPES)
        cat = rng.choice(CATEGORIES)
        configs.append({
            "bucket": bucket,
            "shape": shape,
            "dtype_name": dtype_name,
            "dtype": dtype,
            "stride_cat": cat,
        })
    return configs


def is_skippable_shape(shape: tuple[int, ...]) -> str | None:
    """Return a reason string if this shape should be skipped for RMSNorm."""
    if not shape:
        return "scalar"
    if shape[-1] == 0:
        return "empty_reduce_dim"
    if any(d == 0 for d in shape):
        return "has_zero_dim"
    return None


def run_one(spec: dict[str, Any], seed: int, mps: MPSBackend) -> dict[str, Any]:
    shape = spec["shape"]
    dtype = spec["dtype"]
    dtype_name = spec["dtype_name"]
    cat = spec["stride_cat"]

    out: dict[str, Any] = {
        "shape": list(shape),
        "dtype": dtype_name,
        "stride_cat": cat,
        "bucket": spec["bucket"],
        "seed": seed,
    }

    skip = is_skippable_shape(shape)
    if skip is not None:
        out["status"] = "UNSUPPORTED"
        out["reason"] = skip
        return out

    # bf16 skip rule: if MPS reports bf16 unsupported. Apple Silicon supports
    # bf16 since Ventura; we only skip if the build can't allocate it.
    if dtype is torch.bfloat16:
        try:
            torch.zeros(1, dtype=torch.bfloat16, device="mps")
        except Exception as e:  # pragma: no cover  -- guard for old MPS builds
            out["status"] = "UNSUPPORTED"
            out["reason"] = f"bf16_mps_alloc_failed: {e!r}"[:200]
            return out

    try:
        x_cpu = fuzz_strides_for_category(shape, dtype, cat, device="cpu", seed=seed)
    except Exception as e:
        out["status"] = "ERROR_BUILD_CPU"
        out["reason"] = repr(e)[:200]
        return out

    try:
        x_mps = fuzz_strides_for_category(shape, dtype, cat, device="mps", seed=seed)
    except Exception as e:
        out["status"] = "ERROR_BUILD_MPS"
        out["reason"] = repr(e)[:200]
        return out

    try:
        y_cpu = rmsnorm(x_cpu)
    except Exception as e:
        out["status"] = "UNSUPPORTED_CPU"
        out["reason"] = repr(e)[:200]
        return out

    try:
        y_mps = rmsnorm(x_mps)
        mps.synchronize()
    except Exception as e:
        msg = repr(e)[:200]
        # Heuristic: MPS-side missing-op errors → UNSUPPORTED, not ERROR.
        if "MPS" in msg and ("not implemented" in msg or "doesn't support" in msg):
            out["status"] = "UNSUPPORTED_MPS"
        else:
            out["status"] = "UNSUPPORTED_MPS"
        out["reason"] = msg
        return out

    try:
        abs_err, rel_err, denom_mag = err_metrics(y_mps, y_cpu)
    except Exception as e:
        out["status"] = "ERROR_COMPARE"
        out["reason"] = repr(e)[:200]
        return out

    # Non-finite output on either side counts as a hard MPS bug separately.
    finite_mps = bool(torch.isfinite(y_mps.detach().cpu().to(torch.float32)).all().item())
    finite_cpu = bool(torch.isfinite(y_cpu.detach().cpu().to(torch.float32)).all().item())

    k_dim = shape[-1]
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")

    # Filtered ratios: rel_err is suppressed when denom is below floor.
    abs_factor = abs_err / atol if atol > 0 else float("inf")
    rel_factor = (rel_err / rtol) if (rtol > 0 and denom_mag >= DENOM_FLOOR) else 0.0
    max_factor = max(abs_factor, rel_factor)

    if not (finite_mps and finite_cpu):
        bucket = "NON_FINITE"
    elif max_factor > 10.0:
        bucket = "OVER_10X"
    elif max_factor > 1.0:
        bucket = "RECAL"
    else:
        bucket = "OK"

    out["status"] = bucket
    out["abs_err"] = abs_err
    out["rel_err"] = rel_err
    out["denom_mag"] = denom_mag
    out["atol"] = atol
    out["rtol"] = rtol
    out["k_dim"] = k_dim
    out["abs_factor"] = abs_factor
    out["rel_factor"] = rel_factor
    out["max_factor"] = max_factor
    out["finite_mps"] = finite_mps
    out["finite_cpu"] = finite_cpu
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mps = MPSBackend()
    if not mps.is_available():
        print("SKIPPED: torch.mps not available", file=sys.stderr)
        return 0

    print(
        f"torch={torch.__version__} mps_available=True "
        f"target_iterations={TARGET_ITERATIONS} seeds={SEEDS} configs={N_CONFIGS}",
        flush=True,
    )

    configs = build_configs(N_CONFIGS, CONFIG_SAMPLER_SEED)
    start = time.time()
    results: list[dict[str, Any]] = []

    # config_key -> {seed: result}
    by_config: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)

    attempted = 0
    for ci, spec in enumerate(configs):
        for seed in SEEDS:
            if time.time() - start > BUDGET_SECONDS:
                print(f"BUDGET EXPIRED at config={ci} seed={seed}", flush=True)
                break
            attempted += 1
            try:
                r = run_one(spec, seed, mps)
            except Exception as e:
                r = {
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype_name"],
                    "stride_cat": spec["stride_cat"],
                    "bucket": spec["bucket"],
                    "seed": seed,
                    "status": "FATAL",
                    "reason": repr(e)[:200],
                    "trace": traceback.format_exc()[-400:],
                }
                # Halt on unexpected fatal — per spec.
                print(f"FATAL at config={ci} seed={seed}: {r['reason']}", file=sys.stderr, flush=True)
                results.append(r)
                # Continue collecting to allow clean write-out, but mark as halted.
                break
            results.append(r)
            ck = (tuple(r["shape"]), r["dtype"], r["stride_cat"])
            by_config[ck][seed] = r

            if attempted <= 5 or r["status"] in ("OVER_10X", "RECAL", "NON_FINITE"):
                print(
                    f"  [{attempted:04d}] cfg={ci:03d} seed={seed} "
                    f"{r['status']:10s} shape={spec['shape']} "
                    f"dtype={spec['dtype_name']} stride={spec['stride_cat']} "
                    f"abs={r.get('abs_err','-')} rel={r.get('rel_err','-')} "
                    f"factor={r.get('max_factor','-')}",
                    flush=True,
                )
        else:
            continue
        break

    completed = sum(1 for r in results if r["status"] in ("OK", "RECAL", "OVER_10X", "NON_FINITE"))

    # Aggregate by config across seeds: count over_10x and recal frequency.
    filable_configs: list[dict[str, Any]] = []
    recal_configs: list[dict[str, Any]] = []
    nonfinite_configs: list[dict[str, Any]] = []
    for ck, by_seed in by_config.items():
        over10 = [r for r in by_seed.values() if r["status"] == "OVER_10X"]
        recals = [r for r in by_seed.values() if r["status"] == "RECAL"]
        nonfin = [r for r in by_seed.values() if r["status"] == "NON_FINITE"]
        if len(nonfin) >= 1:
            # Worst seed sample
            nonfin.sort(key=lambda r: r.get("max_factor", 0.0), reverse=True)
            nonfinite_configs.append({
                "config_key": ck,
                "n_nonfinite_seeds": len(nonfin),
                "worst": nonfin[0],
            })
        if len(over10) >= 3:
            over10.sort(key=lambda r: r.get("max_factor", 0.0), reverse=True)
            filable_configs.append({
                "config_key": ck,
                "n_seeds_over_10x": len(over10),
                "worst": over10[0],
                "all_factors": [r.get("max_factor", 0.0) for r in over10],
            })
        elif over10:
            # Treat 1-2 seed >10x as RECAL (not reproducible enough to file).
            over10.sort(key=lambda r: r.get("max_factor", 0.0), reverse=True)
            recal_configs.append({
                "config_key": ck,
                "n_seeds_over_10x": len(over10),
                "n_seeds_recal": len(recals),
                "worst": over10[0],
                "demoted_from": "OVER_10X_<3_seeds",
            })
        elif recals:
            recals.sort(key=lambda r: r.get("max_factor", 0.0), reverse=True)
            recal_configs.append({
                "config_key": ck,
                "n_seeds_recal": len(recals),
                "worst": recals[0],
            })

    # Global maxima across all completed runs.
    completed_runs = [r for r in results if r["status"] in ("OK", "RECAL", "OVER_10X")]
    max_abs = max((r.get("abs_err", 0.0) for r in completed_runs), default=0.0)
    # max_rel ignoring near-zero artifacts:
    max_rel_filtered = max(
        (r.get("rel_err", 0.0) for r in completed_runs
         if r.get("denom_mag", 0.0) >= DENOM_FLOOR),
        default=0.0,
    )
    max_rel_raw = max((r.get("rel_err", 0.0) for r in completed_runs), default=0.0)

    # Top 3 repros — prioritize FILABLE configs, fall back to RECAL.
    filable_configs.sort(key=lambda d: d["worst"].get("max_factor", 0.0), reverse=True)
    recal_configs.sort(key=lambda d: d["worst"].get("max_factor", 0.0), reverse=True)
    top_pool = filable_configs + recal_configs
    top3 = top_pool[:3]

    # Per-bucket breakdowns.
    per_bucket: dict[str, int] = defaultdict(int)
    per_dtype: dict[str, int] = defaultdict(int)
    per_stride: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    for r in results:
        per_bucket[r["bucket"]] += 1
        per_dtype[r["dtype"]] += 1
        per_stride[r["stride_cat"]] += 1
        status_counts[r["status"]] += 1

    elapsed = time.time() - start
    print(
        f"\nDone in {elapsed:.1f}s. attempted={attempted} completed={completed} "
        f"filable={len(filable_configs)} recal={len(recal_configs)} "
        f"nonfinite={len(nonfinite_configs)}",
        flush=True,
    )
    print(f"max_abs={max_abs:.3e} max_rel_filtered={max_rel_filtered:.3e} "
          f"max_rel_raw={max_rel_raw:.3e}", flush=True)
    print("status_counts:", dict(status_counts), flush=True)

    summary = {
        "kernel": "rmsnorm",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "divergences_filable": len(filable_configs),
        "divergences_recalibration": len(recal_configs),
        "non_finite_configs": len(nonfinite_configs),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel_filtered,
        "max_rel_err_raw_unfiltered": max_rel_raw,
        "status_counts": dict(status_counts),
        "per_bucket": dict(per_bucket),
        "per_dtype": dict(per_dtype),
        "per_stride_cat": dict(per_stride),
        "top_3_repros": [
            {
                "shape": list(d["worst"]["shape"]),
                "dtype": d["worst"]["dtype"],
                "stride_cat": d["worst"]["stride_cat"],
                "bucket": d["worst"]["bucket"],
                "abs_err": d["worst"].get("abs_err"),
                "rel_err": d["worst"].get("rel_err"),
                "denom_mag": d["worst"].get("denom_mag"),
                "atol": d["worst"].get("atol"),
                "rtol": d["worst"].get("rtol"),
                "max_factor": d["worst"].get("max_factor"),
                "n_seeds_over_10x": d.get("n_seeds_over_10x", 0),
                "n_seeds_recal": d.get("n_seeds_recal", 0),
                "verdict": ("FILABLE" if d in filable_configs else "RECALIBRATION"),
            }
            for d in top3
        ],
        "elapsed_seconds": round(elapsed, 2),
        "torch_version": torch.__version__,
        "device": "mps",
        "seeds": list(SEEDS),
        "config_count": N_CONFIGS,
        "denom_floor": DENOM_FLOOR,
        "filable_threshold": "max_factor > 10x AND reproduces on >= 3 seeds (denom-floor filter on rel)",
        "recal_threshold": "1x < max_factor <= 10x OR (>10x but <3 seeds)",
    }
    if nonfinite_configs:
        summary["non_finite_top"] = [
            {
                "shape": list(d["worst"]["shape"]),
                "dtype": d["worst"]["dtype"],
                "stride_cat": d["worst"]["stride_cat"],
                "n_nonfinite_seeds": d["n_nonfinite_seeds"],
            }
            for d in nonfinite_configs[:3]
        ]

    summary["recommended_filing_target"] = (
        "pytorch/pytorch" if (filable_configs or nonfinite_configs) else "none"
    )

    print(json.dumps(summary, indent=2, default=str), flush=True)

    # Persist full results for the writer.
    with open("/tmp/fuzz_rmsnorm_v2_summary.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
