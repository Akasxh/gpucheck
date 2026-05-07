"""Fuzz torch.nn.functional.mish: MPS vs CPU reference (v2 spec).

v2 divergence filtering:
  - max_rel_err > 10x tol counts only if denom_magnitude >= 1e-6
  - max_abs_err > 10x tol always counts
  - >=3 reproducing seeds => FILABLE
  - 1-5x tol => TOLERANCE_RECALIBRATION
  - <1x tol => OK
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-mish/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_mish.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.mish"
N_ITERS = 1000
SEEDS_BASE = [0, 1, 2, 3, 4]
WALL_BUDGET_S = 12 * 60 - 60  # leave 60 s for results

# Shape buckets — keep modest, mish is elementwise
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


def _max_rel_err_and_denom(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_max).

    denom_magnitude = |b| at the index of max relative error. This lets us
    suppress relative-error 'spikes' that come from comparing against tiny
    reference values (near-zero-denominator artifact).
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    rel = diff / denom
    idx = rel.argmax()
    return float(rel.flatten()[idx].item()), float(b.abs().flatten()[idx].item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _config_key(shape: tuple[int, ...], dtype_name: str, stride_cat: str) -> str:
    return f"{tuple(shape)}|{dtype_name}|{stride_cat}"


def _classify(
    abs_err: float, rel_err: float, denom: float, atol: float, rtol: float,
) -> str:
    """Return one of: OK, RECALIBRATION, DIVERGENCE_CANDIDATE."""
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = rel_err / rtol if rtol > 0 else 0.0

    # Suppress relative-spike-from-tiny-denom artifact
    rel_counts = denom >= 1e-6

    abs_div = abs_ratio > 10.0
    rel_div = rel_counts and (rel_ratio > 10.0)
    if abs_div or rel_div:
        return "DIVERGENCE_CANDIDATE"

    abs_recal = 1.0 < abs_ratio <= 5.0
    rel_recal = rel_counts and (1.0 < rel_ratio <= 5.0)
    if abs_recal or rel_recal:
        return "RECALIBRATION"

    return "OK"


def _run_once(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride_cat: str,
    seed: int,
) -> dict | None:
    """Run a single fuzz iteration. Returns metrics dict or None if unsupported."""
    try:
        x_cpu = fuzz_strides_for_category(shape, dtype, stride_cat, device="cpu", seed=seed)
        try:
            x_mps = fuzz_strides_for_category(shape, dtype, stride_cat, device="mps", seed=seed)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            return {"unsupported": True, "stage": "build_mps", "msg": str(exc).splitlines()[0][:200]}

        if x_cpu.numel() == 0:
            return {"empty": True, "x_cpu_layout": _stride_tag(x_cpu), "x_mps_layout": _stride_tag(x_mps)}

        try:
            y_mps = F.mish(x_mps)
            torch.mps.synchronize()
            y_mps_cpu = y_mps.detach().to("cpu")
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            return {"unsupported": True, "stage": "run_mps", "msg": str(exc).splitlines()[0][:200]}

        try:
            y_cpu = F.mish(x_cpu)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            return {"unsupported": True, "stage": "run_cpu", "msg": str(exc).splitlines()[0][:200]}

        rel, denom = _max_rel_err_and_denom(y_mps_cpu, y_cpu)
        absdiff = _max_abs_err(y_mps_cpu, y_cpu)
        return {
            "ok": True,
            "max_abs_err": absdiff,
            "max_rel_err": rel,
            "denom_at_max_rel": denom,
            "x_cpu_layout": _stride_tag(x_cpu),
            "x_mps_layout": _stride_tag(x_mps),
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"error": True, "msg": repr(exc)[:300]}


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
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz - SKIPPED (MPS not available)\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    rng = random.Random(0xBEEF)  # deterministic config sampler
    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_errored = 0
    iters_empty = 0

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0  # only counts rel errs that pass denom filter

    # config -> dict of per-classification per-seed metric records
    # config_key -> {"shape": ..., "dtype": ..., "stride_cat": ...,
    #                "candidates": {seed -> rec}, "recalibrations": {seed -> rec}}
    config_state: dict[str, dict] = {}

    per_bucket = {b: 0 for b in BUCKET_NAMES}
    per_dtype = {d: 0 for d in DTYPE_NAMES}
    per_stride = {s: 0 for s in STRIDE_CATEGORIES}

    for i in range(N_ITERS):
        if time.monotonic() - started > WALL_BUDGET_S:
            print(f"[budget] aborting at iter {i}/{N_ITERS}", file=sys.stderr)
            break

        iters_attempted += 1

        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        dtype = DTYPES_BY_NAME[dtype_name]
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        # cycle seeds 0..4
        seed = SEEDS_BASE[i % len(SEEDS_BASE)]

        per_bucket[bucket] += 1
        per_dtype[dtype_name] += 1
        per_stride[stride_cat] += 1

        out = _run_once(shape, dtype, stride_cat, seed)
        if out is None:
            iters_errored += 1
            continue
        if out.get("error"):
            print(f"[ERROR] iter={i} {bucket}/{dtype_name}/{stride_cat} shape={shape} seed={seed}: {out['msg']}", file=sys.stderr)
            iters_errored += 1
            # Halt rule: report and exit non-zero so caller sees it.
            return 2
        if out.get("unsupported"):
            iters_unsupported += 1
            print(f"[unsupported-{out['stage']}] iter={i} {bucket}/{dtype_name}/{stride_cat} shape={shape}: {out['msg']}", file=sys.stderr)
            continue
        if out.get("empty"):
            iters_completed += 1
            iters_empty += 1
            continue

        iters_completed += 1
        atol, rtol = compute_tolerance(dtype, device_type="mps")
        abs_err = out["max_abs_err"]
        rel_err = out["max_rel_err"]
        denom = out["denom_at_max_rel"]
        max_abs_err_global = max(max_abs_err_global, abs_err)
        # only count rel_err in global if denom passes filter
        if denom >= 1e-6:
            max_rel_err_global = max(max_rel_err_global, rel_err)

        verdict = _classify(abs_err, rel_err, denom, atol, rtol)
        if verdict == "OK":
            continue

        key = _config_key(shape, dtype_name, stride_cat)
        st = config_state.setdefault(key, {
            "shape": list(shape),
            "shape_bucket": bucket,
            "dtype": dtype_name,
            "stride_cat": stride_cat,
            "atol": atol,
            "rtol": rtol,
            "candidates": {},
            "recalibrations": {},
        })
        rec = {
            "iter": i,
            "seed": seed,
            "max_abs_err": abs_err,
            "max_rel_err": rel_err,
            "denom_at_max_rel": denom,
            "atol": atol,
            "rtol": rtol,
            "abs_ratio": abs_err / atol if atol > 0 else 0.0,
            "rel_ratio": (rel_err / rtol if rtol > 0 else 0.0) if denom >= 1e-6 else 0.0,
            "x_cpu_layout": out["x_cpu_layout"],
            "x_mps_layout": out["x_mps_layout"],
        }
        if verdict == "DIVERGENCE_CANDIDATE":
            st["candidates"][seed] = rec
            print(f"[CAND] iter={i} {bucket}/{dtype_name}/{stride_cat} shape={shape} seed={seed} abs={abs_err:.3e}({abs_err/atol:.1f}x) rel={rel_err:.3e}({rel_err/rtol:.1f}x) denom={denom:.2e}", file=sys.stderr)
        elif verdict == "RECALIBRATION":
            st["recalibrations"][seed] = rec

    # --- Reproduction pass: for each config with candidates, fill in missing seeds ---
    repro_extra_iters = 0
    for key, st in list(config_state.items()):
        if not st["candidates"]:
            continue
        if time.monotonic() - started > WALL_BUDGET_S:
            break
        for seed in SEEDS_BASE:
            if seed in st["candidates"]:
                continue
            if time.monotonic() - started > WALL_BUDGET_S:
                break
            shape_t = tuple(st["shape"])
            dtype = DTYPES_BY_NAME[st["dtype"]]
            stride_cat = st["stride_cat"]
            out = _run_once(shape_t, dtype, stride_cat, seed)
            repro_extra_iters += 1
            if out is None or out.get("error") or out.get("unsupported") or out.get("empty"):
                continue
            atol = st["atol"]; rtol = st["rtol"]
            abs_err = out["max_abs_err"]; rel_err = out["max_rel_err"]; denom = out["denom_at_max_rel"]
            if denom >= 1e-6:
                max_rel_err_global = max(max_rel_err_global, rel_err)
            max_abs_err_global = max(max_abs_err_global, abs_err)
            verdict = _classify(abs_err, rel_err, denom, atol, rtol)
            if verdict == "DIVERGENCE_CANDIDATE":
                st["candidates"][seed] = {
                    "iter": -1,  # repro pass
                    "seed": seed,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "denom_at_max_rel": denom,
                    "atol": atol, "rtol": rtol,
                    "abs_ratio": abs_err / atol if atol > 0 else 0.0,
                    "rel_ratio": (rel_err / rtol if rtol > 0 else 0.0) if denom >= 1e-6 else 0.0,
                    "x_cpu_layout": out["x_cpu_layout"],
                    "x_mps_layout": out["x_mps_layout"],
                }

    # --- Classify final divergences ---
    filable: list[dict] = []
    recalibration: list[dict] = []
    for key, st in config_state.items():
        seeds_with_div = sorted(st["candidates"].keys())
        if len(seeds_with_div) >= 3:
            # FILABLE: pick worst record as representative
            worst = max(st["candidates"].values(),
                        key=lambda r: max(r["abs_ratio"], r["rel_ratio"]))
            filable.append({
                "config_key": key,
                "shape": st["shape"],
                "shape_bucket": st["shape_bucket"],
                "dtype": st["dtype"],
                "stride_cat": st["stride_cat"],
                "seeds_reproducing": seeds_with_div,
                "n_seeds_reproducing": len(seeds_with_div),
                "worst_seed_record": worst,
            })
        elif st["candidates"]:
            # Saw >10x but not reproducible across >=3 seeds. Demote to recalibration only
            # if the best seed lands in the 1-5x band; otherwise it stays as a "single-seed
            # spike" which we report but don't file.
            worst = max(st["candidates"].values(),
                        key=lambda r: max(r["abs_ratio"], r["rel_ratio"]))
            recalibration.append({
                "config_key": key,
                "shape": st["shape"],
                "shape_bucket": st["shape_bucket"],
                "dtype": st["dtype"],
                "stride_cat": st["stride_cat"],
                "seeds_reproducing": seeds_with_div,
                "n_seeds_reproducing": len(seeds_with_div),
                "worst_seed_record": worst,
                "demoted_reason": "fewer than 3 seeds reproduce >10x error",
            })
        elif st["recalibrations"]:
            worst = max(st["recalibrations"].values(),
                        key=lambda r: max(r["abs_ratio"], r["rel_ratio"]))
            seeds_seen = sorted(st["recalibrations"].keys())
            recalibration.append({
                "config_key": key,
                "shape": st["shape"],
                "shape_bucket": st["shape_bucket"],
                "dtype": st["dtype"],
                "stride_cat": st["stride_cat"],
                "seeds_reproducing": seeds_seen,
                "n_seeds_reproducing": len(seeds_seen),
                "worst_seed_record": worst,
            })

    # Sort filable by worst ratio (descending), tiebreak by smallest numel
    def _numel(s: list[int]) -> int:
        n = 1
        for d in s:
            n *= max(d, 1)
        return n

    filable.sort(
        key=lambda d: (
            -max(d["worst_seed_record"]["abs_ratio"], d["worst_seed_record"]["rel_ratio"]),
            _numel(d["shape"]),
        ),
    )
    top_3_repros = filable[:3] if filable else recalibration[:3]

    elapsed = time.monotonic() - started

    summary = {
        "kernel": KERNEL,
        "status": "OK",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_errored": iters_errored,
        "iters_empty_skipped": iters_empty,
        "repro_extra_iters": repro_extra_iters,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": top_3_repros,
        "per_shape_bucket": per_bucket,
        "per_dtype": per_dtype,
        "per_stride_category": per_stride,
        "wall_time_sec": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": SEEDS_BASE,
        "iters_target": N_ITERS,
        "filtering_rules": {
            "divergence_threshold": "10x tolerance",
            "rel_err_denom_floor": 1e-6,
            "recalibration_band_x_tol": [1.0, 5.0],
            "filable_min_seeds": 3,
        },
        "recommended_filing_target": "pytorch/pytorch" if filable else "none",
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    return 0


def _write_markdown(path: Path, s: dict) -> None:
    L = []
    L.append(f"# {s['kernel']} - MPS fuzz report (v2)")
    L.append("")
    L.append(f"- **Kernel:** `{s['kernel']}`")
    L.append(f"- **Status:** {s['status']}")
    L.append(f"- **Iterations attempted:** {s['iters_attempted']} / {s['iters_target']} target")
    L.append(f"- **Iterations completed:** {s['iters_completed']}")
    L.append(f"- **Iterations unsupported:** {s['iters_unsupported']}")
    L.append(f"- **Iterations errored:** {s['iters_errored']}")
    L.append(f"- **Iterations empty (numel==0, skipped):** {s['iters_empty_skipped']}")
    L.append(f"- **Reproduction extra iters:** {s['repro_extra_iters']}")
    L.append(f"- **Divergences (FILABLE, >=3 seeds, >10x tol, denom-filtered):** {s['divergences_filable']}")
    L.append(f"- **Divergences (TOLERANCE_RECALIBRATION, 1-5x tol or non-reproducible):** {s['divergences_recalibration']}")
    L.append(f"- **MPS-vs-CPU max absolute error:** {s['max_abs_err']:.3e}")
    L.append(f"- **MPS-vs-CPU max relative error (denom-filtered):** {s['max_rel_err']:.3e}")
    L.append(f"- **MPS-vs-CUDA max relative error:** N/A (CUDA mocked - no NVIDIA GPU on host)")
    L.append(f"- **Recommended filing target:** `{s['recommended_filing_target']}`")
    L.append(f"- **Elapsed:** {s['wall_time_sec']} s")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}, seeds: {s['seeds']}")
    L.append("")
    L.append("## Sampling distribution")
    L.append("")
    L.append("| dimension | counts |")
    L.append("|---|---|")
    L.append(f"| shape bucket | {s['per_shape_bucket']} |")
    L.append(f"| dtype | {s['per_dtype']} |")
    L.append(f"| stride category | {s['per_stride_category']} |")
    L.append("")
    L.append("## Filtering rules (v2)")
    L.append("")
    L.append("- A divergence candidate requires `max_abs_err > 10x atol` OR")
    L.append("  (`max_rel_err > 10x rtol` AND `denom_magnitude >= 1e-6`).")
    L.append("- A FILABLE divergence has the same (shape, dtype, stride_category) tuple")
    L.append("  reproducing at >10x tolerance on >=3 of seeds {0,1,2,3,4}.")
    L.append("- A TOLERANCE_RECALIBRATION entry sits in the 1-5x band, or shows >10x on")
    L.append("  fewer than 3 seeds (single-seed spikes / non-reproducible).")
    L.append("- Anything below 1x tolerance is OK and not surfaced.")
    L.append("")
    L.append("## Top 3 minimal repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No divergences exceeded gpucheck's per-dtype tolerance "
                 "(MPS 2x multiplier applied; see `assertions/tolerances.py`)._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            w = r["worst_seed_record"]
            L.append(f"### Repro #{i}")
            L.append("")
            L.append(f"- **shape:** `{tuple(r['shape'])}` (bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride category:** `{r['stride_cat']}`")
            L.append(f"- **seeds reproducing:** {r['seeds_reproducing']} ({r['n_seeds_reproducing']}/5)")
            L.append(f"- **worst seed:** {w['seed']}")
            L.append(f"- **max abs err:** {w['max_abs_err']:.3e} (atol={w['atol']:.2e}, ratio={w['abs_ratio']:.2f}x)")
            L.append(f"- **max rel err:** {w['max_rel_err']:.3e} (rtol={w['rtol']:.2e}, ratio={w['rel_ratio']:.2f}x)")
            L.append(f"- **denom at max-rel:** {w['denom_at_max_rel']:.3e}")
            L.append(f"- **CPU layout:** `{w['x_cpu_layout']}`")
            L.append(f"- **MPS layout:** `{w['x_mps_layout']}`")
            L.append("")
    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.mish` on CPU (FP32-promoted comparison).")
    L.append("- Tolerance source: `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (CUDA base tol x MPS 2x multiplier).")
    L.append("- mish is elementwise; no sqrt(k/128) matmul scaling applied.")
    L.append("- Stride categories sampled: row_major, column_major, broadcast,")
    L.append("  transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).")
    L.append("- `torch.mps.synchronize()` is called after each MPS forward to defeat")
    L.append("  async kernel completion before pulling the result back to CPU for compare.")
    L.append("- CUDA mocked: no NVIDIA GPU on host - cross-device CUDA-vs-MPS comparison is N/A.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
