"""Fuzz torch.nn.functional.log_softmax: MPS vs CPU reference.

Spec (kernel-fuzzer-log_softmax v2):
- 1000 iterations across seeds 0,1,2,3,4 (200 configs x 5 seeds)
- dtypes fp32/fp16/bf16 (skip if MPS unsupported)
- Divergence filtering:
  * max_rel_err > 10x tolerance counts only if denom_magnitude >= 1e-6
  * max_abs_err > 10x tolerance always counts
  * FILABLE only if reproducible across >=3 seeds
  * 1-5x tolerance => TOLERANCE_RECALIBRATION (xfail recommendation)
  * <1x tolerance => OK
- 12 minute wall budget; on timeout, write what we have and exit cleanly.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# Use the worktree's gpucheck source
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-log_softmax/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_log_softmax.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.log_softmax"
N_CONFIGS = 200
SEEDS = (0, 1, 2, 3, 4)
N_ITERS_TARGET = N_CONFIGS * len(SEEDS)
WALL_BUDGET_S = 12 * 60 - 45  # 11:15 hard cap; leave 45s for write/cleanup
CONFIG_RNG_SEED = 0xC0DEC0DE  # used only for config selection (not data RNG)

# Divergence multipliers
RECALIBRATION_LOWER = 1.0
RECALIBRATION_UPPER = 5.0
DIVERGENCE_FACTOR = 10.0
NEAR_ZERO_DENOM = 1e-6
REPRO_THRESHOLD = 3  # seeds out of 5

# ---- Sampling space (must include a non-empty last dim for log_softmax) ----
DEGENERATE = [(1,), (1, 1), (1, 1, 1), (16, 1), (1, 16)]  # all reducible
NON_TILE = [
    (TILE_SIZES[0] - 1,),       # 31
    (TILE_SIZES[0] + 1,),       # 33
    (TILE_SIZES[1] - 1, 16),    # 63x16
    (TILE_SIZES[1] + 3, 16),    # 67x16
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),  # 127x129
    (TILE_SIZES[2] + 1, 16),    # 129x16
    (8, 33),                    # batch with non-tile reduction dim
    (4, 8, 33),
]
PRIME_S = [(p,) for p in PRIMES] + [
    (PRIMES[0], PRIMES[1]),     # 7x13
    (PRIMES[2], 16),            # 31x16
    (4, PRIMES[3]),             # 4x127
    (8, 4, PRIMES[2]),          # 8x4x31
]
POW2_BOUNDARY = [(v,) for v in POWER_OF_2_BOUNDARIES] + [
    (4, 128), (4, 129), (4, 255), (4, 256), (4, 257), (4, 513),
]
LARGE = [(LARGE_DIMS[0],), (LARGE_DIMS[1],), (256, 256), (1024, 64), (16, 4096)]
MIXED = [(127, 16), (1024, 3), (7, 128), (33, 128, 4), (2, 5, 7, 11)]

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


def _max_rel_err_with_denom(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, |b| at the worst rel-err element)."""
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    abs_b = b.abs()
    diff = (a - b).abs()
    # Avoid /0; pick the worst rel-err with a guard, but report the
    # ACTUAL |b_i| at that index so the divergence-filter rule can decide.
    denom_safe = abs_b.clamp_min(1e-30)
    rel = diff / denom_safe
    flat_rel = rel.flatten()
    flat_b = abs_b.flatten()
    idx = int(flat_rel.argmax().item())
    return float(flat_rel[idx].item()), float(flat_b[idx].item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, "
        f"contig={t.is_contiguous()}"
    )


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _classify(
    abs_err: float, rel_err: float, denom: float, atol: float, rtol: float,
) -> str:
    """Return one of: ok, recalibration, divergence_candidate.

    Rules (per spec):
      - <1x tolerance => OK
      - 1-5x tolerance => RECALIBRATION
      - >10x tolerance => DIVERGENCE_CANDIDATE, but rel-err only counts
        if denom >= 1e-6 (otherwise near-zero artifact)
      - 5-10x or above-threshold-but-near-zero-denom => RECALIBRATION
        (treat as a softer signal, still worth an xfail conversation)
    """
    abs_factor = abs_err / atol if atol > 0 else float("inf")
    rel_factor = rel_err / rtol if rtol > 0 else float("inf")

    abs_diverge = abs_factor > DIVERGENCE_FACTOR
    rel_diverge = (rel_factor > DIVERGENCE_FACTOR) and (denom >= NEAR_ZERO_DENOM)

    if abs_diverge or rel_diverge:
        return "divergence_candidate"

    worst_factor = max(abs_factor, rel_factor)
    if worst_factor >= RECALIBRATION_LOWER:
        return "recalibration"
    return "ok"


def _build_configs(rng: "random.Random") -> list[dict]:
    """Pick N_CONFIGS deterministic (bucket, shape, dtype, stride_cat) rows."""
    configs: list[dict] = []
    for cfg_id in range(N_CONFIGS):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        # Skip shapes that are empty along the last dim (log_softmax of an
        # empty reduction dim is undefined / errors).
        if len(shape) == 0 or shape[-1] == 0:
            shape = (16,)
            bucket = "degenerate"
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        configs.append({
            "cfg_id": cfg_id,
            "bucket": bucket,
            "shape": tuple(shape),
            "dtype_name": dtype_name,
            "stride_cat": stride_cat,
        })
    return configs


def _run_one(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride_cat: str,
    seed: int,
) -> tuple[str, dict]:
    """Run a single (config, seed) trial.

    Returns (status, payload). status in:
      ok, recalibration, divergence_candidate, unsupported, skipped_empty
    payload is a dict with the metric/error info.
    """
    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "unsupported", {"phase": "build_cpu", "msg": str(exc)[:200]}

    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "unsupported", {"phase": "build_mps", "msg": str(exc)[:200]}

    if x_cpu.numel() == 0:
        return "skipped_empty", {}

    # log_softmax over last dim. Skip degenerate dim-of-size-0.
    if x_cpu.shape and x_cpu.shape[-1] == 0:
        return "skipped_empty", {}

    try:
        y_mps = F.log_softmax(x_mps, dim=-1)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "unsupported", {"phase": "mps", "msg": str(exc)[:200]}

    try:
        y_cpu = F.log_softmax(x_cpu, dim=-1)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "unsupported", {"phase": "cpu", "msg": str(exc)[:200]}

    abs_err = _max_abs_err(y_mps_cpu, y_cpu)
    rel_err, denom = _max_rel_err_with_denom(y_mps_cpu, y_cpu)

    # log_softmax reduces over the last dim -> use shape[-1] as k_dim
    k_dim = shape[-1] if shape else None
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
    status = _classify(abs_err, rel_err, denom, atol, rtol)

    return status, {
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "denom_magnitude": denom,
        "atol": atol,
        "rtol": rtol,
        "abs_factor": abs_err / atol if atol > 0 else float("inf"),
        "rel_factor": rel_err / rtol if rtol > 0 else float("inf"),
        "x_layout_cpu": _stride_tag(x_cpu),
        "x_layout_mps": _stride_tag(x_mps),
    }


def main() -> int:
    import random
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()

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
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    cfg_rng = random.Random(CONFIG_RNG_SEED)
    configs = _build_configs(cfg_rng)

    # results[cfg_id] = {seed -> (status, payload)}
    results: dict[int, dict[int, tuple[str, dict]]] = {c["cfg_id"]: {} for c in configs}
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_skipped_empty = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    timed_out = False

    for cfg in configs:
        if time.monotonic() - started > WALL_BUDGET_S:
            timed_out = True
            print(
                f"[budget] aborting at cfg {cfg['cfg_id']}/{len(configs)} "
                f"(elapsed {time.monotonic() - started:.1f}s)",
                file=sys.stderr,
            )
            break

        shape = cfg["shape"]
        dtype = DTYPES_BY_NAME[cfg["dtype_name"]]
        stride_cat = cfg["stride_cat"]

        for seed in SEEDS:
            if time.monotonic() - started > WALL_BUDGET_S:
                timed_out = True
                break
            iters_attempted += 1
            try:
                status, payload = _run_one(shape, dtype, stride_cat, seed)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Per spec: "Halt and report any process error immediately."
                print(
                    f"[ERROR] cfg={cfg['cfg_id']} seed={seed} "
                    f"shape={shape} dtype={cfg['dtype_name']} "
                    f"stride={stride_cat}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                _emit_partial(
                    results, configs, iters_attempted, iters_completed,
                    iters_unsupported, iters_skipped_empty,
                    max_abs_err_global, max_rel_err_global,
                    started, error=repr(exc),
                )
                return 2

            results[cfg["cfg_id"]][seed] = (status, payload)
            if status == "unsupported":
                iters_unsupported += 1
            elif status == "skipped_empty":
                iters_skipped_empty += 1
            else:
                iters_completed += 1
                if "max_abs_err" in payload:
                    max_abs_err_global = max(max_abs_err_global, payload["max_abs_err"])
                    max_rel_err_global = max(max_rel_err_global, payload["max_rel_err"])

    elapsed = time.monotonic() - started
    _emit_partial(
        results, configs, iters_attempted, iters_completed,
        iters_unsupported, iters_skipped_empty,
        max_abs_err_global, max_rel_err_global,
        started, error=None, timed_out=timed_out,
    )
    print(
        f"[done] iters_attempted={iters_attempted} "
        f"iters_completed={iters_completed} elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


def _emit_partial(
    results: dict[int, dict[int, tuple[str, dict]]],
    configs: list[dict],
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_skipped_empty: int,
    max_abs_err_global: float,
    max_rel_err_global: float,
    started: float,
    *,
    error: str | None = None,
    timed_out: bool = False,
) -> None:
    """Aggregate per-config across seeds and write outputs."""
    elapsed = time.monotonic() - started

    # Per-config aggregation
    filable: list[dict] = []
    recalibration: list[dict] = []
    for cfg in configs:
        seed_results = results[cfg["cfg_id"]]
        if not seed_results:
            continue
        statuses = {s: r[0] for s, r in seed_results.items()}
        payloads = {s: r[1] for s, r in seed_results.items() if r[1] and "max_abs_err" in r[1]}
        if not payloads:
            continue

        n_div = sum(1 for st in statuses.values() if st == "divergence_candidate")
        n_recal = sum(1 for st in statuses.values() if st == "recalibration")

        worst_abs = max(p["max_abs_err"] for p in payloads.values())
        worst_rel = max(p["max_rel_err"] for p in payloads.values())
        worst_abs_factor = max(p["abs_factor"] for p in payloads.values())
        worst_rel_factor = max(p["rel_factor"] for p in payloads.values())
        # Take the seed with the largest abs_factor for the layout / repro field
        repro_seed = max(
            payloads, key=lambda s: payloads[s]["abs_factor"],
        )
        repro_payload = payloads[repro_seed]

        record = {
            "cfg_id": cfg["cfg_id"],
            "shape": list(cfg["shape"]),
            "shape_bucket": cfg["bucket"],
            "dtype": cfg["dtype_name"],
            "stride_category": cfg["stride_cat"],
            "n_seeds_run": len(seed_results),
            "n_seeds_divergence_candidate": n_div,
            "n_seeds_recalibration": n_recal,
            "max_abs_err": worst_abs,
            "max_rel_err": worst_rel,
            "max_abs_factor": worst_abs_factor,
            "max_rel_factor": worst_rel_factor,
            "denom_magnitude": repro_payload["denom_magnitude"],
            "atol": repro_payload["atol"],
            "rtol": repro_payload["rtol"],
            "x_layout_cpu": repro_payload["x_layout_cpu"],
            "x_layout_mps": repro_payload["x_layout_mps"],
            "repro_seed": repro_seed,
        }

        if n_div >= REPRO_THRESHOLD:
            filable.append(record)
        elif n_div + n_recal >= 1:
            # any recalibration tier signal -> add to recalibration bucket
            recalibration.append(record)

    # Sort by (severity, smallness) for "minimal repro" ordering
    filable.sort(
        key=lambda r: (-r["max_abs_factor"], -r["max_rel_factor"], _numel(tuple(r["shape"]))),
    )
    recalibration.sort(
        key=lambda r: (-r["max_abs_factor"], -r["max_rel_factor"], _numel(tuple(r["shape"]))),
    )
    top3 = filable[:3] if filable else recalibration[:3]

    summary = {
        "kernel": KERNEL,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_skipped_empty": iters_skipped_empty,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "elapsed_seconds": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "n_configs_planned": N_CONFIGS,
        "timed_out": timed_out,
        "error": error,
        "top_3_repros": top3,
    }

    _write_markdown(RESULTS_MD, summary, filable, recalibration)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(
    path: Path, s: dict, filable: list[dict], recalibration: list[dict],
) -> None:
    lines: list[str] = []
    lines.append(f"# Fuzz results - kernel: `{s['kernel']}`")
    lines.append("")
    lines.append("**Backends:** MPS (real, Apple Silicon) vs CPU reference (same dtype).")
    lines.append("CUDA: not present - mocked / not compared.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- kernel: `{s['kernel']}`")
    lines.append(
        f"- iters_attempted: **{s['iters_attempted']}** / "
        f"target {N_ITERS_TARGET}"
    )
    lines.append(f"- iters_completed: **{s['iters_completed']}**")
    lines.append(f"- iters_unsupported: {s['iters_unsupported']}")
    lines.append(f"- iters_skipped_empty: {s['iters_skipped_empty']}")
    lines.append(f"- divergences_filable: **{s['divergences_filable']}**")
    lines.append(f"- divergences_recalibration: **{s['divergences_recalibration']}**")
    lines.append(f"- max_abs_err: `{s['max_abs_err']:.3e}`")
    lines.append(f"- max_rel_err: `{s['max_rel_err']:.3e}`")
    lines.append(f"- runtime: `{s['elapsed_seconds']}s` (budget {WALL_BUDGET_S}s, "
                 f"timed_out={s['timed_out']})")
    lines.append(f"- torch: `{s['torch_version']}`, host: {s['host']}")
    lines.append(f"- seeds: {s['seeds']}, configs planned: {s['n_configs_planned']}")
    if s.get("error"):
        lines.append(f"- **ERROR:** `{s['error']}`")
    lines.append("")

    lines.append("## Divergence-filtering rules (spec)")
    lines.append("")
    lines.append(
        "- **OK** if max_abs_err < 1x atol AND max_rel_err < 1x rtol."
    )
    lines.append(
        "- **TOLERANCE_RECALIBRATION** if 1x <= worst factor <= 5x "
        "(recommend xfail entry)."
    )
    lines.append(
        "- **FILABLE** only if max_abs_factor > 10x OR "
        "(max_rel_factor > 10x AND denom_magnitude >= 1e-6) AND the same "
        "(shape, dtype, stride_category) reproduces in >=3 of 5 seeds."
    )
    lines.append("")

    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append(
            "_None - every (shape, dtype, stride_category) trial stayed within "
            "1x the MPS-overlay tolerance from `gpucheck.assertions.tolerances`._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            tier = "FILABLE" if r in filable else "RECALIBRATION"
            lines.append(f"### Repro #{i} ({tier})")
            lines.append("")
            lines.append(
                f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)"
            )
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(
                f"- **max abs err:** {r['max_abs_err']:.3e}  "
                f"(atol={r['atol']:.2e}, factor={r['max_abs_factor']:.2f}x)"
            )
            lines.append(
                f"- **max rel err:** {r['max_rel_err']:.3e}  "
                f"(rtol={r['rtol']:.2e}, factor={r['max_rel_factor']:.2f}x)"
            )
            lines.append(f"- **denom_magnitude:** {r['denom_magnitude']:.3e}")
            lines.append(
                f"- **seed reproducibility:** "
                f"{r['n_seeds_divergence_candidate']}/5 seeds = "
                f"divergence_candidate, "
                f"{r['n_seeds_recalibration']}/5 = recalibration"
            )
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append(f"- **worst-seed:** {r['repro_seed']}")
            lines.append("")

    if recalibration:
        lines.append("## Recalibration candidates (xfail recommendations)")
        lines.append("")
        lines.append(
            "| dtype | shape | stride | abs_factor | rel_factor | "
            "n_div_seeds | n_recal_seeds |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|"
        )
        for r in recalibration[:20]:
            lines.append(
                f"| {r['dtype']} | {tuple(r['shape'])} | {r['stride_category']} | "
                f"{r['max_abs_factor']:.2f} | {r['max_rel_factor']:.2f} | "
                f"{r['n_seeds_divergence_candidate']} | "
                f"{r['n_seeds_recalibration']} |"
            )
        lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Reference: `torch.nn.functional.log_softmax(x_cpu, dim=-1)` at the "
        "SAME dtype as MPS."
    )
    lines.append(
        "- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, "
        "k_dim=shape[-1], device_type='mps')` (k_dim from the reduction axis)."
    )
    lines.append(
        "- Per-seed classification: ok / recalibration / divergence_candidate. "
        "Per-config: 5 seeds total, FILABLE iff >=3 seeds are "
        "divergence_candidate."
    )
    lines.append(
        "- Stride categories: row_major, column_major, broadcast, transpose, "
        "slice, non_contig, gather (see `gpucheck.fuzzing.strides`)."
    )
    lines.append(
        "- log_softmax over the last dim is exercised (well-defined for any "
        "shape with shape[-1] >= 1)."
    )

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
