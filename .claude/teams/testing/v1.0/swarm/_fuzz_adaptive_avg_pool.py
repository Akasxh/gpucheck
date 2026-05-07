"""V2 fuzz: adaptive_avg_pool2d on MPS vs CPU reference.

Targets ``torch.nn.functional.adaptive_avg_pool2d`` which expects a
4-D ``(N, C, H, W)`` input and an ``output_size`` (int | tuple).
Spec parity with _fuzz_gelu_v2.py:

- 500 configs sampled deterministically with config-RNG (per task brief).
- Each config replayed across 5 data seeds (0,1,2,3,4).
- Per-run buckets:
    * abs > 10·atol AND rel > 10·rtol AND |y_ref|@argmax_rel >= 1e-6
                                 -> FILABLE_HIT
    * abs > 10·atol only          -> CRIT_ABS_ONLY
    * rel > 10·rtol (denom-gated) -> CRIT_REL_ONLY
    * abs in [atol, 5·atol)       -> RECAL
    * rel in [rtol, 5·rtol)
      AND |y_ref|@argmax_rel
        >= 1e-6                   -> RECAL
    * else                        -> OK
- Per-config verdict:
    * FILABLE       -> FILABLE_HIT on >= 3 / 5 seeds.
    * RECALIBRATION -> RECAL on >= 3 / 5 seeds (CRIT_*_ONLY rolled in
                       when reproduced >= 3 seeds), and not FILABLE.
- 8 minute wall budget (per task brief). Writes partial results on
  time-out.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-adaptive-avg-pool/src")
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_adaptive-avg-pool.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.adaptive_avg_pool2d"
N_CONFIGS = 500
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # 8min budget, leave 30s for writing
CONFIG_SEED = 0xADAA1AAB  # adaptive-avg-pool

# ---------------------------------------------------------------------------
# Sampling space — 4-D shapes (N, C, H, W) with H,W >= 1.
# Buckets mirror the v2 conventions but specialised to pooling-valid 4-D.
# ---------------------------------------------------------------------------

DEGENERATE: list[tuple[int, ...]] = [
    (1, 1, 1, 1), (1, 1, 1, 4), (1, 1, 4, 1), (1, 4, 1, 1),
    (16, 1, 1, 1), (1, 1, 2, 2),
]
NON_TILE: list[tuple[int, ...]] = [
    (1, 16, 17, 17), (1, 16, 31, 33), (1, 32, 31, 31), (1, 4, 33, 33),
    (1, 8, 65, 63), (2, 17, 33, 17),
]
PRIMES: list[tuple[int, ...]] = [
    (1, 7, 11, 13), (1, 11, 17, 19), (3, 5, 7, 11), (1, 13, 23, 29),
    (1, 7, 7, 7), (2, 3, 5, 7),
]
POW2_BOUNDARY: list[tuple[int, ...]] = [
    (1, 32, 32, 32), (1, 64, 63, 63), (1, 64, 65, 65), (1, 16, 128, 128),
    (1, 32, 127, 129), (1, 8, 256, 256),
]
LARGE: list[tuple[int, ...]] = [
    (1, 16, 224, 224), (4, 32, 112, 112), (1, 3, 256, 256),
    (1, 64, 56, 56), (2, 128, 28, 28),
]
MIXED: list[tuple[int, ...]] = [
    (2, 16, 56, 56), (4, 64, 14, 14), (8, 128, 7, 7),
    (1, 64, 35, 47), (3, 11, 19, 23),
]

SHAPE_BUCKETS: dict[str, list[tuple[int, ...]]] = {
    "degenerate": DEGENERATE,
    "non_tile_aligned": NON_TILE,
    "prime": PRIMES,
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

# Output sizes exercise: global-pool, divides-cleanly, indivisible, equal-to-input.
OUTPUT_PATTERNS: list[str] = [
    "global",         # (1, 1)
    "tiny",           # (2, 2)
    "small",          # (7, 7)
    "half",           # (H//2, W//2)
    "asymmetric",     # (1, W) / (H, 1) — collapse one axis
    "indivisible",    # (h_out, w_out) where input not divisible by output
    "identity",       # (H, W) — passthrough; tests trivial cases
]


def _resolve_output_size(
    shape: tuple[int, ...], pattern: str, rng: random.Random,
) -> tuple[int, int] | None:
    """Resolve a pattern label into a concrete (h_out, w_out) for this shape.

    Returns None if the pattern is incompatible with the shape (e.g. half of
    a degenerate H=1 input is still 1, which is fine; but indivisible
    requires H >= 2). Caller must skip None.
    """
    H, W = shape[-2], shape[-1]
    if H < 1 or W < 1:
        return None
    if pattern == "global":
        return (1, 1)
    if pattern == "tiny":
        return (min(2, H), min(2, W))
    if pattern == "small":
        return (min(7, H), min(7, W))
    if pattern == "half":
        return (max(1, H // 2), max(1, W // 2))
    if pattern == "asymmetric":
        # Pick which axis collapses.
        if rng.random() < 0.5:
            return (1, W)
        return (H, 1)
    if pattern == "indivisible":
        # Pick an output size that does not evenly divide H or W.
        # Skip if H or W < 2 (only divisor is 1).
        if H < 3 and W < 3:
            return None
        h_out = max(1, H - 1) if H >= 2 else 1
        w_out = max(1, W - 1) if W >= 2 else 1
        # Prefer non-divisors; nudge if accidentally divides.
        if H >= 4 and H % h_out == 0:
            h_out = H - 2 if H - 2 >= 1 else 1
        if W >= 4 and W % w_out == 0:
            w_out = W - 2 if W - 2 >= 1 else 1
        return (max(1, h_out), max(1, w_out))
    if pattern == "identity":
        return (H, W)
    return None


def _err_metrics(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float, float]:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a - b).abs()
    abs_err = float(diff.max().item())
    denom = b.abs()
    safe = denom.clamp_min(1e-12)
    rel = diff / safe
    rel_flat = rel.reshape(-1)
    idx = int(rel_flat.argmax().item())
    rel_err = float(rel_flat[idx].item())
    denom_at_relmax = float(denom.reshape(-1)[idx].item())
    return abs_err, rel_err, denom_at_relmax


def _classify(
    abs_err: float, rel_err: float, denom_at_relmax: float,
    atol: float, rtol: float,
) -> str:
    rel_gate = denom_at_relmax >= 1e-6
    abs_critical = abs_err > 10.0 * atol
    rel_critical = rel_err > 10.0 * rtol and rel_gate
    if abs_critical and rel_critical:
        return "FILABLE_HIT"
    if abs_critical:
        return "CRIT_ABS_ONLY"
    if rel_critical:
        return "CRIT_REL_ONLY"
    abs_recal = atol <= abs_err < 5.0 * atol
    rel_recal = (rtol <= rel_err < 5.0 * rtol) and rel_gate
    if abs_recal or rel_recal:
        return "RECAL"
    return "OK"


def _sample_configs(rng: random.Random, n: int) -> list[dict]:
    cfgs = []
    for cid in range(n):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        out_pattern = rng.choice(OUTPUT_PATTERNS)
        cfgs.append({
            "cid": cid,
            "shape": tuple(shape),
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "shape_bucket": bucket,
            "output_pattern": out_pattern,
        })
    return cfgs


def _run_one(
    cfg: dict, data_seed: int, rng: random.Random,
) -> tuple[str, float, float, float, float, float, str | None,
           tuple[int, int] | None]:
    """Run a single (config, seed) pair.

    Returns (status, abs_err, rel_err, denom_at_relmax, atol, rtol,
             unsupported_reason, resolved_output_size).
    status in {OK, RECAL, CRIT_*, FILABLE_HIT, UNSUPPORTED, EMPTY, SKIP}.
    """
    shape = cfg["shape"]
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]
    out_size = _resolve_output_size(shape, cfg["output_pattern"], rng)
    if out_size is None:
        return "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0, "incompatible-output-size", None

    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return ("UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0,
                f"cpu-build:{exc!s:.180}", out_size)

    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return ("UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0,
                f"mps-build:{exc!s:.180}", out_size)

    if x_cpu.numel() == 0:
        return "EMPTY", 0.0, 0.0, 0.0, 0.0, 0.0, None, out_size

    try:
        y_mps = F.adaptive_avg_pool2d(x_mps, out_size)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return ("UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0,
                f"mps-op:{exc!s:.180}", out_size)

    try:
        y_cpu = F.adaptive_avg_pool2d(x_cpu, out_size)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return ("UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0,
                f"cpu-op:{exc!s:.180}", out_size)

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    # adaptive_avg_pool reduces over a kH·kW window; treat that window as
    # the k-dim for tolerance scaling so giant-pool configs (e.g.
    # 256×256 -> 1×1, k=65536) get a sqrt(k/128) atol bump.
    H, W = shape[-2], shape[-1]
    h_out, w_out = out_size
    avg_k = max(1, (H // max(1, h_out)) * (W // max(1, w_out)))
    atol, rtol = compute_tolerance(dtype, k_dim=avg_k, device_type="mps")
    status = _classify(abs_err, rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None, out_size


def _emit_skipped(reason: str, torch_version: str) -> None:
    rec = {
        "agent": "kernel-fuzzer-adaptive-avg-pool-v2",
        "kernel": KERNEL,
        "status": "SKIPPED",
        "reason": reason,
        "iters_attempted": 0,
        "iters_completed": 0,
        "divergences_filable": 0,
        "divergences_recalibration": 0,
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "top_3_repros": [],
        "torch_version": torch_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_MD.write_text(
        f"# {KERNEL} v2 fuzz - SKIPPED\n\n{reason}\n",
    )
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        _emit_skipped("torch.backends.mps.is_available() is False",
                      torch.__version__)
        return 0

    cfg_rng = random.Random(CONFIG_SEED)
    configs = _sample_configs(cfg_rng, N_CONFIGS)
    # Per-call rng (deterministic from cid+seed) for output-pattern resolution
    # of asymmetric outputs (the only nondeterministic branch).

    # Probe bf16 support.
    bf16_supported = True
    try:
        probe = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16, device="mps")
        _ = F.adaptive_avg_pool2d(probe, (2, 2))
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        bf16_supported = False
        print(f"[probe] bf16 unsupported on MPS for adaptive_avg_pool2d: "
              f"{exc!s:.160}", file=sys.stderr)

    started = time.monotonic()
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_empty = 0
    iters_skipped_dtype = 0
    iters_skipped_pattern = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    timed_out = False

    per_config: dict[int, dict] = {}

    schedule: list[tuple[dict, int]] = []
    for c in configs:
        for s in DATA_SEEDS:
            schedule.append((c, s))

    for c, s in schedule:
        if time.monotonic() - started > WALL_BUDGET_S:
            print(
                f"[budget] timeout at iter {iters_attempted} "
                f"(elapsed {time.monotonic()-started:.1f}s)",
                file=sys.stderr,
            )
            timed_out = True
            break

        if c["dtype"] == "bfloat16" and not bf16_supported:
            iters_skipped_dtype += 1
            iters_attempted += 1
            continue

        iters_attempted += 1
        # Deterministic per-(cid,seed) rng for output-pattern asymmetric branch.
        per_run_rng = random.Random((c["cid"] << 8) ^ s ^ CONFIG_SEED)
        try:
            status, abs_err, rel_err, denom_at_relmax, atol, rtol, why, out_size = (
                _run_one(c, s, per_run_rng)
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 -- halt rule per spec
            print(f"[ERROR] cfg={c} seed={s}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2

        if status == "SKIP":
            iters_skipped_pattern += 1
            continue
        if status == "UNSUPPORTED":
            iters_unsupported += 1
            continue
        if status == "EMPTY":
            iters_empty += 1
            iters_completed += 1
            continue

        iters_completed += 1
        max_abs_err_global = max(max_abs_err_global, abs_err)
        max_rel_err_global = max(max_rel_err_global, rel_err)

        slot = per_config.setdefault(c["cid"], {
            "cfg": c, "runs": {}, "atol": atol, "rtol": rtol,
            "out_size": out_size,
        })
        slot["runs"][s] = {
            "status": status, "abs_err": abs_err, "rel_err": rel_err,
            "denom_at_relmax": denom_at_relmax, "seed": s,
        }
        if status == "FILABLE_HIT":
            print(
                f"[FILABLE_HIT] cid={c['cid']} seed={s} {c['shape_bucket']}/"
                f"{c['dtype']}/{c['stride_category']}/"
                f"{c['output_pattern']}->{out_size} shape={c['shape']} "
                f"abs={abs_err:.3e} rel={rel_err:.3e} "
                f"denom@rel={denom_at_relmax:.3e}",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started

    filable: list[dict] = []
    recal: list[dict] = []
    for cid, slot in per_config.items():
        runs = slot["runs"]
        if not runs:
            continue
        n_runs = len(runs)
        n_filable_hit = sum(
            1 for r in runs.values() if r["status"] == "FILABLE_HIT"
        )
        n_crit_abs = sum(
            1 for r in runs.values() if r["status"] == "CRIT_ABS_ONLY"
        )
        n_crit_rel = sum(
            1 for r in runs.values() if r["status"] == "CRIT_REL_ONLY"
        )
        n_recal = sum(1 for r in runs.values() if r["status"] == "RECAL")
        n_recal_eq = (
            n_recal
            + (n_crit_abs if n_crit_abs >= 3 else 0)
            + (n_crit_rel if n_crit_rel >= 3 else 0)
        )
        max_abs = max(r["abs_err"] for r in runs.values())
        max_rel = max(r["rel_err"] for r in runs.values())
        max_denom = max(r["denom_at_relmax"] for r in runs.values())
        record = {
            "cid": cid,
            "cfg": slot["cfg"],
            "out_size": slot["out_size"],
            "atol": slot["atol"], "rtol": slot["rtol"],
            "n_runs": n_runs, "n_filable_hit": n_filable_hit,
            "n_crit_abs": n_crit_abs, "n_crit_rel": n_crit_rel,
            "n_recal": n_recal,
            "max_abs_err": max_abs, "max_rel_err": max_rel,
            "max_denom": max_denom,
            "per_seed": runs,
        }
        if n_filable_hit >= 3:
            filable.append(record)
        elif n_recal_eq >= 3:
            recal.append(record)

    filable.sort(
        key=lambda d: (-d["n_filable_hit"], -d["max_abs_err"], -d["max_rel_err"]),
    )
    recal.sort(
        key=lambda d: (
            -(d["n_recal"] + d["n_crit_abs"] + d["n_crit_rel"]),
            -d["max_abs_err"], -d["max_rel_err"],
        ),
    )
    top3 = (filable + recal)[:3]

    summary = {
        "agent": "kernel-fuzzer-adaptive-avg-pool-v2",
        "kernel": KERNEL,
        "op_path": KERNEL,
        "device_under_test": "mps",
        "reference": "cpu_fp32",
        "cuda_backend": "mocked",
        "torch_version": torch.__version__,
        "config_seed": f"0x{CONFIG_SEED:X}",
        "data_seeds": list(DATA_SEEDS),
        "n_configs_planned": N_CONFIGS,
        "n_configs_run": len(per_config),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_empty": iters_empty,
        "iters_skipped_dtype": iters_skipped_dtype,
        "iters_skipped_pattern": iters_skipped_pattern,
        "bf16_supported": bf16_supported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": [
            {
                "cid": r["cid"],
                "shape": list(r["cfg"]["shape"]),
                "dtype": r["cfg"]["dtype"],
                "stride_category": r["cfg"]["stride_category"],
                "shape_bucket": r["cfg"]["shape_bucket"],
                "output_pattern": r["cfg"]["output_pattern"],
                "output_size": list(r["out_size"]) if r["out_size"] else None,
                "n_filable_hit_seeds": r["n_filable_hit"],
                "n_crit_abs_only_seeds": r["n_crit_abs"],
                "n_crit_rel_only_seeds": r["n_crit_rel"],
                "n_recal_seeds": r["n_recal"],
                "n_runs": r["n_runs"],
                "max_abs_err": r["max_abs_err"],
                "max_rel_err": r["max_rel_err"],
                "max_denom_magnitude": r["max_denom"],
                "atol": r["atol"], "rtol": r["rtol"],
                "verdict": (
                    "FILABLE" if r["n_filable_hit"] >= 3 else "RECALIBRATION"
                ),
                "per_seed_status": {
                    s: rec["status"] for s, rec in r["per_seed"].items()
                },
            }
            for r in top3
        ],
        "elapsed_s": round(elapsed, 2),
        "timed_out": timed_out,
        "wall_budget_s": WALL_BUDGET_S,
        "host": "darwin/arm64 (Apple Silicon)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_md": str(RESULTS_MD),
    }

    _write_markdown(RESULTS_MD, summary, filable, recal)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    print(
        f"[done] iters_completed={iters_completed} "
        f"filable={len(filable)} recal={len(recal)} "
        f"max_abs={max_abs_err_global:.3e} max_rel={max_rel_err_global:.3e} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


def _write_markdown(
    path: Path, s: dict, filable: list[dict], recal: list[dict],
) -> None:
    L: list[str] = []
    L.append(f"# {s['kernel']} - MPS fuzz v2 report")
    L.append("")
    L.append(f"- **kernel:** `{s['kernel']}`")
    L.append(f"- **device under test:** `{s['device_under_test']}` "
             f"(reference: `{s['reference']}`, "
             f"CUDA backend: `{s['cuda_backend']}`)")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    L.append(f"- **iters_empty (numel==0):** {s['iters_empty']}")
    L.append(f"- **iters_skipped_dtype (bf16 unsupported):** "
             f"{s['iters_skipped_dtype']}")
    L.append(f"- **iters_skipped_pattern (output-size incompatible):** "
             f"{s['iters_skipped_pattern']}")
    L.append(f"- **bf16 supported on MPS for adaptive_avg_pool2d:** "
             f"{s['bf16_supported']}")
    L.append(f"- **n_configs_planned:** {s['n_configs_planned']} x "
             f"seeds={s['data_seeds']}")
    L.append(f"- **n_configs_run (saw >=1 seed completed):** "
             f"{s['n_configs_run']}")
    L.append(f"- **divergences_filable (FILABLE_HIT on >=3 seeds):** "
             f"{s['divergences_filable']}")
    L.append(f"- **divergences_recalibration (RECAL on >=3 seeds, not "
             f"filable):** {s['divergences_recalibration']}")
    L.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    L.append(f"- **max_rel_err:** {s['max_rel_err']:.3e}")
    L.append(f"- **elapsed:** {s['elapsed_s']} s "
             f"(budget {s['wall_budget_s']}s, timed_out={s['timed_out']})")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    L.append(f"- **config_seed:** {s['config_seed']}, "
             f"data_seeds: {s['data_seeds']}")
    L.append("")
    L.append("## Divergence rules (v2)")
    L.append("")
    L.append("Per-run status:")
    L.append("")
    L.append("- `FILABLE_HIT` - `abs_err > 10.atol` AND "
             "`rel_err > 10.rtol AND |y_ref|@argmax_rel >= 1e-6`.")
    L.append("- `CRIT_ABS_ONLY` - only the abs criterion fires.")
    L.append("- `CRIT_REL_ONLY` - only the denom-gated rel criterion fires.")
    L.append("- `RECAL` - `abs_err in [atol, 5.atol)` OR "
             "(`rel_err in [rtol, 5.rtol)` AND `|y_ref|@argmax_rel >= 1e-6`).")
    L.append("- `OK` - below 1x tolerance.")
    L.append("")
    L.append("Per-config verdict (across the 5 data seeds):")
    L.append("")
    L.append("- **FILABLE** - `FILABLE_HIT` on >= 3 of 5 seeds.")
    L.append("- **RECALIBRATION** - `RECAL` (or `CRIT_*_ONLY` >= 3 seeds) on "
             ">= 3 of 5 seeds and not FILABLE.")
    L.append("")
    L.append("Tolerances from `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, k_dim=(H/h_out)*(W/w_out), device_type='mps')` "
             "- includes the 2x MPS multiplier and the sqrt(k/128) "
             "accumulator-error scale, where k is the average pool window "
             "size. Convention matches the rest of swarm-v2.")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} - {r['verdict']}")
            L.append("")
            L.append(f"- **shape (N,C,H,W):** `{tuple(r['shape'])}` "
                     f"(bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride_category:** `{r['stride_category']}`")
            L.append(f"- **output_pattern:** `{r['output_pattern']}` -> "
                     f"`{tuple(r['output_size']) if r['output_size'] else None}`")
            L.append(f"- **per-seed status:** `{r['per_seed_status']}`")
            L.append(
                f"- **filable_hit / crit_abs_only / crit_rel_only / recal "
                f"/ n_runs:** "
                f"{r['n_filable_hit_seeds']} / {r['n_crit_abs_only_seeds']} "
                f"/ {r['n_crit_rel_only_seeds']} / {r['n_recal_seeds']} "
                f"/ {r['n_runs']}",
            )
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                     f"(atol={r['atol']:.2e}, 10x={10*r['atol']:.2e})")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                     f"(rtol={r['rtol']:.2e}, 10x={10*r['rtol']:.2e})")
            L.append(f"- **max_denom_magnitude (|y_ref|@argmax_rel):** "
                     f"{r['max_denom_magnitude']:.3e}")
            L.append("")

    if filable:
        L.append("## All FILABLE configs")
        L.append("")
        L.append("| cid | shape | dtype | stride | out | "
                 "filable_hits/n_runs | max_abs | max_rel |")
        L.append("|-----|-------|-------|--------|-----|"
                 "--------------------|---------|---------|")
        for r in filable:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` | "
                f"`{tuple(r['out_size']) if r['out_size'] else None}` "
                f"| {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |",
            )
        L.append("")

    if recal:
        L.append("## RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | shape | dtype | stride | out | "
                 "recal/crit_abs/crit_rel of n_runs | max_abs | max_rel |")
        L.append("|-----|-------|-------|--------|-----|"
                 "----------------------------------|---------|---------|")
        for r in recal[:50]:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` | "
                f"`{tuple(r['out_size']) if r['out_size'] else None}` "
                f"| {r['n_recal']}/{r['n_crit_abs']}/{r['n_crit_rel']} "
                f"of {r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |",
            )
        if len(recal) > 50:
            L.append("")
            L.append(
                f"_... {len(recal) - 50} more recalibration entries omitted._",
            )
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.adaptive_avg_pool2d` on CPU, "
             "comparison promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()` "
             "(see `src/gpucheck/backends/mps.py`).")
    L.append("- bf16 probed once at startup; if unsupported, those configs "
             "are skipped (counted under iters_skipped_dtype).")
    L.append("- Output-size patterns (global / tiny / small / half / "
             "asymmetric / indivisible / identity) skipped when "
             "incompatible with input H,W (counted under "
             "iters_skipped_pattern).")
    L.append("- CUDA cross-device comparison is N/A (no NVIDIA GPU on host).")
    L.append("- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: "
             f"{list(STRIDE_CATEGORIES)}.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
