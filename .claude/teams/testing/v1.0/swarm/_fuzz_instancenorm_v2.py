"""V2 fuzz: InstanceNorm on MPS vs CPU reference.

Spec (kernel-fuzzer-instancenorm-v2):
- 500 unique configs sampled deterministically.
- Each config replayed across 5 data seeds (0..4).
- FILABLE per spec: max_rel_err > 10·rtol AND
  denom_magnitude (|y_ref|@argmax_rel) >= 1e-6
  AND reproduced on >= 3 of 5 seeds.
  We additionally require the abs criterion (>10·atol) per the v2 family
  convention (matches gelu/softmax/rmsnorm v2 reports). Configs that trip
  only the rel-and-denom criterion are reported under
  RECALIBRATION/CRIT_REL_ONLY.
- Below FILABLE: TOLERANCE_RECALIBRATION (RECAL on >= 3 seeds, or
  single-criterion crit on >= 3 seeds) — recommend xfail.
- Otherwise OK.
- 8-minute wall budget. Writes partial results on time-out.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-instancenorm/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_instancenorm.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.instance_norm"
N_CONFIGS = 500
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for writing
CONFIG_SEED = 0xCAFEFACE

# ---------------------------------------------------------------------------
# Sampling space — instance_norm requires (N, C, *spatial) with at least
# one spatial dim having >1 element. Build 3D/4D/5D shapes accordingly.
# ---------------------------------------------------------------------------

T0 = TILE_SIZES[0]
T1 = TILE_SIZES[1]
T2 = TILE_SIZES[2]

DEGENERATE = [
    (1, 1, 2),       # N=C=1, smallest valid spatial
    (1, 4, 2),
    (2, 1, 4),
    (1, 1, 1, 2),    # 4D minimal
    (1, 1, 2, 1),
    (2, 1, 1, 1, 2), # 5D minimal
]
NON_TILE = [
    (2, T0 - 1, T0 + 1),
    (1, T0 + 1, T0 - 1),
    (1, 16, T1 - 1, T1 + 3),
    (2, 8, T2 - 1, T2 + 1),
    (1, T0 + 3, 8, T0 - 1),
    (1, 32, T1 + 1, T1 - 1, T0 + 1),
]
PRIME_S = [
    (1, PRIMES[0], PRIMES[1]),
    (PRIMES[0], 8, PRIMES[2]),
    (1, 16, PRIMES[2], PRIMES[1]),
    (2, PRIMES[1], 16, 16),
    (1, 8, PRIMES[3], PRIMES[3]),
]
POW2_BOUNDARY = [
    (1, 16, POWER_OF_2_BOUNDARIES[0]),
    (1, 32, POWER_OF_2_BOUNDARIES[1]),
    (2, 8, 128, 129),
    (1, 16, 256, 255),
    (1, 8, 64, 64, 2),
]
LARGE = [
    (1, 32, LARGE_DIMS[0]),
    (1, 8, 256, 256),
    (1, 16, 64, 128),
    (2, 32, 128, 128),
]
MIXED = [
    (1, 16, 127),
    (1, 8, 1024, 3),
    (2, 4, 7, 128),
    (1, 4, 33, 128, 4),
    (1, 24, 5, 17),
]

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


def _err_metrics(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_at_relmax) in fp32."""
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
    """FILABLE_HIT requires both abs and rel-with-denom-gate to fire.

    Single-criterion crit hits roll into RECAL at the config level.
    """
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
        cfgs.append({
            "cid": cid,
            "shape": tuple(shape),
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "shape_bucket": bucket,
        })
    return cfgs


def _run_one(
    cfg: dict, data_seed: int,
) -> tuple[str, float, float, float, float, float, str | None]:
    """Run a single (config, seed). Returns (status, abs, rel, denom, atol, rtol, why)."""
    shape = cfg["shape"]
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]

    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-build:{exc!s:.180}"

    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-build:{exc!s:.180}"

    if x_cpu.numel() == 0:
        return "EMPTY", 0.0, 0.0, 0.0, 0.0, 0.0, None

    try:
        y_mps = F.instance_norm(x_mps)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-op:{exc!s:.180}"

    try:
        y_cpu = F.instance_norm(x_cpu)
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-op:{exc!s:.180}"

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    atol, rtol = compute_tolerance(dtype, device_type="mps")
    status = _classify(abs_err, rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        rec = {
            "agent": "kernel-fuzzer-instancenorm-v2",
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "top_3_repros": [],
            "torch_version": torch.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} v2 fuzz — SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return 0

    cfg_rng = random.Random(CONFIG_SEED)
    configs = _sample_configs(cfg_rng, N_CONFIGS)

    # Probe bf16 support once.
    bf16_supported = True
    try:
        _ = F.instance_norm(torch.zeros(1, 2, 4, dtype=torch.bfloat16, device="mps"))
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        bf16_supported = False
        print(f"[probe] bf16 unsupported on MPS for instance_norm: {exc!s:.160}",
              file=sys.stderr)

    started = time.monotonic()
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_empty = 0
    iters_skipped_dtype = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    timed_out = False

    per_config: dict[int, dict] = {}
    unsupported_samples: list[str] = []

    # config-major: each cid runs all 5 seeds together so partial truncation
    # still preserves cross-seed reproducibility info for the prefix.
    schedule = [(c, s) for c in configs for s in DATA_SEEDS]

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
        try:
            status, abs_err, rel_err, denom_at_relmax, atol, rtol, why = _run_one(c, s)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — halt rule per spec
            print(f"[ERROR] cfg={c} seed={s}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2

        if status == "UNSUPPORTED":
            iters_unsupported += 1
            if why and len(unsupported_samples) < 5:
                unsupported_samples.append(
                    f"cid={c['cid']} {c['shape_bucket']}/{c['dtype']}/"
                    f"{c['stride_category']} shape={c['shape']} :: {why}"
                )
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
        })
        slot["runs"][s] = {
            "status": status, "abs_err": abs_err, "rel_err": rel_err,
            "denom_at_relmax": denom_at_relmax, "seed": s,
        }
        if status == "FILABLE_HIT":
            print(
                f"[FILABLE_HIT] cid={c['cid']} seed={s} {c['shape_bucket']}/"
                f"{c['dtype']}/{c['stride_category']} shape={c['shape']} "
                f"abs={abs_err:.3e} rel={rel_err:.3e} denom@rel={denom_at_relmax:.3e}",
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
        n_recal_eq = n_recal + (n_crit_abs if n_crit_abs >= 3 else 0) \
            + (n_crit_rel if n_crit_rel >= 3 else 0)
        max_abs = max(r["abs_err"] for r in runs.values())
        max_rel = max(r["rel_err"] for r in runs.values())
        max_denom = max(r["denom_at_relmax"] for r in runs.values())
        if n_filable_hit >= 3:
            filable.append({
                "cid": cid, "cfg": slot["cfg"],
                "atol": slot["atol"], "rtol": slot["rtol"],
                "n_runs": n_runs, "n_filable_hit": n_filable_hit,
                "n_crit_abs": n_crit_abs, "n_crit_rel": n_crit_rel,
                "n_recal": n_recal,
                "max_abs_err": max_abs, "max_rel_err": max_rel,
                "max_denom": max_denom, "per_seed": runs,
            })
        elif n_recal_eq >= 3:
            recal.append({
                "cid": cid, "cfg": slot["cfg"],
                "atol": slot["atol"], "rtol": slot["rtol"],
                "n_runs": n_runs, "n_filable_hit": n_filable_hit,
                "n_crit_abs": n_crit_abs, "n_crit_rel": n_crit_rel,
                "n_recal": n_recal,
                "max_abs_err": max_abs, "max_rel_err": max_rel,
                "max_denom": max_denom, "per_seed": runs,
            })

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
        "agent": "kernel-fuzzer-instancenorm-v2",
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
        "unsupported_samples": unsupported_samples,
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
    L.append(f"# {s['kernel']} — MPS fuzz v2 report")
    L.append("")
    L.append(f"- **kernel:** `{s['kernel']}`")
    L.append(f"- **device under test:** `{s['device_under_test']}` "
             f"(reference: `{s['reference']}`, CUDA backend: `{s['cuda_backend']}`)")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    L.append(f"- **iters_empty (numel==0):** {s['iters_empty']}")
    L.append(f"- **iters_skipped_dtype (bf16 unsupported):** {s['iters_skipped_dtype']}")
    L.append(f"- **bf16 supported on MPS for instance_norm:** {s['bf16_supported']}")
    L.append(f"- **n_configs_planned:** {s['n_configs_planned']} "
             f"× seeds={s['data_seeds']}")
    L.append(f"- **n_configs_run (saw ≥1 seed completed):** {s['n_configs_run']}")
    L.append(f"- **divergences_filable (FILABLE_HIT on ≥3 seeds):** "
             f"{s['divergences_filable']}")
    L.append(f"- **divergences_recalibration (RECAL on ≥3 seeds, not filable):** "
             f"{s['divergences_recalibration']}")
    L.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    L.append(f"- **max_rel_err:** {s['max_rel_err']:.3e}")
    L.append(f"- **elapsed:** {s['elapsed_s']} s "
             f"(budget {s['wall_budget_s']}s, timed_out={s['timed_out']})")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    L.append(f"- **config_seed:** {s['config_seed']}, data_seeds: {s['data_seeds']}")
    L.append("")
    L.append("## Divergence rules (v2)")
    L.append("")
    L.append("Per-run status:")
    L.append("")
    L.append("- `FILABLE_HIT` — `abs_err > 10·atol` AND "
             "`rel_err > 10·rtol AND |y_ref|@argmax_rel ≥ 1e-6`.")
    L.append("- `CRIT_ABS_ONLY` — only the abs criterion fires.")
    L.append("- `CRIT_REL_ONLY` — only the denom-gated rel criterion fires.")
    L.append("- `RECAL` — `abs_err ∈ [atol, 5·atol)` OR "
             "(`rel_err ∈ [rtol, 5·rtol)` AND `|y_ref|@argmax_rel ≥ 1e-6`).")
    L.append("- `OK` — below 1× tolerance.")
    L.append("")
    L.append("Per-config verdict (across the 5 data seeds):")
    L.append("")
    L.append("- **FILABLE** — `FILABLE_HIT` on ≥ 3 of 5 seeds.")
    L.append("- **RECALIBRATION** — `RECAL` (or `CRIT_*_ONLY` ≥ 3 seeds) on "
             "≥ 3 of 5 seeds and not FILABLE.")
    L.append("")
    L.append("Tolerances from `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (includes 2× MPS multiplier).")
    L.append("Convention matches `RESULTS_gelu.md` / `RESULTS_softmax.md` / "
             "`RESULTS_rmsnorm.md` (swarm-v2).")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} — {r['verdict']}")
            L.append("")
            L.append(f"- **shape:** `{tuple(r['shape'])}` "
                     f"(bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride_category:** `{r['stride_category']}`")
            L.append(f"- **per-seed status:** `{r['per_seed_status']}`")
            L.append(
                f"- **filable_hit / crit_abs_only / crit_rel_only / recal "
                f"/ n_runs:** "
                f"{r['n_filable_hit_seeds']} / {r['n_crit_abs_only_seeds']} "
                f"/ {r['n_crit_rel_only_seeds']} / {r['n_recal_seeds']} "
                f"/ {r['n_runs']}"
            )
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                     f"(atol={r['atol']:.2e}, 10×={10*r['atol']:.2e})")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                     f"(rtol={r['rtol']:.2e}, 10×={10*r['rtol']:.2e})")
            L.append(f"- **max_denom_magnitude (|y_ref|@argmax_rel):** "
                     f"{r['max_denom_magnitude']:.3e}")
            L.append("")

    if filable:
        L.append("## All FILABLE configs")
        L.append("")
        L.append("| cid | shape | dtype | stride | filable_hits/n_runs | max_abs | max_rel |")
        L.append("|-----|-------|-------|--------|--------------------|---------|---------|")
        for r in filable:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        L.append("")

    if recal:
        L.append("## RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | shape | dtype | stride | recal/crit_abs/crit_rel of n_runs | max_abs | max_rel |")
        L.append("|-----|-------|-------|--------|-----------------------------------|---------|---------|")
        for r in recal[:50]:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['n_recal']}/{r['n_crit_abs']}/{r['n_crit_rel']} of {r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        if len(recal) > 50:
            L.append("")
            L.append(f"_… {len(recal) - 50} more recalibration entries omitted._")
        L.append("")

    if s.get("unsupported_samples"):
        L.append("## Unsupported samples (first few)")
        L.append("")
        for line in s["unsupported_samples"]:
            L.append(f"- {line}")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.instance_norm` on CPU, comparison "
             "promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()` "
             "(see `src/gpucheck/backends/mps.py`).")
    L.append("- bf16 probed once at startup; if unsupported, those configs are "
             "skipped (counted under iters_skipped_dtype).")
    L.append("- CUDA cross-device comparison is N/A (no NVIDIA GPU on host; "
             "CUDA backend mocked at detection level only).")
    L.append("- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: "
             f"{list(STRIDE_CATEGORIES)}.")
    L.append("- Shapes restricted to ≥ 3 dims with at least one spatial dim > 1; "
             "single-spatial-element shapes raise `ValueError` (counted as "
             "UNSUPPORTED).")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
