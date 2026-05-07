"""V2 fuzz: torch.baddbmm on MPS vs CPU reference.

Spec (per swarm-v2 convention used by gelu/softmax/rmsnorm fuzzers):
- 100 configs × 5 seeds = 500 total iterations.
- Per-run divergence buckets:
    * abs > 10·atol AND
      (rel > 10·rtol AND |y_ref|@argmax_rel >= 1e-6)  → FILABLE_HIT
    * only abs criterion → CRIT_ABS_ONLY
    * only denom-gated rel criterion → CRIT_REL_ONLY
    * abs ∈ [atol, 5·atol) OR
      (rel ∈ [rtol, 5·rtol) AND |y_ref|@argmax_rel >= 1e-6)  → RECAL
    * else → OK
- Per-config verdict: FILABLE iff FILABLE_HIT on ≥3 seeds.
  RECALIBRATION iff (RECAL or single-criterion crit hits) on ≥3 seeds and
  not FILABLE.
- 8-minute wall budget. Halts on process error per spec.

baddbmm: out = beta·input + alpha·(batch1 @ batch2)
  batch1: (b, n, m)   batch2: (b, m, p)   input: (b, n, p)
  contraction dim k = m (used for tolerance sqrt(k/128) scaling).
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-baddbmm/src")
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_baddbmm.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.baddbmm"
N_CONFIGS = 100
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for writeout
CONFIG_SEED = 0xBADD8888

# ---------------------------------------------------------------------------
# Sampling space — keep tensor sizes bounded so 500 runs fit in 8 minutes.
# ---------------------------------------------------------------------------

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
NON_TILE_ALIGNED = [9, 18, 27, 36, 45, 54, 99, 130]
LARGE_DIMS = [192, 256, 320]
DEGEN = [1]  # zero-dim handled separately as EMPTY

BATCH_BUCKETS = {
    "small": [1, 2, 3, 4],
    "medium": [8, 12, 16],
    "prime": [3, 5, 7, 11],
    "boundary": [1, 2, 32, 33],
}

DIM_BUCKETS: dict[str, list[int]] = {
    "degenerate": DEGEN,
    "non_tile_aligned": NON_TILE_ALIGNED,
    "prime": PRIMES,
    "power_of_2_boundary": POW2_BOUNDARY,
    "large": LARGE_DIMS,
}
BUCKET_NAMES = list(DIM_BUCKETS.keys())
BATCH_BUCKET_NAMES = list(BATCH_BUCKETS.keys())

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())

ALPHA_BETA_PAIRS = [
    (1.0, 1.0),
    (1.0, 0.0),    # pure matmul, beta=0
    (0.5, 0.5),
    (2.0, 1.0),
    (1.0, -1.0),   # subtract input
    (0.0, 1.0),    # input passthrough (alpha=0): degenerate but valid
    (-0.25, 1.5),
]


def _err_metrics(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_at_relmax). Both compared in fp32."""
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
    """Sample n unique-ish baddbmm configs."""
    cfgs = []
    for cid in range(n):
        b_bucket = rng.choice(BATCH_BUCKET_NAMES)
        b = rng.choice(BATCH_BUCKETS[b_bucket])
        # Sample n, m, p from possibly different dim buckets — keeps coverage
        # broad without exploding the config count.
        bucket_n = rng.choice(BUCKET_NAMES)
        bucket_m = rng.choice(BUCKET_NAMES)
        bucket_p = rng.choice(BUCKET_NAMES)
        n_ = rng.choice(DIM_BUCKETS[bucket_n])
        m_ = rng.choice(DIM_BUCKETS[bucket_m])
        p_ = rng.choice(DIM_BUCKETS[bucket_p])
        # Memory cap: reject configs whose result tensor exceeds ~16M elements.
        # 16M * 4B = 64 MiB per tensor — comfortable on most M-series.
        if b * max(n_, 1) * max(p_, 1) > 16_000_000:
            n_ = min(n_, 64)
            p_ = min(p_, 64)
        if b * max(n_, 1) * max(m_, 1) > 16_000_000:
            m_ = min(m_, 64)
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        alpha, beta = rng.choice(ALPHA_BETA_PAIRS)
        cfgs.append({
            "cid": cid,
            "b": b, "n": n_, "m": m_, "p": p_,
            "shape_batch1": (b, n_, m_),
            "shape_batch2": (b, m_, p_),
            "shape_input": (b, n_, p_),
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "shape_bucket": f"{b_bucket}/{bucket_n}/{bucket_m}/{bucket_p}",
            "alpha": alpha,
            "beta": beta,
        })
    return cfgs


def _run_one(
    cfg: dict, data_seed: int,
) -> tuple[str, float, float, float, float, float, str | None]:
    """Run a single (config, seed) pair.

    Returns (status, abs_err, rel_err, denom_at_relmax, atol, rtol, unsupported_reason).
    status in {OK, RECAL, FILABLE_HIT, CRIT_ABS_ONLY, CRIT_REL_ONLY, UNSUPPORTED, EMPTY}.
    """
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]
    alpha, beta = cfg["alpha"], cfg["beta"]
    sh1 = cfg["shape_batch1"]
    sh2 = cfg["shape_batch2"]
    shi = cfg["shape_input"]

    # Build inputs on CPU. Use the same seed for all three tensors but with a
    # small offset so we don't get correlated structure between batch1/batch2/input.
    try:
        b1_cpu = fuzz_strides_for_category(
            sh1, dtype, stride_cat, device="cpu", seed=data_seed,
        )
        b2_cpu = fuzz_strides_for_category(
            sh2, dtype, stride_cat, device="cpu", seed=data_seed + 1009,
        )
        in_cpu = fuzz_strides_for_category(
            shi, dtype, stride_cat, device="cpu", seed=data_seed + 2017,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-build:{exc!s:.180}"

    try:
        b1_mps = fuzz_strides_for_category(
            sh1, dtype, stride_cat, device="mps", seed=data_seed,
        )
        b2_mps = fuzz_strides_for_category(
            sh2, dtype, stride_cat, device="mps", seed=data_seed + 1009,
        )
        in_mps = fuzz_strides_for_category(
            shi, dtype, stride_cat, device="mps", seed=data_seed + 2017,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-build:{exc!s:.180}"

    if b1_cpu.numel() == 0 or b2_cpu.numel() == 0 or in_cpu.numel() == 0:
        return "EMPTY", 0.0, 0.0, 0.0, 0.0, 0.0, None

    # MPS path
    try:
        y_mps = torch.baddbmm(in_mps, b1_mps, b2_mps, beta=beta, alpha=alpha)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-op:{exc!s:.180}"

    # CPU reference
    try:
        y_cpu = torch.baddbmm(in_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-op:{exc!s:.180}"

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    # Tolerance: matmul-style with k=m (contraction dim).
    atol, rtol = compute_tolerance(dtype, k_dim=cfg["m"], device_type="mps")
    status = _classify(abs_err, rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        rec = {
            "agent": "kernel-fuzzer-baddbmm-v2",
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

    # Probe bf16 baddbmm support on MPS.
    bf16_supported = True
    try:
        _ = torch.baddbmm(
            torch.zeros(2, 4, 4, dtype=torch.bfloat16, device="mps"),
            torch.zeros(2, 4, 4, dtype=torch.bfloat16, device="mps"),
            torch.zeros(2, 4, 4, dtype=torch.bfloat16, device="mps"),
        )
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        bf16_supported = False
        print(f"[probe] bf16 unsupported on MPS for baddbmm: {exc!s:.160}",
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

    # Config-major: each config gets all 5 seeds run together so the
    # reproducibility-across-seeds verdict is preserved on truncation.
    schedule = []
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
                f"{c['dtype']}/{c['stride_category']} "
                f"b1={c['shape_batch1']} b2={c['shape_batch2']} "
                f"alpha={c['alpha']} beta={c['beta']} "
                f"abs={abs_err:.3e} rel={rel_err:.3e} denom@rel={denom_at_relmax:.3e}",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started

    # Aggregate per-config
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
        n_recal_eq = (n_recal
                      + (n_crit_abs if n_crit_abs >= 3 else 0)
                      + (n_crit_rel if n_crit_rel >= 3 else 0))
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
    top3_pool = filable + recal
    top3 = top3_pool[:3]

    summary = {
        "agent": "kernel-fuzzer-baddbmm-v2",
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
                "shape_batch1": list(r["cfg"]["shape_batch1"]),
                "shape_batch2": list(r["cfg"]["shape_batch2"]),
                "shape_input": list(r["cfg"]["shape_input"]),
                "alpha": r["cfg"]["alpha"], "beta": r["cfg"]["beta"],
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
    L.append(f"- **bf16 supported on MPS for baddbmm:** {s['bf16_supported']}")
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
             "dtype, k_dim=m, device_type='mps')` — k_dim is the contraction dim "
             "(m), which drives the matmul `sqrt(k/128)` scaling. The 2× MPS "
             "overlay multiplier is included.")
    L.append("Convention matches `RESULTS_softmax.md` / `RESULTS_rmsnorm.md` / "
             "`RESULTS_gelu.md` (swarm-v2).")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} — {r['verdict']}")
            L.append("")
            L.append(f"- **batch1 shape:** `{tuple(r['shape_batch1'])}`")
            L.append(f"- **batch2 shape:** `{tuple(r['shape_batch2'])}`")
            L.append(f"- **input shape:** `{tuple(r['shape_input'])}`")
            L.append(f"- **alpha / beta:** {r['alpha']} / {r['beta']}")
            L.append(f"- **shape_bucket (b/n/m/p):** `{r['shape_bucket']}`")
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
        L.append("| cid | b1 | b2 | dtype | stride | α | β | filable_hits/n_runs | max_abs | max_rel |")
        L.append("|-----|----|----|-------|--------|---|---|---------------------|---------|---------|")
        for r in filable:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape_batch1'])}` | "
                f"`{tuple(r['cfg']['shape_batch2'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['cfg']['alpha']} | {r['cfg']['beta']} "
                f"| {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        L.append("")

    if recal:
        L.append("## RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | b1 | b2 | dtype | stride | α | β | recal/crit_abs/crit_rel of n_runs | max_abs | max_rel |")
        L.append("|-----|----|----|-------|--------|---|---|-----------------------------------|---------|---------|")
        for r in recal[:50]:
            L.append(
                f"| {r['cid']} | `{tuple(r['cfg']['shape_batch1'])}` | "
                f"`{tuple(r['cfg']['shape_batch2'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['cfg']['alpha']} | {r['cfg']['beta']} "
                f"| {r['n_recal']}/{r['n_crit_abs']}/{r['n_crit_rel']} of {r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        if len(recal) > 50:
            L.append("")
            L.append(f"_… {len(recal) - 50} more recalibration entries omitted._")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- `torch.baddbmm(input, batch1, batch2, beta=β, alpha=α)` "
             "computes `β·input + α·(batch1 @ batch2)` over a batch axis.")
    L.append("- Reference: same call on CPU; comparison promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()` "
             "(see `src/gpucheck/backends/mps.py`).")
    L.append("- bf16 probed once at startup; if unsupported, those configs are "
             "skipped (counted under iters_skipped_dtype).")
    L.append("- CUDA cross-device comparison is N/A (no NVIDIA GPU on host).")
    L.append("- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: "
             f"{list(STRIDE_CATEGORIES)}.")
    L.append("- Each (batch1, batch2, input) triple uses the same stride "
             "category but different per-tensor seeds (offsets +0/+1009/+2017) "
             "to avoid correlated structure across operands.")
    L.append("- Tolerance k_dim = m (contraction dim) so matmul `sqrt(k/128)` "
             "scaling tracks the accumulator depth.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
