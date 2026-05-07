"""V2 fuzz: avg_pool2d on MPS vs CPU reference.

Spec (per dispatch):
- 500 configs sampled deterministically with config-RNG.
- Each config replayed across 5 data seeds (0,1,2,3,4) — gives ≥3-seed gate.
- FILABLE bucket per spec:
    rel_err > 10*rtol AND denom_magnitude >= 1e-6 AND >=3/5 seeds repro
- TOLERANCE_RECALIBRATION: rel_err in [rtol, 5*rtol) AND denom>=1e-6 (>=3 seeds)
- 8-minute wall budget. Writes partial results on time-out.
- Halt and report process error immediately.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-avg-pool/src")
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_avg-pool.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.avg_pool2d"
N_CONFIGS = 500
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for writing
CONFIG_SEED = 0xAVE9001 if False else 0xA0E09001  # 0xA0E09001 = "avgpool"-ish

# ---------------------------------------------------------------------------
# Sampling space — must be 3D (C,H,W) or 4D (N,C,H,W) for avg_pool2d.
# H,W must be >= kernel_size; we filter at config build time.
# ---------------------------------------------------------------------------

# Spatial sizes by bucket
SPATIAL_DEGENERATE = [1, 2]
SPATIAL_NON_TILE = [31, 33, 63, 65]
SPATIAL_PRIME = [7, 13, 31, 127]
SPATIAL_POW2 = [127, 128, 129, 255, 256, 257]
SPATIAL_LARGE = [512, 1024]
SPATIAL_MIXED = [16, 32, 33, 127]

SHAPE_BUCKETS: dict[str, list[int]] = {
    "degenerate": SPATIAL_DEGENERATE,
    "non_tile_aligned": SPATIAL_NON_TILE,
    "prime": SPATIAL_PRIME,
    "power_of_2_boundary": SPATIAL_POW2,
    "large": SPATIAL_LARGE,
    "mixed": SPATIAL_MIXED,
}
BUCKET_NAMES = list(SHAPE_BUCKETS.keys())

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())

# avg_pool2d hyperparams sampled per config
KERNEL_SIZES = [2, 3, 4, 5]
STRIDES = [1, 2, 3]
PADDINGS = [0, 1]
COUNT_INCLUDE_PADS = [True, False]
NDIMS = [3, 4]  # (C,H,W) or (N,C,H,W)
N_CHANNELS = [1, 3, 8, 16]
N_BATCHES = [1, 2, 4]


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
    rel_err: float, denom_at_relmax: float, atol: float, rtol: float,
) -> str:
    """Per spec: FILABLE_HIT iff rel_err > 10*rtol AND denom>=1e-6.

    RECAL: rel_err in [rtol, 5*rtol) AND denom>=1e-6.
    OK: below.
    """
    rel_gate = denom_at_relmax >= 1e-6
    if rel_err > 10.0 * rtol and rel_gate:
        return "FILABLE_HIT"
    if rtol <= rel_err < 5.0 * rtol and rel_gate:
        return "RECAL"
    return "OK"


def _sample_configs(rng: random.Random, n: int) -> list[dict]:
    cfgs = []
    for cid in range(n):
        bucket = rng.choice(BUCKET_NAMES)
        spatial_pool = SHAPE_BUCKETS[bucket]
        h = rng.choice(spatial_pool)
        w = rng.choice(spatial_pool)
        ndim = rng.choice(NDIMS)
        c = rng.choice(N_CHANNELS)
        if ndim == 4:
            n_batch = rng.choice(N_BATCHES)
            shape = (n_batch, c, h, w)
        else:
            shape = (c, h, w)

        # Pool params — constrain kernel_size <= min(H, W) (or 1 for degenerate)
        max_k = max(1, min(h, w))
        valid_ks = [k for k in KERNEL_SIZES if k <= max_k]
        if not valid_ks:
            valid_ks = [1]
        ksize = rng.choice(valid_ks)
        # stride must divide cleanly enough; just pick any
        stride = rng.choice(STRIDES)
        # padding must be <= ksize/2 per torch
        max_pad = max(0, ksize // 2)
        valid_pads = [p for p in PADDINGS if p <= max_pad]
        if not valid_pads:
            valid_pads = [0]
        padding = rng.choice(valid_pads)
        cip = rng.choice(COUNT_INCLUDE_PADS)

        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        cfgs.append({
            "cid": cid,
            "shape": tuple(shape),
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "shape_bucket": bucket,
            "kernel_size": ksize,
            "stride_arg": stride,
            "padding": padding,
            "count_include_pad": cip,
        })
    return cfgs


def _run_one(
    cfg: dict, data_seed: int,
) -> tuple[str, float, float, float, float, float, str | None]:
    """Returns (status, abs_err, rel_err, denom, atol, rtol, unsupported_reason)."""
    shape = cfg["shape"]
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]
    ksize = cfg["kernel_size"]
    stride_arg = cfg["stride_arg"]
    padding = cfg["padding"]
    cip = cfg["count_include_pad"]

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
        y_mps = F.avg_pool2d(
            x_mps, kernel_size=ksize, stride=stride_arg, padding=padding,
            count_include_pad=cip,
        )
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-op:{exc!s:.180}"

    try:
        y_cpu = F.avg_pool2d(
            x_cpu, kernel_size=ksize, stride=stride_arg, padding=padding,
            count_include_pad=cip,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-op:{exc!s:.180}"

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    atol, rtol = compute_tolerance(dtype, device_type="mps")
    status = _classify(rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        rec = {
            "agent": "kernel-fuzzer-avg-pool-v2",
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

    # Probe bf16 support once: avg_pool2d may or may not be implemented for bf16 on MPS.
    bf16_supported = True
    try:
        probe = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16, device="mps")
        _ = F.avg_pool2d(probe, kernel_size=2)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        bf16_supported = False
        print(f"[probe] bf16 unsupported on MPS for avg_pool2d: {exc!s:.160}",
              file=sys.stderr)

    fp16_supported = True
    try:
        probe = torch.zeros(1, 1, 4, 4, dtype=torch.float16, device="mps")
        _ = F.avg_pool2d(probe, kernel_size=2)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        fp16_supported = False
        print(f"[probe] fp16 unsupported on MPS for avg_pool2d: {exc!s:.160}",
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

    # Config-major schedule: each config sees all 5 seeds before moving on,
    # so partial truncation still preserves cross-seed reproducibility info
    # for the prefix of configs covered.
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
        if c["dtype"] == "float16" and not fp16_supported:
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
                f"{c['dtype']}/{c['stride_category']} shape={c['shape']} "
                f"k={c['kernel_size']} s={c['stride_arg']} p={c['padding']} "
                f"abs={abs_err:.3e} rel={rel_err:.3e} denom@rel={denom_at_relmax:.3e}",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started

    # Aggregate per-config: per-seed status counts.
    filable: list[dict] = []
    recal: list[dict] = []
    for cid, slot in per_config.items():
        runs = slot["runs"]
        if not runs:
            continue
        n_runs = len(runs)
        n_filable_hit = sum(1 for r in runs.values() if r["status"] == "FILABLE_HIT")
        n_recal = sum(1 for r in runs.values() if r["status"] == "RECAL")
        max_abs = max(r["abs_err"] for r in runs.values())
        max_rel = max(r["rel_err"] for r in runs.values())
        max_denom = max(r["denom_at_relmax"] for r in runs.values())
        if n_filable_hit >= 3:
            filable.append({
                "cid": cid, "cfg": slot["cfg"],
                "atol": slot["atol"], "rtol": slot["rtol"],
                "n_runs": n_runs, "n_filable_hit": n_filable_hit,
                "n_recal": n_recal,
                "max_abs_err": max_abs, "max_rel_err": max_rel,
                "max_denom": max_denom,
                "per_seed": runs,
            })
        elif n_recal >= 3:
            recal.append({
                "cid": cid, "cfg": slot["cfg"],
                "atol": slot["atol"], "rtol": slot["rtol"],
                "n_runs": n_runs, "n_filable_hit": n_filable_hit,
                "n_recal": n_recal,
                "max_abs_err": max_abs, "max_rel_err": max_rel,
                "max_denom": max_denom,
                "per_seed": runs,
            })

    filable.sort(
        key=lambda d: (-d["n_filable_hit"], -d["max_rel_err"], -d["max_abs_err"]),
    )
    recal.sort(
        key=lambda d: (-d["n_recal"], -d["max_rel_err"], -d["max_abs_err"]),
    )
    top3_pool = filable + recal
    top3 = top3_pool[:3]

    summary = {
        "agent": "kernel-fuzzer-avg-pool-v2",
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
        "fp16_supported": fp16_supported,
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
                "kernel_size": r["cfg"]["kernel_size"],
                "stride_arg": r["cfg"]["stride_arg"],
                "padding": r["cfg"]["padding"],
                "count_include_pad": r["cfg"]["count_include_pad"],
                "n_filable_hit_seeds": r["n_filable_hit"],
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
    L.append(f"- **iters_skipped_dtype:** {s['iters_skipped_dtype']}")
    L.append(f"- **bf16 supported on MPS for avg_pool2d:** {s['bf16_supported']}")
    L.append(f"- **fp16 supported on MPS for avg_pool2d:** {s['fp16_supported']}")
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
    L.append("## Divergence rules (per dispatch spec)")
    L.append("")
    L.append("Per-run status:")
    L.append("")
    L.append("- `FILABLE_HIT` — `rel_err > 10·rtol` AND `|y_ref|@argmax_rel ≥ 1e-6`.")
    L.append("- `RECAL` — `rel_err ∈ [rtol, 5·rtol)` AND `|y_ref|@argmax_rel ≥ 1e-6`.")
    L.append("- `OK` — below 1× rel tolerance, or denom-gated out.")
    L.append("")
    L.append("Per-config verdict (across the 5 data seeds):")
    L.append("")
    L.append("- **FILABLE** — `FILABLE_HIT` on ≥ 3 of 5 seeds.")
    L.append("- **TOLERANCE_RECALIBRATION** — `RECAL` on ≥ 3 of 5 seeds and not FILABLE.")
    L.append("")
    L.append("Tolerances from `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (includes the 2× MPS multiplier).")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} — {r['verdict']}")
            L.append("")
            L.append(f"- **shape:** `{tuple(r['shape'])}` (bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride_category:** `{r['stride_category']}`")
            L.append(f"- **pool params:** kernel_size={r['kernel_size']}, "
                     f"stride={r['stride_arg']}, padding={r['padding']}, "
                     f"count_include_pad={r['count_include_pad']}")
            L.append(f"- **per-seed status:** `{r['per_seed_status']}`")
            L.append(
                f"- **filable_hit / recal / n_runs:** "
                f"{r['n_filable_hit_seeds']} / {r['n_recal_seeds']} / {r['n_runs']}"
            )
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                     f"(atol={r['atol']:.2e}, 10×={10*r['atol']:.2e})")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                     f"(rtol={r['rtol']:.2e}, 10×={10*r['rtol']:.2e})")
            L.append(f"- **max_denom_magnitude:** {r['max_denom_magnitude']:.3e}")
            L.append("")

    if filable:
        L.append("## All FILABLE configs")
        L.append("")
        L.append("| cid | shape | dtype | stride | k/s/p | filable_hits/n_runs | max_rel | max_abs |")
        L.append("|-----|-------|-------|--------|-------|--------------------|---------|---------|")
        for r in filable:
            cf = r["cfg"]
            L.append(
                f"| {r['cid']} | `{tuple(cf['shape'])}` | "
                f"`{cf['dtype']}` | `{cf['stride_category']}` "
                f"| {cf['kernel_size']}/{cf['stride_arg']}/{cf['padding']} "
                f"| {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_rel_err']:.3e} | {r['max_abs_err']:.3e} |"
            )
        L.append("")

    if recal:
        L.append("## TOLERANCE_RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | shape | dtype | stride | k/s/p | recal/n_runs | max_rel | max_abs |")
        L.append("|-----|-------|-------|--------|-------|--------------|---------|---------|")
        for r in recal[:50]:
            cf = r["cfg"]
            L.append(
                f"| {r['cid']} | `{tuple(cf['shape'])}` | "
                f"`{cf['dtype']}` | `{cf['stride_category']}` "
                f"| {cf['kernel_size']}/{cf['stride_arg']}/{cf['padding']} "
                f"| {r['n_recal']}/{r['n_runs']} "
                f"| {r['max_rel_err']:.3e} | {r['max_abs_err']:.3e} |"
            )
        if len(recal) > 50:
            L.append("")
            L.append(f"_… {len(recal) - 50} more recalibration entries omitted._")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.avg_pool2d` on CPU, comparison "
             "promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()`.")
    L.append("- bf16 / fp16 probed once at startup; if unsupported, those configs are "
             "skipped (counted under iters_skipped_dtype).")
    L.append("- avg_pool2d requires NCHW or CHW shapes; configs sample ndim ∈ {3,4} "
             "and constrain `kernel_size <= min(H,W)` and `padding <= kernel_size/2`.")
    L.append("- Stride-category builders that produce unsupported layouts for "
             "avg_pool2d (e.g. column-major / broadcast on a 4D NCHW tensor) "
             "fall through to UNSUPPORTED rather than being silently dropped.")
    L.append("- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: "
             f"{list(STRIDE_CATEGORIES)}.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
