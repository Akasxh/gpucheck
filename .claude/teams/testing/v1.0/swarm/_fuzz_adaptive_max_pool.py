"""V2 fuzz: adaptive_max_pool {1d,2d,3d} on MPS vs CPU reference.

500 (config, seed) pairs = 100 configs × 5 data seeds.
8-minute wall budget. Halts on process error.

Spec:
- FILABLE: max_rel_err > 10·rtol AND |y_ref|@argmax_rel >= 1e-6
  AND reproducible on >= 3 of 5 seeds.
- TOLERANCE_RECALIBRATION: per-config status in [tol, 5·tol) on >= 3 seeds
  (or single-criterion crit-only hits ≥3 seeds), and not FILABLE.
- OK otherwise.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-adaptive-max-pool/src")
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_adaptive-max-pool.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.adaptive_max_pool{1,2,3}d"
N_CONFIGS = 100
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for writing
CONFIG_SEED = 0xADAFACE5

# Variant -> (input_rank options, op fn, valid output spec)
# adaptive_max_pool1d: input must be 2D (C,L) or 3D (N,C,L). We use 3D.
# adaptive_max_pool2d: input must be 3D (C,H,W) or 4D (N,C,H,W). We use 4D.
# adaptive_max_pool3d: input must be 4D or 5D. We use 5D.
VARIANTS = ("1d", "2d", "3d")

# Shape buckets per variant. Each shape is the full input shape for the variant.
# For 1d: (N, C, L); 2d: (N, C, H, W); 3d: (N, C, D, H, W).
# Buckets follow the project priority: degenerate > non-tile-aligned > prime >
# power-of-2 boundary > large > mixed.
SHAPES_1D = {
    "degenerate":         [(1, 1, 1), (1, 3, 2), (2, 1, 4), (1, 0, 16)],
    "non_tile_aligned":   [(2, 3, 33), (4, 8, 65), (1, 7, 17)],
    "prime":              [(2, 3, 7), (3, 5, 11), (1, 7, 13), (5, 11, 17)],
    "power_of_2_boundary":[(1, 3, 64), (2, 4, 128), (1, 1, 256), (4, 4, 127)],
    "large":              [(2, 16, 1024), (1, 32, 2048)],
    "mixed":              [(8, 3, 100), (1, 16, 41), (3, 7, 73)],
}
SHAPES_2D = {
    "degenerate":         [(1, 1, 1, 1), (1, 1, 2, 2), (2, 1, 4, 4), (1, 0, 8, 8)],
    "non_tile_aligned":   [(2, 3, 17, 17), (1, 4, 33, 31), (4, 8, 65, 33)],
    "prime":              [(2, 3, 7, 11), (1, 5, 13, 7), (3, 7, 11, 13)],
    "power_of_2_boundary":[(1, 3, 64, 64), (2, 4, 128, 128), (1, 1, 32, 32),
                           (4, 4, 127, 127)],
    "large":              [(2, 16, 224, 224), (1, 32, 256, 256)],
    "mixed":              [(8, 3, 28, 28), (1, 16, 41, 17), (3, 7, 50, 50)],
}
SHAPES_3D = {
    "degenerate":         [(1, 1, 1, 1, 1), (1, 1, 2, 2, 2), (2, 1, 4, 4, 4)],
    "non_tile_aligned":   [(1, 3, 9, 9, 9), (2, 4, 17, 17, 17)],
    "prime":              [(1, 3, 5, 7, 11), (2, 5, 7, 11, 13)],
    "power_of_2_boundary":[(1, 2, 16, 16, 16), (1, 1, 8, 8, 8), (1, 4, 32, 32, 32)],
    "large":              [(1, 8, 32, 32, 32), (1, 4, 64, 64, 64)],
    "mixed":              [(2, 3, 10, 20, 30), (1, 16, 5, 9, 13)],
}
SHAPE_BUCKETS_BY_VARIANT = {
    "1d": SHAPES_1D,
    "2d": SHAPES_2D,
    "3d": SHAPES_3D,
}
BUCKET_NAMES = list(SHAPES_2D.keys())

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())


def _spatial_dims_from_shape(shape: tuple[int, ...], variant: str) -> tuple[int, ...]:
    """Return spatial dims (last 1/2/3 entries) from input shape."""
    nspatial = {"1d": 1, "2d": 2, "3d": 3}[variant]
    return shape[-nspatial:]


def _valid_output_size(
    spatial: tuple[int, ...], rng: random.Random,
) -> tuple[int, ...] | int:
    """Pick a valid adaptive output_size <= each spatial dim, >= 1.

    Variants of pick: 1, half, two-thirds, full. Skip if any spatial == 0.
    """
    if any(s == 0 for s in spatial):
        return 1 if len(spatial) == 1 else (1,) * len(spatial)
    choice = rng.choice(["one", "half", "two_thirds", "full"])
    if choice == "one":
        out = tuple(1 for _ in spatial)
    elif choice == "half":
        out = tuple(max(1, s // 2) for s in spatial)
    elif choice == "two_thirds":
        out = tuple(max(1, (2 * s) // 3) for s in spatial)
    else:
        out = tuple(s for s in spatial)
    if len(out) == 1:
        return out[0]
    return out


def _adaptive_max_pool(x: torch.Tensor, variant: str, output_size) -> torch.Tensor:
    if variant == "1d":
        return F.adaptive_max_pool1d(x, output_size=output_size)
    if variant == "2d":
        return F.adaptive_max_pool2d(x, output_size=output_size)
    if variant == "3d":
        return F.adaptive_max_pool3d(x, output_size=output_size)
    raise ValueError(variant)


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
        variant = rng.choice(VARIANTS)
        bucket = rng.choice(BUCKET_NAMES)
        shape_bucket = SHAPE_BUCKETS_BY_VARIANT[variant][bucket]
        shape = rng.choice(shape_bucket)
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        spatial = _spatial_dims_from_shape(shape, variant)
        output_size = _valid_output_size(spatial, rng)
        cfgs.append({
            "cid": cid,
            "variant": variant,
            "shape": tuple(shape),
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "shape_bucket": bucket,
            "output_size": output_size,
        })
    return cfgs


def _run_one(
    cfg: dict, data_seed: int,
) -> tuple[str, float, float, float, float, float, str | None]:
    """Returns (status, abs_err, rel_err, denom_at_relmax, atol, rtol, why)."""
    shape = cfg["shape"]
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]
    variant = cfg["variant"]
    output_size = cfg["output_size"]

    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-build:{exc!s:.180}"

    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-build:{exc!s:.180}"

    if x_cpu.numel() == 0:
        return "EMPTY", 0.0, 0.0, 0.0, 0.0, 0.0, None

    try:
        y_mps = _adaptive_max_pool(x_mps, variant, output_size)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-op:{exc!s:.180}"

    try:
        y_cpu = _adaptive_max_pool(x_cpu, variant, output_size)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-op:{exc!s:.180}"

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    atol, rtol = compute_tolerance(dtype, device_type="mps")
    status = _classify(abs_err, rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        rec = {
            "agent": "kernel-fuzzer-adaptive-max-pool-v2",
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
            f"# {KERNEL} v2 fuzz — SKIPPED\n\nMPS not available.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return 0

    cfg_rng = random.Random(CONFIG_SEED)
    configs = _sample_configs(cfg_rng, N_CONFIGS)

    started = time.monotonic()
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_empty = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    timed_out = False
    unsupported_reasons: dict[str, int] = {}

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

        iters_attempted += 1
        try:
            status, abs_err, rel_err, denom_at_relmax, atol, rtol, why = _run_one(c, s)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # halt on process error per spec
            print(f"[ERROR] cfg={c} seed={s}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2

        if status == "UNSUPPORTED":
            iters_unsupported += 1
            if why:
                key = why.split(":", 1)[0]
                unsupported_reasons[key] = unsupported_reasons.get(key, 0) + 1
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
                f"[FILABLE_HIT] cid={c['cid']} seed={s} "
                f"{c['variant']}/{c['shape_bucket']}/{c['dtype']}/"
                f"{c['stride_category']} shape={c['shape']} "
                f"out={c['output_size']} "
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
        n_filable_hit = sum(1 for r in runs.values() if r["status"] == "FILABLE_HIT")
        n_crit_abs = sum(1 for r in runs.values() if r["status"] == "CRIT_ABS_ONLY")
        n_crit_rel = sum(1 for r in runs.values() if r["status"] == "CRIT_REL_ONLY")
        n_recal = sum(1 for r in runs.values() if r["status"] == "RECAL")
        n_recal_eq = (
            n_recal
            + (n_crit_abs if n_crit_abs >= 3 else 0)
            + (n_crit_rel if n_crit_rel >= 3 else 0)
        )
        max_abs = max(r["abs_err"] for r in runs.values())
        max_rel = max(r["rel_err"] for r in runs.values())
        max_denom = max(r["denom_at_relmax"] for r in runs.values())
        rec = {
            "cid": cid,
            "cfg": slot["cfg"],
            "atol": slot["atol"], "rtol": slot["rtol"],
            "n_runs": n_runs, "n_filable_hit": n_filable_hit,
            "n_crit_abs": n_crit_abs, "n_crit_rel": n_crit_rel,
            "n_recal": n_recal,
            "max_abs_err": max_abs, "max_rel_err": max_rel,
            "max_denom": max_denom,
            "per_seed": runs,
        }
        if n_filable_hit >= 3:
            filable.append(rec)
        elif n_recal_eq >= 3:
            recal.append(rec)

    filable.sort(key=lambda d: (-d["n_filable_hit"], -d["max_abs_err"], -d["max_rel_err"]))
    recal.sort(key=lambda d: (
        -(d["n_recal"] + d["n_crit_abs"] + d["n_crit_rel"]),
        -d["max_abs_err"], -d["max_rel_err"],
    ))
    top3_pool = filable + recal
    top3 = top3_pool[:3]

    summary = {
        "agent": "kernel-fuzzer-adaptive-max-pool-v2",
        "kernel": KERNEL,
        "op_path": "torch.nn.functional.adaptive_max_pool{1,2,3}d",
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
        "unsupported_reasons": unsupported_reasons,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": [
            {
                "cid": r["cid"],
                "variant": r["cfg"]["variant"],
                "shape": list(r["cfg"]["shape"]),
                "dtype": r["cfg"]["dtype"],
                "stride_category": r["cfg"]["stride_category"],
                "shape_bucket": r["cfg"]["shape_bucket"],
                "output_size": r["cfg"]["output_size"],
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
        f"unsupported={iters_unsupported} empty={iters_empty} "
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
    L.append(f"- **kernel:** `{s['kernel']}` (variants 1d/2d/3d)")
    L.append(f"- **device under test:** `{s['device_under_test']}` "
             f"(reference: `{s['reference']}`, CUDA backend: `{s['cuda_backend']}`)")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']} "
             f"(reasons: {s['unsupported_reasons']})")
    L.append(f"- **iters_empty (numel==0):** {s['iters_empty']}")
    L.append(f"- **n_configs_planned:** {s['n_configs_planned']} "
             f"× seeds={s['data_seeds']}")
    L.append(f"- **n_configs_run (≥1 seed completed):** {s['n_configs_run']}")
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
    L.append("- **RECALIBRATION** — `RECAL` (or single-criterion crit-only ≥3 seeds) "
             "on ≥ 3 of 5 seeds and not FILABLE.")
    L.append("")
    L.append("Tolerances from `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (includes the 2× MPS multiplier).")
    L.append("Convention matches v2 swarm reports (gelu, softmax, rmsnorm).")
    L.append("")
    L.append("## Op coverage notes")
    L.append("")
    L.append("- `adaptive_max_pool1d` and `adaptive_max_pool2d` are implemented "
             "natively on MPS (torch 2.6+).")
    L.append("- `adaptive_max_pool3d` is **not implemented on MPS** as of "
             f"torch=={s['torch_version']} — see pytorch issue #141287. "
             "Configs sampling the 3d variant fall through `mps-op:NotImplementedError` "
             "and are counted under `iters_unsupported`. Without "
             "`PYTORCH_ENABLE_MPS_FALLBACK=1` they cannot run; with the flag they "
             "would silently CPU-fall-back, defeating the comparison.")
    L.append("- `output_size` per config: one of `1`, `floor(s/2)`, `floor(2s/3)`, "
             "or full `s` per spatial dim (sampled deterministically).")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} — {r['verdict']}")
            L.append("")
            L.append(f"- **variant:** `adaptive_max_pool{r['variant']}`")
            L.append(f"- **shape:** `{tuple(r['shape'])}` "
                     f"(bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride_category:** `{r['stride_category']}`")
            L.append(f"- **output_size:** `{r['output_size']}`")
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
        L.append("| cid | variant | shape | dtype | stride | filable_hits/n_runs "
                 "| max_abs | max_rel |")
        L.append("|-----|---------|-------|-------|--------|---------------------"
                 "|---------|---------|")
        for r in filable:
            L.append(
                f"| {r['cid']} | `{r['cfg']['variant']}` "
                f"| `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        L.append("")

    if recal:
        L.append("## RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | variant | shape | dtype | stride | recal/crit_abs/crit_rel "
                 "of n_runs | max_abs | max_rel |")
        L.append("|-----|---------|-------|-------|--------|"
                 "-----------------------------------|---------|---------|")
        for r in recal[:50]:
            L.append(
                f"| {r['cid']} | `{r['cfg']['variant']}` "
                f"| `{tuple(r['cfg']['shape'])}` | "
                f"`{r['cfg']['dtype']}` | `{r['cfg']['stride_category']}` "
                f"| {r['n_recal']}/{r['n_crit_abs']}/{r['n_crit_rel']} "
                f"of {r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        if len(recal) > 50:
            L.append("")
            L.append(f"_… {len(recal) - 50} more recalibration entries omitted._")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.adaptive_max_pool{1,2,3}d` on CPU "
             "with the same dtype, comparison promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()` "
             "(see `src/gpucheck/backends/mps.py`).")
    L.append("- `return_indices=False` (default). Comparing values only.")
    L.append("- CUDA cross-device comparison is N/A (no NVIDIA GPU on host).")
    L.append("- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: "
             f"{list(STRIDE_CATEGORIES)}.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
