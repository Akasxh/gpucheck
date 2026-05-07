"""Fuzz the CELU kernel (v2): MPS vs CPU reference.

Spec (v2):
- 1000 iterations distributed across seeds 0,1,2,3,4
- ~200 unique (shape, dtype, stride_cat) configs, each replayed for all 5 seeds
- Filtering rules:
  * max_abs_err > 10× tolerance      -> always counts as DIVERGENCE
  * max_rel_err > 10× tolerance      -> DIVERGENCE only if denom_magnitude >= 1e-6
  * 1× <= max_err <= 10× tolerance   -> RECALIBRATION (xfail-eligible)
  * < 1×                              -> OK
- FILABLE = DIVERGENCE on >=3 of 5 seeds for the same config.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-celu/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_celu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.celu"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_CONFIGS = 200  # 200 configs * 5 seeds = 1000 iters
TIME_BUDGET_S = 10 * 60  # leave ~2 min for write + safety

# Filing thresholds (multipliers on per-dtype atol/rtol)
RECAL_LO = 1.0
DIVERGE_HI = 10.0
DENOM_FLOOR = 1e-6  # below this, large rel_err is "near-zero artifact"
FILABLE_MIN_SEEDS = 3

# Shape buckets — same priority taxonomy as v1, but trim large dims to keep
# elementwise CELU fast (we need 1000 iters in budget).
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

# CELU alpha values to sweep — the kernel is parameterised, so we should hit
# the default (1.0) plus a non-default to expose alpha-handling bugs.
CELU_ALPHAS: tuple[float, ...] = (1.0, 0.5, 2.0)


def _max_rel_err_with_denom(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_max).

    Computes per-element rel = |a-b| / max(|b|, 1e-12). The denom magnitude at
    the argmax tells us whether a large rel_err is a near-zero artefact.
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    rel = diff / denom
    idx = int(rel.argmax().item())
    flat_b = b.reshape(-1)
    return float(rel.reshape(-1)[idx].item()), float(flat_b[idx].abs().item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(
    abs_err: float, rel_err: float, denom_at_rel_max: float, atol: float, rtol: float,
) -> str:
    """Return one of: 'OK', 'RECALIBRATION', 'DIVERGENCE'."""
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = rel_err / rtol if rtol > 0 else 0.0

    # DIVERGENCE: abs > 10x always; rel > 10x only with healthy denom.
    abs_diverge = abs_ratio > DIVERGE_HI
    rel_diverge = rel_ratio > DIVERGE_HI and denom_at_rel_max >= DENOM_FLOOR
    if abs_diverge or rel_diverge:
        return "DIVERGENCE"

    if abs_ratio >= RECAL_LO or rel_ratio >= RECAL_LO:
        return "RECALIBRATION"
    return "OK"


def _build_configs(seed: int) -> list[dict]:
    """Deterministically build N_CONFIGS unique configurations."""
    rng = random.Random(seed)
    configs: list[dict] = []
    seen = set()
    attempts = 0
    while len(configs) < N_CONFIGS and attempts < N_CONFIGS * 20:
        attempts += 1
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        alpha = rng.choice(CELU_ALPHAS)
        key = (bucket, tuple(shape), dtype_name, stride_cat, alpha)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "config_id": len(configs),
            "shape_bucket": bucket,
            "shape": tuple(shape),
            "dtype_name": dtype_name,
            "stride_category": stride_cat,
            "alpha": alpha,
        })
    return configs


def _run_one(
    cfg: dict, data_seed: int,
) -> dict:
    """Run a single (config, seed) iteration. Returns a dict with status."""
    shape = cfg["shape"]
    dtype = DTYPES_BY_NAME[cfg["dtype_name"]]
    stride_cat = cfg["stride_category"]
    alpha = cfg["alpha"]

    out: dict = {
        "config_id": cfg["config_id"],
        "data_seed": data_seed,
        "status": "OK",
    }

    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        out["status"] = "UNSUPPORTED_BUILD_CPU"
        out["err"] = str(exc).splitlines()[0][:200]
        return out

    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=data_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
        out["status"] = "UNSUPPORTED_BUILD_MPS"
        out["err"] = str(exc).splitlines()[0][:200]
        return out

    if x_cpu.numel() == 0:
        out["status"] = "EMPTY"
        return out

    try:
        y_mps = F.celu(x_mps, alpha=alpha)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        out["status"] = "UNSUPPORTED_MPS_OP"
        out["err"] = str(exc).splitlines()[0][:200]
        return out

    try:
        y_cpu = F.celu(x_cpu, alpha=alpha)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        out["status"] = "UNSUPPORTED_CPU_OP"
        out["err"] = str(exc).splitlines()[0][:200]
        return out

    rel, denom = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
    abs_err = _max_abs_err(y_mps_cpu, y_cpu)
    atol, rtol = compute_tolerance(dtype, device_type="mps")

    out["max_abs_err"] = abs_err
    out["max_rel_err"] = rel
    out["denom_at_rel_max"] = denom
    out["atol"] = atol
    out["rtol"] = rtol
    out["x_layout_cpu"] = _stride_tag(x_cpu)
    out["x_layout_mps"] = _stride_tag(x_mps)
    out["status"] = _classify(abs_err, rel, denom, atol, rtol)
    return out


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
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz (v2) — SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()
    configs = _build_configs(seed=0xCE10)  # config-set seed independent of data seeds

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_empty = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    # Per-config seed votes
    diverge_seeds: dict[int, list[int]] = defaultdict(list)
    recal_seeds: dict[int, list[int]] = defaultdict(list)
    per_config_results: dict[int, list[dict]] = defaultdict(list)

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}
    aborted = False

    for cfg in configs:
        if time.monotonic() - started > TIME_BUDGET_S:
            print(f"[budget] aborting at config {cfg['config_id']}/{len(configs)}",
                  file=sys.stderr)
            aborted = True
            break
        for data_seed in SEEDS:
            iters_attempted += 1
            try:
                rec = _run_one(cfg, data_seed)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — halt rule
                print(f"[ERROR] cfg={cfg} seed={data_seed}: {exc!r}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return 2

            status = rec["status"]
            if status.startswith("UNSUPPORTED"):
                iters_unsupported += 1
                continue
            if status == "EMPTY":
                iters_completed += 1
                continue

            iters_completed += 1
            per_bucket_counts[cfg["shape_bucket"]] += 1
            per_dtype_counts[cfg["dtype_name"]] += 1
            per_stride_counts[cfg["stride_category"]] += 1

            max_abs_err_global = max(max_abs_err_global, rec["max_abs_err"])
            max_rel_err_global = max(max_rel_err_global, rec["max_rel_err"])

            per_config_results[cfg["config_id"]].append(rec)
            if status == "DIVERGENCE":
                diverge_seeds[cfg["config_id"]].append(data_seed)
            elif status == "RECALIBRATION":
                recal_seeds[cfg["config_id"]].append(data_seed)

    # ---- Classify per-config ----
    filable: list[dict] = []
    recalibration: list[dict] = []
    config_by_id = {c["config_id"]: c for c in configs}

    for cid, seeds_hit in diverge_seeds.items():
        if len(seeds_hit) >= FILABLE_MIN_SEEDS:
            cfg = config_by_id[cid]
            recs = per_config_results[cid]
            # Use the worst rec (largest abs_err) as canonical repro
            worst = max(recs, key=lambda r: r.get("max_abs_err", 0.0))
            filable.append({
                "config_id": cid,
                "shape": list(cfg["shape"]),
                "shape_bucket": cfg["shape_bucket"],
                "dtype": cfg["dtype_name"],
                "stride_category": cfg["stride_category"],
                "alpha": cfg["alpha"],
                "diverge_seed_count": len(seeds_hit),
                "diverge_seeds": seeds_hit,
                "max_abs_err": worst["max_abs_err"],
                "max_rel_err": worst["max_rel_err"],
                "denom_at_rel_max": worst["denom_at_rel_max"],
                "atol": worst["atol"],
                "rtol": worst["rtol"],
                "x_layout_cpu": worst["x_layout_cpu"],
                "x_layout_mps": worst["x_layout_mps"],
            })

    for cid, seeds_hit in recal_seeds.items():
        # Recalibration is a softer signal: report any config with >=3 seeds
        # in 1-10x band, OR any config with mixed RECAL+DIVERGE signals.
        total_signal = len(seeds_hit) + len(diverge_seeds.get(cid, []))
        if total_signal >= FILABLE_MIN_SEEDS and cid not in {f["config_id"] for f in filable}:
            cfg = config_by_id[cid]
            recs = per_config_results[cid]
            worst = max(recs, key=lambda r: r.get("max_abs_err", 0.0))
            recalibration.append({
                "config_id": cid,
                "shape": list(cfg["shape"]),
                "shape_bucket": cfg["shape_bucket"],
                "dtype": cfg["dtype_name"],
                "stride_category": cfg["stride_category"],
                "alpha": cfg["alpha"],
                "recal_seed_count": len(seeds_hit),
                "recal_seeds": seeds_hit,
                "max_abs_err": worst["max_abs_err"],
                "max_rel_err": worst["max_rel_err"],
                "denom_at_rel_max": worst["denom_at_rel_max"],
                "atol": worst["atol"],
                "rtol": worst["rtol"],
            })

    # Sort filable by abs_err (descending) and minimal numel for clean repros
    filable_sorted = sorted(
        filable,
        key=lambda d: (-d["max_abs_err"], _numel(d["shape"])),
    )
    top3 = filable_sorted[:3] if filable_sorted else _top3_recal(recalibration)

    elapsed = time.monotonic() - started
    summary = {
        "kernel": KERNEL,
        "status": "ABORTED_BUDGET" if aborted else "OK",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "divergences_filable": len(filable_sorted),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": top3,
        "elapsed_seconds": round(elapsed, 2),
        "n_configs": len(configs),
        "seeds": list(SEEDS),
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "filing_target": ("pytorch/pytorch" if filable_sorted else "none"),
    }

    _write_markdown(RESULTS_MD, summary, recalibration)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    return 0


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _top3_recal(recal_list: list[dict]) -> list[dict]:
    return sorted(recal_list, key=lambda d: -d["max_abs_err"])[:3]


def _write_markdown(path: Path, s: dict, recal_list: list[dict]) -> None:
    lines: list[str] = []
    lines.append(f"# {s['kernel']} — MPS fuzz report (v2)")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **Iters attempted:** {s['iters_attempted']}")
    lines.append(f"- **Iters completed:** {s['iters_completed']}")
    lines.append(f"- **Iters unsupported:** {s['iters_unsupported']}")
    lines.append(f"- **Divergences (FILABLE — ≥3 seeds, >10× tol):** "
                 f"{s['divergences_filable']}")
    lines.append(f"- **Divergences (RECALIBRATION — 1-10× tol):** "
                 f"{s['divergences_recalibration']}")
    lines.append(f"- **max_abs_err (global):** {s['max_abs_err']:.3e}")
    lines.append(f"- **max_rel_err (global):** {s['max_rel_err']:.3e}")
    lines.append(f"- **Configs × seeds:** {s['n_configs']} × {len(s['seeds'])} = "
                 f"{s['n_configs'] * len(s['seeds'])} planned")
    lines.append(f"- **Seeds:** {s['seeds']}")
    lines.append(f"- **Elapsed:** {s['elapsed_seconds']} s")
    lines.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    lines.append(f"- **Filing target:** `{s['filing_target']}`")
    lines.append("")
    lines.append("## Filtering rules")
    lines.append("")
    lines.append("- `max_abs_err > 10× atol` → DIVERGENCE (always).")
    lines.append("- `max_rel_err > 10× rtol` → DIVERGENCE only when "
                 "`denom_at_rel_max >= 1e-6` (else near-zero artefact).")
    lines.append("- `1× ≤ err ≤ 10× tol` → RECALIBRATION (xfail-eligible).")
    lines.append("- `< 1× tol` → OK.")
    lines.append("- A config is **FILABLE** iff DIVERGENCE on ≥3 of 5 seeds.")
    lines.append("")
    lines.append("## Sampling distribution (completed iters)")
    lines.append("")
    lines.append("| dimension | counts |")
    lines.append("|---|---|")
    lines.append(f"| shape bucket | {s['per_shape_bucket']} |")
    lines.append(f"| dtype | {s['per_dtype']} |")
    lines.append(f"| stride category | {s['per_stride_category']} |")
    lines.append("")
    lines.append("## Top 3 repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append("_No FILABLE divergences and no RECALIBRATION configs hit "
                     "≥3 seeds. CELU on MPS matches CPU within "
                     "gpucheck's per-dtype tolerance (with 2× MPS overlay)._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}` "
                         f"(bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(f"- **alpha:** {r['alpha']}")
            lines.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                         f"(atol={r['atol']:.2e}, "
                         f"ratio={r['max_abs_err']/r['atol']:.2f}×)")
            lines.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                         f"(rtol={r['rtol']:.2e}, "
                         f"ratio={r['max_rel_err']/r['rtol']:.2f}×)")
            lines.append(f"- **denom_at_rel_max:** "
                         f"{r['denom_at_rel_max']:.3e}")
            if "diverge_seed_count" in r:
                lines.append(f"- **divergent seeds:** {r['diverge_seeds']} "
                             f"({r['diverge_seed_count']}/5)")
            elif "recal_seed_count" in r:
                lines.append(f"- **recalibration seeds:** {r['recal_seeds']} "
                             f"({r['recal_seed_count']}/5) "
                             "[no FILABLE — top repro pulled from RECAL bucket]")
            if "x_layout_cpu" in r:
                lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
                lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append("")

    if recal_list:
        lines.append("## RECALIBRATION configs (xfail candidates)")
        lines.append("")
        lines.append("| dtype | shape | stride | alpha | abs_err | atol | "
                     "abs_ratio | seeds_hit |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(recal_list, key=lambda d: -d["max_abs_err"])[:20]:
            lines.append(
                f"| {r['dtype']} | {tuple(r['shape'])} | "
                f"{r['stride_category']} | {r['alpha']} | "
                f"{r['max_abs_err']:.2e} | {r['atol']:.1e} | "
                f"{r['max_abs_err']/r['atol']:.2f}× | "
                f"{r['recal_seed_count']}/5 |"
            )
        lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.nn.functional.celu` on CPU, FP32-promoted "
                 "for error metrics.")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance"
                 "(dtype, device_type='mps')` (carries the v1.0 MPS 2× overlay).")
    lines.append("- Stride categories: row_major, column_major, broadcast, "
                 "transpose, slice, non_contig, gather "
                 "(`gpucheck.fuzzing.strides`).")
    lines.append("- Alpha sweep: 1.0 (default), 0.5, 2.0 — exposes "
                 "alpha-handling bugs distinct from input-handling.")
    lines.append("- CUDA backend mocked (no NVIDIA GPU on host); cross-device "
                 "MPS-vs-CUDA comparison N/A by spec.")
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
