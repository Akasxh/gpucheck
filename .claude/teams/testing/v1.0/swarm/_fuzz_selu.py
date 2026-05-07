"""Fuzz the SELU kernel: MPS vs CPU reference, with v2 divergence filtering.

1000 iterations across seeds 0,1,2,3,4 — 200 distinct (shape, dtype, stride)
configs replayed under each of the 5 master seeds, so per-config reproducibility
across seeds can be measured directly.

Filtering rules (v2):
- max_rel_err > 10x rtol counts as divergence ONLY if denom_magnitude >= 1e-6
- max_abs_err > 10x atol always counts
- Both must reproduce under >=3 distinct master seeds to be FILABLE
- 1x..5x tolerance => TOLERANCE_RECALIBRATION (recommend xfail entry)
- < 1x tolerance => OK
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-selu/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_selu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.selu"
N_CONFIGS = 200
MASTER_SEEDS = (0, 1, 2, 3, 4)
N_ITERS = N_CONFIGS * len(MASTER_SEEDS)  # 1000
TIME_BUDGET_S = 12 * 60 - 30  # leave 30s for write + jsonl
CONFIG_RNG_SEED = 0xCEFE_5E10  # fixed: regenerates the SAME 200 configs every run

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


def _err_metrics(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_at_rel_max, ref_global_max).

    `denom_at_rel_max` is |b| at the element where rel-error peaks — this is
    the magnitude that decides whether the rel-error spike is a real divergence
    or a near-zero-denominator artifact.
    """
    if a_cpu.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    diff = (a - b).abs()
    abs_err = float(diff.max().item())
    denom = b.abs().clamp_min(1e-30)
    rel = diff / denom
    rel_flat = rel.flatten()
    idx = int(rel_flat.argmax().item())
    rel_err = float(rel_flat[idx].item())
    denom_at_rel_max = float(b.abs().flatten()[idx].item())
    ref_global_max = float(b.abs().max().item())
    return abs_err, rel_err, denom_at_rel_max, ref_global_max


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(abs_err: float, rel_err: float, denom_mag: float,
              atol: float, rtol: float) -> tuple[str, str]:
    """Return (bucket, dominant_signal).

    bucket ∈ {OK, RECALIBRATION, DIVERGENCE_CANDIDATE}
    dominant_signal: which axis pushed it past the threshold (or 'none').
    """
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = rel_err / rtol if rtol > 0 else 0.0

    abs_filable = abs_ratio > 10.0
    rel_filable = (rel_ratio > 10.0) and (denom_mag >= 1e-6)
    if abs_filable or rel_filable:
        sig = "abs" if abs_filable and abs_ratio >= rel_ratio else "rel"
        return ("DIVERGENCE_CANDIDATE", sig)

    if abs_ratio > 1.0 and abs_ratio <= 5.0:
        return ("RECALIBRATION", "abs")
    if rel_ratio > 1.0 and rel_ratio <= 5.0 and denom_mag >= 1e-6:
        return ("RECALIBRATION", "rel")
    if 5.0 < abs_ratio <= 10.0:
        return ("RECALIBRATION", "abs")
    if 5.0 < rel_ratio <= 10.0 and denom_mag >= 1e-6:
        return ("RECALIBRATION", "rel")

    return ("OK", "none")


def _build_configs() -> list[dict]:
    """Deterministic list of 200 (shape, dtype, stride_category, bucket) configs."""
    rng = random.Random(CONFIG_RNG_SEED)
    configs = []
    for cidx in range(N_CONFIGS):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        configs.append({
            "cidx": cidx,
            "bucket": bucket,
            "shape": tuple(shape),
            "dtype_name": dtype_name,
            "stride_cat": stride_cat,
        })
    return configs


def _data_seed(master_seed: int, cidx: int) -> int:
    """Combine a master seed with the config index into a torch-friendly seed."""
    return (master_seed * 1_000_003 + cidx * 7919 + 0x9E3779B1) & 0x7FFFFFFF


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "agent": "kernel-fuzzer-selu",
            "status": "SKIPPED",
            "reason": "torch.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available on this host.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()
    configs = _build_configs()

    iters_attempted = 0
    iters_completed = 0
    unsupported = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    # Per-config aggregation: cidx -> {seeds_seen: set, samples: [...]}
    per_config: dict[int, dict] = {}

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    aborted_early = False
    for master_seed in MASTER_SEEDS:
        if aborted_early:
            break
        for cfg in configs:
            if time.monotonic() - started > TIME_BUDGET_S:
                print(
                    f"[budget] aborting at iter {iters_attempted}/{N_ITERS} "
                    f"(elapsed {time.monotonic() - started:.1f}s)",
                    file=sys.stderr,
                )
                aborted_early = True
                break

            iters_attempted += 1
            cidx = cfg["cidx"]
            bucket = cfg["bucket"]
            shape = cfg["shape"]
            dtype_name = cfg["dtype_name"]
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = cfg["stride_cat"]
            data_seed = _data_seed(master_seed, cidx)

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            try:
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=data_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    unsupported += 1
                    print(f"[unsupported-cpu-build] iter={iters_attempted} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                          file=sys.stderr)
                    continue

                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=data_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError, ValueError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    unsupported += 1
                    print(f"[unsupported-mps-build] iter={iters_attempted} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                          file=sys.stderr)
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.selu(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    unsupported += 1
                    print(f"[unsupported-mps-op] iter={iters_attempted} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                          file=sys.stderr)
                    continue

                try:
                    y_cpu = F.selu(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    unsupported += 1
                    print(f"[unsupported-cpu-op] iter={iters_attempted} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                          file=sys.stderr)
                    continue

                iters_completed += 1

                absdiff, rel, denom_mag, ref_max = _err_metrics(y_mps_cpu, y_cpu)
                max_rel_err_global = max(max_rel_err_global, rel)
                max_abs_err_global = max(max_abs_err_global, absdiff)

                atol, rtol = compute_tolerance(dtype, device_type="mps")
                bucket_class, sig = _classify(absdiff, rel, denom_mag, atol, rtol)

                rec = {
                    "master_seed": master_seed,
                    "data_seed": data_seed,
                    "cidx": cidx,
                    "bucket": bucket,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "max_abs_err": absdiff,
                    "max_rel_err": rel,
                    "denom_mag": denom_mag,
                    "ref_global_max": ref_max,
                    "atol": atol,
                    "rtol": rtol,
                    "abs_ratio": (absdiff / atol) if atol > 0 else 0.0,
                    "rel_ratio": (rel / rtol) if rtol > 0 else 0.0,
                    "class": bucket_class,
                    "signal": sig,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                }
                slot = per_config.setdefault(
                    cidx,
                    {
                        "cidx": cidx,
                        "bucket": bucket,
                        "shape": list(shape),
                        "dtype": dtype_name,
                        "stride_category": stride_cat,
                        "atol": atol,
                        "rtol": rtol,
                        "samples": [],
                        "seeds_with_divergence": set(),
                        "seeds_with_recal": set(),
                    },
                )
                slot["samples"].append(rec)
                if bucket_class == "DIVERGENCE_CANDIDATE":
                    slot["seeds_with_divergence"].add(master_seed)
                if bucket_class == "RECALIBRATION":
                    slot["seeds_with_recal"].add(master_seed)

                if bucket_class != "OK":
                    print(f"[{bucket_class}] iter={iters_attempted} seed={master_seed} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                          f"abs={absdiff:.3e} rel={rel:.3e} denom={denom_mag:.3e} "
                          f"(atol={atol:.2e} rtol={rtol:.2e}, sig={sig})",
                          file=sys.stderr)

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[ERROR] iter={iters_attempted} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape}: {exc!r}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return 2

    elapsed = time.monotonic() - started

    # Classify each config: FILABLE if divergence under >=3 master seeds.
    filable_configs: list[dict] = []
    recalibration_configs: list[dict] = []
    for slot in per_config.values():
        n_div_seeds = len(slot["seeds_with_divergence"])
        n_recal_seeds = len(slot["seeds_with_recal"])
        if n_div_seeds >= 3:
            worst = max(slot["samples"], key=lambda r: max(r["abs_ratio"], r["rel_ratio"]))
            filable_configs.append({**worst,
                                    "n_divergent_seeds": n_div_seeds,
                                    "n_total_seeds": len(MASTER_SEEDS)})
        elif n_div_seeds >= 1 or n_recal_seeds >= 1:
            worst = max(slot["samples"], key=lambda r: max(r["abs_ratio"], r["rel_ratio"]))
            if worst["class"] != "OK":
                recalibration_configs.append({
                    **worst,
                    "n_divergent_seeds": n_div_seeds,
                    "n_recal_seeds": n_recal_seeds,
                    "n_total_seeds": len(MASTER_SEEDS),
                })

    filable_configs.sort(
        key=lambda r: (-max(r["abs_ratio"], r["rel_ratio"]), _numel(r["shape"])),
    )
    recalibration_configs.sort(
        key=lambda r: (-max(r["abs_ratio"], r["rel_ratio"]), _numel(r["shape"])),
    )

    top3 = filable_configs[:3] if filable_configs else recalibration_configs[:3]

    summary = {
        "agent": "kernel-fuzzer-selu",
        "kernel": KERNEL,
        "status": "OK" if not aborted_early else "PARTIAL",
        "backend_mps": True,
        "backend_cuda": "mocked (no NVIDIA GPU on host)",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": unsupported,
        "divergences_filable": len(filable_configs),
        "divergences_recalibration": len(recalibration_configs),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "n_distinct_configs_seen": len(per_config),
        "master_seeds": list(MASTER_SEEDS),
        "filing_rule": "FILABLE = divergence reproduces under >=3 master seeds",
        "top_3_repros": top3,
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "recommended_filing_target": _recommend_target(filable_configs),
        "elapsed_s": round(elapsed, 2),
        "torch_version": torch.__version__,
        "host": "darwin/arm64 (Apple Silicon)",
        "config_rng_seed_hex": f"0x{CONFIG_RNG_SEED:x}",
    }

    _write_markdown(RESULTS_MD, summary, filable_configs, recalibration_configs)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    return 0


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _recommend_target(filable: list[dict]) -> str:
    if not filable:
        return "none"
    return "pytorch/pytorch"


def _write_markdown(path: Path, s: dict,
                    filable: list[dict], recal: list[dict]) -> None:
    L: list[str] = []
    L.append(f"# {s['kernel']} — MPS fuzz report (v2)")
    L.append("")
    L.append(f"- **kernel:** `{s['kernel']}`")
    L.append(f"- **status:** {s['status']}")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    L.append(f"- **divergences_filable:** {s['divergences_filable']}")
    L.append(f"- **divergences_recalibration:** {s['divergences_recalibration']}")
    L.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    L.append(f"- **max_rel_err:** {s['max_rel_err']:.3e}")
    L.append(f"- **n_distinct_configs_seen:** {s['n_distinct_configs_seen']} / {N_CONFIGS}")
    L.append(f"- **master_seeds:** {s['master_seeds']}")
    L.append(f"- **filing_rule:** {s['filing_rule']}")
    L.append(f"- **recommended_filing_target:** `{s['recommended_filing_target']}`")
    L.append(f"- **elapsed:** {s['elapsed_s']} s (budget {TIME_BUDGET_S}s)")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}, "
             f"config_seed: {s['config_rng_seed_hex']}")
    L.append("")
    L.append("## Sampling distribution")
    L.append("")
    L.append("| dimension | counts |")
    L.append("|---|---|")
    L.append(f"| shape bucket | {s['per_shape_bucket']} |")
    L.append(f"| dtype | {s['per_dtype']} |")
    L.append(f"| stride category | {s['per_stride_category']} |")
    L.append("")
    L.append("## Top 3 minimal repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No divergences in any bucket — selu(MPS) matched CPU within 1× tolerance "
                 "for every (shape, dtype, stride) config._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i}  ({r['class']}, signal={r['signal']})")
            L.append("")
            L.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride category:** `{r['stride_category']}`")
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e}  (atol={r['atol']:.2e}, "
                     f"abs_ratio={r['abs_ratio']:.2f}×)")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e}  (rtol={r['rtol']:.2e}, "
                     f"rel_ratio={r['rel_ratio']:.2f}×)")
            L.append(f"- **denom_mag:** {r['denom_mag']:.3e}  "
                     f"(threshold for rel-divergence: ≥1e-6)")
            n_div = r.get("n_divergent_seeds", 0)
            n_recal = r.get("n_recal_seeds", 0)
            n_tot = r.get("n_total_seeds", len(MASTER_SEEDS))
            L.append(f"- **reproducibility:** divergent in {n_div}/{n_tot} master seeds, "
                     f"recalibration in {n_recal}/{n_tot}")
            L.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            L.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            L.append(f"- **example data_seed:** {r['data_seed']} "
                     f"(master_seed={r['master_seed']}, cidx={r['cidx']})")
            L.append("")

    if filable:
        L.append("## All FILABLE configs (≥3 seeds reproduced)")
        L.append("")
        L.append("| cidx | shape | dtype | stride | abs_ratio× | rel_ratio× | denom_mag | seeds |")
        L.append("|------|-------|-------|--------|------------|------------|-----------|-------|")
        for r in filable:
            L.append(
                f"| {r['cidx']} | `{tuple(r['shape'])}` | {r['dtype']} | "
                f"{r['stride_category']} | {r['abs_ratio']:.2f} | "
                f"{r['rel_ratio']:.2f} | {r['denom_mag']:.2e} | "
                f"{r['n_divergent_seeds']}/{r['n_total_seeds']} |"
            )
        L.append("")

    if recal:
        L.append("## TOLERANCE_RECALIBRATION configs (1×–10× over tolerance, "
                 "or <3 seed reproduction)")
        L.append("")
        L.append("| cidx | shape | dtype | stride | abs_ratio× | rel_ratio× | "
                 "denom_mag | div_seeds | recal_seeds |")
        L.append("|------|-------|-------|--------|------------|------------|"
                 "-----------|-----------|-------------|")
        for r in recal[:25]:
            L.append(
                f"| {r['cidx']} | `{tuple(r['shape'])}` | {r['dtype']} | "
                f"{r['stride_category']} | {r['abs_ratio']:.2f} | "
                f"{r['rel_ratio']:.2f} | {r['denom_mag']:.2e} | "
                f"{r['n_divergent_seeds']}/{r['n_total_seeds']} | "
                f"{r['n_recal_seeds']}/{r['n_total_seeds']} |"
            )
        if len(recal) > 25:
            L.append(f"| … | … (+{len(recal) - 25} more) | | | | | | | |")
        L.append("")
        L.append("**Recommendation:** for any config in this table that is reproducibly "
                 "1×–5× over tolerance, add a curated entry to "
                 "`[tool.gpucheck.mps.xfail]` rather than further inflating the global "
                 "MPS multiplier. (See `assertions/tolerances.py:35` for the current 2× overlay.)")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- **Reference:** `torch.nn.functional.selu` on CPU (FP32 promotion in error calc).")
    L.append("- **Tolerance:** "
             "`gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` "
             "(2× MPS overlay applied).")
    L.append("- **SELU is element-wise**, so no `sqrt(k/128)` matmul scaling.")
    L.append("- **Divergence buckets** (per the v2 spec):")
    L.append("    - `OK` — error ≤ 1× tolerance.")
    L.append("    - `RECALIBRATION` — error 1×–10× tolerance, OR rel-error >10× rtol but "
             "denom_mag < 1e-6 (a near-zero artifact, not real numerics).")
    L.append("    - `DIVERGENCE_CANDIDATE` — abs > 10× atol, OR (rel > 10× rtol AND "
             "denom_mag ≥ 1e-6).")
    L.append("    - `FILABLE` — a `DIVERGENCE_CANDIDATE` reproduced under ≥3 of the "
             "5 master seeds.")
    L.append("- **Stride categories drawn:** "
             "row_major, column_major, broadcast, transpose, slice, non_contig, gather "
             "(see `gpucheck.fuzzing.strides`).")
    L.append("- **CUDA backend mocked** (no NVIDIA GPU on host); MPS-vs-CUDA comparison N/A.")
    L.append(
        "- **Reproducibility:** the 200 (shape, dtype, stride) configs are deterministic "
        f"from `CONFIG_RNG_SEED=0x{CONFIG_RNG_SEED:x}`. Each config is run once per "
        "master_seed in {0, 1, 2, 3, 4} (5 x 200 = 1000 iterations)."
    )
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
