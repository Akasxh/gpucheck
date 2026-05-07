"""Fuzz the torch.tanh kernel: MPS vs CPU reference (v2 swarm spec).

Divergence filtering (v2):
  - max_abs_err > 10x tolerance     -> FILABLE candidate (always)
  - max_rel_err > 10x tolerance AND |y_ref| >= 1e-6 -> FILABLE candidate
    (the denom_magnitude guard suppresses near-zero artifacts where tanh
     crosses through 0 and any tiny absolute drift balloons the relative)
  - 1x < ratio <= 5x                 -> TOLERANCE_RECALIBRATION
  - ratio <= 1x                      -> OK
  - FILABLE only counts as a real bug when reproducible across >= 3 seeds.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Ensure gpucheck (worktree) is importable
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-tanh/src")
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.shapes import (  # noqa: E402
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_tanh.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.tanh"
N_ITERS = 1000
SEEDS = (0, 1, 2, 3, 4)
TIME_BUDGET_S = 12 * 60 - 30  # leave 30 s for writing results
MASTER_SEED = 0xC0FFEE  # for sampling combos / inputs

# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------

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


def _max_rel_err_with_denom(
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    denom_floor: float = 1e-6,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_max).

    denom_magnitude_at_max is |b| at the worst-case rel-err position; the v2
    spec uses this to decide whether the rel-err is signal or near-zero noise.
    """
    a = a_cpu.to(torch.float32).flatten()
    b = b_cpu.to(torch.float32).flatten()
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    rel = diff / denom
    max_idx = int(rel.argmax().item())
    return float(rel[max_idx].item()), float(b.abs()[max_idx].item())


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(
    abs_err: float, rel_err: float, denom_mag: float, atol: float, rtol: float,
) -> str:
    """Return one of: OK, RECALIBRATION, FILABLE_CANDIDATE."""
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    rel_ratio = rel_err / rtol if rtol > 0 else 0.0

    # Apply denom_magnitude guard to rel-err: low denom -> ignore rel signal
    if denom_mag < 1e-6:
        rel_ratio = 0.0

    worst_ratio = max(abs_ratio, rel_ratio)

    if worst_ratio <= 1.0:
        return "OK"
    if worst_ratio <= 5.0:
        return "RECALIBRATION"
    if worst_ratio > 10.0:
        return "FILABLE_CANDIDATE"
    # 5 < ratio <= 10: between recalibration and filable; treat as recalibration
    return "RECALIBRATION"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.mps.is_available() is False",
            "iterations_attempted": 0,
            "iterations_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    rng = random.Random(MASTER_SEED)
    started = time.monotonic()

    iterations_attempted = 0
    iterations_completed = 0
    unsupported = 0

    # Track every classified divergence by combo key for cross-seed dedup
    # combo_key = (bucket, shape_tuple, dtype_name, stride_cat)
    combo_seeds: dict[tuple, set[int]] = defaultdict(set)
    combo_records: dict[tuple, list[dict]] = defaultdict(list)
    combo_class: dict[tuple, str] = {}  # most severe class seen

    max_rel_err_global = 0.0
    max_abs_err_global = 0.0
    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    for i in range(N_ITERS):
        if time.monotonic() - started > TIME_BUDGET_S:
            print(
                f"[budget] aborting at iter {i}/{N_ITERS} "
                f"(elapsed {time.monotonic() - started:.1f}s)",
                file=sys.stderr,
            )
            break

        iterations_attempted += 1

        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        dtype = DTYPES_BY_NAME[dtype_name]
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        seed = rng.choice(SEEDS)

        per_bucket_counts[bucket] += 1
        per_dtype_counts[dtype_name] += 1
        per_stride_counts[stride_cat] += 1

        try:
            try:
                x_cpu = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="cpu", seed=seed,
                )
                x_mps = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="mps", seed=seed,
                )
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-build] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            if x_cpu.numel() == 0:
                iterations_completed += 1
                continue

            try:
                y_mps = torch.tanh(x_mps)
                torch.mps.synchronize()
                y_mps_cpu = y_mps.detach().to("cpu")
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-mps] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            try:
                y_cpu = torch.tanh(x_cpu)
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                msg = str(exc).splitlines()[0][:200]
                unsupported += 1
                print(
                    f"[unsupported-cpu] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {msg}",
                    file=sys.stderr,
                )
                continue

            iterations_completed += 1

            rel, denom_mag = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
            absdiff = _max_abs_err(y_mps_cpu, y_cpu)
            max_rel_err_global = max(max_rel_err_global, rel)
            max_abs_err_global = max(max_abs_err_global, absdiff)

            atol, rtol = compute_tolerance(dtype, device_type="mps")
            klass = _classify(absdiff, rel, denom_mag, atol, rtol)

            if klass == "OK":
                continue

            combo_key = (bucket, tuple(shape), dtype_name, stride_cat)
            combo_seeds[combo_key].add(seed)

            rec = {
                "iter": i,
                "shape": list(shape),
                "dtype": dtype_name,
                "stride_category": stride_cat,
                "shape_bucket": bucket,
                "max_abs_err": absdiff,
                "max_rel_err": rel,
                "denom_magnitude_at_max_rel": denom_mag,
                "atol": atol,
                "rtol": rtol,
                "abs_ratio": absdiff / atol if atol > 0 else 0.0,
                "rel_ratio_effective": (
                    (rel / rtol) if (rtol > 0 and denom_mag >= 1e-6) else 0.0
                ),
                "x_layout_cpu": _stride_tag(x_cpu),
                "x_layout_mps": _stride_tag(x_mps),
                "seed": seed,
                "class": klass,
            }
            combo_records[combo_key].append(rec)
            # Promote class severity if a worse one shows up across seeds.
            sev = {"RECALIBRATION": 1, "FILABLE_CANDIDATE": 2}
            if combo_key not in combo_class or sev[klass] > sev[combo_class[combo_key]]:
                combo_class[combo_key] = klass

            print(
                f"[{klass}] iter={i} seed={seed} {bucket}/{dtype_name}/{stride_cat} "
                f"shape={shape} abs={absdiff:.3e} rel={rel:.3e} denom={denom_mag:.3e} "
                f"(atol={atol:.2e} rtol={rtol:.2e})",
                file=sys.stderr,
            )

        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] iter={i} {bucket}/{dtype_name}/{stride_cat} "
                f"shape={shape}: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return 2

    elapsed = time.monotonic() - started

    # Apply 3-seed reproducibility gate to FILABLE_CANDIDATE -> FILABLE
    filable_combos: list[dict] = []
    recalibration_combos: list[dict] = []
    for combo_key, klass in combo_class.items():
        seeds_seen = combo_seeds[combo_key]
        records = combo_records[combo_key]
        worst = max(records, key=lambda r: max(r["abs_ratio"], r["rel_ratio_effective"]))
        entry = {
            "combo": {
                "shape_bucket": combo_key[0],
                "shape": list(combo_key[1]),
                "dtype": combo_key[2],
                "stride_category": combo_key[3],
            },
            "class": klass,
            "seeds_diverged": sorted(seeds_seen),
            "n_seeds_diverged": len(seeds_seen),
            "worst_repro": worst,
            "all_repro_seeds": [r["seed"] for r in records],
        }
        if klass == "FILABLE_CANDIDATE" and len(seeds_seen) >= 3:
            entry["class"] = "FILABLE"
            filable_combos.append(entry)
        elif klass == "FILABLE_CANDIDATE":
            # Failed reproducibility gate -> downgrade to RECALIBRATION
            entry["class"] = "RECALIBRATION_UNREPRODUCED_FILABLE"
            recalibration_combos.append(entry)
        else:
            recalibration_combos.append(entry)

    # Sort by severity (abs_ratio + rel_ratio_effective) for "top 3"
    def _score(entry: dict) -> float:
        w = entry["worst_repro"]
        return max(w["abs_ratio"], w["rel_ratio_effective"])

    filable_sorted = sorted(filable_combos, key=_score, reverse=True)
    recal_sorted = sorted(recalibration_combos, key=_score, reverse=True)
    top3 = (filable_sorted + recal_sorted)[:3]

    summary = {
        "kernel": KERNEL,
        "status": "OK",
        "iters_attempted": iterations_attempted,
        "iters_completed": iterations_completed,
        "iters_unsupported": unsupported,
        "divergences_filable": len(filable_combos),
        "divergences_recalibration": len(recalibration_combos),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "elapsed_seconds": round(elapsed, 2),
        "top_3_repros": top3,
        "all_filable": filable_sorted,
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "recommended_filing_target": (
            "pytorch/pytorch" if filable_combos else "none"
        ),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "master_seed": MASTER_SEED,
        "input_seeds": list(SEEDS),
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    return 0


def _write_markdown(path: Path, s: dict) -> None:
    lines = []
    lines.append(f"# {s['kernel']} - MPS fuzz report (v2)")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    lines.append(f"- **iters_attempted:** {s['iters_attempted']}")
    lines.append(f"- **iters_completed:** {s['iters_completed']}")
    lines.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    lines.append(f"- **divergences_filable:** {s['divergences_filable']}")
    lines.append(f"- **divergences_recalibration:** {s['divergences_recalibration']}")
    lines.append(f"- **max_abs_err (global):** {s['max_abs_err']:.3e}")
    lines.append(f"- **max_rel_err (global):** {s['max_rel_err']:.3e}")
    lines.append(f"- **elapsed:** {s['elapsed_seconds']} s")
    lines.append(f"- **recommended filing target:** `{s['recommended_filing_target']}`")
    lines.append(
        f"- **torch:** {s['torch_version']}, host: {s['host']}, "
        f"master_seed: 0x{s['master_seed']:x}, input_seeds: {s['input_seeds']}"
    )
    lines.append("")
    lines.append("## Sampling distribution")
    lines.append("")
    lines.append("| dimension | counts |")
    lines.append("|---|---|")
    lines.append(f"| shape bucket | {s['per_shape_bucket']} |")
    lines.append(f"| dtype | {s['per_dtype']} |")
    lines.append(f"| stride category | {s['per_stride_category']} |")
    lines.append("")
    lines.append("## Divergence classification (v2 spec)")
    lines.append("")
    lines.append("- **FILABLE** — ratio > 10x tolerance (abs always; rel only when "
                 "|y_ref| >= 1e-6) AND reproducible across >= 3 seeds.")
    lines.append("- **RECALIBRATION** — 1x < ratio <= 5x tolerance, OR a >10x "
                 "FILABLE_CANDIDATE that failed the >= 3-seed reproducibility gate.")
    lines.append("- **OK** — ratio <= 1x tolerance.")
    lines.append("")
    lines.append("## Top 3 repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append("_No divergences exceeded the 1x tolerance threshold (with MPS "
                     "2x multiplier from `assertions/tolerances.py:35`)._")
    else:
        for i, entry in enumerate(s["top_3_repros"], 1):
            c = entry["combo"]
            w = entry["worst_repro"]
            lines.append(f"### Repro #{i} — {entry['class']}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(c['shape'])}`  (bucket: `{c['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{c['dtype']}`")
            lines.append(f"- **stride category:** `{c['stride_category']}`")
            lines.append(f"- **seeds diverged:** {entry['seeds_diverged']} "
                         f"({entry['n_seeds_diverged']}/5)")
            lines.append(f"- **max abs err:** {w['max_abs_err']:.3e}  "
                         f"(atol={w['atol']:.2e}, ratio={w['abs_ratio']:.2f}x)")
            lines.append(f"- **max rel err:** {w['max_rel_err']:.3e}  "
                         f"(rtol={w['rtol']:.2e}, "
                         f"effective_ratio={w['rel_ratio_effective']:.2f}x)")
            lines.append(f"- **denom magnitude at max-rel:** {w['denom_magnitude_at_max_rel']:.3e}  "
                         f"(>=1e-6 required for rel signal to count)")
            lines.append(f"- **CPU layout:** `{w['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{w['x_layout_mps']}`")
            lines.append(f"- **seed at worst:** {w['seed']}")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Reference: `torch.tanh` on CPU (FP32 promotion for accuracy comparison).")
    lines.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, "
                 "device_type='mps')` — applies the PROVISIONAL 2x MPS overlay.")
    lines.append("- `tanh` is elementwise with output bounded in [-1, 1]; no `k_dim` "
                 "scaling applies. Near-zero outputs (`tanh(0) = 0`) make rel-err noisy, "
                 "so the v2 spec's `denom_magnitude >= 1e-6` guard is essential here.")
    lines.append("- Stride categories: row_major, column_major, broadcast, transpose, "
                 "slice, non_contig, gather (see `gpucheck.fuzzing.strides`).")
    lines.append("- bf16 IS exercised — torch.tanh supports bf16 on MPS in torch 2.11.")
    lines.append("- CUDA backend is mocked (no NVIDIA GPU present); cross-device "
                 "MPS-vs-CUDA comparison reported as N/A by spec.")
    lines.append("")
    if s["all_filable"]:
        lines.append("## All FILABLE combos")
        lines.append("")
        for entry in s["all_filable"]:
            c = entry["combo"]
            w = entry["worst_repro"]
            lines.append(f"- `{c['dtype']}/{c['stride_category']}/{tuple(c['shape'])}` "
                         f"seeds={entry['seeds_diverged']} "
                         f"abs_ratio={w['abs_ratio']:.2f}x "
                         f"rel_ratio={w['rel_ratio_effective']:.2f}x")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
