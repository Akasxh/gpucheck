"""Fuzz torch.nn.functional.hardsigmoid: MPS vs CPU reference.

Spec (v2 swarm, kernel-fuzzer-hardsigmoid):
- 1000 total iters distributed across seeds 0..4 (200 per seed)
- dtypes fp32 / fp16 / bf16 (all supported by MPS hardsigmoid as of torch 2.11)
- divergence classification:
    * max_rel_err  > 10x tol  AND denom_magnitude >= 1e-6   -> FILABLE candidate
    * max_abs_err  > 10x tol                                -> FILABLE candidate
      (FILABLE only if reproducible across >=3 of the 5 seeds for the same
       (dtype, stride_category, shape_bucket) signature)
    * 1x..5x tol                                            -> TOLERANCE_RECALIBRATION
    * <1x tol                                               -> OK
- 12 minute wall budget. Write what we have on timeout.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-hardsigmoid/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_hardsigmoid.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.hardsigmoid"
SEEDS = (0, 1, 2, 3, 4)
ITERS_PER_SEED = 200  # 5 * 200 = 1000
WALL_BUDGET_S = 12 * 60 - 30  # leave 30 s for writing results

# Sampling space (kept small/cheap — hardsigmoid is elementwise saturating).
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


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    if a_cpu.numel() == 0:
        return 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    return float((a - b).abs().max().item())


def _max_rel_err_filtered(a_cpu: torch.Tensor, b_cpu: torch.Tensor,
                          denom_floor: float = 1e-6) -> tuple[float, float]:
    """Return (max_rel_err, denom_at_max).

    Only samples whose |reference| (denom magnitude) >= ``denom_floor``
    contribute. This avoids near-zero-denominator artifacts. If no samples
    qualify, returns (0.0, 0.0).
    """
    if a_cpu.numel() == 0:
        return 0.0, 0.0
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    denom = b.abs()
    mask = denom >= denom_floor
    if not bool(mask.any()):
        return 0.0, 0.0
    diff = (a - b).abs()[mask]
    den = denom[mask]
    rel = diff / den
    idx = int(rel.argmax().item())
    return float(rel[idx].item()), float(den[idx].item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available on host.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_errored = 0

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0  # already denom-filtered

    # records: list[dict] all over-tolerance hits with classification
    records: list[dict] = []
    # signature -> set of seeds that hit max_*_err > 10x tol (FILABLE candidate band)
    filable_seed_hits: dict[tuple, set[int]] = defaultdict(set)
    # signature -> sample worst record (for repro picks)
    worst_per_sig: dict[tuple, dict] = {}

    per_bucket = {b: 0 for b in BUCKET_NAMES}
    per_dtype = {d: 0 for d in DTYPE_NAMES}
    per_stride = {s: 0 for s in STRIDE_CATEGORIES}

    for seed in SEEDS:
        rng = random.Random(seed)
        for i in range(ITERS_PER_SEED):
            if time.monotonic() - started > WALL_BUDGET_S:
                print(f"[budget] aborting at seed={seed} iter={i}",
                      file=sys.stderr)
                break

            iters_attempted += 1

            bucket = rng.choice(BUCKET_NAMES)
            shape = rng.choice(SHAPE_BUCKETS[bucket])
            dtype_name = rng.choice(DTYPE_NAMES)
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = rng.choice(STRIDE_CATEGORIES)

            per_bucket[bucket] += 1
            per_dtype[dtype_name] += 1
            per_stride[stride_cat] += 1

            inner_seed = rng.randrange(2**31 - 1)

            try:
                x_cpu = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="cpu", seed=inner_seed,
                )
                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=inner_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(f"[unsupported-build] seed={seed} iter={i} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                          f"{str(exc).splitlines()[0][:200]}", file=sys.stderr)
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.hardsigmoid(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(f"[unsupported-mps] seed={seed} iter={i} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                          f"{str(exc).splitlines()[0][:200]}", file=sys.stderr)
                    continue

                try:
                    y_cpu = F.hardsigmoid(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(f"[unsupported-cpu] seed={seed} iter={i} "
                          f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                          f"{str(exc).splitlines()[0][:200]}", file=sys.stderr)
                    continue

                iters_completed += 1

                abs_err = _max_abs_err(y_mps_cpu, y_cpu)
                rel_err, denom_at_max = _max_rel_err_filtered(
                    y_mps_cpu, y_cpu, denom_floor=1e-6,
                )
                max_abs_err_global = max(max_abs_err_global, abs_err)
                max_rel_err_global = max(max_rel_err_global, rel_err)

                atol, rtol = compute_tolerance(dtype, device_type="mps")

                # Classification thresholds (per task spec).
                #   FILABLE    -> abs_err > 10*atol  OR  (rel_err > 10*rtol AND denom>=1e-6)
                #   RECALIB    -> any error in (1x .. 5x] tol band
                #   OK         -> below 1x tol
                abs_ratio = abs_err / atol if atol > 0 else 0.0
                rel_ratio = rel_err / rtol if rtol > 0 else 0.0
                # rel only counts when the denom mask actually qualified
                rel_band_active = rel_err > 0.0  # denom>=1e-6 sample exists

                filable_band = (abs_ratio > 10.0) or (rel_band_active and rel_ratio > 10.0)
                recalib_band = (
                    (1.0 < abs_ratio <= 5.0)
                    or (rel_band_active and 1.0 < rel_ratio <= 5.0)
                )

                if filable_band or recalib_band:
                    classification = "FILABLE_CANDIDATE" if filable_band else "TOLERANCE_RECALIBRATION"
                    sig = (dtype_name, stride_cat, bucket, tuple(shape))
                    rec = {
                        "seed": seed,
                        "iter": i,
                        "shape": list(shape),
                        "shape_bucket": bucket,
                        "dtype": dtype_name,
                        "stride_category": stride_cat,
                        "max_abs_err": abs_err,
                        "max_rel_err": rel_err,
                        "denom_at_max_rel": denom_at_max,
                        "abs_ratio_to_atol": abs_ratio,
                        "rel_ratio_to_rtol": rel_ratio,
                        "atol": atol,
                        "rtol": rtol,
                        "classification": classification,
                        "x_layout_cpu": _stride_tag(x_cpu),
                        "x_layout_mps": _stride_tag(x_mps),
                        "inner_seed": inner_seed,
                    }
                    records.append(rec)

                    if filable_band:
                        filable_seed_hits[sig].add(seed)
                        prev = worst_per_sig.get(sig)
                        # keep the worst (max max_abs_err) per signature
                        if (prev is None) or (rec["max_abs_err"] > prev["max_abs_err"]):
                            worst_per_sig[sig] = rec

                        print(f"[FILABLE-CAND] seed={seed} iter={i} "
                              f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                              f"abs={abs_err:.3e}/{atol:.2e} rel={rel_err:.3e}/{rtol:.2e}",
                              file=sys.stderr)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                iters_errored += 1
                print(f"[ERROR] seed={seed} iter={i} {bucket}/{dtype_name}/{stride_cat} "
                      f"shape={shape}: {exc!r}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # Halt-and-report rule: surface the failure non-zero.
                _emit(
                    started,
                    iters_attempted, iters_completed, iters_unsupported, iters_errored,
                    max_abs_err_global, max_rel_err_global,
                    records, filable_seed_hits, worst_per_sig,
                    per_bucket, per_dtype, per_stride,
                    status=f"ERROR ({type(exc).__name__})",
                )
                return 2
        else:
            continue
        # only reach here if inner break (budget) -> stop outer too
        break

    # FILABLE = >=3 distinct seeds hit the same signature in the >10x band.
    filable_signatures = {
        sig: seeds for sig, seeds in filable_seed_hits.items() if len(seeds) >= 3
    }

    # Build top-3 repros: pick worst record per filable signature, sorted by abs ratio.
    filable_repros = sorted(
        (worst_per_sig[sig] | {"reproducing_seeds": sorted(filable_seed_hits[sig])}
         for sig in filable_signatures),
        key=lambda r: (-r["abs_ratio_to_atol"], _numel(r["shape"])),
    )
    # Fallback: if no FILABLE, surface top-3 worst RECALIB hits as informative.
    if not filable_repros:
        recal = [r for r in records if r["classification"] == "TOLERANCE_RECALIBRATION"]
        recal.sort(key=lambda r: (-max(r["abs_ratio_to_atol"], r["rel_ratio_to_rtol"]),
                                  _numel(r["shape"])))
        top3 = recal[:3]
        top3_label = "top_3_recalibration_hits"
    else:
        top3 = filable_repros[:3]
        top3_label = "top_3_filable_repros"

    divergences_filable = len(filable_signatures)
    divergences_recalibration = sum(
        1 for r in records if r["classification"] == "TOLERANCE_RECALIBRATION"
    )

    _emit(
        started,
        iters_attempted, iters_completed, iters_unsupported, iters_errored,
        max_abs_err_global, max_rel_err_global,
        records, filable_seed_hits, worst_per_sig,
        per_bucket, per_dtype, per_stride,
        status="OK",
        top3=top3, top3_label=top3_label,
        divergences_filable=divergences_filable,
        divergences_recalibration=divergences_recalibration,
        filable_signatures=filable_signatures,
    )
    return 0


def _emit(
    started: float,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_errored: int,
    max_abs_err_global: float,
    max_rel_err_global: float,
    records: list[dict],
    filable_seed_hits: dict,
    worst_per_sig: dict,
    per_bucket: dict,
    per_dtype: dict,
    per_stride: dict,
    *,
    status: str,
    top3: list[dict] | None = None,
    top3_label: str = "top_3_repros",
    divergences_filable: int = 0,
    divergences_recalibration: int = 0,
    filable_signatures: dict | None = None,
) -> None:
    elapsed = round(time.monotonic() - started, 2)
    summary = {
        "kernel": KERNEL,
        "status": status,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_errored": iters_errored,
        "divergences_filable": divergences_filable,
        "divergences_recalibration": divergences_recalibration,
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "elapsed_seconds": elapsed,
        "torch_version": torch.__version__,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "iters_per_seed": ITERS_PER_SEED,
        "per_shape_bucket": per_bucket,
        "per_dtype": per_dtype,
        "per_stride_category": per_stride,
        "top_3_repros": top3 or [],
    }

    _write_md(
        RESULTS_MD, summary,
        records_count=len(records),
        top3_label=top3_label,
        filable_signatures=filable_signatures or {},
    )
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_md(path: Path, s: dict, *,
              records_count: int,
              top3_label: str,
              filable_signatures: dict) -> None:
    L: list[str] = []
    L.append(f"# {s['kernel']} - MPS fuzz report")
    L.append("")
    L.append(f"- **Kernel:** `{s['kernel']}`")
    L.append(f"- **Status:** {s['status']}")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    L.append(f"- **iters_errored:** {s['iters_errored']}")
    L.append(f"- **divergences_filable:** {s['divergences_filable']}")
    L.append(f"- **divergences_recalibration:** {s['divergences_recalibration']}")
    L.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    L.append(f"- **max_rel_err (denom>=1e-6):** {s['max_rel_err']:.3e}")
    L.append(f"- **Elapsed:** {s['elapsed_seconds']} s")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    L.append(f"- **Seeds:** {s['seeds']} x {s['iters_per_seed']} iters/seed")
    L.append(f"- **Total over-tolerance hits (any band):** {records_count}")
    L.append("")
    L.append("## Sampling distribution")
    L.append("")
    L.append("| dimension | counts |")
    L.append("|---|---|")
    L.append(f"| shape bucket | {s['per_shape_bucket']} |")
    L.append(f"| dtype | {s['per_dtype']} |")
    L.append(f"| stride category | {s['per_stride_category']} |")
    L.append("")
    L.append("## Divergence classification")
    L.append("")
    L.append("- **FILABLE_CANDIDATE:** max_abs_err > 10*atol, OR "
             "(max_rel_err > 10*rtol AND denom_magnitude >= 1e-6).")
    L.append("- **FILABLE (in `divergences_filable`):** a FILABLE_CANDIDATE "
             "signature `(dtype, stride_category, shape_bucket, shape)` that "
             "reproduces across **>=3 of 5 seeds**.")
    L.append("- **TOLERANCE_RECALIBRATION:** error in the 1x..5x tol band -> "
             "recommend an `[tool.gpucheck.mps.xfail]` or multiplier nudge.")
    L.append("- **OK:** below 1x tol.")
    L.append("")
    if filable_signatures:
        L.append("### FILABLE signatures (reproduced across >=3 seeds)")
        L.append("")
        for sig, seeds in filable_signatures.items():
            dn, sc, bk, sh = sig
            L.append(f"- dtype=`{dn}`, stride=`{sc}`, bucket=`{bk}`, "
                     f"shape=`{sh}`, seeds={sorted(seeds)}")
        L.append("")
    L.append(f"## {top3_label.replace('_', ' ').title()}")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_None - no FILABLE or RECALIBRATION hits in this run._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i}")
            L.append("")
            L.append(f"- **classification:** `{r.get('classification','?')}`")
            L.append(f"- **shape:** `{tuple(r['shape'])}` (bucket `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride category:** `{r['stride_category']}`")
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e}  "
                     f"(atol={r['atol']:.2e}, ratio={r['abs_ratio_to_atol']:.2f}x)")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e}  "
                     f"(rtol={r['rtol']:.2e}, ratio={r['rel_ratio_to_rtol']:.2f}x)")
            L.append(f"- **denom at max-rel:** {r['denom_at_max_rel']:.3e}")
            L.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            L.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            L.append(f"- **outer seed:** {r['seed']}, inner seed: {r['inner_seed']}")
            if "reproducing_seeds" in r:
                L.append(f"- **reproducing seeds:** {r['reproducing_seeds']}")
            L.append("")
    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.hardsigmoid` on CPU "
             "(comparison done in fp32 promotion).")
    L.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (MPS overlay applied).")
    L.append("- `max_rel_err` is computed only over samples where "
             "|reference| >= 1e-6, removing near-zero-denominator artifacts.")
    L.append("- hardsigmoid saturates to 0 for x<=-3 and to 1 for x>=3, so "
             "exact-zero outputs are common and would otherwise inflate "
             "rel-err with /eps division.")
    L.append("- 1000 iters across seeds 0..4 (200 per seed); FILABLE requires "
             ">=3 seeds reproducing the same (dtype, stride, bucket, shape).")
    L.append("- CUDA backend mocked (no NVIDIA GPU on host); MPS-vs-CUDA "
             "cross-device check is N/A.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
