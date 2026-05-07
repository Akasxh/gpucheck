"""v2 fuzzer for torch.nn.functional.silu on MPS vs CPU.

Spec (v2 swarm protocol):
- 1000 iterations across seeds {0,1,2,3,4}
- dtypes: fp32, fp16, bf16
- Divergence filtering:
    * max_rel_err > 10x tolerance counts ONLY if denom_magnitude >= 1e-6
      (avoids near-zero-denominator artifacts)
    * max_abs_err > 10x tolerance always counts
    * Reproducible across >=3 seeds  -> FILABLE
    * 1x..5x tolerance               -> TOLERANCE_RECALIBRATION bucket (xfail candidate)
    * <1x tolerance                  -> OK
- 12 minute wall budget.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Use the merged release/v1.0 worktree
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-silu/src")
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
RESULTS_MD = OUT_DIR / "RESULTS_silu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.silu"
N_ITERS = 1000
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 12 * 60 - 30  # leave 30s for write + safety margin

# ---------------------------------------------------------------------------
# Shape buckets — mirror v1 fuzzer (gelu) but capped to fit memory in 1000 iters
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
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_max).

    denom_magnitude is |b| at the location of the max-rel-err element. If that
    location has |b| < 1e-6 the relative error is a near-zero artifact and the
    v2 spec says it should NOT count toward divergence.
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)  # tiny floor to avoid /0 in computation
    rel = diff / denom
    flat_idx = int(torch.argmax(rel.flatten()).item())
    rel_max = float(rel.flatten()[flat_idx].item())
    denom_at_max = float(b.abs().flatten()[flat_idx].item())
    return rel_max, denom_at_max


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _signature(shape: tuple[int, ...], dtype: str, stride_cat: str) -> str:
    """Repro signature stable across seeds — used to count seed-replication."""
    return f"{stride_cat}|{dtype}|{tuple(shape)}"


def _classify(
    abs_err: float, rel_err: float, denom_mag: float, atol: float, rtol: float,
) -> str:
    """Return one of: OK, TOLERANCE_RECALIBRATION, DIVERGENCE.

    - max_abs > 10*atol            -> DIVERGENCE (always)
    - max_rel > 10*rtol AND denom>=1e-6 -> DIVERGENCE
    - max_abs > 1*atol or max_rel > 1*rtol (up to 5x) -> TOLERANCE_RECALIBRATION
    - else                          -> OK
    """
    div_abs = abs_err > 10.0 * atol
    div_rel = (rel_err > 10.0 * rtol) and (denom_mag >= 1e-6)
    if div_abs or div_rel:
        return "DIVERGENCE"
    recal_abs = (abs_err > atol) and (abs_err <= 5.0 * atol)
    recal_rel = (rel_err > rtol) and (rel_err <= 5.0 * rtol) and (denom_mag >= 1e-6)
    if recal_abs or recal_rel:
        return "TOLERANCE_RECALIBRATION"
    return "OK"


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
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "top_3_repros": [],
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0

    # Records bucketed by signature — for FILABLE (>=3 seed) determination
    div_by_sig: dict[str, list[dict]] = defaultdict(list)
    recal_by_sig: dict[str, list[dict]] = defaultdict(list)
    per_dtype_max_rel: dict[str, float] = defaultdict(float)
    per_stride_max_rel: dict[str, float] = defaultdict(float)
    per_bucket_counts: dict[str, int] = defaultdict(int)
    per_dtype_counts: dict[str, int] = defaultdict(int)
    per_stride_counts: dict[str, int] = defaultdict(int)

    iters_per_seed = N_ITERS // len(SEEDS)
    leftover = N_ITERS - iters_per_seed * len(SEEDS)

    aborted = False
    for seed_idx, seed in enumerate(SEEDS):
        rng = random.Random(seed)
        n_this_seed = iters_per_seed + (1 if seed_idx < leftover else 0)
        for it in range(n_this_seed):
            if time.monotonic() - started > WALL_BUDGET_S:
                print(
                    f"[budget] aborting at seed={seed} iter={it} "
                    f"(elapsed {time.monotonic()-started:.1f}s)",
                    file=sys.stderr,
                )
                aborted = True
                break
            iters_attempted += 1

            bucket = rng.choice(BUCKET_NAMES)
            shape = rng.choice(SHAPE_BUCKETS[bucket])
            dtype_name = rng.choice(DTYPE_NAMES)
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = rng.choice(STRIDE_CATEGORIES)

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            tensor_seed = rng.randrange(2**31 - 1)

            try:
                x_cpu = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="cpu", seed=tensor_seed,
                )
                try:
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=tensor_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.silu(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.silu(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                rel, denom_mag = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
                absdiff = _max_abs_err(y_mps_cpu, y_cpu)

                # Effectively-counted rel error per spec (zero out near-denom artifacts)
                effective_rel = rel if denom_mag >= 1e-6 else 0.0
                if absdiff > max_abs_err_global:
                    max_abs_err_global = absdiff
                if effective_rel > max_rel_err_global:
                    max_rel_err_global = effective_rel
                if effective_rel > per_dtype_max_rel[dtype_name]:
                    per_dtype_max_rel[dtype_name] = effective_rel
                if effective_rel > per_stride_max_rel[stride_cat]:
                    per_stride_max_rel[stride_cat] = effective_rel

                atol, rtol = compute_tolerance(dtype, device_type="mps")
                klass = _classify(absdiff, rel, denom_mag, atol, rtol)

                if klass == "OK":
                    continue

                rec = {
                    "seed": seed,
                    "tensor_seed": tensor_seed,
                    "iter": it,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "shape_bucket": bucket,
                    "max_abs_err": absdiff,
                    "max_rel_err": rel,
                    "denom_magnitude_at_max_rel": denom_mag,
                    "atol": atol,
                    "rtol": rtol,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                    "classification": klass,
                }
                sig = _signature(shape, dtype_name, stride_cat)
                if klass == "DIVERGENCE":
                    div_by_sig[sig].append(rec)
                else:  # TOLERANCE_RECALIBRATION
                    recal_by_sig[sig].append(rec)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Halt on unknown error per spec, but write what we have
                print(
                    f"[ERROR] seed={seed} iter={it} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                _write_results(
                    aborted=True,
                    error_msg=f"{exc!r}",
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported=iters_unsupported,
                    div_by_sig=div_by_sig,
                    recal_by_sig=recal_by_sig,
                    max_abs_err=max_abs_err_global,
                    max_rel_err=max_rel_err_global,
                    per_bucket_counts=per_bucket_counts,
                    per_dtype_counts=per_dtype_counts,
                    per_stride_counts=per_stride_counts,
                    per_dtype_max_rel=per_dtype_max_rel,
                    per_stride_max_rel=per_stride_max_rel,
                    elapsed=time.monotonic() - started,
                )
                return 2
        if aborted:
            break

    elapsed = time.monotonic() - started
    _write_results(
        aborted=aborted,
        error_msg=None,
        iters_attempted=iters_attempted,
        iters_completed=iters_completed,
        iters_unsupported=iters_unsupported,
        div_by_sig=div_by_sig,
        recal_by_sig=recal_by_sig,
        max_abs_err=max_abs_err_global,
        max_rel_err=max_rel_err_global,
        per_bucket_counts=per_bucket_counts,
        per_dtype_counts=per_dtype_counts,
        per_stride_counts=per_stride_counts,
        per_dtype_max_rel=per_dtype_max_rel,
        per_stride_max_rel=per_stride_max_rel,
        elapsed=elapsed,
    )
    return 0


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _split_filable(
    div_by_sig: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Return (filable_records, non_filable_div_records).

    A signature is FILABLE iff it appears under >=3 distinct seeds.
    """
    filable: list[dict] = []
    non_filable: list[dict] = []
    for sig, recs in div_by_sig.items():
        seeds_seen = {r["seed"] for r in recs}
        if len(seeds_seen) >= 3:
            filable.extend(recs)
        else:
            non_filable.extend(recs)
    return filable, non_filable


def _top_3_repros(records: list[dict]) -> list[dict]:
    """Pick top 3 minimal repros: largest abs err, smallest tensor."""
    if not records:
        return []
    sorted_recs = sorted(
        records,
        key=lambda r: (-r["max_abs_err"], _numel(r["shape"])),
    )
    return sorted_recs[:3]


def _write_results(
    *,
    aborted: bool,
    error_msg: str | None,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    div_by_sig: dict[str, list[dict]],
    recal_by_sig: dict[str, list[dict]],
    max_abs_err: float,
    max_rel_err: float,
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    per_dtype_max_rel: dict[str, float],
    per_stride_max_rel: dict[str, float],
    elapsed: float,
) -> None:
    filable, non_filable = _split_filable(div_by_sig)

    # For recalibration: anything that's classified as TOLERANCE_RECALIBRATION,
    # plus DIVERGENCE records that didn't reach the >=3-seed bar.
    recal_records: list[dict] = []
    for recs in recal_by_sig.values():
        recal_records.extend(recs)
    recal_records.extend(non_filable)

    top3 = _top_3_repros(filable) or _top_3_repros(recal_records)

    status = "ABORTED" if aborted else "OK"
    if error_msg:
        status = "ERROR"

    summary = {
        "kernel": KERNEL,
        "status": status,
        "error": error_msg,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal_records),
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "top_3_repros": top3,
        "per_dtype_max_rel": dict(per_dtype_max_rel),
        "per_stride_max_rel": dict(per_stride_max_rel),
        "per_shape_bucket_counts": dict(per_bucket_counts),
        "per_dtype_counts": dict(per_dtype_counts),
        "per_stride_counts": dict(per_stride_counts),
        "elapsed_seconds": round(elapsed, 2),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "n_iters_target": N_ITERS,
        "spec_version": "v2",
        "filing_target": "pytorch/pytorch" if filable else "none",
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(path: Path, s: dict) -> None:
    lines: list[str] = []
    lines.append(f"# {s['kernel']} — MPS fuzz report (v2)")
    lines.append("")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    if s.get("error"):
        lines.append(f"- **Error:** `{s['error']}`")
    lines.append(f"- **Iters attempted:** {s['iters_attempted']} / target {s['n_iters_target']}")
    lines.append(f"- **Iters completed:** {s['iters_completed']}")
    lines.append(f"- **Iters unsupported:** {s['iters_unsupported']}")
    lines.append(f"- **Divergences (FILABLE, >=3 seeds):** {s['divergences_filable']}")
    lines.append(
        f"- **Divergences (TOLERANCE_RECALIBRATION):** {s['divergences_recalibration']}"
    )
    lines.append(f"- **Max abs err:** {s['max_abs_err']:.3e}")
    lines.append(f"- **Max rel err (denom>=1e-6):** {s['max_rel_err']:.3e}")
    lines.append(f"- **Filing target:** `{s['filing_target']}`")
    lines.append(f"- **Elapsed:** {s['elapsed_seconds']} s")
    lines.append(
        f"- **torch:** {s['torch_version']}, host: {s['host']}, seeds: {s['seeds']}"
    )
    lines.append("")
    lines.append("## Sampling distribution")
    lines.append("")
    lines.append("| dimension | counts |")
    lines.append("|---|---|")
    lines.append(f"| shape bucket | {s['per_shape_bucket_counts']} |")
    lines.append(f"| dtype | {s['per_dtype_counts']} |")
    lines.append(f"| stride category | {s['per_stride_counts']} |")
    lines.append("")
    lines.append("## Per-axis max relative error (denom-filtered)")
    lines.append("")
    lines.append("| dtype | max_rel_err |")
    lines.append("|---|---|")
    for k in sorted(s["per_dtype_max_rel"]):
        lines.append(f"| `{k}` | {s['per_dtype_max_rel'][k]:.3e} |")
    lines.append("")
    lines.append("| stride_category | max_rel_err |")
    lines.append("|---|---|")
    for k in sorted(s["per_stride_max_rel"]):
        lines.append(f"| `{k}` | {s['per_stride_max_rel'][k]:.3e} |")
    lines.append("")
    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append(
            "_No FILABLE or RECALIBRATION-bucket records — all errors fell within "
            "1x the dtype tolerance (assertions/tolerances.py with MPS 2x multiplier)._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i} ({r['classification']})")
            lines.append("")
            lines.append(
                f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)"
            )
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride_category']}`")
            lines.append(
                f"- **max abs err:** {r['max_abs_err']:.3e}  (atol={r['atol']:.2e})"
            )
            lines.append(
                f"- **max rel err:** {r['max_rel_err']:.3e}  (rtol={r['rtol']:.2e})"
            )
            lines.append(
                f"- **denom |b| at max rel:** {r['denom_magnitude_at_max_rel']:.3e}"
            )
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append(f"- **seed:** {r['seed']}, tensor_seed: {r['tensor_seed']}")
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Reference: `torch.nn.functional.silu` on CPU promoted to FP32 for "
        "the error comparison."
    )
    lines.append(
        "- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
        "dtype, device_type='mps')`."
    )
    lines.append(
        "- Classification: max_abs > 10x atol => DIVERGENCE; max_rel > 10x rtol "
        "with |b|>=1e-6 => DIVERGENCE; otherwise the [1x, 5x] band is "
        "TOLERANCE_RECALIBRATION; below 1x is OK."
    )
    lines.append(
        "- A DIVERGENCE only becomes FILABLE if its (stride, dtype, shape) "
        "signature reproduces under >=3 distinct seeds."
    )
    lines.append(
        "- Stride categories: row_major, column_major, broadcast, transpose, "
        "slice, non_contig, gather (see `gpucheck.fuzzing.strides`)."
    )
    lines.append(
        "- Fuzzer: `_fuzz_silu_v2.py` in this directory."
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
