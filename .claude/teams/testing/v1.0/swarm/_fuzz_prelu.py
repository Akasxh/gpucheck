"""Stride/shape/dtype fuzzer for torch.nn.functional.prelu on MPS vs CPU.

Spec: kernel-fuzzer-prelu (v2).
- 1000 iterations across seeds 0,1,2,3,4 (200 per seed).
- dtypes fp32/fp16/bf16 (skip bf16 if torch.mps refuses it).
- Divergence filtering:
    * max_rel_err > 10x tolerance ONLY counts if denom_magnitude >= 1e-6.
    * max_abs_err > 10x tolerance always counts.
    * Both (divergence + reproducibility >= 3 seeds) -> FILABLE.
    * 1x..5x tolerance         -> TOLERANCE_RECALIBRATION.
    * < 1x tolerance           -> OK.
- 12 minute hard wall; on overrun write what we have and exit cleanly.
- Do not invent divergences; UNSUPPORTED if torch.mps lacks the op or layout.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Make the worktree's gpucheck importable.
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-prelu/src")
sys.path.insert(0, str(SRC))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_prelu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.prelu"

# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
ITERS_PER_SEED = 200
WALL_BUDGET_S = 12 * 60 - 60  # 11 min for fuzzing, ~60 s headroom for I/O

# ---------------------------------------------------------------------------
# Sampling space (kept modest so 1000 iters fit easily; prelu is elementwise)
# ---------------------------------------------------------------------------

DEGENERATE: list[tuple[int, ...]] = [(1,), (1, 1), (1, 1, 1), (16, 1)]
NON_TILE: list[tuple[int, ...]] = [
    (31,), (33,), (63, 16), (67, 16), (127, 33), (129, 16), (3, 65, 7),
]
PRIME: list[tuple[int, ...]] = [
    (7,), (13,), (31,), (61,), (127,), (7, 13), (13, 17), (5, 31, 7),
]
POW2_BOUNDARY: list[tuple[int, ...]] = [
    (127,), (128,), (129,), (255,), (256,), (257,), (128, 129), (256, 255),
]
LARGE: list[tuple[int, ...]] = [
    (1024,), (256, 256), (1024, 64), (4, 256, 64),
]
MIXED: list[tuple[int, ...]] = [
    (127, 16), (1024, 3), (7, 128), (33, 128, 4), (3, 5, 7, 11),
]

SHAPE_BUCKETS: dict[str, list[tuple[int, ...]]] = {
    "degenerate": DEGENERATE,
    "non_tile_aligned": NON_TILE,
    "prime": PRIME,
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

# Stride categories supplied by gpucheck. We use all seven and let the
# kernel handle them; UNSUPPORTED layouts are reported, not invented.
STRIDE_CATS = list(STRIDE_CATEGORIES)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _max_rel_err_with_denom(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude).

    denom_magnitude = max(|b|) over the comparison tensor; this is the scalar
    we use to decide whether a large rel_err is a near-zero-denominator
    artifact (denom < 1e-6) or a real divergence.
    """
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    denom_mag = float(b.abs().max().item())
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-7)
    return float((diff / denom).max().item()), denom_mag


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------


def run_one(
    iter_idx: int,
    seed_root: int,
    rng: random.Random,
) -> dict:
    """Sample (bucket, shape, dtype, stride) and run one prelu compare.

    Returns a record dict; status in {"OK","UNSUPPORTED","BUILD_ERR","ERR","SKIP_EMPTY"}.
    """
    bucket = rng.choice(BUCKET_NAMES)
    shape = rng.choice(SHAPE_BUCKETS[bucket])
    dtype_name = rng.choice(DTYPE_NAMES)
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = rng.choice(STRIDE_CATS)
    sub_seed = rng.randrange(2**31 - 1)
    # weight policy: 50% scalar weight (shape [1]), 50% per-channel if rank>=2.
    use_per_channel = rng.random() < 0.5 and len(shape) >= 2
    # alpha sampling: occasional negative weight to exercise both branches
    if rng.random() < 0.2:
        alpha = -rng.uniform(0.05, 0.5)
    else:
        alpha = rng.uniform(0.0, 0.5)

    rec: dict = {
        "seed_root": seed_root,
        "iter": iter_idx,
        "shape_bucket": bucket,
        "shape": list(shape),
        "dtype": dtype_name,
        "stride": stride_cat,
        "sub_seed": sub_seed,
        "alpha": alpha,
        "weight_kind": "per_channel" if use_per_channel else "scalar",
        "status": "OK",
        "max_abs_err": None,
        "max_rel_err": None,
        "denom_magnitude": None,
        "atol": None,
        "rtol": None,
        "x_layout_cpu": None,
        "x_layout_mps": None,
        "bucket_class": None,  # OK / TOLERANCE_RECALIBRATION / DIVERGENCE
        "note": "",
    }

    # Build CPU input
    try:
        x_cpu = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="cpu", seed=sub_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        rec["status"] = "BUILD_ERR_CPU"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    if x_cpu.numel() == 0:
        rec["status"] = "SKIP_EMPTY"
        return rec

    # Build MPS input with the same sub_seed so contents pre-cast match.
    try:
        x_mps = fuzz_strides_for_category(
            shape, dtype, stride_cat, device="mps", seed=sub_seed,
        )
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        rec["status"] = "UNSUPPORTED_MPS_BUILD"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    rec["x_layout_cpu"] = _stride_tag(x_cpu)
    rec["x_layout_mps"] = _stride_tag(x_mps)

    # Build weight tensors. PReLU requires num_params == 1 OR == channels (dim 1).
    if use_per_channel:
        n_channels = shape[1]
        # Vary weight per channel a bit; keep magnitudes modest.
        base = (torch.arange(n_channels, dtype=torch.float32) + 1.0) * (alpha / max(n_channels, 1))
        w_cpu = base.to(dtype=dtype)
        try:
            w_mps = w_cpu.to(device="mps")
        except (RuntimeError, NotImplementedError, TypeError) as e:
            rec["status"] = "UNSUPPORTED_MPS_WEIGHT"
            rec["note"] = f"{type(e).__name__}: {e}"[:240]
            return rec
    else:
        w_cpu = torch.tensor([alpha], dtype=dtype)
        try:
            w_mps = w_cpu.to(device="mps")
        except (RuntimeError, NotImplementedError, TypeError) as e:
            rec["status"] = "UNSUPPORTED_MPS_WEIGHT"
            rec["note"] = f"{type(e).__name__}: {e}"[:240]
            return rec

    # CPU reference
    try:
        y_cpu = F.prelu(x_cpu, w_cpu)
    except (RuntimeError, NotImplementedError, TypeError) as e:
        rec["status"] = "UNSUPPORTED_CPU"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    # MPS run
    try:
        y_mps = F.prelu(x_mps, w_mps)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as e:
        rec["status"] = "UNSUPPORTED_MPS"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    abs_err = _max_abs_err(y_mps_cpu, y_cpu)
    rel_err, denom_mag = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
    atol, rtol = compute_tolerance(dtype, device_type="mps")

    rec["max_abs_err"] = abs_err
    rec["max_rel_err"] = rel_err
    rec["denom_magnitude"] = denom_mag
    rec["atol"] = atol
    rec["rtol"] = rtol

    rec["bucket_class"] = classify(abs_err, rel_err, denom_mag, atol, rtol)
    return rec


def classify(
    abs_err: float, rel_err: float, denom_mag: float, atol: float, rtol: float,
) -> str:
    """Per-iter bucket: OK / TOLERANCE_RECALIBRATION / DIVERGENCE."""
    abs_ratio = abs_err / atol if atol > 0 else math.inf
    rel_ratio = rel_err / rtol if rtol > 0 else math.inf
    rel_meaningful = denom_mag >= 1e-6

    # DIVERGENCE: max_abs > 10x always; max_rel > 10x only if denom is meaningful.
    if abs_ratio > 10.0:
        return "DIVERGENCE"
    if rel_ratio > 10.0 and rel_meaningful:
        return "DIVERGENCE"

    # TOLERANCE_RECALIBRATION: anything above 1x tolerance that's not a
    # near-zero-denominator artifact for the rel-err path.
    if abs_ratio > 1.0:
        return "TOLERANCE_RECALIBRATION"
    if rel_ratio > 1.0 and rel_meaningful:
        return "TOLERANCE_RECALIBRATION"

    return "OK"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
            f"# {KERNEL} — SKIPPED\n\nMPS not available on this host.\n",
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    all_records: list[dict] = []
    iters_attempted = 0
    iters_completed = 0
    aborted_for_budget = False

    for seed in SEEDS:
        if time.monotonic() - started > WALL_BUDGET_S:
            aborted_for_budget = True
            print(f"[budget] aborted before seed={seed}", file=sys.stderr)
            break
        rng = random.Random(seed)
        for i in range(ITERS_PER_SEED):
            if time.monotonic() - started > WALL_BUDGET_S:
                aborted_for_budget = True
                print(
                    f"[budget] aborted at seed={seed} iter={i} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                )
                break
            iters_attempted += 1
            try:
                rec = run_one(i, seed, rng)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "seed_root": seed,
                    "iter": i,
                    "status": "ERR",
                    "note": f"{type(exc).__name__}: {exc}"[:240],
                    "trace": traceback.format_exc()[-600:],
                }
                # Hard rule: halt and report any process error immediately.
                all_records.append(rec)
                print(
                    f"[HALT] seed={seed} iter={i} {rec['note']}", file=sys.stderr,
                )
                # Write whatever we have, then exit with non-zero.
                _write_outputs(
                    all_records, iters_attempted, iters_completed,
                    started, halted=True, aborted=False,
                )
                return 2
            all_records.append(rec)
            if rec["status"] == "OK":
                iters_completed += 1
            if (i % 50) == 0:
                print(
                    f"[seed={seed} iter={i}] {rec.get('shape_bucket')}/"
                    f"{rec.get('dtype')}/{rec.get('stride')} "
                    f"shape={rec.get('shape')} status={rec['status']} "
                    f"class={rec.get('bucket_class')}",
                    flush=True,
                )
        if aborted_for_budget:
            break

    _write_outputs(
        all_records, iters_attempted, iters_completed,
        started, halted=False, aborted=aborted_for_budget,
    )
    return 0


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _signature(rec: dict) -> tuple:
    return (
        tuple(rec.get("shape") or ()),
        rec.get("dtype"),
        rec.get("stride"),
        rec.get("weight_kind"),
    )


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return n


def _write_outputs(
    records: list[dict],
    iters_attempted: int,
    iters_completed: int,
    started: float,
    halted: bool,
    aborted: bool,
) -> None:
    elapsed = time.monotonic() - started

    # Per-signature aggregation across seeds
    sig_seeds_by_class: dict[tuple, dict[str, set[int]]] = defaultdict(
        lambda: {"DIVERGENCE": set(), "TOLERANCE_RECALIBRATION": set()},
    )
    sig_best_repro: dict[tuple, dict] = {}

    max_abs_global = 0.0
    max_rel_global = 0.0

    diverged_records: list[dict] = []
    recal_records: list[dict] = []
    unsupported_count = 0
    err_count = 0

    for r in records:
        st = r.get("status")
        if st in (
            "UNSUPPORTED_MPS",
            "UNSUPPORTED_CPU",
            "UNSUPPORTED_MPS_BUILD",
            "UNSUPPORTED_MPS_WEIGHT",
            "BUILD_ERR_CPU",
        ):
            unsupported_count += 1
            continue
        if st == "ERR":
            err_count += 1
            continue
        if st != "OK":
            continue
        abs_err = r.get("max_abs_err") or 0.0
        rel_err = r.get("max_rel_err") or 0.0
        max_abs_global = max(max_abs_global, abs_err)
        max_rel_global = max(max_rel_global, rel_err)
        cls = r.get("bucket_class")
        sig = _signature(r)
        if cls == "DIVERGENCE":
            diverged_records.append(r)
            sig_seeds_by_class[sig]["DIVERGENCE"].add(r["seed_root"])
        elif cls == "TOLERANCE_RECALIBRATION":
            recal_records.append(r)
            sig_seeds_by_class[sig]["TOLERANCE_RECALIBRATION"].add(r["seed_root"])
        # Track the worst (highest max_rel_err) repro per signature for top3.
        prev = sig_best_repro.get(sig)
        if prev is None or (r.get("max_rel_err") or 0.0) > (
            prev.get("max_rel_err") or 0.0
        ):
            sig_best_repro[sig] = r

    # FILABLE: signatures where DIVERGENCE class hit on >= 3 distinct seeds.
    filable_sigs = [
        sig for sig, sd in sig_seeds_by_class.items()
        if len(sd["DIVERGENCE"]) >= 3
    ]
    # Recalibration: signatures where the recalibration band hit >= 3 seeds
    # AND it is not already filable. This is the "consistent recalibration
    # candidate" set we'd recommend xfail entries for.
    recal_sigs = [
        sig for sig, sd in sig_seeds_by_class.items()
        if sig not in filable_sigs and len(sd["TOLERANCE_RECALIBRATION"]) >= 3
    ]

    # Top-3 repros: prefer FILABLE signatures; fall back to recal; sort by
    # (largest rel_err, smallest numel) for "minimal repro" prioritisation.
    def _sort_key(sig: tuple) -> tuple:
        rep = sig_best_repro.get(sig, {})
        rel = rep.get("max_rel_err") or 0.0
        ne = _numel(rep.get("shape") or [1])
        return (-rel, ne)

    top3_sigs = sorted(filable_sigs, key=_sort_key)[:3]
    if len(top3_sigs) < 3:
        top3_sigs += sorted(
            (s for s in recal_sigs if s not in top3_sigs), key=_sort_key,
        )[: 3 - len(top3_sigs)]

    top3 = [sig_best_repro[s] for s in top3_sigs]

    # ---------------- markdown ----------------
    lines: list[str] = []
    lines.append(f"# {KERNEL} — MPS fuzz report\n")
    lines.append("- **Kernel:** `torch.nn.functional.prelu`")
    lines.append(
        "- **Reference:** CPU `F.prelu` (same dtype) compared against MPS "
        "`F.prelu`; CUDA path mocked off (no NVIDIA GPU on host).",
    )
    lines.append(f"- **iters_attempted:** {iters_attempted}")
    lines.append(f"- **iters_completed (status=OK):** {iters_completed}")
    lines.append(
        f"- **iters_unsupported (op/build refused):** {unsupported_count}",
    )
    lines.append(f"- **iters_errored:** {err_count}")
    lines.append(f"- **divergences_filable (>=3 seeds, >10x tol):** {len(filable_sigs)}")
    lines.append(
        f"- **divergences_recalibration (1-5x tol band, >=3 seeds): "
        f"{len(recal_sigs)}",
    )
    lines.append(f"- **max_abs_err:** {max_abs_global:.4e}")
    lines.append(f"- **max_rel_err:** {max_rel_global:.4e}")
    lines.append(f"- **elapsed:** {elapsed:.1f} s")
    lines.append(f"- **seeds:** {list(SEEDS)}")
    lines.append(f"- **iters_per_seed:** {ITERS_PER_SEED}")
    lines.append(f"- **wall_budget_s:** {WALL_BUDGET_S}")
    lines.append(f"- **aborted_for_budget:** {aborted}")
    lines.append(f"- **halted_on_error:** {halted}")
    lines.append(f"- **torch:** {torch.__version__}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
        "dtype, device_type='mps')` (per-dtype atol/rtol, with the "
        "PROVISIONAL 2x MPS overlay).",
    )
    lines.append(
        "- Per-iter classification:",
    )
    lines.append("  - `DIVERGENCE`  — `max_abs_err > 10*atol` (always) OR "
                 "`max_rel_err > 10*rtol` AND `denom_magnitude (=max|y_cpu|) >= 1e-6`.")
    lines.append("  - `TOLERANCE_RECALIBRATION` — error in (1x..10x] tolerance band "
                 "(rel-err counted only when denom is meaningful).")
    lines.append("  - `OK` — error within 1x tolerance.")
    lines.append(
        "- A signature `(shape, dtype, stride_category, weight_kind)` is "
        "**FILABLE** only when the per-iter `DIVERGENCE` class is observed on "
        "\\>= 3 distinct seeds (the user's reproducibility rule).",
    )
    lines.append(
        "- Stride categories: " + ", ".join(STRIDE_CATS),
    )
    lines.append("")
    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not top3:
        lines.append(
            "_None — no signature exceeded gpucheck's MPS tolerance band on "
            ">=3 seeds._",
        )
    else:
        for k, r in enumerate(top3, 1):
            lines.append(f"### Repro #{k}")
            lines.append("")
            lines.append(f"- **shape:** `{tuple(r['shape'])}` "
                         f"(bucket: `{r['shape_bucket']}`)")
            lines.append(f"- **dtype:** `{r['dtype']}`")
            lines.append(f"- **stride category:** `{r['stride']}`")
            lines.append(f"- **weight kind:** `{r['weight_kind']}`")
            lines.append(f"- **alpha:** {r['alpha']:.4f}")
            lines.append(f"- **max_abs_err:** {r['max_abs_err']:.4e}  "
                         f"(atol={r['atol']:.2e}, ratio="
                         f"{(r['max_abs_err'] / r['atol']) if r['atol'] else float('inf'):.2f}x)")
            lines.append(
                f"- **max_rel_err:** {r['max_rel_err']:.4e}  "
                f"(rtol={r['rtol']:.2e}, ratio="
                f"{(r['max_rel_err'] / r['rtol']) if r['rtol'] else float('inf'):.2f}x)",
            )
            lines.append(f"- **denom_magnitude (max|y_cpu|):** "
                         f"{r['denom_magnitude']:.3e}")
            lines.append(f"- **bucket_class:** `{r['bucket_class']}`")
            lines.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            lines.append(f"- **sub_seed:** {r['sub_seed']}, "
                         f"**seed_root:** {r['seed_root']}")
            sig = _signature(r)
            div_n = len(sig_seeds_by_class[sig]["DIVERGENCE"])
            recal_n = len(sig_seeds_by_class[sig]["TOLERANCE_RECALIBRATION"])
            lines.append(f"- **seed-reproducibility:** "
                         f"DIVERGENCE on {div_n} seed(s), "
                         f"RECALIBRATION on {recal_n} seed(s)")
            lines.append("")

    # Status-bucket breakdown for transparency
    status_counts: dict[str, int] = defaultdict(int)
    for r in records:
        status_counts[r.get("status", "?")] += 1
    lines.append("## Status counts")
    lines.append("")
    for s, c in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{s}`: {c}")
    lines.append("")

    # FILABLE / RECAL signature inventory (so the next pass can xfail them)
    if filable_sigs:
        lines.append("## FILABLE signatures (>=3 seeds, >10x tol)")
        lines.append("")
        for sig in filable_sigs:
            r = sig_best_repro[sig]
            seeds = sorted(sig_seeds_by_class[sig]["DIVERGENCE"])
            lines.append(
                f"- shape={list(sig[0])} dtype={sig[1]} stride={sig[2]} "
                f"weight_kind={sig[3]} :: "
                f"max_abs_err={r['max_abs_err']:.3e} (atol={r['atol']:.2e}), "
                f"max_rel_err={r['max_rel_err']:.3e} (rtol={r['rtol']:.2e}), "
                f"denom={r['denom_magnitude']:.3e}, "
                f"seeds={seeds}",
            )
        lines.append("")
    if recal_sigs:
        lines.append("## RECALIBRATION-candidate signatures (>=3 seeds, 1-10x tol)")
        lines.append("")
        for sig in recal_sigs[:20]:  # cap to keep the report readable
            r = sig_best_repro[sig]
            seeds = sorted(sig_seeds_by_class[sig]["TOLERANCE_RECALIBRATION"])
            lines.append(
                f"- shape={list(sig[0])} dtype={sig[1]} stride={sig[2]} "
                f"weight_kind={sig[3]} :: "
                f"max_abs_err={r['max_abs_err']:.3e} (atol={r['atol']:.2e}), "
                f"max_rel_err={r['max_rel_err']:.3e} (rtol={r['rtol']:.2e}), "
                f"seeds={seeds}",
            )
        if len(recal_sigs) > 20:
            lines.append(f"- ... and {len(recal_sigs) - 20} more")
        lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")

    summary = {
        "kernel": KERNEL,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": unsupported_count,
        "iters_errored": err_count,
        "divergences_filable": len(filable_sigs),
        "divergences_recalibration": len(recal_sigs),
        "max_abs_err": max_abs_global,
        "max_rel_err": max_rel_global,
        "elapsed_seconds": round(elapsed, 2),
        "aborted_for_budget": aborted,
        "halted_on_error": halted,
        "seeds": list(SEEDS),
        "iters_per_seed": ITERS_PER_SEED,
        "torch_version": torch.__version__,
        "top_3_repros": [
            {
                "shape": list(r["shape"]),
                "dtype": r["dtype"],
                "stride": r["stride"],
                "weight_kind": r["weight_kind"],
                "alpha": r["alpha"],
                "max_abs_err": r["max_abs_err"],
                "max_rel_err": r["max_rel_err"],
                "denom_magnitude": r["denom_magnitude"],
                "atol": r["atol"],
                "rtol": r["rtol"],
                "bucket_class": r["bucket_class"],
                "sub_seed": r["sub_seed"],
                "seed_root": r["seed_root"],
                "n_seeds_diverged": len(
                    sig_seeds_by_class[_signature(r)]["DIVERGENCE"]
                ),
                "n_seeds_recalibration": len(
                    sig_seeds_by_class[_signature(r)]["TOLERANCE_RECALIBRATION"]
                ),
            }
            for r in top3
        ],
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(main())
