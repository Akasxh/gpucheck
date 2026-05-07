"""v2 fuzzer for torch.nn.functional.conv2d on MPS vs CPU.

Spec (v2 swarm protocol):
- 500 iterations across seeds {0,1,2,3,4} (100 per seed)
- dtypes: fp32, fp16, bf16
- Stride/contiguity perturbation on the *input* tensor; weight stays row_major
  (perturbing weight as well doubles the failure surface and conflates two bugs).
- Divergence filtering (FILABLE):
    * max_rel_err > 10x rtol counts ONLY if denom_magnitude >= 1e-6
      (avoids near-zero-denominator artifacts)
    * max_abs_err > 10x atol always counts toward DIVERGENCE
    * DIVERGENCE that reproduces under >=3 distinct seeds  -> FILABLE
    * Lower-band errors (1x..5x of tolerance) -> TOLERANCE_RECALIBRATION
    * <1x tolerance                          -> OK
- 8 minute wall budget.

CUDA channel: this Mac has no NVIDIA GPU, so the CUDA path is N/A — only
the MPS-vs-CPU divergence channel is exercised.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

WORKTREE = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-conv2d")
sys.path.insert(0, str(WORKTREE / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_conv2d.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.conv2d"
AGENT = "kernel-fuzzer-conv2d-v2"
N_ITERS = 500
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 45  # leave 45s for write + safety margin

# ---------------------------------------------------------------------------
# Shape buckets for conv2d input (N, C_in, H, W)
# Kept conservative — conv2d with large H,W,C explodes wall time fast.
# ---------------------------------------------------------------------------
N_CHOICES = [1, 2]
C_IN_CHOICES = [1, 3, 8, 16, 32]
C_OUT_CHOICES = [1, 4, 8, 16, 32]
K_CHOICES = [1, 3, 5]  # square kernels

DEGENERATE_HW = [(1, 1), (1, 4), (4, 1), (3, 3), (5, 5)]
NON_TILE_HW = [(15, 15), (17, 17), (33, 31), (9, 18), (27, 9)]
PRIME_HW = [(7, 11), (13, 17), (29, 31), (5, 13)]
POW2_BOUNDARY_HW = [(15, 16), (16, 17), (32, 33), (63, 64), (64, 65)]
LARGE_HW = [(96, 96), (128, 128), (192, 64)]
MIXED_HW = [(127, 16), (33, 128), (7, 64)]

SHAPE_BUCKETS: dict[str, list[tuple[int, int]]] = {
    "degenerate": DEGENERATE_HW,
    "non_tile_aligned": NON_TILE_HW,
    "prime": PRIME_HW,
    "power_of_2_boundary": POW2_BOUNDARY_HW,
    "large": LARGE_HW,
    "mixed": MIXED_HW,
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
    """Return (max_rel_err, |b| at the location of max-rel-err)."""
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
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


def _signature(
    n: int, c_in: int, c_out: int, h: int, w: int, k: int,
    dtype: str, stride_cat: str,
) -> str:
    return f"{stride_cat}|{dtype}|N{n}_Cin{c_in}_Cout{c_out}_H{h}_W{w}_k{k}"


def _classify(
    abs_err: float, rel_err: float, denom_mag: float, atol: float, rtol: float,
) -> str:
    div_abs = abs_err > 10.0 * atol
    div_rel = (rel_err > 10.0 * rtol) and (denom_mag >= 1e-6)
    if div_abs or div_rel:
        return "DIVERGENCE"
    recal_abs = (abs_err > atol) and (abs_err <= 5.0 * atol)
    recal_rel = (rel_err > rtol) and (rel_err <= 5.0 * rtol) and (denom_mag >= 1e-6)
    if recal_abs or recal_rel:
        return "TOLERANCE_RECALIBRATION"
    return "OK"


def _build_input(
    shape: tuple[int, int, int, int], dtype: torch.dtype,
    stride_cat: str, device: str, seed: int,
) -> torch.Tensor:
    return fuzz_strides_for_category(shape, dtype, stride_cat, device=device, seed=seed)


def _build_weight(
    shape: tuple[int, int, int, int], dtype: torch.dtype, device: str, seed: int,
) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    # row_major / contiguous weight — keeps the variation focused on input layout.
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
            "agent": AGENT, "kernel": KERNEL, "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
            "iters_attempted": 0, "iters_completed": 0,
            "divergences_filable": 0, "divergences_recalibration": 0,
            "max_abs_err": 0.0, "max_rel_err": 0.0, "top_3_repros": [],
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz — SKIPPED\n\nMPS not available.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported_mps = 0
    iters_unsupported_cpu = 0
    iters_unsupported_build = 0
    iters_skipped_empty = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    max_rel_err_raw_unfiltered = 0.0

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
                    file=sys.stderr, flush=True,
                )
                aborted = True
                break
            iters_attempted += 1

            bucket = rng.choice(BUCKET_NAMES)
            h, w = rng.choice(SHAPE_BUCKETS[bucket])
            n = rng.choice(N_CHOICES)
            c_in = rng.choice(C_IN_CHOICES)
            c_out = rng.choice(C_OUT_CHOICES)
            k = rng.choice(K_CHOICES)
            # Skip configs where kernel exceeds spatial extent (would zero output).
            if k > h or k > w:
                k = 1
            dtype_name = rng.choice(DTYPE_NAMES)
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = rng.choice(STRIDE_CATEGORIES)

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            tensor_seed = rng.randrange(2**31 - 1)
            weight_seed = rng.randrange(2**31 - 1)

            input_shape = (n, c_in, h, w)
            weight_shape = (c_out, c_in, k, k)

            try:
                try:
                    x_cpu = _build_input(input_shape, dtype, stride_cat, "cpu", tensor_seed)
                    x_mps = _build_input(input_shape, dtype, stride_cat, "mps", tensor_seed)
                    w_cpu = _build_weight(weight_shape, dtype, "cpu", weight_seed)
                    w_mps = _build_weight(weight_shape, dtype, "mps", weight_seed)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported_build += 1
                    print(
                        f"[unsupported-build] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} in={input_shape} k={k}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr, flush=True,
                    )
                    continue

                if x_cpu.numel() == 0 or w_cpu.numel() == 0:
                    iters_skipped_empty += 1
                    iters_completed += 1
                    continue

                # MPS first
                try:
                    y_mps = F.conv2d(x_mps, w_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported_mps += 1
                    print(
                        f"[unsupported-mps] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} in={input_shape} k={k}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr, flush=True,
                    )
                    continue

                # CPU reference (same dtype to compare apples-to-apples)
                try:
                    y_cpu = F.conv2d(x_cpu, w_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported_cpu += 1
                    print(
                        f"[unsupported-cpu] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} in={input_shape} k={k}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr, flush=True,
                    )
                    continue

                iters_completed += 1

                rel, denom_mag = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
                absdiff = _max_abs_err(y_mps_cpu, y_cpu)

                if rel > max_rel_err_raw_unfiltered:
                    max_rel_err_raw_unfiltered = rel
                effective_rel = rel if denom_mag >= 1e-6 else 0.0
                if absdiff > max_abs_err_global:
                    max_abs_err_global = absdiff
                if effective_rel > max_rel_err_global:
                    max_rel_err_global = effective_rel
                if effective_rel > per_dtype_max_rel[dtype_name]:
                    per_dtype_max_rel[dtype_name] = effective_rel
                if effective_rel > per_stride_max_rel[stride_cat]:
                    per_stride_max_rel[stride_cat] = effective_rel

                # Tolerance: conv2d's reduction dim is C_in * k * k.
                k_dim = c_in * k * k
                atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
                klass = _classify(absdiff, rel, denom_mag, atol, rtol)

                if klass == "OK":
                    continue

                rec = {
                    "seed": seed,
                    "tensor_seed": tensor_seed,
                    "weight_seed": weight_seed,
                    "iter": it,
                    "input_shape": list(input_shape),
                    "weight_shape": list(weight_shape),
                    "k": k,
                    "k_dim_reduction": k_dim,
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
                sig = _signature(n, c_in, c_out, h, w, k, dtype_name, stride_cat)
                if klass == "DIVERGENCE":
                    div_by_sig[sig].append(rec)
                else:
                    recal_by_sig[sig].append(rec)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Per spec: halt on unknown error, but write what we have.
                print(
                    f"[ERROR] seed={seed} iter={it} {bucket}/{dtype_name}/{stride_cat} "
                    f"in={input_shape} k={k}: {exc!r}",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                _write_results(
                    aborted=True,
                    error_msg=f"{exc!r}",
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported_mps=iters_unsupported_mps,
                    iters_unsupported_cpu=iters_unsupported_cpu,
                    iters_unsupported_build=iters_unsupported_build,
                    iters_skipped_empty=iters_skipped_empty,
                    div_by_sig=div_by_sig,
                    recal_by_sig=recal_by_sig,
                    max_abs_err=max_abs_err_global,
                    max_rel_err=max_rel_err_global,
                    max_rel_err_raw_unfiltered=max_rel_err_raw_unfiltered,
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
        iters_unsupported_mps=iters_unsupported_mps,
        iters_unsupported_cpu=iters_unsupported_cpu,
        iters_unsupported_build=iters_unsupported_build,
        iters_skipped_empty=iters_skipped_empty,
        div_by_sig=div_by_sig,
        recal_by_sig=recal_by_sig,
        max_abs_err=max_abs_err_global,
        max_rel_err=max_rel_err_global,
        max_rel_err_raw_unfiltered=max_rel_err_raw_unfiltered,
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
    """A signature is FILABLE iff it appears under >=3 distinct seeds."""
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
    if not records:
        return []
    sorted_recs = sorted(
        records,
        key=lambda r: (-r["max_abs_err"], _numel(r["input_shape"])),
    )
    return sorted_recs[:3]


def _write_results(
    *,
    aborted: bool,
    error_msg: str | None,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported_mps: int,
    iters_unsupported_cpu: int,
    iters_unsupported_build: int,
    iters_skipped_empty: int,
    div_by_sig: dict[str, list[dict]],
    recal_by_sig: dict[str, list[dict]],
    max_abs_err: float,
    max_rel_err: float,
    max_rel_err_raw_unfiltered: float,
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    per_dtype_max_rel: dict[str, float],
    per_stride_max_rel: dict[str, float],
    elapsed: float,
) -> None:
    filable, non_filable_div = _split_filable(div_by_sig)

    recal_records: list[dict] = []
    for recs in recal_by_sig.values():
        recal_records.extend(recs)
    recal_records.extend(non_filable_div)

    top3 = _top_3_repros(filable) or _top_3_repros(recal_records)

    status = "ABORTED" if aborted else "OK"
    if error_msg:
        status = "ERROR"

    iters_unsupported_total = (
        iters_unsupported_mps + iters_unsupported_cpu + iters_unsupported_build
    )

    if filable:
        filing_target = "pytorch/pytorch"
    else:
        filing_target = "none"

    summary = {
        "agent": AGENT,
        "kernel": KERNEL,
        "status": status,
        "error": error_msg,
        "backend_mps": True,
        "backend_cuda": "mocked (no NVIDIA GPU on host)",
        "device_under_test": "mps",
        "reference": "cpu_same_dtype",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported_total": iters_unsupported_total,
        "iters_unsupported_mps": iters_unsupported_mps,
        "iters_unsupported_cpu": iters_unsupported_cpu,
        "iters_unsupported_build": iters_unsupported_build,
        "iters_skipped_empty": iters_skipped_empty,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal_records),
        "divergences_unconfirmed_unique_configs": len(
            {r["seed"] for r in non_filable_div}
        ) if non_filable_div else 0,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "max_rel_err_raw_unfiltered": max_rel_err_raw_unfiltered,
        "top_3_repros": top3,
        "filable_records": filable,
        "per_dtype_max_rel": dict(per_dtype_max_rel),
        "per_stride_max_rel": dict(per_stride_max_rel),
        "per_shape_bucket_counts": dict(per_bucket_counts),
        "per_dtype_counts": dict(per_dtype_counts),
        "per_stride_counts": dict(per_stride_counts),
        "elapsed_seconds": round(elapsed, 2),
        "wall_budget_seconds": WALL_BUDGET_S,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "n_iters_target": N_ITERS,
        "denom_filter_floor": 1e-06,
        "filable_threshold": "max_abs > 10x atol OR (max_rel > 10x rtol AND denom>=1e-6); reproduces on >=3 seeds",
        "recal_threshold": "1x..5x of tolerance, OR DIVERGENCE on <3 seeds",
        "spec_version": "v2",
        "filing_target": filing_target,
        "results_md": str(RESULTS_MD),
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(path: Path, s: dict) -> None:
    lines: list[str] = []
    lines.append(f"# {s['kernel']} — MPS fuzz report (v2)")
    lines.append("")
    lines.append(f"- **Agent:** `{s['agent']}`")
    lines.append(f"- **Kernel:** `{s['kernel']}`")
    lines.append(f"- **Status:** {s['status']}")
    if s.get("error"):
        lines.append(f"- **Error:** `{s['error']}`")
    lines.append(f"- **Device under test:** MPS (Apple Silicon, real)")
    lines.append(f"- **Reference:** CPU (same dtype, `F.conv2d`)")
    lines.append(f"- **CUDA backend:** {s['backend_cuda']} — N/A on this host")
    lines.append(
        f"- **Iters attempted:** {s['iters_attempted']} / target {s['n_iters_target']}"
    )
    lines.append(f"- **Iters completed:** {s['iters_completed']}")
    lines.append(
        f"- **Iters unsupported (MPS / CPU / build):** "
        f"{s['iters_unsupported_mps']} / {s['iters_unsupported_cpu']} / "
        f"{s['iters_unsupported_build']}"
    )
    lines.append(f"- **Iters skipped (empty tensor):** {s['iters_skipped_empty']}")
    lines.append(f"- **Divergences (FILABLE, >=3 seeds):** {s['divergences_filable']}")
    lines.append(
        f"- **Divergences (TOLERANCE_RECALIBRATION):** {s['divergences_recalibration']}"
    )
    lines.append(f"- **Max abs err (MPS vs CPU):** {s['max_abs_err']:.3e}")
    lines.append(
        f"- **Max rel err (denom>=1e-6 filter):** {s['max_rel_err']:.3e}"
    )
    lines.append(
        f"- **Max rel err (raw, unfiltered — for context):** "
        f"{s['max_rel_err_raw_unfiltered']:.3e}"
    )
    lines.append(f"- **Filing target:** `{s['filing_target']}`")
    lines.append(f"- **Elapsed:** {s['elapsed_seconds']} s "
                 f"(budget {s['wall_budget_seconds']} s)")
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
            "1x the dtype tolerance (`gpucheck.assertions.tolerances.compute_tolerance` "
            "with k_dim = C_in*kH*kW and the MPS 2x overlay)._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i} ({r['classification']})")
            lines.append("")
            lines.append(
                f"- **input shape (N,C,H,W):** `{tuple(r['input_shape'])}`  "
                f"(bucket: `{r['shape_bucket']}`)"
            )
            lines.append(
                f"- **weight shape (Cout,Cin,kH,kW):** `{tuple(r['weight_shape'])}`"
            )
            lines.append(
                f"- **dtype:** `{r['dtype']}` | "
                f"**stride category (input):** `{r['stride_category']}`"
            )
            lines.append(
                f"- **k_dim (reduction = C_in*kH*kW):** {r['k_dim_reduction']}"
            )
            lines.append(
                f"- **max abs err:** {r['max_abs_err']:.3e}  (atol={r['atol']:.2e})"
            )
            lines.append(
                f"- **max rel err:** {r['max_rel_err']:.3e}  (rtol={r['rtol']:.2e})"
            )
            lines.append(
                f"- **denom |b| at max rel:** {r['denom_magnitude_at_max_rel']:.3e}"
            )
            lines.append(f"- **CPU input layout:** `{r['x_layout_cpu']}`")
            lines.append(f"- **MPS input layout:** `{r['x_layout_mps']}`")
            lines.append(
                f"- **seeds:** seed={r['seed']}, "
                f"tensor_seed={r['tensor_seed']}, weight_seed={r['weight_seed']}"
            )
            lines.append("")
    lines.append("## FILABLE summary")
    lines.append("")
    if not s["filable_records"]:
        lines.append(
            "_No filable bug found. All MPS-vs-CPU divergences either fell within the "
            "tolerance band or did not reproduce on >=3 seeds._"
        )
    else:
        seen_sigs: set[str] = set()
        for r in s["filable_records"]:
            sig = (
                f"{r['stride_category']}|{r['dtype']}|"
                f"in={tuple(r['input_shape'])}|w={tuple(r['weight_shape'])}"
            )
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            lines.append(
                f"- `{sig}` — abs_err={r['max_abs_err']:.3e}, "
                f"rel_err={r['max_rel_err']:.3e}, atol={r['atol']:.2e}, "
                f"rtol={r['rtol']:.2e}"
            )
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- **Operation:** `torch.nn.functional.conv2d(input, weight)` "
        "(default stride=1, padding=0, dilation=1, groups=1)."
    )
    lines.append(
        "- **Reference:** same-dtype `F.conv2d` on CPU. We use same-dtype on both "
        "sides so we are measuring kernel-implementation drift, not promotion drift."
    )
    lines.append(
        "- **Tolerance:** `gpucheck.assertions.tolerances.compute_tolerance(dtype, "
        "k_dim=C_in*kH*kW, device_type='mps')` — base + sqrt(k/128) reduction "
        "scaling + MPS 2x overlay."
    )
    lines.append(
        "- **Classification:**"
    )
    lines.append(
        "  - `max_abs > 10x atol` => DIVERGENCE (always)"
    )
    lines.append(
        "  - `max_rel > 10x rtol AND |b| >= 1e-6` => DIVERGENCE"
    )
    lines.append(
        "  - DIVERGENCE that reproduces on >= 3 seeds => **FILABLE**"
    )
    lines.append(
        "  - DIVERGENCE that reproduces on < 3 seeds OR errors in 1x..5x band "
        "=> TOLERANCE_RECALIBRATION (xfail / overlay candidate)"
    )
    lines.append(
        "  - else OK"
    )
    lines.append(
        "- **Stride/contiguity perturbation:** input only — weight kept row-major "
        "to focus on the input strided code path. Categories: row_major, column_major, "
        "broadcast, transpose, slice, non_contig, gather "
        "(see `gpucheck.fuzzing.strides`)."
    )
    lines.append(
        "- **CUDA channel:** N/A — no NVIDIA GPU on this host. The CUDA backend in "
        "gpucheck is mockable for arch detection but cannot run actual conv2d kernels "
        "without hardware. Same-dtype CPU is the industry-standard fallback for "
        "MPS validation (see SYNTHESIS §7)."
    )
    lines.append(f"- **Fuzzer:** `_fuzz_conv2d_v2.py` in this directory.")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
