"""v2 fuzzer for depthwise torch.nn.functional.conv2d on MPS vs CPU.

Spec (v2 swarm protocol):
- 500 iterations across seeds {0,1,2,3,4}
- dtypes: fp32, fp16, bf16
- Input shapes are 4D (N, C, H, W) sized to fit a conv kernel
- Weight shape is (C, 1, kH, kW); groups = C  (canonical depthwise)
- Stride-category fuzzing applied to the input tensor; weight kept contiguous
- Conv params (stride/padding/dilation) sampled per-iter
- Divergence filtering:
    * max_rel_err > 10x tolerance counts ONLY if denom_magnitude >= 1e-6
    * max_abs_err > 10x tolerance always counts
    * Reproducible across >=3 seeds  -> FILABLE
    * 1x..5x tolerance               -> TOLERANCE_RECALIBRATION
    * <1x tolerance                  -> OK
- 8 minute wall budget.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-conv-depthwise/src")
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_conv-depthwise.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.conv2d[depthwise]"
N_ITERS = 500
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30

# Channel counts kept modest — depthwise output is C·multiplier, multiplier=1 here
CHANNELS_SMALL = [1, 3, 4, 8]
CHANNELS_MED = [16, 32, 64]
CHANNELS_PRIME = [3, 5, 7, 11, 13]

# Spatial sizes per bucket — H and W can differ
H_DEGEN = [1, 2, 3]
H_NON_TILE = [15, 17, 31, 33, 63, 65]
H_PRIME = [7, 11, 13, 17, 19, 23, 29]
H_POW2 = [16, 32, 64, 128]
H_LARGE = [96, 128, 160]
H_MIXED = [9, 24, 48, 56, 80]

KERNEL_SIZES = [(1, 1), (3, 3), (5, 5), (7, 7), (1, 3), (3, 1), (1, 5), (5, 1)]

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())

BUCKET_NAMES = [
    "degenerate",
    "non_tile_aligned",
    "prime",
    "power_of_2_boundary",
    "large",
    "mixed",
]


def _sample_input_shape(rng: random.Random, bucket: str) -> tuple[int, int, int, int]:
    """Return (N, C, H, W) for the bucket.

    Sizes are kept small enough that any kernel up to 7x7 will fit (the caller
    truncates the kernel if H or W is too small).
    """
    n = rng.choice([1, 1, 1, 2])
    if bucket == "degenerate":
        c = rng.choice([1, 1, 2])
        h = rng.choice(H_DEGEN)
        w = rng.choice(H_DEGEN)
    elif bucket == "non_tile_aligned":
        c = rng.choice(CHANNELS_SMALL + CHANNELS_MED)
        h = rng.choice(H_NON_TILE)
        w = rng.choice(H_NON_TILE)
    elif bucket == "prime":
        c = rng.choice(CHANNELS_PRIME)
        h = rng.choice(H_PRIME)
        w = rng.choice(H_PRIME)
    elif bucket == "power_of_2_boundary":
        c = rng.choice(CHANNELS_SMALL + CHANNELS_MED)
        h = rng.choice(H_POW2)
        w = rng.choice(H_POW2)
        # Sometimes shift by ±1 to land on the boundary
        if rng.random() < 0.5:
            h = max(1, h + rng.choice([-1, 1]))
        if rng.random() < 0.5:
            w = max(1, w + rng.choice([-1, 1]))
    elif bucket == "large":
        c = rng.choice(CHANNELS_SMALL + CHANNELS_MED)
        h = rng.choice(H_LARGE)
        w = rng.choice(H_LARGE)
    else:  # mixed
        c = rng.choice(CHANNELS_SMALL + CHANNELS_MED + CHANNELS_PRIME)
        h = rng.choice(H_MIXED)
        w = rng.choice(H_MIXED)
    return n, c, h, w


def _sample_kernel(rng: random.Random, h: int, w: int) -> tuple[int, int]:
    candidates = [(kh, kw) for (kh, kw) in KERNEL_SIZES if kh <= h and kw <= w]
    if not candidates:
        return 1, 1
    return rng.choice(candidates)


def _sample_conv_params(
    rng: random.Random, h: int, w: int, kh: int, kw: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    stride_h = rng.choice([1, 1, 1, 2])
    stride_w = rng.choice([1, 1, 1, 2])
    pad_h = rng.choice([0, 0, kh // 2])
    pad_w = rng.choice([0, 0, kw // 2])
    dil_h = 1
    dil_w = 1
    # Only enable dilation if input large enough so output stays positive
    eff_kh = (kh - 1) * dil_h + 1
    eff_kw = (kw - 1) * dil_w + 1
    if h + 2 * pad_h - eff_kh < 1 or w + 2 * pad_w - eff_kw < 1:
        return (1, 1), (0, 0), (1, 1)
    return (stride_h, stride_w), (pad_h, pad_w), (dil_h, dil_w)


def _max_rel_err_with_denom(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float]:
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
    shape: tuple[int, ...],
    dtype: str,
    stride_cat: str,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> str:
    return f"{stride_cat}|{dtype}|{tuple(shape)}|k={kernel}|s={stride}|p={padding}"


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

    div_by_sig: dict[str, list[dict]] = defaultdict(list)
    recal_by_sig: dict[str, list[dict]] = defaultdict(list)
    per_dtype_max_rel: dict[str, float] = defaultdict(float)
    per_stride_max_rel: dict[str, float] = defaultdict(float)
    per_kernel_max_rel: dict[str, float] = defaultdict(float)
    per_bucket_counts: dict[str, int] = defaultdict(int)
    per_dtype_counts: dict[str, int] = defaultdict(int)
    per_stride_counts: dict[str, int] = defaultdict(int)
    per_kernel_counts: dict[str, int] = defaultdict(int)

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
            n, c, h, w = _sample_input_shape(rng, bucket)
            kh, kw = _sample_kernel(rng, h, w)
            stride_hw, pad_hw, dil_hw = _sample_conv_params(rng, h, w, kh, kw)
            dtype_name = rng.choice(DTYPE_NAMES)
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = rng.choice(STRIDE_CATEGORIES)

            input_shape = (n, c, h, w)
            weight_shape = (c, 1, kh, kw)
            kernel_tag = f"{kh}x{kw}"

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1
            per_kernel_counts[kernel_tag] += 1

            tensor_seed = rng.randrange(2**31 - 1)
            weight_seed = tensor_seed ^ 0xA5A5_5A5A

            try:
                x_cpu = fuzz_strides_for_category(
                    input_shape, dtype, stride_cat, device="cpu", seed=tensor_seed,
                )
                try:
                    x_mps = fuzz_strides_for_category(
                        input_shape, dtype, stride_cat, device="mps", seed=tensor_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={input_shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                # Weight: contiguous; deterministic from weight_seed
                gen_cpu = torch.Generator(device="cpu").manual_seed(weight_seed)
                w_cpu = torch.empty(weight_shape, dtype=dtype)
                if dtype == torch.float32:
                    w_cpu.normal_(mean=0.0, std=0.5, generator=gen_cpu)
                else:
                    tmp = torch.empty(weight_shape, dtype=torch.float32)
                    tmp.normal_(mean=0.0, std=0.5, generator=gen_cpu)
                    w_cpu = tmp.to(dtype)
                w_mps = w_cpu.to("mps")

                try:
                    y_mps = F.conv2d(
                        x_mps,
                        w_mps,
                        bias=None,
                        stride=stride_hw,
                        padding=pad_hw,
                        dilation=dil_hw,
                        groups=c,
                    )
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={input_shape} "
                        f"k={kernel_tag} s={stride_hw} p={pad_hw}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.conv2d(
                        x_cpu,
                        w_cpu,
                        bias=None,
                        stride=stride_hw,
                        padding=pad_hw,
                        dilation=dil_hw,
                        groups=c,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] seed={seed} iter={it} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={input_shape}: "
                        f"{str(exc).splitlines()[0][:200]}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                rel, denom_mag = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
                absdiff = _max_abs_err(y_mps_cpu, y_cpu)

                effective_rel = rel if denom_mag >= 1e-6 else 0.0
                if absdiff > max_abs_err_global:
                    max_abs_err_global = absdiff
                if effective_rel > max_rel_err_global:
                    max_rel_err_global = effective_rel
                if effective_rel > per_dtype_max_rel[dtype_name]:
                    per_dtype_max_rel[dtype_name] = effective_rel
                if effective_rel > per_stride_max_rel[stride_cat]:
                    per_stride_max_rel[stride_cat] = effective_rel
                if effective_rel > per_kernel_max_rel[kernel_tag]:
                    per_kernel_max_rel[kernel_tag] = effective_rel

                # MatMul-style sqrt(k/128) scaling: depthwise reduces over kH*kW
                # elements per output, so use that as the inner-dim k.
                atol, rtol = compute_tolerance(
                    dtype, device_type="mps", reduction_dim=kh * kw,
                )
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
                    "kernel": [kh, kw],
                    "conv_stride": list(stride_hw),
                    "padding": list(pad_hw),
                    "dilation": list(dil_hw),
                    "groups": c,
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
                sig = _signature(
                    input_shape, dtype_name, stride_cat,
                    (kh, kw), stride_hw, pad_hw,
                )
                if klass == "DIVERGENCE":
                    div_by_sig[sig].append(rec)
                else:
                    recal_by_sig[sig].append(rec)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[ERROR] seed={seed} iter={it} {bucket}/{dtype_name}/{stride_cat} "
                    f"shape={input_shape} k={kernel_tag}: {exc!r}",
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
                    per_kernel_counts=per_kernel_counts,
                    per_dtype_max_rel=per_dtype_max_rel,
                    per_stride_max_rel=per_stride_max_rel,
                    per_kernel_max_rel=per_kernel_max_rel,
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
        per_kernel_counts=per_kernel_counts,
        per_dtype_max_rel=per_dtype_max_rel,
        per_stride_max_rel=per_stride_max_rel,
        per_kernel_max_rel=per_kernel_max_rel,
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
    filable: list[dict] = []
    non_filable: list[dict] = []
    for _sig, recs in div_by_sig.items():
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
    iters_unsupported: int,
    div_by_sig: dict[str, list[dict]],
    recal_by_sig: dict[str, list[dict]],
    max_abs_err: float,
    max_rel_err: float,
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    per_kernel_counts: dict[str, int],
    per_dtype_max_rel: dict[str, float],
    per_stride_max_rel: dict[str, float],
    per_kernel_max_rel: dict[str, float],
    elapsed: float,
) -> None:
    filable, non_filable = _split_filable(div_by_sig)

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
        "per_kernel_max_rel": dict(per_kernel_max_rel),
        "per_shape_bucket_counts": dict(per_bucket_counts),
        "per_dtype_counts": dict(per_dtype_counts),
        "per_stride_counts": dict(per_stride_counts),
        "per_kernel_counts": dict(per_kernel_counts),
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
    lines.append(f"| kernel | {s['per_kernel_counts']} |")
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
    lines.append("| kernel | max_rel_err |")
    lines.append("|---|---|")
    for k in sorted(s["per_kernel_max_rel"]):
        lines.append(f"| `{k}` | {s['per_kernel_max_rel'][k]:.3e} |")
    lines.append("")
    lines.append("## Top 3 minimal repros")
    lines.append("")
    if not s["top_3_repros"]:
        lines.append(
            "_No FILABLE or RECALIBRATION-bucket records — all errors fell within "
            "1x the dtype tolerance (assertions/tolerances.py with MPS 2x multiplier "
            "and sqrt(kH*kW/128) scaling)._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            lines.append(f"### Repro #{i} ({r['classification']})")
            lines.append("")
            lines.append(
                f"- **input shape:** `{tuple(r['input_shape'])}`  "
                f"(bucket: `{r['shape_bucket']}`)"
            )
            lines.append(
                f"- **weight shape:** `{tuple(r['weight_shape'])}` "
                f"(groups={r['groups']})"
            )
            lines.append(
                f"- **kernel:** `{tuple(r['kernel'])}`, "
                f"stride: `{tuple(r['conv_stride'])}`, "
                f"padding: `{tuple(r['padding'])}`, "
                f"dilation: `{tuple(r['dilation'])}`"
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
            lines.append(
                f"- **seed:** {r['seed']}, tensor_seed: {r['tensor_seed']}, "
                f"weight_seed: {r['weight_seed']}"
            )
            lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- Reference: `torch.nn.functional.conv2d` on CPU, depthwise (groups=C), "
        "promoted to FP32 for the error comparison."
    )
    lines.append(
        "- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
        "dtype, device_type='mps', reduction_dim=kH*kW)` — kH·kW is the inner "
        "reduction dimension of depthwise conv per output element."
    )
    lines.append(
        "- Classification: max_abs > 10x atol => DIVERGENCE; max_rel > 10x rtol "
        "with |b|>=1e-6 => DIVERGENCE; otherwise the [1x, 5x] band is "
        "TOLERANCE_RECALIBRATION; below 1x is OK."
    )
    lines.append(
        "- A DIVERGENCE only becomes FILABLE if its (stride, dtype, shape, "
        "kernel, conv_stride, padding) signature reproduces under >=3 distinct seeds."
    )
    lines.append(
        "- Stride categories applied to the input only; weight is contiguous "
        "to isolate input-layout effects on MPS depthwise conv."
    )
    lines.append(
        "- Fuzzer: `_fuzz_conv_depthwise.py` in this directory."
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
