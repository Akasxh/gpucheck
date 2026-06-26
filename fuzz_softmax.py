"""v2 fuzzer for softmax — MPS vs CPU reference, with FILABLE / RECALIBRATION
divergence buckets per the testing-team v2 swarm spec.

Design
------
- 200 distinct (shape, dtype, stride_cat) samples are drawn from a meta-RNG.
- Each sample is replayed under seeds 0,1,2,3,4 (value-generator seeds), giving
  exactly 1000 (sample, seed) iterations.
- Per-iteration we record max_abs_err, max_rel_err, and the |reference| value
  at the location of the worst relative error ("denom_at_worst").
- Per-sample classification:
    * a seed is "violating" iff
        max_abs_err >= 10*atol  OR
       (max_rel_err >= 10*rtol  AND  denom_at_worst >= 1e-6)
    * sample = FILABLE       if violating_seeds >= 3
    * sample = RECALIBRATION if any seed has 1*tol <= ratio < 10*tol
                              (or <3 seeds violate — flaky / value-dependent)
    * sample = OK            otherwise

Tolerance source
----------------
gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=last_dim,
device_type='mps') — k_dim scaling because softmax reduces over the last dim.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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

KERNEL = "softmax"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_SAMPLES = 200                  # distinct (shape, dtype, stride) tuples
N_ITERATIONS = N_SAMPLES * len(SEEDS)  # 1000
WALL_BUDGET_S = 12 * 60 - 30     # 30s safety margin under 12-min hard cap

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_MD = OUT_DIR / f"RESULTS_{KERNEL}.md"
OUT_JSONL = OUT_DIR / "swarm.jsonl"

SHAPE_BUCKETS = (
    "degenerate",
    "prime",
    "power_of_2_boundary",
    "non_tile_aligned",
    "large",
)

DTYPES: tuple[tuple[str, "torch.dtype"], ...] = (
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
)


def mps_ready() -> bool:
    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def sample_shape(rng: random.Random) -> tuple[tuple[int, ...], str]:
    bucket = rng.choice(SHAPE_BUCKETS)
    if bucket == "degenerate":
        choices: list[tuple[int, ...]] = [
            (1, 1), (2, 1), (1, 16), (16, 1),
            (1, 7), (1, 128), (1, 257),
        ]
        return rng.choice(choices), bucket
    if bucket == "prime":
        return (rng.choice(PRIMES), rng.choice(PRIMES)), bucket
    if bucket == "power_of_2_boundary":
        v1 = min(rng.choice(POWER_OF_2_BOUNDARIES), 513)
        v2 = min(rng.choice(POWER_OF_2_BOUNDARIES), 513)
        return (v1, v2), bucket
    if bucket == "non_tile_aligned":
        v1 = max(rng.choice(TILE_SIZES) + rng.choice((-1, 1, 3)), 1)
        v2 = max(rng.choice(TILE_SIZES) + rng.choice((-1, 1, 3)), 1)
        return (v1, v2), bucket
    big = min(rng.choice(LARGE_DIMS), 4096)
    batch = rng.choice((1, 8, 32))
    return (batch, big), bucket


@dataclass
class IterResult:
    seed: int
    max_abs_err: float
    max_rel_err: float
    denom_at_worst_rel: float   # |b[i*]| where i* = argmax(rel_err)
    abs_ratio: float            # max_abs_err / atol
    rel_ratio: float            # max_rel_err / rtol
    violating: bool             # filable-class violation


@dataclass
class SampleRecord:
    sample_idx: int
    shape: tuple[int, ...]
    dtype_name: str
    stride: str
    bucket: str
    atol: float
    rtol: float
    iters: list[IterResult] = field(default_factory=list)
    bucket_class: str = "PENDING"  # OK / RECALIBRATION / FILABLE / UNSUPPORTED

    @property
    def violating_seed_count(self) -> int:
        return sum(1 for r in self.iters if r.violating)

    @property
    def max_abs_err(self) -> float:
        return max((r.max_abs_err for r in self.iters), default=0.0)

    @property
    def max_rel_err(self) -> float:
        return max((r.max_rel_err for r in self.iters), default=0.0)

    @property
    def max_abs_ratio(self) -> float:
        return max((r.abs_ratio for r in self.iters), default=0.0)

    @property
    def max_rel_ratio_with_denom(self) -> float:
        # Largest rel_ratio among iterations whose denom passes the gate.
        return max(
            (r.rel_ratio for r in self.iters if r.denom_at_worst_rel >= 1e-6),
            default=0.0,
        )


@dataclass
class Stats:
    attempted: int = 0
    completed: int = 0
    skipped_empty: int = 0
    unsupported: int = 0
    errors: int = 0
    samples: list[SampleRecord] = field(default_factory=list)
    max_rel_err_overall: float = 0.0
    max_abs_err_overall: float = 0.0


def softmax_op(t: torch.Tensor) -> torch.Tensor:
    return torch.softmax(t, dim=-1)


def per_iter_errors(out_mps: torch.Tensor, out_cpu: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, denom_at_worst_rel)."""
    a = out_mps.detach().to("cpu", dtype=torch.float32).contiguous()
    b = out_cpu.detach().to(torch.float32).contiguous()
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    abs_diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-30)
    rel = abs_diff / denom
    # Mask non-finite (defensive — softmax shouldn't produce NaN/Inf, but bf16
    # underflow can in edge shapes).
    finite = torch.isfinite(abs_diff) & torch.isfinite(rel)
    if not finite.any():
        return 0.0, 0.0, 0.0
    abs_diff = abs_diff.where(finite, torch.zeros_like(abs_diff))
    rel = rel.where(finite, torch.zeros_like(rel))
    max_abs = float(abs_diff.max().item())
    max_rel = float(rel.max().item())
    # Find index of worst rel; report |b| there.
    flat_rel = rel.reshape(-1)
    idx = int(flat_rel.argmax().item())
    denom_at_worst = float(b.reshape(-1).abs()[idx].item())
    return max_abs, max_rel, denom_at_worst


def run_one_seed(
    sample: SampleRecord,
    seed: int,
    stats: Stats,
) -> str:
    """Returns 'completed' | 'unsupported' | 'error' | 'skipped'."""
    if any(d == 0 for d in sample.shape):
        stats.skipped_empty += 1
        return "skipped"

    dtype = dict(DTYPES)[sample.dtype_name]
    try:
        t_cpu = fuzz_strides_for_category(
            sample.shape, dtype, sample.stride, device="cpu", seed=seed,
        )
        t_mps = fuzz_strides_for_category(
            sample.shape, dtype, sample.stride, device="mps", seed=seed,
        )
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e).lower()
        if "not implement" in msg or "not supported" in msg or "mps" in msg:
            stats.unsupported += 1
            return "unsupported"
        stats.errors += 1
        return "error"

    try:
        out_mps = softmax_op(t_mps)
        torch.mps.synchronize()
        out_cpu = softmax_op(t_cpu)
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e).lower()
        if "not implement" in msg or "not supported" in msg:
            stats.unsupported += 1
            return "unsupported"
        stats.errors += 1
        return "error"

    abs_e, rel_e, denom_w = per_iter_errors(out_mps, out_cpu)
    abs_ratio = abs_e / max(sample.atol, 1e-30)
    rel_ratio = rel_e / max(sample.rtol, 1e-30)
    violating = (abs_ratio >= 10.0) or (rel_ratio >= 10.0 and denom_w >= 1e-6)

    sample.iters.append(IterResult(
        seed=seed,
        max_abs_err=abs_e,
        max_rel_err=rel_e,
        denom_at_worst_rel=denom_w,
        abs_ratio=abs_ratio,
        rel_ratio=rel_ratio,
        violating=violating,
    ))
    stats.completed += 1
    stats.max_abs_err_overall = max(stats.max_abs_err_overall, abs_e)
    stats.max_rel_err_overall = max(stats.max_rel_err_overall, rel_e)
    # Free MPS tensors aggressively to keep allocator pressure low.
    del t_mps, out_mps
    return "completed"


def classify_sample(sample: SampleRecord) -> str:
    """Bucket per spec.

    FILABLE        — violating_seed_count >= 3 (i.e. >=10x exceeded reproducibly)
    RECALIBRATION  — any seed has 1x <= ratio < 10x (either rel-with-denom or abs)
                     OR <3 violations at >=10x (flaky / value-dependent)
    OK             — every seed strictly below 1x tolerance (with denom gate on rel)
    """
    if not sample.iters:
        return "UNSUPPORTED"

    n_viol = sample.violating_seed_count

    def gated_rel_ratio(r: IterResult) -> float:
        return r.rel_ratio if r.denom_at_worst_rel >= 1e-6 else 0.0

    max_ratio_per_seed = [
        max(r.abs_ratio, gated_rel_ratio(r)) for r in sample.iters
    ]
    overall_max = max(max_ratio_per_seed) if max_ratio_per_seed else 0.0

    if n_viol >= 3:
        return "FILABLE"
    if any(1.0 <= mr for mr in max_ratio_per_seed):
        # Includes the 1-5x recalibration band, the 5-10x grey band, and the
        # >=10x flaky/non-reproducible band — all recommend a tolerance/xfail
        # action rather than an upstream filing.
        return "RECALIBRATION"
    _ = overall_max  # for debug visibility
    return "OK"


def write_outputs(
    stats: Stats,
    elapsed: float,
    end_reason: str,
    samples_attempted: int,
) -> None:
    filable = [s for s in stats.samples if s.bucket_class == "FILABLE"]
    recalib = [s for s in stats.samples if s.bucket_class == "RECALIBRATION"]
    unsupp_samples = [s for s in stats.samples if s.bucket_class == "UNSUPPORTED"]
    ok_samples = [s for s in stats.samples if s.bucket_class == "OK"]

    # Top 3 minimal repros: rank FILABLE first by violating_seed_count desc then
    # max_abs_ratio desc; if fewer than 3, fill with worst RECALIBRATION samples.
    def filable_key(s: SampleRecord) -> tuple[float, float, float]:
        return (-s.violating_seed_count, -s.max_abs_ratio, -s.max_rel_ratio_with_denom)

    def recalib_key(s: SampleRecord) -> tuple[float, float]:
        max_seed_ratio = max(
            (max(r.abs_ratio, r.rel_ratio if r.denom_at_worst_rel >= 1e-6 else 0.0)
             for r in s.iters), default=0.0,
        )
        return (-max_seed_ratio, -s.max_abs_err)

    sorted_filable = sorted(filable, key=filable_key)
    sorted_recalib = sorted(recalib, key=recalib_key)
    top3 = (sorted_filable + sorted_recalib)[:3]

    # ---------------------- Markdown ----------------------
    md: list[str] = []
    md.append(f"# Fuzz results — kernel: `{KERNEL}` (v2)\n")
    md.append("**Backends:** MPS (Apple Silicon) vs CPU reference (same dtype). "
              "CUDA: not present.")
    md.append("")
    md.append("## Summary\n")
    md.append(f"- **kernel**: `torch.softmax(x, dim=-1)`")
    md.append(f"- **iters_attempted**: {stats.attempted} (samples × seeds = "
              f"{samples_attempted} × {len(SEEDS)})")
    md.append(f"- **iters_completed**: {stats.completed}")
    md.append(f"- **iters_skipped_empty**: {stats.skipped_empty}")
    md.append(f"- **iters_unsupported**: {stats.unsupported}")
    md.append(f"- **iters_errored**: {stats.errors}")
    md.append(f"- **divergences_filable** (≥3-of-5 seeds at ≥10× tol, denom-gated): "
              f"**{len(filable)}**")
    md.append(f"- **divergences_recalibration** (1×–<10× tol, or <3-seed flaky): "
              f"**{len(recalib)}**")
    md.append(f"- samples OK : {len(ok_samples)}")
    md.append(f"- samples UNSUPPORTED : {len(unsupp_samples)}")
    md.append(f"- **max_abs_err** (MPS vs CPU, any iter): `{stats.max_abs_err_overall:.3e}`")
    md.append(f"- **max_rel_err** (MPS vs CPU, any iter): `{stats.max_rel_err_overall:.3e}`")
    md.append(f"- runtime: `{elapsed:.1f}s` (budget {WALL_BUDGET_S}s; end={end_reason})")
    md.append(f"- torch: `{torch.__version__}`, seeds: `{list(SEEDS)}`")
    md.append("")

    md.append("## Top 3 minimal repros\n")
    if not top3:
        md.append("_None — every (shape, dtype, stride) sample stayed below 1× the "
                  "MPS-overlay tolerance from `compute_tolerance(..., device_type='mps')`._")
    else:
        md.append("| # | class | shape | dtype | stride | bucket | "
                  "viol/5 | max_abs_err | max_rel_err | denom@worst | atol | rtol |")
        md.append("|---|-------|-------|-------|--------|--------|"
                  "--------|-------------|-------------|-------------|------|------|")
        for i, s in enumerate(top3, 1):
            worst = max(
                s.iters,
                key=lambda r: max(
                    r.abs_ratio,
                    r.rel_ratio if r.denom_at_worst_rel >= 1e-6 else 0.0,
                ),
                default=None,
            )
            denom_w = worst.denom_at_worst_rel if worst else 0.0
            md.append(
                f"| {i} | {s.bucket_class} | `{tuple(s.shape)}` | {s.dtype_name} | "
                f"{s.stride} | {s.bucket} | {s.violating_seed_count}/{len(s.iters)} | "
                f"{s.max_abs_err:.3e} | {s.max_rel_err:.3e} | {denom_w:.3e} | "
                f"{s.atol:.2e} | {s.rtol:.2e} |"
            )
    md.append("")

    md.append("## Filing recommendation\n")
    if filable:
        md.append("**pytorch/pytorch** — divergence between `torch.softmax` on MPS "
                  "and the same op on CPU at identical dtype is reproducible across "
                  "≥3 seeds at ≥10× the gpucheck MPS-overlay tolerance.")
    elif recalib:
        md.append("**xfail / tolerance-recalibration** — flag in `[tool.gpucheck.mps.xfail]` "
                  "or recalibrate `_MPS_TOLERANCE_MULTIPLIERS` for the offending "
                  "dtype × stride combination. No upstream PyTorch bug indicated.")
    else:
        md.append("**none** — MPS softmax stayed within the v1.0 MPS-overlay tolerance.")
    md.append("")

    md.append("## Method notes\n")
    md.append("- Reference: `torch.softmax(x.cpu(), dim=-1)` at the SAME dtype as MPS.")
    md.append("- Tolerance: `compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')`.")
    md.append("- Per-iter violation: `abs_ratio >= 10` OR (`rel_ratio >= 10` AND "
              "`|ref| at worst-rel index >= 1e-6`).")
    md.append("- Sample classification: FILABLE if ≥3-of-5 seeds violate; "
              "RECALIBRATION if any seed reaches 1×–<10× tol (or fewer than 3 seeds "
              "violate at ≥10×, i.e. value-dependent flakiness); else OK.")
    md.append(f"- Stride categories sampled: {', '.join(STRIDE_CATEGORIES)}")
    md.append(f"- Shape buckets: {', '.join(SHAPE_BUCKETS)}")
    md.append("")

    # FILABLE detail (if any)
    if filable:
        md.append("## FILABLE divergences (full list)\n")
        md.append("| shape | dtype | stride | bucket | viol/5 | max_abs_err | max_rel_err | atol | rtol |")
        md.append("|-------|-------|--------|--------|--------|-------------|-------------|------|------|")
        for s in sorted_filable:
            md.append(
                f"| `{tuple(s.shape)}` | {s.dtype_name} | {s.stride} | {s.bucket} | "
                f"{s.violating_seed_count}/5 | {s.max_abs_err:.3e} | {s.max_rel_err:.3e} | "
                f"{s.atol:.2e} | {s.rtol:.2e} |"
            )
        md.append("")

    OUT_MD.write_text("\n".join(md))

    # ---------------------- swarm.jsonl ----------------------
    rec = {
        "agent": "kernel-fuzzer-softmax-v2",
        "kernel": KERNEL,
        "op_path": "torch.softmax",
        "device_under_test": "mps",
        "reference": "cpu_same_dtype",
        "torch_version": torch.__version__,
        "seeds": list(SEEDS),
        "samples_attempted": samples_attempted,
        "iters_attempted": stats.attempted,
        "iters_completed": stats.completed,
        "iters_skipped_empty": stats.skipped_empty,
        "iters_unsupported": stats.unsupported,
        "iters_errored": stats.errors,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalib),
        "samples_ok": len(ok_samples),
        "samples_unsupported": len(unsupp_samples),
        "max_abs_err": stats.max_abs_err_overall,
        "max_rel_err": stats.max_rel_err_overall,
        "elapsed_s": elapsed,
        "end_reason": end_reason,
        "wall_budget_s": WALL_BUDGET_S,
        "top_3_repros": [
            {
                "class": s.bucket_class,
                "shape": list(s.shape),
                "dtype": s.dtype_name,
                "stride": s.stride,
                "bucket": s.bucket,
                "violating_seeds": s.violating_seed_count,
                "total_seeds": len(s.iters),
                "max_abs_err": s.max_abs_err,
                "max_rel_err": s.max_rel_err,
                "atol": s.atol,
                "rtol": s.rtol,
                "per_seed": [
                    {
                        "seed": r.seed,
                        "max_abs_err": r.max_abs_err,
                        "max_rel_err": r.max_rel_err,
                        "denom_at_worst": r.denom_at_worst_rel,
                        "abs_ratio": r.abs_ratio,
                        "rel_ratio": r.rel_ratio,
                        "violating": r.violating,
                    }
                    for r in s.iters
                ],
            }
            for s in top3
        ],
        "filing_target": (
            "pytorch/pytorch" if filable
            else ("xfail/recalibration" if recalib else "none")
        ),
        "results_md": str(OUT_MD),
    }
    with OUT_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> int:
    if not mps_ready():
        OUT_MD.write_text(
            f"# {KERNEL} fuzz — SKIPPED (MPS unavailable)\n\n"
            "torch.backends.mps.is_available()/is_built() returned False.\n"
        )
        with OUT_JSONL.open("a") as f:
            f.write(json.dumps({
                "agent": "kernel-fuzzer-softmax-v2",
                "kernel": KERNEL,
                "status": "SKIPPED",
                "reason": "torch.mps unavailable",
            }) + "\n")
        return 0

    meta_rng = random.Random(0xC0FFEE_50FD)
    stats = Stats()
    t_start = time.monotonic()
    end_reason = "completed"

    # Pre-build sample list deterministically.
    samples: list[SampleRecord] = []
    for i in range(N_SAMPLES):
        shape, bucket = sample_shape(meta_rng)
        dtype_name, dtype = meta_rng.choice(DTYPES)
        stride_cat = meta_rng.choice(STRIDE_CATEGORIES)
        k_dim = shape[-1] if shape else 0
        atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
        samples.append(SampleRecord(
            sample_idx=i,
            shape=shape,
            dtype_name=dtype_name,
            stride=stride_cat,
            bucket=bucket,
            atol=atol,
            rtol=rtol,
        ))
    stats.samples = samples

    samples_attempted = 0
    for s_idx, sample in enumerate(samples):
        samples_attempted += 1
        # Run all 5 seeds for this sample.
        any_unsupported = False
        for seed in SEEDS:
            if time.monotonic() - t_start > WALL_BUDGET_S:
                end_reason = "timeout"
                break
            stats.attempted += 1
            outcome = run_one_seed(sample, seed, stats)
            if outcome == "unsupported":
                any_unsupported = True
                # bail out on remaining seeds for this sample — torch.mps is
                # consistent per dtype/op; no point retrying.
                break
        if end_reason == "timeout":
            break
        if any_unsupported and not sample.iters:
            sample.bucket_class = "UNSUPPORTED"
        else:
            sample.bucket_class = classify_sample(sample)

        if s_idx and s_idx % 25 == 0:
            elapsed = time.monotonic() - t_start
            n_fil = sum(1 for s in samples if s.bucket_class == "FILABLE")
            n_rec = sum(1 for s in samples if s.bucket_class == "RECALIBRATION")
            print(
                f"[softmax] sample={s_idx}/{N_SAMPLES} attempted={stats.attempted} "
                f"completed={stats.completed} fil={n_fil} rec={n_rec} "
                f"unsup={stats.unsupported} elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Finalize any pending samples with no iters (timeout case).
    for sample in samples:
        if sample.bucket_class == "PENDING":
            if sample.iters:
                sample.bucket_class = classify_sample(sample)
            else:
                # Never ran — treat as not-attempted; drop from counts by leaving
                # PENDING -> we'll filter out below. Actually to keep counts clean,
                # mark UNSUPPORTED only if we know the op failed. Leave PENDING
                # so it's excluded from FILABLE/RECALIBRATION/OK.
                pass

    # Strip PENDINGs from stats.samples for output classification clarity.
    stats.samples = [s for s in samples if s.bucket_class != "PENDING"]

    elapsed = time.monotonic() - t_start
    write_outputs(stats, elapsed, end_reason, samples_attempted)
    print(
        f"[softmax] DONE attempted={stats.attempted} completed={stats.completed} "
        f"fil={sum(1 for s in stats.samples if s.bucket_class == 'FILABLE')} "
        f"rec={sum(1 for s in stats.samples if s.bucket_class == 'RECALIBRATION')} "
        f"elapsed={elapsed:.1f}s end={end_reason}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
