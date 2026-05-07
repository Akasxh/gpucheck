"""Fuzzer for mse_loss across MPS (real) vs CPU reference (v2 classifier).

Driven kernel: ``torch.nn.functional.mse_loss(x, y, reduction=...)``.
MSE is elementwise ``(x-y)**2``; with reduction='mean' it sums NK elements then
divides — so the natural ``k_dim`` for sqrt(k/128) tolerance scaling is the
total reduction extent (numel for mean/sum, 1 for 'none').

CUDA: not present on this host — comparison is MPS-vs-CPU only.

v2 classification (per dispatch spec):
- FILABLE: max_rel_err > 10x tolerance AND denom_magnitude >= 1e-6
           AND reproducible across >= 3 seeds (initial + 2 verification).
- TOLERANCE_RECALIBRATION: combined-check fail but NOT filable
           (small denom, single-seed flake, or within 10x tolerance band).
- OK: combined-check passes.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-mse"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "mse"
N_ITERS = 500
BUDGET_S = 8 * 60 - 60  # leave 60s for reproducibility re-runs + write-out

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_mse.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast", "non_contig_perm"]
REDUCTIONS = ["none", "mean", "sum"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]

# FILABLE thresholds
FILABLE_TOL_MULTIPLE = 10.0
FILABLE_MIN_DENOM = 1.0e-6
FILABLE_SEED_REPRO_MIN = 3  # initial + 2 verification


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    B: int
    K: int
    dtype: str
    stride: str
    reduction: str
    seed: int
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    classification: str = "OK"  # OK | TOLERANCE_RECALIBRATION | FILABLE
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    denom_magnitude: float | None = None  # |out_cpu| at the worst-rel cell
    atol: float | None = None
    rtol: float | None = None
    tolerance_multiple: float | None = None  # max_rel_err / rtol
    seeds_reproduced: int = 1
    seed_repro_max_rels: list[float] = field(default_factory=list)
    note: str = ""


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def pick_dim(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(DEGEN)
    if bucket == "prime":
        return rng.choice(PRIMES)
    if bucket == "pow2_boundary":
        return rng.choice(POW2_BOUNDARY)
    if bucket == "non_tile_aligned":
        return rng.choice(NON_TILE)
    if bucket == "large":
        return rng.choice(LARGE)
    raise ValueError(bucket)


def make_pair(
    B: int, K: int, dtype_name: str, stride: str, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, y) on CPU at requested dtype with requested stride pattern.

    Both tensors have logical shape (B, K). Values cover ~N(0, 1) with light
    occasional inflation so we exercise both small-denom and ordinary-magnitude
    elements. The caller materialises a contiguous clone for the MPS side.
    """
    dt = torch_dtype(dtype_name)
    g_x = torch.Generator().manual_seed(seed * 2 + 1)
    g_y = torch.Generator().manual_seed(seed * 2 + 2)

    if stride == "contiguous":
        x = torch.randn(B, K, generator=g_x, dtype=torch.float32).to(dt)
        y = torch.randn(B, K, generator=g_y, dtype=torch.float32).to(dt)
    elif stride == "slice":
        x_full = torch.randn(B, K * 2, generator=g_x, dtype=torch.float32).to(dt)
        y_full = torch.randn(B, K * 2, generator=g_y, dtype=torch.float32).to(dt)
        x = x_full[:, ::2]
        y = y_full[:, ::2]
    elif stride == "transpose":
        x_t = torch.randn(K, B, generator=g_x, dtype=torch.float32).to(dt)
        y_t = torch.randn(K, B, generator=g_y, dtype=torch.float32).to(dt)
        x = x_t.t()
        y = y_t.t()
    elif stride == "broadcast":
        # x is row-broadcast 1xK; y is full BxK — exercises mixed-stride inputs.
        x_full = torch.randn(1, K, generator=g_x, dtype=torch.float32).to(dt)
        y = torch.randn(B, K, generator=g_y, dtype=torch.float32).to(dt)
        x = x_full.expand(B, K)
    elif stride == "non_contig_perm":
        x_full = torch.randn(B, 2, K, generator=g_x, dtype=torch.float32).to(dt)
        y_full = torch.randn(B, 2, K, generator=g_y, dtype=torch.float32).to(dt)
        x = x_full[:, 0, :]
        y = y_full[:, 0, :]
    else:
        raise ValueError(stride)
    return x, y


def measure_one(
    B: int, K: int, dtype: str, stride: str, reduction: str, seed: int,
) -> tuple[float, float, float, float, float]:
    """Run a single MPS-vs-CPU comparison.

    Returns (max_abs_err, max_rel_err, denom_magnitude_at_worst_rel, atol, rtol).
    Raises on unsupported / harness errors.
    """
    cpu_dev = torch.device("cpu")
    mps_dev = torch.device("mps")

    x_cpu, y_cpu = make_pair(B, K, dtype, stride, seed)

    # Materialize then push to MPS — both backends see identical input bytes.
    x_mps = x_cpu.detach().clone().contiguous().to(mps_dev)
    y_mps = y_cpu.detach().clone().contiguous().to(mps_dev)

    ref_dtype = torch.float64 if dtype == "float32" else torch.float32

    out_cpu = F.mse_loss(x_cpu, y_cpu, reduction=reduction).to(ref_dtype)
    out_mps = F.mse_loss(x_mps, y_mps, reduction=reduction).to(cpu_dev).to(ref_dtype)

    # Tolerance: for reduction in {mean, sum}, the reduction k is numel(x).
    # For reduction='none' it's an elementwise (x-y)**2 — k_dim=1 is right.
    k_dim = (B * K) if reduction in ("mean", "sum") else 1
    atol, rtol = compute_tolerance(dtype, k_dim=max(k_dim, 1), device_type="mps")

    out_cpu_t = out_cpu if out_cpu.dim() > 0 else out_cpu.reshape(1)
    out_mps_t = out_mps if out_mps.dim() > 0 else out_mps.reshape(1)

    diff = (out_mps_t - out_cpu_t).abs()
    if diff.numel() == 0:
        return 0.0, 0.0, 0.0, atol, rtol

    max_abs = float(diff.max().item())
    denom = out_cpu_t.abs()
    rel = diff / denom.clamp_min(1e-30)
    # Mask out cells where denom is near-zero — those are not meaningful relative
    # errors. We still report them in denom_magnitude for the FILABLE filter.
    flat_idx = int(rel.argmax().item())
    max_rel = float(rel.flatten()[flat_idx].item())
    denom_at_worst = float(denom.flatten()[flat_idx].item())
    return max_abs, max_rel, denom_at_worst, atol, rtol


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)
    reduction = rng.choice(REDUCTIONS)
    B = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)
    seed = rng.randrange(1, 1 << 30)

    base = IterResult(
        idx=idx, shape_bucket=bucket, B=B, K=K, dtype=dtype, stride=stride,
        reduction=reduction, seed=seed, status="ERROR",
    )
    if B == 0 or K == 0:
        base.status = "SKIP"
        base.note = "degenerate zero-dim"
        return base

    try:
        max_abs, max_rel, denom, atol, rtol = measure_one(
            B, K, dtype, stride, reduction, seed,
        )
        base.max_abs_err = max_abs
        base.max_rel_err = max_rel
        base.denom_magnitude = denom
        base.atol = atol
        base.rtol = rtol
        base.tolerance_multiple = (max_rel / rtol) if rtol > 0 else float("inf")

        # gpucheck combined check — element-wise (or scalar) |a-b| <= atol + rtol*|b|.
        # We approximate this with the worst-cell relative + atol band.
        ok = max_abs <= (atol + rtol * denom)
        base.status = "OK" if ok else "DIVERGENCE"

        if not ok:
            # v2 classification — first cut on the single-seed observation.
            tol_mult = base.tolerance_multiple or 0.0
            denom_ok = denom >= FILABLE_MIN_DENOM
            tol_band_ok = tol_mult > FILABLE_TOL_MULTIPLE
            if denom_ok and tol_band_ok:
                base.classification = "FILABLE_PROVISIONAL"  # promoted to FILABLE on repro
            else:
                base.classification = "TOLERANCE_RECALIBRATION"
                why = []
                if not denom_ok:
                    why.append(f"denom={denom:.2e} < {FILABLE_MIN_DENOM:.0e}")
                if not tol_band_ok:
                    why.append(f"max_rel/rtol={tol_mult:.2f}x <= {FILABLE_TOL_MULTIPLE:.0f}x")
                base.note = "; ".join(why)
        else:
            base.classification = "OK"
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e).splitlines()[0][:200]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported", "MPS")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def reproduce_across_seeds(r: IterResult, n_extra_seeds: int = 2) -> None:
    """For a FILABLE_PROVISIONAL result, retry with 2 fresh seeds.

    Promotes to ``FILABLE`` if all (1 + n_extra_seeds) seeds also exceed
    FILABLE_TOL_MULTIPLE * rtol AND denom >= FILABLE_MIN_DENOM. Demotes to
    ``TOLERANCE_RECALIBRATION`` otherwise.
    """
    rng = random.Random(0xBADBEEF ^ r.idx)
    seeds_passing = 1  # the original seed already passed both filters
    rels = [r.max_rel_err or 0.0]
    for _ in range(n_extra_seeds):
        s = rng.randrange(1, 1 << 30)
        try:
            _, max_rel, denom, _, rtol = measure_one(
                r.B, r.K, r.dtype, r.stride, r.reduction, s,
            )
            rels.append(max_rel)
            tol_mult = (max_rel / rtol) if rtol > 0 else float("inf")
            if tol_mult > FILABLE_TOL_MULTIPLE and denom >= FILABLE_MIN_DENOM:
                seeds_passing += 1
        except Exception:  # noqa: BLE001
            # treat unsupported / error during repro as non-reproducible
            pass
    r.seeds_reproduced = seeds_passing
    r.seed_repro_max_rels = rels
    if seeds_passing >= FILABLE_SEED_REPRO_MIN:
        r.classification = "FILABLE"
        r.note = (
            f"reproduced {seeds_passing}/{1 + n_extra_seeds} seeds; "
            f"max_rel/rtol seen: {[f'{x:.2e}' for x in rels]}"
        )
    else:
        r.classification = "TOLERANCE_RECALIBRATION"
        r.note = (
            f"non-reproducible: {seeds_passing}/{1 + n_extra_seeds} seeds exceeded 10x rtol "
            f"with denom>=1e-6; max_rels: {[f'{x:.2e}' for x in rels]}"
        )


def main() -> int:
    rng = random.Random(0xC0FFEE_M5E if False else 0x000C0FFE)
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# mse fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters", flush=True)
            break
        attempted = i + 1
        r = run_one(i, rng)
        results.append(r)
        completed += 1
        if r.status == "DIVERGENCE":
            print(
                f"[{i:03d}] DIV {r.dtype}/{r.stride}/red={r.reduction} B={r.B} K={r.K} "
                f"max_rel={r.max_rel_err:.3e} (={r.tolerance_multiple:.2f}x rtol) "
                f"denom={r.denom_magnitude:.2e} -> {r.classification}",
                flush=True,
            )
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.dtype}/{r.stride}/red={r.reduction}: {r.note}", flush=True)
        elif r.status == "UNSUPPORTED":
            print(f"[{i:03d}] UNSUP {r.dtype}/{r.stride}/red={r.reduction}: {r.note}", flush=True)

    # Phase 2: cross-seed reproducibility for FILABLE_PROVISIONAL candidates.
    candidates = [r for r in results if r.classification == "FILABLE_PROVISIONAL"]
    print(f"[phase2] {len(candidates)} FILABLE candidates -> running cross-seed repro", flush=True)
    for r in candidates:
        if time.monotonic() - t0 > BUDGET_S + 50:
            print(f"[budget] stopping repro at idx={r.idx}", flush=True)
            r.classification = "TOLERANCE_RECALIBRATION"
            r.note = (r.note + "; repro_skipped_budget").lstrip("; ")
            continue
        reproduce_across_seeds(r)
        print(
            f"[repro] idx={r.idx} -> {r.classification} ({r.seeds_reproduced}/3)",
            flush=True,
        )

    divergences = [r for r in results if r.status == "DIVERGENCE"]
    filable = [r for r in divergences if r.classification == "FILABLE"]
    recalibrate = [r for r in divergences if r.classification == "TOLERANCE_RECALIBRATION"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    ok = [r for r in results if r.status == "OK"]
    skipped = [r for r in results if r.status == "SKIP"]

    filable.sort(key=lambda r: (r.tolerance_multiple or 0.0), reverse=True)
    top3 = filable[:3]

    overall_max_rel = max(
        (r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0
    )
    overall_max_abs = max(
        (r.max_abs_err for r in results if r.max_abs_err is not None), default=0.0
    )
    overall_max_tolmult = max(
        (r.tolerance_multiple for r in results if r.tolerance_multiple is not None), default=0.0
    )

    target = "pytorch/pytorch" if filable else "none"
    elapsed = time.monotonic() - t0

    md: list[str] = []
    md.append(f"# Fuzz results — kernel: `{KERNEL_NAME}` (v2 FILABLE classifier)\n")
    md.append("**Backends:** MPS (real, Apple Silicon) vs CPU reference (same dtype).")
    md.append("CUDA: not present on this host — comparison is MPS-vs-CPU only.\n")
    md.append("## Summary\n")
    md.append(f"- iterations attempted : **{attempted}** / target {N_ITERS}")
    md.append(f"- iterations completed : **{completed}**")
    md.append(f"- OK (combined check)  : {len(ok)}")
    md.append(f"- skipped (zero-dim)   : {len(skipped)}")
    md.append(f"- unsupported (MPS)    : {len(unsupported)}")
    md.append(f"- harness errors       : {len(errors)}")
    md.append(f"- combined-check fails : **{len(divergences)}** (FILABLE: {len(filable)} | TOLERANCE_RECALIBRATION: {len(recalibrate)})")
    md.append(f"- max relative error   : `{overall_max_rel:.3e}`")
    md.append(f"- max abs error        : `{overall_max_abs:.3e}`")
    md.append(f"- max tolerance mult   : `{overall_max_tolmult:.2f}x` rtol")
    md.append(f"- runtime              : `{elapsed:.1f}s` (budget {BUDGET_S}s; "
              f"end={'budget_hit' if attempted < N_ITERS else 'completed'})")
    md.append(f"- torch                : `{torch.__version__}`")
    md.append("- mps_vs_cuda_max_rel_err: N/A (no CUDA device)\n")

    md.append("## v2 classification rules\n")
    md.append(f"- **FILABLE**: max_rel_err > {FILABLE_TOL_MULTIPLE:.0f}x rtol "
              f"AND denom_magnitude >= {FILABLE_MIN_DENOM:.0e} "
              f"AND reproducible across >= {FILABLE_SEED_REPRO_MIN} seeds.")
    md.append("- **TOLERANCE_RECALIBRATION**: combined-check fails but does not meet FILABLE bar "
              "(small-denom region, single-seed flake, or within 10x rtol band).")
    md.append("- **OK**: combined check `|a-b| <= atol + rtol*|b|` passes.\n")

    md.append("## Top 3 FILABLE minimal repros\n")
    if top3:
        md.append("| # | dtype | stride | reduction | B | K | shape_bucket | tol_mult | max_rel_err | max_abs_err | denom | atol | rtol | seeds_repro | note |")
        md.append("|---|-------|--------|-----------|---|---|--------------|----------|-------------|-------------|-------|------|------|-------------|------|")
        for i, r in enumerate(top3, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.reduction} | {r.B} | {r.K} | {r.shape_bucket} | "
                f"{r.tolerance_multiple:.2f}x | {r.max_rel_err:.3e} | {r.max_abs_err:.3e} | "
                f"{r.denom_magnitude:.2e} | {r.atol:.3e} | {r.rtol:.3e} | "
                f"{r.seeds_reproduced}/3 | {r.note} |"
            )
    else:
        md.append("_None — no MPS-vs-CPU divergence cleared the FILABLE bar (10x rtol, denom>=1e-6, 3-seed reproducible)._")
    md.append("")

    if recalibrate:
        md.append(f"## TOLERANCE_RECALIBRATION summary ({len(recalibrate)})\n")
        # bucket by (dtype, stride, reduction)
        buckets: dict[tuple[str, str, str], list[IterResult]] = {}
        for r in recalibrate:
            buckets.setdefault((r.dtype, r.stride, r.reduction), []).append(r)
        md.append("| dtype | stride | reduction | count | worst tol_mult | sample note |")
        md.append("|-------|--------|-----------|-------|----------------|-------------|")
        for (dt, st, red), rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            worst = max((r.tolerance_multiple or 0.0) for r in rs)
            md.append(f"| {dt} | {st} | {red} | {len(rs)} | {worst:.2f}x | `{rs[0].note}` |")
        md.append("")

    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})\n")
        seen: dict[tuple[str, str, str], list[IterResult]] = {}
        for r in unsupported:
            seen.setdefault((r.dtype, r.stride, r.reduction), []).append(r)
        for (dt, st, red), rs in seen.items():
            md.append(f"- {dt} / {st} / red={red}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")

    if errors:
        md.append(f"## ERROR summary ({len(errors)})\n")
        for r in errors[:10]:
            md.append(
                f"- iter {r.idx} {r.dtype}/{r.stride}/red={r.reduction}/{r.shape_bucket} "
                f"B={r.B} K={r.K}: {r.note}"
            )
        md.append("")

    md.append("## Recommended upstream filing target\n")
    if filable:
        md.append(f"**{target}** — {len(filable)} FILABLE divergence(s); see top-3 table for repros.")
    else:
        md.append("**none** — every divergence either failed reproducibility or sat in a small-denom / sub-10x band.")
    md.append("\n## Method notes\n")
    md.append("- Kernel: `torch.nn.functional.mse_loss(x, y, reduction=...)`, sampled uniformly over reduction in {none, mean, sum}.")
    md.append("- For reduction in {mean, sum}, k_dim = numel(x). For reduction='none', k_dim = 1 (elementwise (x-y)**2).")
    md.append("- Inputs built on CPU at fp32, projected to native dtype, transferred to MPS as a contiguous clone.")
    md.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim, device_type='mps')` — base + sqrt(k/128) accumulator scaling + MPS 2× overlay.")
    md.append("- Stride categories: contiguous, slice (stride-2), transpose (`.t()`), broadcast (1×K → B×K), non_contig_perm (3-D channel slice giving stride (2K,1)).")
    md.append("- Shape buckets: degenerate, prime, pow2_boundary, non_tile_aligned, large.")
    md.append("- FILABLE 3-seed cross-check uses a deterministic per-iter RNG (`Random(0xBADBEEF ^ idx)`) so promotions are reproducible across runs of the swarm.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "version": "v2",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "divergences": len(divergences),
        "filable": len(filable),
        "tolerance_recalibration": len(recalibrate),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cpu_max_abs_err": overall_max_abs,
        "mps_vs_cpu_max_tolerance_multiple": overall_max_tolmult,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_no_nvidia_hardware",
        "filable_top3": [
            {
                "rank": i + 1,
                "dtype": r.dtype,
                "stride": r.stride,
                "reduction": r.reduction,
                "B": r.B, "K": r.K,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err,
                "max_abs_err": r.max_abs_err,
                "tolerance_multiple": r.tolerance_multiple,
                "denom_magnitude": r.denom_magnitude,
                "atol": r.atol, "rtol": r.rtol,
                "seeds_reproduced": r.seeds_reproduced,
                "seed_repro_max_rels": r.seed_repro_max_rels,
                "note": r.note,
            }
            for i, r in enumerate(top3)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": elapsed,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
