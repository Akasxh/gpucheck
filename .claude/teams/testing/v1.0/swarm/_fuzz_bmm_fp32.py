"""Fuzzer for bmm-fp32: torch.bmm on MPS vs CPU, 500 iters with FILABLE filter.

Filter:
- FILABLE: max_rel_err > 10x rtol(dtype) AND denom_magnitude (|Y_cpu|.max) >= 1e-6
  AND reproducible across >=3 seeds (same shape/dtype/stride, different RNG seeds,
  >=3 of K reseeds also exceed 10x rtol).
- TOLERANCE_RECALIBRATION: max_rel_err > 10x rtol but denom too small or not
  reproducible (sporadic / sample-dependent).
- OK: within 10x rtol baseline.

Tolerance baseline used here is the CUDA-calibrated rtol from gpucheck's
_DEFAULT_TOLERANCES (no MPS overlay), so "10x rtol" reads against the
*calibrated* baseline rather than the already-2x-padded MPS overlay.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-bmm-fp32"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "bmm-fp32"
N_ITERS = 500
BUDGET_S = 8 * 60 - 45  # leave 45s for reseed reruns + writeout
RESEED_K = 4  # number of additional seeds for reproducibility check
RESEED_REQUIRED_HITS = 3  # of RESEED_K seeds, this many must also exceed 10x rtol

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_bmm-fp32.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice_k", "transpose_mk", "broadcast_batch", "noncontig_batch"]
BATCH_SIZES = [1, 2, 3, 4, 8, 16]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [256, 384, 512, 640, 768]
DEGEN = [1, 2]


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    B: int
    M: int
    K: int
    N: int
    dtype: str
    stride: str
    seed: int
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    classification: str = ""  # FILABLE | TOLERANCE_RECALIBRATION | OK | <empty>
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    denom_max: float | None = None  # |Y_cpu|.max — magnitude of reference
    atol: float | None = None
    rtol_baseline: float | None = None  # CUDA-calibrated rtol (no MPS overlay)
    rtol_mps: float | None = None  # MPS-overlay rtol used by gpucheck.assert_close
    reseed_hits: int | None = None  # of RESEED_K reseeds, how many also > 10x rtol
    reseed_max_rels: list[float] = field(default_factory=list)
    note: str = ""


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


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_inputs(B: int, M: int, K: int, N: int, dtype_name: str, stride: str,
                seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (B,M,K) and (B,K,N) tensors on `device` for the requested stride pattern."""
    dt = torch_dtype(dtype_name)
    g = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)

    if stride == "contiguous":
        A = torch.randn(B, M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm = torch.randn(B, K, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
    elif stride == "slice_k":
        A_full = torch.randn(B, M, K * 2, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_full = torch.randn(B, K * 2, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[:, :, ::2]
        Bm = B_full[:, ::2, :]
    elif stride == "transpose_mk":
        A_t = torch.randn(B, K, M, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_t = torch.randn(B, N, K, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.transpose(1, 2)  # -> (B, M, K)
        Bm = B_t.transpose(1, 2)  # -> (B, K, N)
    elif stride == "broadcast_batch":
        # broadcast along batch dim — A has 1xMxK expanded to BxMxK
        A_one = torch.randn(1, M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_one = torch.randn(1, K, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_one.expand(B, M, K)
        Bm = B_one.expand(B, K, N)
    elif stride == "noncontig_batch":
        # Build (2B, M, K) and stride-2 along batch dim → non-contiguous batch
        A_full = torch.randn(2 * B, M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_full = torch.randn(2 * B, K, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[::2]
        Bm = B_full[::2]
    else:
        raise ValueError(stride)
    return A, Bm


def measure_one(B: int, M: int, K: int, N: int, dtype_name: str, stride: str,
                seed: int) -> tuple[float, float, float, float, float, float, float]:
    """Run one bmm on CPU+MPS at the given config; return error metrics.

    Returns: (max_abs, max_rel, denom_max, atol_mps, rtol_mps, atol_baseline, rtol_baseline)
    """
    cpu = torch.device("cpu")
    mps = torch.device("mps")
    A_cpu, B_cpu = make_inputs(B, M, K, N, dtype_name, stride, seed, cpu)
    A_mps = A_cpu.detach().clone().to(mps)
    B_mps = B_cpu.detach().clone().to(mps)

    ref_dtype = torch.float64 if dtype_name == "float32" else torch.float32
    Y_cpu = torch.bmm(A_cpu, B_cpu).to(ref_dtype)
    Y_mps = torch.bmm(A_mps, B_mps).to(cpu).to(ref_dtype)

    diff = (Y_mps - Y_cpu).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = Y_cpu.abs().clamp_min(1e-30)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    denom_max = float(Y_cpu.abs().max().item()) if Y_cpu.numel() else 0.0

    atol_mps, rtol_mps = compute_tolerance(dtype_name, k_dim=K, device_type="mps")
    atol_base, rtol_base = compute_tolerance(dtype_name, k_dim=K, device_type="cpu")
    return max_abs, max_rel, denom_max, atol_mps, rtol_mps, atol_base, rtol_base


def reseed_check(B: int, M: int, K: int, N: int, dtype_name: str, stride: str,
                 base_seed: int, threshold_rel: float) -> tuple[int, list[float]]:
    """Re-run the same config with RESEED_K different seeds.

    Returns (#hits, list of max_rel_err per reseed). A 'hit' is a reseed whose
    max_rel_err also exceeds `threshold_rel`.
    """
    hits = 0
    rels: list[float] = []
    for k in range(RESEED_K):
        s = (base_seed + (k + 1) * 0xCAFE) & 0xFFFFFFFF
        try:
            _, mr, _, _, _, _, _ = measure_one(B, M, K, N, dtype_name, stride, s)
        except Exception:  # noqa: BLE001
            rels.append(float("nan"))
            continue
        rels.append(mr)
        if mr > threshold_rel:
            hits += 1
    return hits, rels


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)
    Bd = rng.choice(BATCH_SIZES)
    M = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)
    N = pick_dim(bucket, rng)
    seed = rng.randrange(1 << 30)

    base = IterResult(
        idx=idx, shape_bucket=bucket, B=Bd, M=M, K=K, N=N, dtype=dtype, stride=stride,
        seed=seed, status="ERROR",
    )

    if M == 0 or K == 0 or N == 0:
        base.status = "SKIP"
        base.classification = "OK"
        base.note = "degenerate zero-dim"
        return base

    try:
        max_abs, max_rel, denom_max, atol_mps, rtol_mps, atol_base, rtol_base = \
            measure_one(Bd, M, K, N, dtype, stride, seed)
        base.max_abs_err = max_abs
        base.max_rel_err = max_rel
        base.denom_max = denom_max
        base.atol = atol_mps
        base.rtol_mps = rtol_mps
        base.rtol_baseline = rtol_base

        # gpucheck-style pass: |a-b| <= atol + rtol*|b|, with MPS overlay.
        # We classify against the CUDA baseline rtol (10x of the calibrated rtol).
        threshold_rel = 10.0 * rtol_base
        within_mps_overlay = max_abs <= atol_mps + rtol_mps * (denom_max if denom_max > 0 else 1.0)

        if max_rel <= threshold_rel:
            base.status = "OK" if within_mps_overlay else "DIVERGENCE"
            base.classification = "OK"
            return base

        # Above 10x baseline rtol — need to gate on denom and reproducibility.
        if denom_max < 1e-6:
            base.status = "DIVERGENCE"
            base.classification = "TOLERANCE_RECALIBRATION"
            base.note = (
                f"max_rel={max_rel:.3e} > 10x rtol={threshold_rel:.3e} but "
                f"denom_max={denom_max:.3e} < 1e-6 (relative error inflated by "
                f"near-zero reference)"
            )
            return base

        # Reproducibility check: re-run with RESEED_K different seeds.
        hits, rels = reseed_check(Bd, M, K, N, dtype, stride, seed, threshold_rel)
        base.reseed_hits = hits
        base.reseed_max_rels = rels
        base.status = "DIVERGENCE"
        if hits >= RESEED_REQUIRED_HITS:
            base.classification = "FILABLE"
            base.note = (
                f"max_rel={max_rel:.3e} > 10x rtol={threshold_rel:.3e}; "
                f"denom={denom_max:.3e}; reseed hits={hits}/{RESEED_K} "
                f"(rels={['%.2e'%r for r in rels]})"
            )
        else:
            base.classification = "TOLERANCE_RECALIBRATION"
            base.note = (
                f"max_rel={max_rel:.3e} > 10x rtol={threshold_rel:.3e}; "
                f"denom={denom_max:.3e}; only {hits}/{RESEED_K} reseeds reproduced "
                f"the breach (rels={['%.2e'%r for r in rels]}) — sample-dependent"
            )
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e).splitlines()[0][:200]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def main() -> int:
    rng = random.Random(0xBEEFCAFE)
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# bmm-fp32 fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
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
        if r.classification == "FILABLE":
            print(f"[{i:03d}] FILABLE {r.dtype} {r.stride} B={r.B} M={r.M} K={r.K} N={r.N} "
                  f"max_rel={r.max_rel_err:.3e} hits={r.reseed_hits}/{RESEED_K}", flush=True)
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.note}", flush=True)
        elif r.status == "UNSUPPORTED":
            print(f"[{i:03d}] UNSUPPORTED {r.dtype} {r.stride} {r.note}", flush=True)

    # Aggregate
    by_class: dict[str, list[IterResult]] = {}
    for r in results:
        by_class.setdefault(r.classification or r.status, []).append(r)

    filable = [r for r in results if r.classification == "FILABLE"]
    tol_recal = [r for r in results if r.classification == "TOLERANCE_RECALIBRATION"]
    ok = [r for r in results if r.classification == "OK"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    skipped = [r for r in results if r.status == "SKIP"]

    overall_max_rel = max((r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0)

    filable.sort(key=lambda r: (r.max_rel_err or 0.0), reverse=True)
    tol_recal.sort(key=lambda r: (r.max_rel_err or 0.0), reverse=True)
    top3_filable = filable[:3]
    top3_tol = tol_recal[:3]

    if filable:
        target = "pytorch/pytorch"
    else:
        target = "none"

    md: list[str] = []
    md.append(f"# bmm-fp32 fuzz results\n")
    md.append(f"- **kernel:** `{KERNEL_NAME}` (`torch.bmm` on MPS vs CPU)")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **FILABLE:** {len(filable)}")
    md.append(f"- **TOLERANCE_RECALIBRATION:** {len(tol_recal)}")
    md.append(f"- **OK:** {len(ok)}")
    md.append(f"- **UNSUPPORTED:** {len(unsupported)}")
    md.append(f"- **ERROR:** {len(errors)}")
    md.append(f"- **SKIP (zero-dim):** {len(skipped)}")
    md.append(f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}")
    md.append(f"- **MPS-vs-CUDA-mock max relative error:** N/A (no NVIDIA hardware on host)")
    md.append(f"- **recommended upstream filing target:** {target}")
    md.append(f"- **wall clock:** {time.monotonic() - t0:.1f}s\n")

    md.append("## FILABLE filter")
    md.append("- `max_rel_err > 10 * rtol_baseline(dtype)` (CUDA-calibrated rtol, no MPS overlay)")
    md.append("- AND `denom_magnitude (|Y_cpu|.max) >= 1e-6` (rules out near-zero reference inflation)")
    md.append(f"- AND reseed check: of {RESEED_K} additional seeds at the same shape/dtype/stride, "
              f">= {RESEED_REQUIRED_HITS} also breach 10x rtol")
    md.append("- Failures of the latter two checks are classified `TOLERANCE_RECALIBRATION` instead.\n")

    md.append("## Top 3 FILABLE repros\n")
    if top3_filable:
        md.append("| # | dtype | stride | B | M | K | N | shape_bucket | seed | max_rel_err | denom_max | rtol_baseline | reseed_hits | note |")
        md.append("|---|-------|--------|---|---|---|---|--------------|------|-------------|-----------|---------------|-------------|------|")
        for i, r in enumerate(top3_filable, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.B} | {r.M} | {r.K} | {r.N} | "
                f"{r.shape_bucket} | {r.seed} | {r.max_rel_err:.3e} | {r.denom_max:.3e} | "
                f"{r.rtol_baseline:.3e} | {r.reseed_hits}/{RESEED_K} | {r.note[:120]} |"
            )
    else:
        md.append("_None — no config breached 10× CUDA-calibrated rtol with reproducible seeds and non-trivial denominator._")
    md.append("")

    md.append("## Top 3 TOLERANCE_RECALIBRATION cases\n")
    if top3_tol:
        md.append("| # | dtype | stride | B | M | K | N | max_rel_err | denom_max | rtol_baseline | reseed_hits | reason |")
        md.append("|---|-------|--------|---|---|---|---|-------------|-----------|---------------|-------------|--------|")
        for i, r in enumerate(top3_tol, 1):
            reason = "denom<1e-6" if (r.denom_max is not None and r.denom_max < 1e-6) else (
                f"reseed_hits={r.reseed_hits}/{RESEED_K} < {RESEED_REQUIRED_HITS}"
            )
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.B} | {r.M} | {r.K} | {r.N} | "
                f"{r.max_rel_err:.3e} | {r.denom_max:.3e} | {r.rtol_baseline:.3e} | "
                f"{r.reseed_hits if r.reseed_hits is not None else '-'}/{RESEED_K} | {reason} |"
            )
    else:
        md.append("_None — no config exceeded 10× rtol baseline._")
    md.append("")

    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})")
        seen: dict[tuple[str, str], list[IterResult]] = {}
        for r in unsupported:
            key = (r.dtype, r.stride)
            seen.setdefault(key, []).append(r)
        for (dt, st), rs in seen.items():
            md.append(f"- {dt} / {st}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")

    if errors:
        md.append(f"## ERROR summary ({len(errors)})")
        for r in errors[:10]:
            md.append(f"- iter {r.idx} {r.dtype}/{r.stride} B={r.B} M={r.M} K={r.K} N={r.N}: {r.note}")
        md.append("")

    md.append("## Methodology")
    md.append("- 500 sampled configs across 5 shape buckets × 3 dtypes × 5 stride patterns × 6 batch sizes.")
    md.append("- Inputs seeded with fp32 RNG on CPU then cast/transferred so MPS and CPU see numerically identical bits.")
    md.append("- Reference Y is computed at fp64 (for fp32 ops) or fp32 (for fp16/bf16 ops) on CPU; the tested error channel is `Y_mps - Y_cpu` at the same low precision.")
    md.append("- Tolerance baseline: `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='cpu')` — CUDA-calibrated, sqrt(K/128)-scaled. The 10× threshold is judged against this baseline (NOT the already-2×-padded MPS overlay) so the FILABLE bar is a real 10× over the calibrated CUDA reference.")
    md.append("- Stride patterns:")
    md.append("  - `contiguous` — vanilla `B×M×K` and `B×K×N`")
    md.append("  - `slice_k` — over-allocate the K dim and slice with `[:, :, ::2]` / `[:, ::2, :]` (stride-2 view)")
    md.append("  - `transpose_mk` — `.transpose(1, 2)` view (non-contiguous M/K)")
    md.append("  - `broadcast_batch` — `1×M×K` and `1×K×N` expanded to `B×M×K` / `B×K×N` (stride-0 along batch dim)")
    md.append("  - `noncontig_batch` — over-allocate batch dim, slice `[::2]` (non-unit batch stride)")
    md.append("- CUDA channel: cannot run on this host (no NVIDIA hardware); flagged as N/A.")
    md.append(f"- torch={torch.__version__}; reseed-K={RESEED_K}; reseed-required={RESEED_REQUIRED_HITS}.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "iters_attempted": attempted,
        "iters_completed": completed,
        "filable": len(filable),
        "tolerance_recalibration": len(tol_recal),
        "ok": len(ok),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_no_nvidia_hardware",
        "filter": {
            "rel_threshold_multiplier": 10.0,
            "denom_min": 1e-6,
            "reseed_k": RESEED_K,
            "reseed_required_hits": RESEED_REQUIRED_HITS,
        },
        "top3_filable": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride,
                "B": r.B, "M": r.M, "K": r.K, "N": r.N,
                "shape_bucket": r.shape_bucket, "seed": r.seed,
                "max_rel_err": r.max_rel_err, "max_abs_err": r.max_abs_err,
                "denom_max": r.denom_max,
                "rtol_baseline": r.rtol_baseline, "rtol_mps": r.rtol_mps,
                "atol_mps": r.atol,
                "reseed_hits": r.reseed_hits,
                "reseed_max_rels": r.reseed_max_rels,
                "note": r.note,
            }
            for i, r in enumerate(top3_filable)
        ],
        "top3_tolerance_recalibration": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride,
                "B": r.B, "M": r.M, "K": r.K, "N": r.N,
                "max_rel_err": r.max_rel_err, "denom_max": r.denom_max,
                "rtol_baseline": r.rtol_baseline,
                "reseed_hits": r.reseed_hits,
                "note": r.note,
            }
            for i, r in enumerate(top3_tol)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": time.monotonic() - t0,
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
