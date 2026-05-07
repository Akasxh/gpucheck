"""Fuzzer v2 for matmul-bf16 on MPS vs CPU with FILABLE filter and seed reproducibility.

Generates 500 iterations of (shape_bucket, dtype, stride_pattern) drawn at random,
runs `A @ B` on MPS and CPU with bit-identical inputs (built on CPU then transferred),
and computes max relative error vs CPU reference.

Classification (per iteration that exceeds gpucheck MPS-overlay tolerance):

    FILABLE                — max_rel_err > 10 * rtol_used
                             AND denom_magnitude_at_argmax >= 1e-6
                             AND reproducible across >=3 seeds (initial + 2 retries)
    TOLERANCE_RECALIBRATION — exceeds tolerance but fails FILABLE gate
    OK                     — within tolerance

Halts on uncaught process error and writes a partial RESULTS file before exiting.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-matmul-bf16"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "matmul-bf16"
N_ITERS = 500
BUDGET_S = 8 * 60 - 45  # leave 45s for reproducibility-check + write-out
REPRO_SEEDS_PER_CANDIDATE = 2  # 2 additional seeds → 3 total
MAX_REPRO_CANDIDATES = 25      # cap repro work to fit budget

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_matmul-bf16.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large", "mixed"]
# matmul-bf16: bias to bf16 but include fp16/fp32 for cross-dtype context (1 in 5 each).
DTYPES_WEIGHTED = ["bfloat16"] * 6 + ["float16"] * 2 + ["float32"] * 2
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "non_contig", "broadcast_K", "broadcast_N"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    M: int
    K: int
    N: int
    dtype: str
    stride: str
    seed: int
    status: str  # OK | TOLERANCE_RECALIBRATION | FILABLE | UNSUPPORTED | ERROR | SKIP
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    atol: float | None = None
    rtol: float | None = None
    denom_at_argmax: float | None = None
    repro_seeds_tested: int = 1
    repro_seeds_passed: int = 1   # number of seeds where divergence reproduced
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
    if bucket == "mixed":
        return rng.choice(PRIMES + POW2_BOUNDARY + NON_TILE + LARGE)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_inputs_cpu(M: int, K: int, N: int, dtype_name: str, stride: str,
                    seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct A (M,K), B (K,N) on CPU with the requested stride pattern.

    bf16/fp16 are produced by casting from a seeded fp32 draw — keeps numerics
    consistent across MPS and CPU since both see the *same* low-precision tensor
    after the cast.
    """
    dt = torch_dtype(dtype_name)
    g_a = torch.Generator().manual_seed(seed)
    g_b = torch.Generator().manual_seed(seed ^ 0x5A5A5A5A)

    if stride == "contiguous":
        A = torch.randn(M, K, generator=g_a, dtype=torch.float32).to(dtype=dt)
        B = torch.randn(K, N, generator=g_b, dtype=torch.float32).to(dtype=dt)
    elif stride == "slice":
        A_full = torch.randn(M, K * 2, generator=g_a, dtype=torch.float32).to(dtype=dt)
        B_full = torch.randn(K * 2, N, generator=g_b, dtype=torch.float32).to(dtype=dt)
        A = A_full[:, ::2]
        B = B_full[::2, :]
    elif stride == "transpose":
        A_t = torch.randn(K, M, generator=g_a, dtype=torch.float32).to(dtype=dt)
        B_t = torch.randn(N, K, generator=g_b, dtype=torch.float32).to(dtype=dt)
        A = A_t.t()
        B = B_t.t()
    elif stride == "non_contig":
        # Inner-stride permutation: build (M, K+2) and slice [:, 1:K+1] — non-1 inner stride
        A_full = torch.randn(M, K + 2, generator=g_a, dtype=torch.float32).to(dtype=dt)
        B_full = torch.randn(K + 2, N, generator=g_b, dtype=torch.float32).to(dtype=dt)
        A = A_full[:, 1:K + 1]
        B = B_full[1:K + 1, :]
    elif stride == "broadcast_K":
        # Replicate a single row over M — degenerate-rank A
        A_row = torch.randn(1, K, generator=g_a, dtype=torch.float32).to(dtype=dt).expand(M, K).contiguous()
        B = torch.randn(K, N, generator=g_b, dtype=torch.float32).to(dtype=dt)
        A = A_row
    elif stride == "broadcast_N":
        # Replicate a single column over N
        A = torch.randn(M, K, generator=g_a, dtype=torch.float32).to(dtype=dt)
        B_col = torch.randn(K, 1, generator=g_b, dtype=torch.float32).to(dtype=dt).expand(K, N).contiguous()
        B = B_col
    else:
        raise ValueError(stride)
    assert A.shape == (M, K) and B.shape == (K, N), f"stride {stride} produced bad shapes"
    return A, B


def run_pair(M: int, K: int, N: int, dtype: str, stride: str, seed: int
             ) -> tuple[float, float, float, float, float]:
    """Run one MPS-vs-CPU comparison. Returns (max_abs, max_rel, atol, rtol, denom_at_argmax)."""
    cpu_dev = torch.device("cpu")
    mps_dev = torch.device("mps")

    A_cpu, B_cpu = make_inputs_cpu(M, K, N, dtype, stride, seed)
    A_mps = A_cpu.detach().clone().to(mps_dev)
    B_mps = B_cpu.detach().clone().to(mps_dev)

    Y_cpu = (A_cpu @ B_cpu)
    Y_mps = (A_mps @ B_mps).to(cpu_dev)

    # Promote both to fp32 for comparison
    Y_cpu_f = Y_cpu.to(torch.float32)
    Y_mps_f = Y_mps.to(torch.float32)

    diff = (Y_mps_f - Y_cpu_f).abs()
    if diff.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    denom = Y_cpu_f.abs().clamp_min(1e-12)
    rel = diff / denom
    max_rel_idx = int(rel.argmax().item())
    max_rel = float(rel.flatten()[max_rel_idx].item())
    max_abs = float(diff.max().item())
    denom_at_argmax = float(Y_cpu_f.abs().flatten()[max_rel_idx].item())

    atol, rtol = compute_tolerance(dtype, k_dim=K, device_type="mps")
    return max_abs, max_rel, atol, rtol, denom_at_argmax


def classify(max_rel: float, atol: float, rtol: float, denom_at_argmax: float,
             repro_passes: int) -> str:
    """Return one of OK | TOLERANCE_RECALIBRATION | FILABLE."""
    # Combined gate: pass if max_rel<=rtol effectively (tol = rtol since denom*rtol covers it)
    if max_rel <= rtol:
        return "OK"
    # Exceeds tolerance — apply FILABLE filter
    if (max_rel > 10 * rtol
            and denom_at_argmax >= 1e-6
            and repro_passes >= 3):
        return "FILABLE"
    return "TOLERANCE_RECALIBRATION"


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES_WEIGHTED)
    stride = rng.choice(STRIDE_PATTERNS)
    M = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)
    N = pick_dim(bucket, rng)
    seed = rng.randrange(1, 1 << 30)

    base = IterResult(
        idx=idx, shape_bucket=bucket, M=M, K=K, N=N, dtype=dtype, stride=stride,
        seed=seed, status="ERROR",
    )

    if M == 0 or K == 0 or N == 0:
        base.status = "SKIP"
        base.note = "zero-dim"
        return base

    try:
        max_abs, max_rel, atol, rtol, denom = run_pair(M, K, N, dtype, stride, seed)
        base.max_abs_err = max_abs
        base.max_rel_err = max_rel
        base.atol = atol
        base.rtol = rtol
        base.denom_at_argmax = denom
        # Pre-classify with 1 seed of evidence (will be upgraded after repro pass)
        if max_rel <= rtol:
            base.status = "OK"
        else:
            # Defer FILABLE/TOLERANCE_RECALIBRATION to post-repro stage
            base.status = "TOLERANCE_RECALIBRATION"
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e)[:200]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage",
                                  "is not currently supported", "MPS does not support")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001 — keep fuzz loop alive
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def reproducibility_pass(r: IterResult, repro_rng: random.Random) -> None:
    """Re-run r with REPRO_SEEDS_PER_CANDIDATE additional seeds and update status."""
    if r.max_rel_err is None or r.rtol is None:
        return
    passes = 1  # initial seed already over-tolerance
    for _ in range(REPRO_SEEDS_PER_CANDIDATE):
        new_seed = repro_rng.randrange(1, 1 << 30)
        try:
            _, max_rel2, _, rtol2, denom2 = run_pair(
                r.M, r.K, r.N, r.dtype, r.stride, new_seed,
            )
        except Exception:
            continue
        # Reproduces if it again exceeds the rtol gate at this shape/dtype/stride.
        if max_rel2 > rtol2:
            passes += 1
    r.repro_seeds_tested = 1 + REPRO_SEEDS_PER_CANDIDATE
    r.repro_seeds_passed = passes
    r.status = classify(r.max_rel_err, r.atol or 0.0, r.rtol, r.denom_at_argmax or 0.0, passes)


def write_partial(results: list[IterResult], attempted: int, completed: int,
                  elapsed: float, error_note: str | None = None) -> None:
    """Write partial RESULTS markdown + JSONL line. Idempotent — overwrites."""
    ok = [r for r in results if r.status == "OK"]
    tol = [r for r in results if r.status == "TOLERANCE_RECALIBRATION"]
    filable = [r for r in results if r.status == "FILABLE"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    skipped = [r for r in results if r.status == "SKIP"]

    candidates = sorted(
        [r for r in results if r.max_rel_err is not None],
        key=lambda r: r.max_rel_err or 0.0, reverse=True,
    )
    top3_filable = [r for r in candidates if r.status == "FILABLE"][:3]
    top3_overall = candidates[:3]
    overall_max_rel = candidates[0].max_rel_err if candidates else 0.0
    target = "pytorch/pytorch" if filable else "none"

    md = []
    md.append(f"# matmul-bf16 fuzz results (v2)\n")
    if error_note:
        md.append(f"> **PROCESS ERROR:** {error_note}\n")
    md.append(f"- **kernel:** {KERNEL_NAME}")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **elapsed:** {elapsed:.2f}s")
    md.append(f"- **OK:** {len(ok)}")
    md.append(f"- **FILABLE:** {len(filable)}")
    md.append(f"- **TOLERANCE_RECALIBRATION:** {len(tol)}")
    md.append(f"- **UNSUPPORTED:** {len(unsupported)}")
    md.append(f"- **ERROR:** {len(errors)}")
    md.append(f"- **SKIP:** {len(skipped)}")
    md.append(f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}")
    md.append(f"- **MPS-vs-CUDA-mock max relative error:** N/A (no NVIDIA hardware on host)")
    md.append(f"- **recommended upstream target:** {target}\n")

    md.append("## FILABLE filter\n")
    md.append(
        "An iteration is FILABLE iff: `max_rel_err > 10 * rtol_used` "
        "AND `|Y_cpu_at_argmax| >= 1e-6` AND it reproduces across at least 3 seeds "
        "(1 original + 2 retries). Otherwise classified TOLERANCE_RECALIBRATION (tol exceeded "
        "but doesn't meet filing gate) or OK (within tol).\n",
    )

    md.append("## Top 3 FILABLE repros\n")
    if top3_filable:
        md.append("| # | dtype | stride | M | K | N | shape_bucket | max_rel_err | rtol_used | "
                  "denom@argmax | seeds_passed/tested |")
        md.append("|---|-------|--------|---|---|---|--------------|-------------|-----------|"
                  "--------------|---------------------|")
        for i, r in enumerate(top3_filable, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.M} | {r.K} | {r.N} | {r.shape_bucket} | "
                f"{r.max_rel_err:.3e} | {r.rtol:.3e} | {r.denom_at_argmax:.3e} | "
                f"{r.repro_seeds_passed}/{r.repro_seeds_tested} |",
            )
    else:
        md.append("_No FILABLE divergences observed within budget._")
    md.append("")

    md.append("## Top 3 by max_rel_err (any classification)\n")
    if top3_overall:
        md.append("| # | status | dtype | stride | M | K | N | max_rel_err | rtol_used | "
                  "denom@argmax | seeds |")
        md.append("|---|--------|-------|--------|---|---|---|-------------|-----------|"
                  "--------------|-------|")
        for i, r in enumerate(top3_overall, 1):
            md.append(
                f"| {i} | {r.status} | {r.dtype} | {r.stride} | {r.M} | {r.K} | {r.N} | "
                f"{(r.max_rel_err or 0.0):.3e} | {(r.rtol or 0.0):.3e} | "
                f"{(r.denom_at_argmax or 0.0):.3e} | {r.repro_seeds_passed}/{r.repro_seeds_tested} |",
            )
    else:
        md.append("_No comparisons completed._")
    md.append("")

    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})")
        seen: dict[tuple[str, str], list[IterResult]] = {}
        for r in unsupported:
            seen.setdefault((r.dtype, r.stride), []).append(r)
        for (dt_, st_), rs in seen.items():
            md.append(f"- {dt_} / {st_}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")
    if errors:
        md.append(f"## ERROR summary ({len(errors)})")
        for r in errors[:10]:
            md.append(
                f"- iter {r.idx} {r.dtype}/{r.stride}/{r.shape_bucket} "
                f"M={r.M} K={r.K} N={r.N}: {r.note}",
            )
        md.append("")

    md.append("## Method notes\n")
    md.append("- Inputs constructed as seeded fp32, cast to dtype on CPU, cloned-and-transferred to MPS.")
    md.append("- Reference: CPU matmul at the *same* low precision (so we measure kernel-level "
              "MPS-vs-CPU divergence, not accumulator vs infinite-precision).")
    md.append("- Tolerance: `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='mps')` — "
              "base + sqrt(K/128) scaling + MPS 2× overlay.")
    md.append("- Reproducibility check: each over-tolerance iteration is re-run with 2 additional seeds "
              "(same M/K/N/dtype/stride). FILABLE requires all 3 seeds to exceed tolerance.")
    md.append("- Stride patterns: contiguous, slice (stride-2 over K), transpose (`.t()` view), "
              "non_contig (1-offset slice), broadcast_K (single-row A), broadcast_N (single-col B).")
    md.append("- Dtype mix is bf16-weighted (60%) with fp16/fp32 (20% each) for cross-dtype context.")
    md.append("- CUDA channel: gpucheck arch detection is mockable but kernels cannot run without an "
              "NVIDIA device — CUDA-vs-MPS comparison is N/A on this host.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))


def append_jsonl(results: list[IterResult], attempted: int, completed: int,
                 elapsed: float, error_note: str | None = None) -> None:
    ok = [r for r in results if r.status == "OK"]
    tol = [r for r in results if r.status == "TOLERANCE_RECALIBRATION"]
    filable = [r for r in results if r.status == "FILABLE"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    skipped = [r for r in results if r.status == "SKIP"]

    candidates = sorted(
        [r for r in results if r.max_rel_err is not None],
        key=lambda r: r.max_rel_err or 0.0, reverse=True,
    )
    overall_max_rel = candidates[0].max_rel_err if candidates else 0.0

    record = {
        "kernel": KERNEL_NAME,
        "version": "v2",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "filable": len(filable),
        "tolerance_recalibration": len(tol),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_no_nvidia_hardware",
        "filable_filter": {
            "max_rel_gt": "10 * rtol_used",
            "denom_at_argmax_ge": 1e-6,
            "min_repro_seeds": 3,
        },
        "top3_filable": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride,
                "M": r.M, "K": r.K, "N": r.N,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err,
                "max_abs_err": r.max_abs_err,
                "atol": r.atol, "rtol": r.rtol,
                "denom_at_argmax": r.denom_at_argmax,
                "repro_seeds_passed": r.repro_seeds_passed,
                "repro_seeds_tested": r.repro_seeds_tested,
                "seed": r.seed,
            }
            for i, r in enumerate([r for r in candidates if r.status == "FILABLE"][:3])
        ],
        "top3_overall": [
            {"rank": i + 1, "status": r.status, "dtype": r.dtype, "stride": r.stride,
             "M": r.M, "K": r.K, "N": r.N, "max_rel_err": r.max_rel_err,
             "rtol": r.rtol, "denom_at_argmax": r.denom_at_argmax,
             "repro_seeds_passed": r.repro_seeds_passed,
             "repro_seeds_tested": r.repro_seeds_tested}
            for i, r in enumerate(candidates[:3])
        ],
        "upstream_target": "pytorch/pytorch" if filable else "none",
        "torch_version": torch.__version__,
        "elapsed_s": elapsed,
        "error_note": error_note,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# matmul-bf16 fuzz (v2) — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "version": "v2", "status": "SKIPPED"}) + "\n")
        return 0

    rng = random.Random(0xBF16BF16)
    repro_rng = random.Random(0xC0FFEEC0)
    torch.manual_seed(0xBF16BF16)

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    last_print = t0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters (elapsed {time.monotonic()-t0:.1f}s)", flush=True)
            break
        attempted = i + 1
        r = run_one(i, rng)
        results.append(r)
        completed += 1
        now = time.monotonic()
        if r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.dtype}/{r.stride} M={r.M} K={r.K} N={r.N}: {r.note}", flush=True)
        elif r.status == "TOLERANCE_RECALIBRATION":
            print(f"[{i:03d}] over-tol {r.dtype}/{r.stride} M={r.M} K={r.K} N={r.N} "
                  f"max_rel={r.max_rel_err:.3e} rtol={r.rtol:.3e} denom={r.denom_at_argmax:.3e}", flush=True)
        elif now - last_print > 15:
            print(f"[progress] {i+1}/{N_ITERS} ({now-t0:.1f}s elapsed)", flush=True)
            last_print = now

    # Reproducibility pass on top over-tolerance candidates
    candidates = sorted(
        [r for r in results if r.status == "TOLERANCE_RECALIBRATION"],
        key=lambda r: r.max_rel_err or 0.0, reverse=True,
    )[:MAX_REPRO_CANDIDATES]
    print(f"[repro] running reproducibility pass on {len(candidates)} candidates", flush=True)
    for r in candidates:
        if time.monotonic() - t0 > BUDGET_S + 30:
            print("[repro] budget exceeded — stopping repro pass", flush=True)
            break
        reproducibility_pass(r, repro_rng)
        print(f"[repro] iter {r.idx} {r.dtype}/{r.stride} M={r.M} K={r.K} N={r.N} → "
              f"{r.status} ({r.repro_seeds_passed}/{r.repro_seeds_tested} seeds)", flush=True)

    elapsed = time.monotonic() - t0
    write_partial(results, attempted, completed, elapsed)
    append_jsonl(results, attempted, completed, elapsed)
    print(f"[done] wrote {RESULTS_MD} and appended JSONL ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        # Halt-and-report: write an error stub so the swarm sees a process error
        try:
            with open(RESULTS_MD, "w") as f:
                f.write(
                    "# matmul-bf16 fuzz (v2) — PROCESS ERROR\n\n"
                    f"```\n{traceback.format_exc()}\n```\n",
                )
            with open(SWARM_JSONL, "a") as f:
                f.write(json.dumps({
                    "kernel": KERNEL_NAME, "version": "v2", "status": "PROCESS_ERROR",
                    "traceback": traceback.format_exc().splitlines()[-5:],
                }) + "\n")
        except Exception:
            pass
        sys.exit(2)
